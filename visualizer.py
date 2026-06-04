"""
显微颗粒检测 — 交互式可视化程序 (GPU加速版)

针对 4060Ti 优化: OpenCL UMat 加速 + 外接圆直径分类

核心改进:
  - minEnclosingCircle 外接圆直径判定颗粒真实大小
  - 轮廓合并 (圆心距 < max(r1,r2) × 1.2 → 同一颗粒)
  - 径向强度剖面 (RIP) 验证环状结构 (衍射暗斑+亮环)
  - Laplacian 焦点过滤 (边缘锐度)
  - 严格直径分类: 1μm = 44±3px, 5μm = 166±3px

操作说明:
  ← →    上/下一帧
  Space   播放/暂停
  ↑ ↓    调整播放速度
  P       切换像素尺寸
  B       切换 圆 / 圆+bbox
  C       切换 圆度过滤 开/关
  S       保存当前帧
  G       切换灰度/二值图
  H       帮助
  +/-     调整直径范围
  R       重置直径范围
  Esc/Q   退出
"""
import cv2
import numpy as np
import os
import glob
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ======================================================================
# 检测器
# ======================================================================
@dataclass
class Particle:
    """检测到的颗粒"""
    x: float = 0.0               # 外接圆圆心 x
    y: float = 0.0               # 外接圆圆心 y
    cx: float = 0.0              # 轮廓质心 x (moments)
    cy: float = 0.0              # 轮廓质心 y
    area: float = 0.0            # 轮廓面积 px² (contourArea)
    perimeter: float = 0.0       # 轮廓周长 px (arcLength)
    circularity: float = 0.0     # 圆度 4πA/P² (1.0=正圆)
    enclosing_diameter: float = 0.0  # minEnclosingCircle 直径 px
    fill_ratio: float = 0.0      # 填充率 area / enclosingCircleArea
    ring_contrast: float = 0.0   # 环比中心亮多少 (归一化)
    ring_consistency: float = 0.0 # 环完整度 [0-1]
    outer_drop: float = 0.0      # 环比外圈亮多少 (归一化)
    equiv_diameter_um: float = 0.0
    type: str = "noise"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class Detector:
    """显微镜颗粒检测器 — 外接圆直径分类"""

    def __init__(self, pixel_size_um: float = 0.03):
        # ---- 预处理参数 ----
        self._bg_sigma: float = 50.0
        self._morph_k: int = 5
        self._median_k: int = 3
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_k, self._morph_k))

        # ---- 直径范围 (px) ----
        # 1μm: 44 ± 3 → [41, 47]
        self._diam_1um_lo: int = 41
        self._diam_1um_hi: int = 47
        # 5μm: 166 ± 3 → [163, 169]
        self._diam_5um_lo: int = 163
        self._diam_5um_hi: int = 169

        # ---- 过滤阈值 ----
        self._min_area: float = 20.0
        self._merge_overlap: float = 1.2
        self._fill_small_min: float = 0.06   # r < 25px
        self._fill_normal_min: float = 0.10
        self._circ_5um_min: float = 0.30
        self._ring_consistency_1um: float = 0.35
        self._ring_contrast_min: float = -0.01
        self._ring_consistency_5um: float = 0.10
        self._fill_5um_min: float = 0.20

        # ---- 焦点过滤 (Laplacian方差阈值) ----
        self._focus_r20: float = 8.0
        self._focus_r30: float = 12.0
        self._focus_r60: float = 6.0
        self._focus_rbig: float = 3.0

        # ---- 运行时状态 ----
        self._use_circularity_filter = True
        self._gpu = cv2.ocl.haveOpenCL()
        self.pixel_size_um = pixel_size_um
        self.stats = {}

    # ---- 属性 ----
    @property
    def use_circularity(self):
        return self._use_circularity_filter

    @use_circularity.setter
    def use_circularity(self, v: bool):
        self._use_circularity_filter = v

    def set_pixel_size(self, um_per_px: float):
        self.pixel_size_um = um_per_px

    def set_diameter_range(self, which: str, lo: int, hi: int):
        """设置外接圆直径范围"""
        if which == "1um":
            self._diam_1um_lo = lo
            self._diam_1um_hi = hi
        else:
            self._diam_5um_lo = lo
            self._diam_5um_hi = hi

    def get_diameter_range(self, which: str) -> Tuple[int, int]:
        if which == "1um":
            return (self._diam_1um_lo, self._diam_1um_hi)
        return (self._diam_5um_lo, self._diam_5um_hi)

    # ---- 预处理管线 ----
    def _preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """背景校正 → 中值滤波 → OTSU → 形态学闭运算"""
        h, w = gray.shape

        # 1. 背景校正 (8x降采样 GPU)
        src = cv2.UMat(gray) if self._gpu else gray
        small = cv2.resize(src, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR)
        s_sigma = self._bg_sigma / 8.0
        small = cv2.GaussianBlur(small, (0, 0), s_sigma, s_sigma,
                                 borderType=cv2.BORDER_REPLICATE)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        corr = cv2.subtract(src, bg, dtype=cv2.CV_16S)
        corr = cv2.add(corr, 128, dtype=cv2.CV_16S)
        corrected = corr.get() if self._gpu else corr
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # 2. MedianBlur
        blurred = cv2.medianBlur(corrected, self._median_k)

        # 3. OTSU 双模投票
        _, bin1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        r1 = cv2.countNonZero(bin1) / bin1.size
        r2 = cv2.countNonZero(bin2) / bin2.size
        use_inv = not (r1 > 0.5 or (r1 < 1e-4 and r2 > 1e-4))
        binary = bin1 if use_inv else bin2

        # 4. Morph close
        if self._gpu:
            _tmp = cv2.UMat(binary)
            _tmp = cv2.morphologyEx(_tmp, cv2.MORPH_CLOSE, self._morph_kernel)
            binary = _tmp.get()
        else:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)

        return binary, corrected

    # ---- 焦点检查 ----
    def _check_focus(self, gray: np.ndarray, cx: float, cy: float,
                     radius: float) -> bool:
        """Laplacian 方差 — 离焦颗粒响应低"""
        h, w = gray.shape
        r = int(radius)
        margin = r + 5
        x1 = max(0, int(cx) - margin)
        y1 = max(0, int(cy) - margin)
        x2 = min(w, int(cx) + margin)
        y2 = min(h, int(cy) + margin)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return True
        patch = gray[y1:y2, x1:x2]
        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()

        if r < 20:
            return lap_var >= self._focus_r20
        elif r < 30:
            return lap_var >= self._focus_r30
        elif r < 60:
            return lap_var >= self._focus_r60
        else:
            return lap_var >= self._focus_rbig

    # ---- 环状结构检测 ----
    def _validate_ring(self, gray: np.ndarray, cx: float, cy: float,
                       radius: float) -> Tuple[float, float, float]:
        """径向强度剖面: 3个同心环各采样24点, 验证暗斑+亮环结构"""
        h, w = gray.shape
        r = int(radius)
        n_samples = 24

        def sample_ring(frac: float) -> np.ndarray:
            rd = max(2, int(r * frac))
            pts = []
            for i in range(n_samples):
                angle = 2.0 * np.pi * i / n_samples
                px = int(cx + rd * np.cos(angle))
                py = int(cy + rd * np.sin(angle))
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                pts.append(float(gray[py, px]))
            return np.array(pts)

        center_vals = sample_ring(0.30)   # 内圈 (暗斑中心)
        ring_vals = sample_ring(0.75)     # 亮环
        outer_vals = sample_ring(1.10)    # 外圈

        mc = np.median(center_vals)
        mr = np.median(ring_vals)
        mo = np.median(outer_vals)

        ring_contrast = (mr - mc) / 255.0
        ring_consistency = float(np.mean(ring_vals > mc))
        outer_drop = (mr - mo) / 255.0

        return ring_contrast, ring_consistency, outer_drop

    # ---- 轮廓合并 ----
    def _merge_overlapping(self, particles: List[Particle]) -> List[Particle]:
        """单次贪心合并: 圆心距 < max(r1,r2) × 1.2 → 同一颗粒"""
        if len(particles) <= 1:
            return particles

        particles.sort(key=lambda p: p.enclosing_diameter, reverse=True)
        merged = []

        for p in particles:
            found = False
            for m in merged:
                dist = np.sqrt((p.x - m.x)**2 + (p.y - m.y)**2)
                max_r = max(p.enclosing_diameter / 2.0, m.enclosing_diameter / 2.0)
                if dist < max_r * self._merge_overlap:
                    # 加权平均圆心 + 最大外接圆
                    total_a = p.area + m.area
                    if total_a > 0:
                        m.x = (p.x * p.area + m.x * m.area) / total_a
                        m.y = (p.y * p.area + m.y * m.area) / total_a
                    if p.enclosing_diameter > m.enclosing_diameter:
                        m.enclosing_diameter = p.enclosing_diameter
                    m.area += p.area
                    found = True
                    break
            if not found:
                merged.append(p)

        return merged

    # ---- 分类 ----
    def _classify(self, p: Particle):
        d = p.enclosing_diameter
        circ = p.circularity
        rc = p.ring_consistency
        rcontrast = p.ring_contrast
        fill = p.fill_ratio

        if d < 20:
            p.type = "noise"
            return

        # -- 1μm: 外接圆直径在设定范围内 --
        if self._diam_1um_lo <= d <= self._diam_1um_hi:
            if rc >= self._ring_consistency_1um or rcontrast > self._ring_contrast_min:
                p.type = "1um_particle"
            else:
                p.type = "noise_rejected"
        # -- 5μm: 外接圆直径在设定范围内 --
        elif self._diam_5um_lo <= d <= self._diam_5um_hi:
            if (not self._use_circularity_filter) or (
                circ >= self._circ_5um_min and
                (rc >= self._ring_consistency_5um or fill >= self._fill_5um_min)
            ):
                p.type = "5um_particle"
            else:
                p.type = "noise_rejected"
        elif d > self._diam_5um_hi:
            if circ >= 0.50:
                p.type = "large_particle"
            else:
                p.type = "noise_rejected"
        else:
            # 不在 1μm 也不在 5μm 范围 → 丢弃
            p.type = "noise_rejected"

    # ---- 检测主入口 ----
    def detect(self, gray: np.ndarray) -> Tuple[List[Particle], np.ndarray]:
        t0 = time.perf_counter()
        h, w = gray.shape

        # 预处理
        binary, _corrected = self._preprocess(gray)

        # findContours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particles_raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity > 1.0:
                circularity = 1.0

            # minEnclosingCircle
            (ec_x, ec_y), ec_r = cv2.minEnclosingCircle(cnt)
            ec_d = ec_r * 2.0
            ec_area = np.pi * ec_r * ec_r
            fill_ratio = area / ec_area if ec_area > 0 else 0.0

            # 填充率过滤
            if ec_r < 25:
                if fill_ratio < self._fill_small_min:
                    continue
            else:
                if fill_ratio < self._fill_normal_min:
                    continue

            # 焦点过滤
            if not self._check_focus(gray, ec_x, ec_y, ec_r):
                continue

            # 轮廓矩质心
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx_m = M['m10'] / M['m00']
                cy_m = M['m01'] / M['m00']
            else:
                cx_m, cy_m = ec_x, ec_y

            bx, by, bw, bh = cv2.boundingRect(cnt)

            p = Particle(
                x=ec_x, y=ec_y,
                cx=cx_m, cy=cy_m,
                area=float(area),
                perimeter=float(perimeter),
                circularity=float(circularity),
                enclosing_diameter=ec_d,
                fill_ratio=fill_ratio,
                bbox=(bx, by, bw, bh),
            )
            particles_raw.append(p)

        # 合并重叠轮廓
        particles = self._merge_overlapping(particles_raw)

        # 环状验证 + 分类
        c1um = c5um = clarge = creject = 0
        final_particles = []
        for p in particles:
            # 径向强度剖面
            p.ring_contrast, p.ring_consistency, p.outer_drop = \
                self._validate_ring(gray, p.x, p.y, p.enclosing_diameter / 2.0)
            # 等效直径
            p.equiv_diameter_um = p.enclosing_diameter * self.pixel_size_um

            self._classify(p)

            if p.type == "1um_particle":
                c1um += 1; final_particles.append(p)
            elif p.type == "5um_particle":
                c5um += 1; final_particles.append(p)
            elif p.type == "large_particle":
                clarge += 1; final_particles.append(p)
            elif p.type == "noise_rejected":
                creject += 1

        t1 = time.perf_counter()
        self.stats = {
            "candidates": len(contours),
            "raw": len(particles_raw),
            "merged": len(particles),
            "1um": c1um, "5um": c5um,
            "large": clarge, "noise_rejected": creject,
            "elapsed_ms": (t1 - t0) * 1000,
        }
        return final_particles, binary


# ======================================================================
# 可视化渲染
# ======================================================================
COLORS = {
    "1um_particle":   (0, 255, 0),     # 绿 — 1μm
    "5um_particle":   (0, 165, 255),   # 橙 — 5μm
    "large_particle": (0, 0, 255),     # 红 — 超大
}


def draw_overlay(vis: np.ndarray, particles: List[Particle],
                 det: Detector, frame_idx: int, total: int,
                 show_help: bool, show_binary: bool, show_bbox: bool,
                 binary: Optional[np.ndarray] = None,
                 speed: float = 1.0):
    h, w = vis.shape[:2]

    if show_binary and binary is not None:
        vis[:] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for p in particles:
        if p.type not in ("1um_particle", "5um_particle", "large_particle"):
            continue

        color = COLORS.get(p.type, (0, 0, 255))
        cx, cy = int(round(p.x)), int(round(p.y))
        r = int(round(p.enclosing_diameter / 2.0))

        # 外接圆标注
        thickness = 2 if p.type == "5um_particle" else 1
        cv2.circle(vis, (cx, cy), max(r, 1), color, thickness)

        # bbox
        if show_bbox:
            bx, by, bw, bh = p.bbox
            cv2.rectangle(vis, (bx, by, bx + bw, by + bh), color, 1)

        # 质心十字
        cross = 4
        cv2.line(vis, (cx - cross, cy), (cx + cross, cy), color, 1)
        cv2.line(vis, (cx, cy - cross), (cx, cy + cross), color, 1)

        # 标签
        tag = "1um" if p.type == "1um_particle" else ("5um" if p.type == "5um_particle" else ">5um")
        cv2.putText(vis, tag, (cx + r + 3, cy - r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # ---- HUD ----
    panel_x = w - 290
    panel_h = 340
    cv2.rectangle(vis, (panel_x - 10, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (panel_x - 10, 0), (w, panel_h), (80, 80, 80), 1)
    cv2.addWeighted(vis[0:panel_h, panel_x - 10:w], 0.5,
                    np.full((panel_h, 300, 3), 0, dtype=np.uint8), 0.5,
                    0, vis[0:panel_h, panel_x - 10:w])

    def put(x, y, text, color=(255, 255, 255), scale=0.42):
        cv2.putText(vis, text, (panel_x + x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    s = det.stats
    d1_lo, d1_hi = det.get_diameter_range("1um")
    d5_lo, d5_hi = det.get_diameter_range("5um")

    put(5, 18,  f"Frame: {frame_idx+1}/{total}", (200, 200, 200))
    put(5, 42,  f"1um (green):    {s['1um']}", COLORS["1um_particle"], 0.55)
    put(5, 68,  f"5um (orange):   {s['5um']}", COLORS["5um_particle"], 0.55)
    put(5, 94,  f">5um (red):     {s.get('large', 0)}", COLORS.get("large_particle", (0,0,255)), 0.45)
    put(5, 116, f"Rejected:       {s.get('noise_rejected', 0)}", (100, 100, 100), 0.4)
    total_p = s['1um'] + s['5um'] + s.get('large', 0)
    put(5, 142, f"Total shown:    {total_p}", (255, 255, 255), 0.5)

    put(5, 168, f"GPU (OpenCL):   {'ON' if det._gpu else 'OFF'}",
        (0, 255, 200) if det._gpu else (128, 128, 128), 0.42)
    put(5, 190, f"Circ. filter:   {'ON' if det._use_circularity_filter else 'OFF'}",
        (200, 200, 0) if det._use_circularity_filter else (128, 128, 128), 0.42)
    put(5, 214, f"Pixel: {det.pixel_size_um:.3f} um/px", (180, 200, 255), 0.42)
    put(5, 234, f"1um diam: [{d1_lo}, {d1_hi}] px", (160, 200, 160), 0.38)
    put(5, 252, f"5um diam: [{d5_lo}, {d5_hi}] px", (160, 160, 200), 0.38)
    put(5, 274, f"Speed: {speed:.1f}x", (200, 200, 200), 0.42)

    status = "PLAYING" if speed > 0 else "PAUSED"
    clr = (0, 255, 0) if speed > 0 else (100, 100, 255)
    put(5, 298, status, clr, 0.5)

    # 底部状态栏
    cv2.rectangle(vis, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, h - 28), (w, h), (60, 60, 60), 1)
    cv2.putText(vis,
                "<- -> nav  Space:play  G:binary  B:bbox  C:circ-filter  P:pixel  +/-:diam  R:reset  S:save  H:help  Q:quit",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1, cv2.LINE_AA)

    if show_help:
        help_lines = [
            "KEYS:",
            "  <- ->     Prev / Next frame",
            "  Space     Play / Pause",
            "  Up/Down   Adjust speed",
            "  P         Cycle pixel size",
            "  C         Toggle circularity filter",
            "  G         Toggle binary view",
            "  B         Toggle bbox overlay",
            "  +/-       Adjust diameter range (+-1px)",
            "  S         Save current frame",
            "  E         Batch export all frames",
            "  R         Reset diameter ranges",
            "  H         Hide this help",
            "  Esc/Q     Quit",
        ]
        box_h = len(help_lines) * 20 + 20
        cv2.rectangle(vis, (10, 10), (420, 10 + box_h), (0, 0, 0, 200), -1)
        cv2.rectangle(vis, (10, 10), (420, 10 + box_h), (100, 100, 100), 1)
        for i, line in enumerate(help_lines):
            cv2.putText(vis, line, (20, 38 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ======================================================================
# 主程序
# ======================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="显微颗粒检测 — GPU加速可视化")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--pixel-size", type=float, default=0.03)
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-gpu", action="store_true", help="禁用GPU加速")
    parser.add_argument("--no-circ-filter", action="store_true", help="禁用圆度过滤")
    args = parser.parse_args()

    img_dir = args.input
    if img_dir is None:
        candidates = []
        for entry in os.listdir("."):
            if os.path.isdir(entry):
                bmps = glob.glob(os.path.join(entry, "*.bmp"))
                if bmps:
                    candidates.append((len(bmps), entry))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            img_dir = candidates[0][1]
            print(f"Auto-selected: {img_dir} ({candidates[0][0]} images)")
        else:
            print("No BMP images found!"); return

    files = sorted(glob.glob(os.path.join(img_dir, "*.bmp")) +
                   glob.glob(os.path.join(img_dir, "*.png")) +
                   glob.glob(os.path.join(img_dir, "*.jpg")))
    files = [f for f in files if os.path.dirname(f) == img_dir]

    if not files:
        print(f"No images in {img_dir}"); return

    print(f"Loaded {len(files)} images")
    first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    print(f"Size: {first.shape[1]}x{first.shape[0]}")

    det = Detector(args.pixel_size)
    if args.no_gpu:
        det._gpu = False
    if args.no_circ_filter:
        det._use_circularity_filter = False

    d1_lo, d1_hi = det.get_diameter_range("1um")
    d5_lo, d5_hi = det.get_diameter_range("5um")

    print(f"OpenCL GPU : {'ENABLED' if det._gpu else 'DISABLED'}")
    print(f"Circ filter: {'ON' if det._use_circularity_filter else 'OFF'}")
    print(f"Pixel size : {det.pixel_size_um:.3f} um/px")
    print(f"1um diam   : [{d1_lo}, {d1_hi}] px")
    print(f"5um diam   : [{d5_lo}, {d5_hi}] px")
    print("Controls: H=help  Space=play  <- -> =navigate  Q=quit")

    os.makedirs(args.output, exist_ok=True)
    idx, speed = 0, args.speed
    show_help, show_binary, show_bbox = False, False, False
    saved = set()
    fps_smooth = 0.0

    cv2.namedWindow("Particle Visualizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Particle Visualizer", 1224, 1024)

    print("Preloading images...")
    grays = []
    for i, f in enumerate(files):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        grays.append(g if g is not None else np.zeros((100, 100), dtype=np.uint8))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"Preloaded {len(grays)} images.")

    while True:
        gray = grays[idx]
        if gray.size < 1000:
            idx = (idx + 1) % len(files); continue

        t0 = time.perf_counter()
        particles, binary = det.detect(gray)
        t1 = time.perf_counter()
        det.stats["elapsed_ms"] = (t1 - t0) * 1000
        det.stats["fps_instant"] = 1.0 / max(t1 - t0, 0.001)
        alpha = 0.3
        fps_smooth = alpha * det.stats["fps_instant"] + (1 - alpha) * fps_smooth

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        draw_overlay(vis, particles, det, idx, len(files),
                     show_help, show_binary, show_bbox, binary, speed)

        cv2.putText(vis, f"Detect: {det.stats['elapsed_ms']:.1f}ms ({fps_smooth:.0f} FPS)",
                    (10, vis.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Particle Visualizer", vis)

        delay = max(1, int(50 / max(speed, 0.1))) if speed > 0 else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('g'):
            show_binary = not show_binary
        elif key == ord('b'):
            show_bbox = not show_bbox
        elif key == ord('c'):
            det._use_circularity_filter = not det._use_circularity_filter
            print(f"Circularity filter: {'ON' if det._use_circularity_filter else 'OFF'}")
        elif key == ord(' '):
            speed = 0.0 if speed > 0 else 1.0
        elif key == 81 or key == 2424832:  # Left
            idx = (idx - 1) % len(files)
            if speed > 0: speed = 0
        elif key == 83 or key == 2555904:  # Right
            idx = (idx + 1) % len(files)
            if speed > 0: speed = 0
        elif key == 82 or key == 2490368:  # Up
            speed = min(speed + 0.5, 10.0)
        elif key == 84 or key == 2621440:  # Down
            speed = max(speed - 0.5, 0.5)
        elif key == ord('p'):
            sizes = [0.03, 0.05, 0.10, 0.02, 0.01]
            cur = det.pixel_size_um
            nxt = sizes[0]
            for s in sizes:
                if abs(s - cur) < 0.001:
                    nxt = sizes[(sizes.index(s) + 1) % len(sizes)]
                    break
            det.set_pixel_size(nxt)
            d1_lo, d1_hi = det.get_diameter_range("1um")
            d5_lo, d5_hi = det.get_diameter_range("5um")
            print(f"Pixel: {det.pixel_size_um:.3f}  1um:[{d1_lo},{d1_hi}]  5um:[{d5_lo},{d5_hi}]")
        elif key == ord('+') or key == ord('='):
            d1_lo, d1_hi = det.get_diameter_range("1um")
            d5_lo, d5_hi = det.get_diameter_range("5um")
            det.set_diameter_range("1um", d1_lo - 1, d1_hi + 1)
            det.set_diameter_range("5um", d5_lo - 1, d5_hi + 1)
            nl, nh = det.get_diameter_range("1um")
            print(f"1um diam: [{nl},{nh}]  5um diam: [{det.get_diameter_range('5um')[0]},{det.get_diameter_range('5um')[1]}]")
        elif key == ord('-'):
            d1_lo, d1_hi = det.get_diameter_range("1um")
            d5_lo, d5_hi = det.get_diameter_range("5um")
            det.set_diameter_range("1um", max(20, d1_lo + 1), max(22, d1_hi - 1))
            det.set_diameter_range("5um", max(50, d5_lo + 1), max(52, d5_hi - 1))
            print(f"1um diam: [{det.get_diameter_range('1um')[0]},{det.get_diameter_range('1um')[1]}]  "
                  f"5um diam: [{det.get_diameter_range('5um')[0]},{det.get_diameter_range('5um')[1]}]")
        elif key == ord('r'):
            det.set_diameter_range("1um", 41, 47)
            det.set_diameter_range("5um", 163, 169)
            print("Diameter ranges reset: 1um [41,47]  5um [163,169]")
        elif key == ord('s'):
            fname = os.path.basename(files[idx])
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(args.output, f"{stem}_detected.png")
            cv2.imwrite(out_path, vis)
            saved.add(idx)
            print(f"Saved: {out_path}")
        elif key == ord('e'):
            print(f"Exporting all {len(files)} frames...")
            for i, f in enumerate(files):
                g = grays[i]
                parts, _ = det.detect(g)
                v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                draw_overlay(v, parts, det, i, len(files), False, False, False, None, 0)
                cv2.putText(v, f"Frame {i+1}", (10, v.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                name = os.path.splitext(os.path.basename(f))[0] + "_detected.png"
                cv2.imwrite(os.path.join(args.output, name), v)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(files)}")
            print("Export done.")

        if speed > 0:
            idx = (idx + 1) % len(files)

    cv2.destroyAllWindows()
    print(f"Exited. Saved {len(saved)} frames to {args.output}/")


if __name__ == "__main__":
    main()
