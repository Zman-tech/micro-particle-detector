"""
显微镜颗粒检测 — 参数校准工具

手动调整预处理参数 (背景校正σ/形态学核/中值滤波)
和检测阈值 (外接圆直径范围), 实时查看各阶段效果。

Usage:
  python calibrate.py                          # Trackbar 交互模式
  python calibrate.py --input ./20260415_165254
  python calibrate.py --sigma 50 --morph 5     # 指定参数启动
  python calibrate.py --batch                  # 批量处理, 输出统计

操作 (Trackbar 模式):
  Trackbars    实时调参
  N / P        上一张 / 下一张
  S            保存当前帧到 results/
  V            打印当前参数
  B            批量测试 (全部图片, 输出统计)
  Q / Esc      退出
"""

import cv2
import numpy as np
import os
import glob
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ======================================================================
# 默认参数 (当前 visualizer.py 实测最佳值)
# ======================================================================
class Params:
    """预处理 & 检测参数集合 — 从 visualizer.py 提取并保存"""
    def __init__(self):
        # ---- 预处理 ----
        self.bg_sigma: float = 50.0       # 背景校正高斯 σ (8×降采样加速)
        self.morph_k: int = 5             # 形态学闭运算核大小 (ELLIPSE)
        self.median_k: int = 3            # 中值滤波核大小

        # ---- 检测阈值 ----
        self.diam_1um_lo: int = 41        # 1μm 外接圆直径下限 (44 ± 3)
        self.diam_1um_hi: int = 47        # 1μm 外接圆直径上限
        self.diam_5um_lo: int = 163       # 5μm 外接圆直径下限 (166 ± 3)
        self.diam_5um_hi: int = 169       # 5μm 外接圆直径上限
        self.min_area: float = 20.0       # 最小轮廓面积 (噪点过滤)
        self.merge_overlap: float = 1.2   # 轮廓合并: 圆心距 < max(r1,r2) × factor

        # ---- 填充率 ----
        self.fill_small_min: float = 0.06  # 小颗粒 (r < 25px) 最低填充率
        self.fill_normal_min: float = 0.10 # 正常颗粒最低填充率

        # ---- 环状结构验证 ----
        self.ring_consistency_1um: float = 0.35
        self.ring_contrast_min: float = -0.01
        self.circ_5um_min: float = 0.30
        self.ring_consistency_5um: float = 0.10
        self.fill_5um_min: float = 0.20

        # ---- 焦点过滤 (Laplacian方差) ----
        self.focus_r20: float = 8.0
        self.focus_r30: float = 12.0
        self.focus_r60: float = 6.0
        self.focus_rbig: float = 3.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def print(self):
        print("=== Current Parameters ===")
        for k, v in self.to_dict().items():
            print(f"  {k}: {v}")
        print()


# ======================================================================
# 检测器 (与 visualizer.py 共享逻辑, 独立副本便于调参)
# ======================================================================
@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    cx: float = 0.0              # 轮廓质心 (moments)
    cy: float = 0.0
    area: float = 0.0
    perimeter: float = 0.0
    circularity: float = 0.0
    enclosing_diameter: float = 0.0
    fill_ratio: float = 0.0
    ring_contrast: float = 0.0
    ring_consistency: float = 0.0
    outer_drop: float = 0.0
    type: str = "noise"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class Detector:
    def __init__(self, params: Params):
        self.p = params
        self._gpu = cv2.ocl.haveOpenCL()
        self._update_morph_kernel()
        self.stats = {}

    def _update_morph_kernel(self):
        k = self.p.morph_k
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _check_focus(self, gray: np.ndarray, cx: float, cy: float,
                     radius: float) -> bool:
        """Laplacian 方差焦点检查"""
        h, w = gray.shape
        r = int(radius)
        margin = r + 5
        x1 = max(0, int(cx) - margin)
        y1 = max(0, int(cy) - margin)
        x2 = min(w, int(cx) + margin)
        y2 = min(h, int(cy) + margin)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return True  # 太小, 不判断
        patch = gray[y1:y2, x1:x2]
        lap = cv2.Laplacian(patch, cv2.CV_64F)
        lap_var = lap.var()

        if r < 20:
            return lap_var >= self.p.focus_r20
        elif r < 30:
            return lap_var >= self.p.focus_r30
        elif r < 60:
            return lap_var >= self.p.focus_r60
        else:
            return lap_var >= self.p.focus_rbig

    def _validate_ring(self, gray: np.ndarray, cx: float, cy: float,
                       radius: float) -> Tuple[float, float, float]:
        """径向强度剖面: 检测衍射暗斑 + 亮环结构"""
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

        center_vals = sample_ring(0.30)
        ring_vals = sample_ring(0.75)
        outer_vals = sample_ring(1.10)

        mc = np.median(center_vals)
        mr = np.median(ring_vals)
        mo = np.median(outer_vals)

        ring_contrast = (mr - mc) / 255.0
        ring_consistency = float(np.mean(ring_vals > mc))
        outer_drop = (mr - mo) / 255.0

        return ring_contrast, ring_consistency, outer_drop

    def _merge_overlapping(self, particles: List[Particle]) -> List[Particle]:
        """单次贪心合并: 圆心距 < max(r1,r2) × factor → 同一颗粒"""
        if len(particles) <= 1:
            return particles

        # 按外接圆直径降序
        particles.sort(key=lambda p: p.enclosing_diameter, reverse=True)
        merged = []

        for p in particles:
            found = False
            for i, m in enumerate(merged):
                dist = np.sqrt((p.x - m.x)**2 + (p.y - m.y)**2)
                max_r = max(p.enclosing_diameter / 2.0, m.enclosing_diameter / 2.0)
                if dist < max_r * self.p.merge_overlap:
                    # 合并: 加权平均圆心 + 取最大外接圆
                    total_area = p.area + m.area
                    if total_area > 0:
                        m.x = (p.x * p.area + m.x * m.area) / total_area
                        m.y = (p.y * p.area + m.y * m.area) / total_area
                    if p.enclosing_diameter > m.enclosing_diameter:
                        m.enclosing_diameter = p.enclosing_diameter
                    m.area += p.area
                    found = True
                    break
            if not found:
                merged.append(p)

        return merged

    def _classify(self, p: Particle):
        d = p.enclosing_diameter
        circ = p.circularity
        rc = p.ring_consistency
        rcontrast = p.ring_contrast
        fill = p.fill_ratio

        if d < 20:
            p.type = "noise"; return

        # -- 1μm: 外接圆直径 41-47 px --
        if self.p.diam_1um_lo <= d <= self.p.diam_1um_hi:
            if rc >= self.p.ring_consistency_1um or rcontrast > self.p.ring_contrast_min:
                p.type = "1um_particle"
            else:
                p.type = "noise_rejected"
        # -- 5μm: 外接圆直径 163-169 px --
        elif self.p.diam_5um_lo <= d <= self.p.diam_5um_hi:
            if circ >= self.p.circ_5um_min and (rc >= self.p.ring_consistency_5um or fill >= self.p.fill_5um_min):
                p.type = "5um_particle"
            else:
                p.type = "noise_rejected"
        elif d > self.p.diam_5um_hi:
            if circ >= 0.50:
                p.type = "large_particle"
            else:
                p.type = "noise_rejected"
        else:
            p.type = "noise_rejected"

    def detect(self, gray: np.ndarray) -> Tuple[List[Particle], np.ndarray,
                                                  np.ndarray, np.ndarray]:
        """返回: particles, binary, corrected, blurred"""
        t0 = time.perf_counter()
        h, w = gray.shape

        # ---- 1. 背景校正 ----
        src = cv2.UMat(gray) if self._gpu else gray
        small = cv2.resize(src, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR)
        s_sigma = self.p.bg_sigma / 8.0
        small = cv2.GaussianBlur(small, (0, 0), s_sigma, s_sigma,
                                 borderType=cv2.BORDER_REPLICATE)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        corr = cv2.subtract(src, bg, dtype=cv2.CV_16S)
        corr = cv2.add(corr, 128, dtype=cv2.CV_16S)
        corrected = corr.get() if self._gpu else corr
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # ---- 2. MedianBlur ----
        blurred = cv2.medianBlur(corrected, self.p.median_k)

        # ---- 3. OTSU ----
        _, bin1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        r1 = cv2.countNonZero(bin1) / bin1.size
        r2 = cv2.countNonZero(bin2) / bin2.size
        use_inv = not (r1 > 0.5 or (r1 < 1e-4 and r2 > 1e-4))
        binary = bin1 if use_inv else bin2

        # ---- 4. Morph close ----
        if self._gpu:
            _tmp = cv2.UMat(binary)
            _tmp = cv2.morphologyEx(_tmp, cv2.MORPH_CLOSE, self._morph_kernel)
            binary = _tmp.get()
        else:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)

        # ---- 5. findContours + minEnclosingCircle ----
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particles_raw = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.p.min_area:
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

            # Fill ratio
            if ec_r < 25:
                if fill_ratio < self.p.fill_small_min:
                    continue
            else:
                if fill_ratio < self.p.fill_normal_min:
                    continue

            # Focus check
            if not self._check_focus(gray, ec_x, ec_y, ec_r):
                continue

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

        # ---- 6. Merge overlapping ----
        particles = self._merge_overlapping(particles_raw)

        # ---- 7. Ring validation + classify ----
        c1um = c5um = clarge = creject = cnoise = 0
        final_particles = []
        for p in particles:
            p.ring_contrast, p.ring_consistency, p.outer_drop = \
                self._validate_ring(gray, p.x, p.y, p.enclosing_diameter / 2.0)
            self._classify(p)

            if p.type == "1um_particle":
                c1um += 1; final_particles.append(p)
            elif p.type == "5um_particle":
                c5um += 1; final_particles.append(p)
            elif p.type == "large_particle":
                clarge += 1; final_particles.append(p)
            elif p.type == "noise_rejected":
                creject += 1
            else:
                cnoise += 1

        t1 = time.perf_counter()
        self.stats = {
            "candidates": len(contours),
            "raw_particles": len(particles_raw),
            "merged": len(particles),
            "1um": c1um, "5um": c5um,
            "large": clarge, "noise_rejected": creject,
            "elapsed_ms": (t1 - t0) * 1000,
        }
        return final_particles, binary, corrected, blurred


# ======================================================================
# 可视化
# ======================================================================
COLORS = {
    "1um_particle":   (0, 255, 0),
    "5um_particle":   (0, 165, 255),
    "large_particle": (0, 0, 255),
}


def make_panel(img: np.ndarray, title: str, target_w: int, target_h: int) -> np.ndarray:
    """将灰度图转为 BGR 并缩放到目标尺寸, 附加标题"""
    if len(img.shape) == 2:
        panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        panel = img.copy()
    panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def draw_result(gray: np.ndarray, particles: List[Particle],
                params: Params, target_w: int, target_h: int) -> np.ndarray:
    """在灰度图上绘制检测结果"""
    scale_x = target_w / gray.shape[1]
    scale_y = target_h / gray.shape[0]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    for p in particles:
        if p.type not in ("1um_particle", "5um_particle", "large_particle"):
            continue
        color = COLORS.get(p.type, (0, 0, 255))
        cx = int(round(p.x * scale_x))
        cy = int(round(p.y * scale_y))
        r = int(round(p.enclosing_diameter / 2.0 * min(scale_x, scale_y)))
        thickness = 2 if p.type == "5um_particle" else 1
        cv2.circle(vis, (cx, cy), max(r, 1), color, thickness)
        # 十字
        cross = 3
        cv2.line(vis, (cx - cross, cy), (cx + cross, cy), color, 1)
        cv2.line(vis, (cx, cy - cross), (cx, cy + cross), color, 1)

    # HUD
    s = params
    lines = [
        f"1um: {sum(1 for p in particles if p.type=='1um_particle')}",
        f"5um: {sum(1 for p in particles if p.type=='5um_particle')}",
        f">5um: {sum(1 for p in particles if p.type=='large_particle')}",
        f"D1:[{s.diam_1um_lo},{s.diam_1um_hi}] D5:[{s.diam_5um_lo},{s.diam_5um_hi}]",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (5, target_h - 60 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


# ======================================================================
# Trackbar 交互模式
# ======================================================================
class Calibrator:
    def __init__(self, files: List[str], params: Params):
        self.files = files
        self.params = params
        self.idx = 0
        self.detector = Detector(params)
        self._grays = []
        self._preload()

    def _preload(self):
        print(f"Preloading {len(self.files)} images...")
        for i, f in enumerate(self.files):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            self._grays.append(g if g is not None else np.zeros((100, 100), dtype=np.uint8))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(self.files)}")
        print("Done.")

    def _on_change(self, val):
        """Trackbar 回调 — 重新处理当前帧"""
        self.detector._update_morph_kernel()
        self._process_current()

    def _process_current(self):
        gray = self._grays[self.idx]
        if gray.size < 1000:
            return
        self._particles, self._binary, self._corrected, self._blurred = \
            self.detector.detect(gray)
        s = self.detector.stats
        print(f"  Frame {self.idx+1}/{len(self.files)} | "
              f"1um:{s['1um']} 5um:{s['5um']} large:{s['large']} "
              f"rej:{s['noise_rejected']} | {s['elapsed_ms']:.1f}ms")

    def run(self):
        cv2.namedWindow("Parameter Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parameter Calibration", 1224, 900)

        # Trackbars
        cv2.createTrackbar("bg_sigma", "Parameter Calibration",
                           int(self.params.bg_sigma), 100, self._on_change)
        cv2.createTrackbar("morph_k", "Parameter Calibration",
                           self.params.morph_k, 15, self._on_change)
        cv2.createTrackbar("median_k", "Parameter Calibration",
                           self.params.median_k, 15, self._on_change)
        cv2.createTrackbar("D1_lo", "Parameter Calibration",
                           self.params.diam_1um_lo, 80, self._on_change)
        cv2.createTrackbar("D1_hi", "Parameter Calibration",
                           self.params.diam_1um_hi, 80, self._on_change)
        cv2.createTrackbar("D5_lo", "Parameter Calibration",
                           self.params.diam_5um_lo, 200, self._on_change)
        cv2.createTrackbar("D5_hi", "Parameter Calibration",
                           self.params.diam_5um_hi, 200, self._on_change)

        gray = self._grays[self.idx]
        pw, ph = 612, 512  # panel size (half of 1224x1024)
        self._particles, self._binary, self._corrected, self._blurred = \
            [], np.zeros((100, 100), np.uint8), gray, gray
        self._process_current()

        while True:
            # Read trackbars
            self.params.bg_sigma = float(cv2.getTrackbarPos("bg_sigma", "Parameter Calibration"))
            self.params.morph_k = max(1, cv2.getTrackbarPos("morph_k", "Parameter Calibration"))
            self.params.median_k = max(1, cv2.getTrackbarPos("median_k", "Parameter Calibration"))
            self.params.diam_1um_lo = cv2.getTrackbarPos("D1_lo", "Parameter Calibration")
            self.params.diam_1um_hi = cv2.getTrackbarPos("D1_hi", "Parameter Calibration")
            self.params.diam_5um_lo = cv2.getTrackbarPos("D5_lo", "Parameter Calibration")
            self.params.diam_5um_hi = cv2.getTrackbarPos("D5_hi", "Parameter Calibration")
            self.detector._update_morph_kernel()

            gray = self._grays[self.idx]
            if gray.size < 1000:
                self.idx = (self.idx + 1) % len(self.files)
                continue

            # Build 2×2 panel
            p1 = make_panel(gray, f"Original [{self.idx+1}/{len(self.files)}]", pw, ph)
            p2 = make_panel(self._corrected, "Background Corrected", pw, ph)
            p3 = make_panel(self._binary, "OTSU + Morph Close", pw, ph)
            p4 = draw_result(gray, self._particles, self.params, pw, ph)

            top = cv2.hconcat([p1, p2])
            bot = cv2.hconcat([p3, p4])
            display = cv2.vconcat([top, bot])

            # Status bar
            s = self.detector.stats
            status = f"bg_sig={self.params.bg_sigma:.0f}  morph={self.params.morph_k}  med={self.params.median_k} | 1um:{s['1um']} 5um:{s['5um']} | {s['elapsed_ms']:.1f}ms"
            cv2.putText(display, status, (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("Parameter Calibration", display)

            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('n'):
                self.idx = (self.idx + 1) % len(self.files)
                self._process_current()
            elif key == ord('p'):
                self.idx = (self.idx - 1) % len(self.files)
                self._process_current()
            elif key == ord('s'):
                out = f"results/calib_{self.idx+1:04d}.png"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(out, display)
                print(f"Saved: {out}")
            elif key == ord('v'):
                self.params.print()
            elif key == ord('b'):
                self._batch_test()
            # Space = reprocess
            elif key == ord(' '):
                self._process_current()

        cv2.destroyAllWindows()

    def _batch_test(self):
        """批量处理全部图片, 输出统计"""
        print(f"\n=== Batch Test ({len(self.files)} images) ===")
        times = []
        total_1um = total_5um = total_large = total_rej = 0
        for i, gray in enumerate(self._grays):
            if gray.size < 1000:
                continue
            t0 = time.perf_counter()
            particles, _, _, _ = self.detector.detect(gray)
            t1 = time.perf_counter()
            s = self.detector.stats
            times.append(s["elapsed_ms"])
            total_1um += s["1um"]
            total_5um += s["5um"]
            total_large += s["large"]
            total_rej += s["noise_rejected"]
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(self.files)}  "
                      f"1um:{s['1um']} 5um:{s['5um']} {s['elapsed_ms']:.1f}ms")

        n = len(times)
        print(f"\n--- Summary (n={n}) ---")
        print(f"  Avg time: {np.mean(times):.1f} ms ({1000/np.mean(times):.0f} FPS)")
        print(f"  P50: {np.median(times):.1f} ms")
        print(f"  1um/frame: {total_1um/n:.1f}  (total: {total_1um})")
        print(f"  5um/frame: {total_5um/n:.1f}  (total: {total_5um})")
        print(f"  large/frame: {total_large/n:.1f}  (total: {total_large})")
        print(f"  rejected/frame: {total_rej/n:.1f}  (total: {total_rej})")
        print()


def main():
    ap = argparse.ArgumentParser(description="显微颗粒检测 — 参数校准工具")
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--morph", type=int, default=None)
    ap.add_argument("--median", type=int, default=None)
    ap.add_argument("--batch", action="store_true", help="批量处理模式")
    args = ap.parse_args()

    # 查找图像目录
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
    print(f"Found {len(files)} images")

    params = Params()
    if args.sigma is not None:
        params.bg_sigma = args.sigma
    if args.morph is not None:
        params.morph_k = args.morph
    if args.median is not None:
        params.median_k = args.median

    if args.batch:
        cal = Calibrator(files, params)
        cal._batch_test()
    else:
        cal = Calibrator(files, params)
        print("\nControls:")
        print("  Trackbars  Adjust parameters")
        print("  N/P        Next/Prev image")
        print("  Space      Reprocess current frame")
        print("  S          Save screenshot")
        print("  V          Print current parameters")
        print("  B          Batch test all images")
        print("  Q/Esc      Quit")
        cal.run()


if __name__ == "__main__":
    main()
