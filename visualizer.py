"""
显微颗粒检测 — 交互式可视化程序 (GPU加速版)

针对 4060Ti 优化: OpenCL UMat 加速 + 圆度联合判定

改进:
  - OTSU 阈值 ×0.7 放宽容差, 捕获大颗粒边缘暗部
  - 圆度 (circularity) = 4π×面积/周长², 区分真颗粒 vs 粘连线团
  - findContours 替代连通域, 直接算周长/圆度
  - UMat (OpenCL) GPU 加速高斯模糊、形态学、阈值

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
  +/-     调整面积阈值容差
  R       重置容差
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
    x: float = 0.0           # 质心 x (轮廓矩精算)
    y: float = 0.0           # 质心 y
    area: float = 0.0        # 面积 px² (contourArea)
    perimeter: float = 0.0   # 周长 px (arcLength)
    circularity: float = 0.0 # 圆度 4πA/P² (1.0=正圆, <0.4=不规则)
    radius_px: float = 0.0   # 等效半径
    equiv_diameter_um: float = 0.0
    type: str = "noise"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class Detector:
    def __init__(self, pixel_size_um: float = 0.03):
        self._tolerance_1um = (0.4, 2.25)
        self._tolerance_5um = (0.5, 2.0)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._use_circularity_filter = True
        self._gpu = cv2.ocl.haveOpenCL()
        self.set_pixel_size(pixel_size_um)
        self.stats = {}

    @property
    def use_circularity(self): return self._use_circularity_filter

    @use_circularity.setter
    def use_circularity(self, v: bool): self._use_circularity_filter = v

    def set_pixel_size(self, um_per_px: float):
        self.pixel_size_um = um_per_px
        px2 = um_per_px ** 2
        self._area_1um_theory = np.pi * 0.25 / px2
        self._area_5um_theory = np.pi * 6.25 / px2
        self._update_ranges()

    def set_tolerance(self, which: str, lo: float, hi: float):
        if which == "1um":
            self._tolerance_1um = (lo, hi)
        else:
            self._tolerance_5um = (lo, hi)
        self._update_ranges()

    def _update_ranges(self):
        lo1, hi1 = self._tolerance_1um
        lo5, hi5 = self._tolerance_5um
        self.a1_lo = self._area_1um_theory * lo1
        self.a1_hi = self._area_1um_theory * hi1
        self.a5_lo = self._area_5um_theory * lo5
        self.a5_hi = self._area_5um_theory * hi5
        if self.a1_hi > self.a5_lo * 0.7:
            self.a1_hi = self.a5_lo * 0.7

    def detect(self, gray: np.ndarray) -> Tuple[List[Particle], np.ndarray]:
        t0 = time.perf_counter()
        h, w = gray.shape

        # ---- 1. 背景校正 (8x降采样) [GPU] ----
        src = cv2.UMat(gray) if self._gpu else gray
        small = cv2.resize(src, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (0, 0), 50.0 / 8, 50.0 / 8,
                                 borderType=cv2.BORDER_REPLICATE)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        corr = cv2.subtract(src, bg, dtype=cv2.CV_16S)
        corr = cv2.add(corr, 128, dtype=cv2.CV_16S)
        corrected = corr.get() if self._gpu else corr
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # ---- 2. MedianBlur ----
        blurred = cv2.medianBlur(corrected, 3)

        # ---- 3. OTSU 二值化 (自动明/暗场) ----
        thresh_val, bin1 = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        r1 = cv2.countNonZero(bin1) / bin1.size
        r2 = cv2.countNonZero(bin2) / bin2.size
        use_inv = not (r1 > 0.5 or (r1 < 1e-4 and r2 > 1e-4))
        binary = bin1 if use_inv else bin2

        # ---- 3.5 形态学闭运算 [GPU] ----
        if self._gpu:
            _tmp = cv2.UMat(binary)
            _tmp = cv2.morphologyEx(_tmp, cv2.MORPH_CLOSE, self._morph_kernel)
            binary = _tmp.get()
        else:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)

        # ---- 4. findContours → 面积 + 周长 + 圆度 + 质心 ----
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particles = []
        c1um = c5um = clarge = creject = cnoise = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20.0:
                cnoise += 1
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                cnoise += 1
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity > 1.0:
                circularity = 1.0

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cnoise += 1
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            radius = np.sqrt(area / np.pi)

            p = Particle(
                x=cx, y=cy,
                area=float(area),
                perimeter=float(perimeter),
                circularity=float(circularity),
                radius_px=float(radius),
                equiv_diameter_um=2.0 * radius * self.pixel_size_um,
                bbox=(x, y, bw, bh),
            )
            self._classify(p)

            if p.type == "1um_particle":   c1um += 1; particles.append(p)
            elif p.type == "5um_particle": c5um += 1; particles.append(p)
            elif p.type == "large_particle": clarge += 1; particles.append(p)
            elif p.type == "noise_rejected": creject += 1
            else:                           cnoise += 1

        t1 = time.perf_counter()
        self.stats = {
            "candidates": len(contours),
            "1um": c1um, "5um": c5um,
            "large": clarge, "noise_rejected": creject,
            "elapsed_ms": (t1 - t0) * 1000,
        }
        return particles, binary

    def _classify(self, p: Particle):
        a = p.area
        circ = p.circularity

        # 面积下限
        if a < 20.0:
            p.type = "noise"; return

        # ---- 圆度过滤 ----
        # 粘连线团(多颗粒合并) → circularity < 0.45 → 丢弃
        # 真颗粒 → circularity ≥ 0.55 → 保留
        if self._use_circularity_filter:
            if circ < 0.35:
                # 极度不规则: 合并碎片/图像噪点 → noise
                p.type = "noise_rejected"; return
            if circ < 0.55 and a > self.a1_hi:
                # 面积大于1μm范围但形状不规则 → 几个1μm颗粒粘在一起
                # 不应归类为5μm → 丢弃
                p.type = "noise_rejected"; return

        # ---- 面积分类 ----
        if self.a1_lo <= a <= self.a1_hi:
            p.type = "1um_particle"
        elif self.a5_lo <= a <= self.a5_hi:
            p.type = "5um_particle"
        elif a < self.a1_lo:
            p.type = "noise"
        elif a > self.a5_hi:
            p.type = "large_particle"
        else:
            # 落在1μm和5μm之间
            mid1 = (self.a1_lo + self.a1_hi) / 2
            rng1 = max((self.a1_hi - self.a1_lo) / 2, 1)
            mid5 = (self.a5_lo + self.a5_hi) / 2
            rng5 = max((self.a5_hi - self.a5_lo) / 2, 1)
            d1 = abs(a - mid1) / rng1
            d5 = abs(a - mid5) / rng5
            p.type = "1um_particle" if d1 <= d5 else "5um_particle"


# ======================================================================
# 可视化渲染
# ======================================================================
COLORS = {
    "1um_particle":     (0, 255, 0),     # 绿 — 小颗粒
    "5um_particle":     (0, 165, 255),   # 橙 — 大颗粒
    "large_particle":   (0, 0, 255),     # 红 — 超大颗粒
    "noise_rejected":   (64, 64, 64),    # (不渲染)
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
        r = int(round(p.radius_px))

        # ---- 圆形标注 (半径 ×1.2 补偿OTSU边缘切除) ----
        r_display = int(round(p.radius_px * 1.20))
        thickness = 2 if p.type == "5um_particle" else 1
        cv2.circle(vis, (cx, cy), r_display, color, thickness)

        # ---- 可选 bbox ----
        if show_bbox:
            bx, by, bw, bh = p.bbox
            cv2.rectangle(vis, (bx, by, bx + bw, by + bh), color, 1)

        # ---- 质心十字 ----
        cross = 4
        cv2.line(vis, (cx - cross, cy), (cx + cross, cy), color, 1)
        cv2.line(vis, (cx, cy - cross), (cx, cy + cross), color, 1)

        # ---- 标签 (含圆度) ----
        tag = "1um" if p.type == "1um_particle" else ("5um" if p.type == "5um_particle" else ">5um")
        cv2.putText(vis, tag, (cx + r + 3, cy - r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # ---- HUD ----
    panel_x = w - 290
    panel_h = 320
    cv2.rectangle(vis, (panel_x - 10, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (panel_x - 10, 0), (w, panel_h), (80, 80, 80), 1)
    cv2.addWeighted(vis[0:panel_h, panel_x - 10:w], 0.5,
                    np.full((panel_h, 300, 3), 0, dtype=np.uint8), 0.5,
                    0, vis[0:panel_h, panel_x - 10:w])

    def put(x, y, text, color=(255, 255, 255), scale=0.42):
        cv2.putText(vis, text, (panel_x + x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    s = det.stats
    put(5, 18,  f"Frame: {frame_idx+1}/{total}", (200, 200, 200))
    put(5, 42,  f"1um (green):    {s['1um']}", COLORS["1um_particle"], 0.55)
    put(5, 68,  f"5um (orange):   {s['5um']}", COLORS["5um_particle"], 0.55)
    put(5, 94,  f">5um (red):     {s.get('large', 0)}", COLORS.get("large_particle", (0,0,255)), 0.45)
    put(5, 116, f"Rejected:       {s.get('noise_rejected', 0)}", (100, 100, 100), 0.4)
    total_p = s['1um'] + s['5um'] + s.get('large', 0)
    put(5, 142, f"Total shown:    {total_p}", (255, 255, 255), 0.5)

    put(5, 168, f"GPU (OpenCL):   {'ON' if det._gpu else 'OFF'}", (0, 255, 200) if det._gpu else (128,128,128), 0.42)
    put(5, 190, f"Circ. filter:   {'ON' if det._use_circularity_filter else 'OFF'}", (200, 200, 0) if det._use_circularity_filter else (128,128,128), 0.42)
    put(5, 214, f"Pixel: {det.pixel_size_um:.3f} um/px", (180, 200, 255), 0.42)
    put(5, 234, f"1um: [{det.a1_lo:.0f}, {det.a1_hi:.0f}] px2", (160, 200, 160), 0.35)
    put(5, 248, f"5um: [{det.a5_lo:.0f}, {det.a5_hi:.0f}] px2", (160, 160, 200), 0.35)
    put(5, 272, f"Speed: {speed:.1f}x", (200, 200, 200), 0.42)

    status = "PLAYING" if speed > 0 else "PAUSED"
    clr = (0, 255, 0) if speed > 0 else (100, 100, 255)
    put(5, 296, status, clr, 0.5)

    # 底部状态栏
    cv2.rectangle(vis, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, h - 28), (w, h), (60, 60, 60), 1)
    cv2.putText(vis,
                "<- -> nav  Space:play  G:binary  B:bbox  C:circ-filter  P:pixel  +/-:tol  R:reset  S:save  H:help  Q:quit",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1, cv2.LINE_AA)

    if show_help:
        help_lines = [
            "KEYS:",
            "  <- ->    Prev / Next frame",
            "  Space    Play / Pause",
            "  Up/Down  Adjust speed",
            "  P        Cycle pixel size",
            "  C        Toggle circularity filter",
            "  G        Toggle binary view",
            "  B        Toggle bbox overlay",
            "  +/-      Adjust area tolerance",
            "  S        Save current frame",
            "  E        Batch export all frames",
            "  R        Reset tolerance",
            "  H        Hide this help",
            "  Esc/Q    Quit",
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

    print(f"OpenCL GPU : {'ENABLED' if det._gpu else 'DISABLED'}")
    print(f"Circ filter: {'ON' if det._use_circularity_filter else 'OFF'}")
    print(f"Pixel size : {det.pixel_size_um:.3f} um/px")
    print(f"1um range  : [{det.a1_lo:.0f}, {det.a1_hi:.0f}] px2")
    print(f"5um range  : [{det.a5_lo:.0f}, {det.a5_hi:.0f}] px2")
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
        elif key == 81 or key == 2424832:
            idx = (idx - 1) % len(files)
            if speed > 0: speed = 0
        elif key == 83 or key == 2555904:
            idx = (idx + 1) % len(files)
            if speed > 0: speed = 0
        elif key == 82 or key == 2490368:
            speed = min(speed + 0.5, 10.0)
        elif key == 84 or key == 2621440:
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
            print(f"Pixel: {det.pixel_size_um:.3f}  1um:[{det.a1_lo:.0f},{det.a1_hi:.0f}]  5um:[{det.a5_lo:.0f},{det.a5_hi:.0f}]")
        elif key == ord('+'):
            t1 = det._tolerance_1um; t5 = det._tolerance_5um
            det.set_tolerance("1um", t1[0] * 1.1, t1[1] * 1.1)
            det.set_tolerance("5um", t5[0] * 1.1, t5[1] * 1.1)
        elif key == ord('-'):
            t1 = det._tolerance_1um; t5 = det._tolerance_5um
            det.set_tolerance("1um", t1[0] * 0.9, t1[1] * 0.9)
            det.set_tolerance("5um", t5[0] * 0.9, t5[1] * 0.9)
        elif key == ord('r'):
            det.set_tolerance("1um", 0.4, 2.25)
            det.set_tolerance("5um", 0.5, 2.0)
            print("Tolerance reset")
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