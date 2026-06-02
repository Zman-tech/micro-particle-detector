"""
显微镜颗粒检测算法验证脚本
算法流程与C++ ParticleDetector完全一致，用于在Python侧快速验证和调参。
"""
import cv2
import numpy as np
import sys
import os
import glob
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    area: float = 0.0
    equiv_diameter_um: float = 0.0
    type: str = "noise"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class ParticleDetector:
    """
    显微镜颗粒检测器 — Python参考实现
    """

    def __init__(self, pixel_size_um: float = 0.03):
        self.pixel_size_um = pixel_size_um
        self._update_thresholds()
        self.stats = {}

    def _update_thresholds(self):
        """根据像素尺寸计算面积阈值"""
        px2 = self.pixel_size_um ** 2

        area_1um_theory = np.pi * 0.25 / px2   # π*(0.5)²
        area_5um_theory = np.pi * 6.25 / px2   # π*(2.5)²

        self.area_1um_min = area_1um_theory * 0.4
        self.area_1um_max = area_1um_theory * 2.25
        self.area_5um_min = area_5um_theory * 0.5
        self.area_5um_max = area_5um_theory * 2.0

        if self.area_1um_max > self.area_5um_min * 0.7:
            self.area_1um_max = self.area_5um_min * 0.7

    def set_pixel_size(self, um_per_pixel: float):
        self.pixel_size_um = um_per_pixel
        self._update_thresholds()

    def _fast_background_correct(self, gray: np.ndarray, sigma: float = 50.0) -> np.ndarray:
        """
        快速背景校正: 8x降采样 → 小σ高斯 → 8x升采样
        对于σ=50的大核模糊，此方法比直接GaussianBlur快 ~50x
        """
        scale = 8
        h, w = gray.shape
        small_h, small_w = h // scale, w // scale

        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # 在小图上模糊
        small_sigma = sigma / scale
        small = cv2.GaussianBlur(small, (0, 0), small_sigma, small_sigma,
                                 borderType=cv2.BORDER_REPLICATE)

        # 升采样回原尺寸
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        corrected = cv2.subtract(gray, bg, dtype=cv2.CV_16S)
        corrected = cv2.add(corrected, 128, dtype=cv2.CV_16S)
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _classify(self, area_px: float, particle: Particle):
        """根据像素面积分类"""
        # 面积 < 20 px² (≈Ø5px) → 传感器噪点/灰尘
        if area_px < 20.0:
            particle.type = "noise"
            return

        diameter_um = 2.0 * np.sqrt(area_px / np.pi) * self.pixel_size_um
        particle.equiv_diameter_um = diameter_um

        if self.area_1um_min <= area_px <= self.area_1um_max:
            particle.type = "1um_particle"
        elif self.area_5um_min <= area_px <= self.area_5um_max:
            particle.type = "5um_particle"
        elif area_px < self.area_1um_min:
            particle.type = "noise"
        elif area_px > self.area_5um_max:
            particle.type = "large_particle"
        else:
            # 中间区域 → 按更近阈值归类
            mid_1um = (self.area_1um_min + self.area_1um_max) / 2.0
            range_1um = (self.area_1um_max - self.area_1um_min) / 2.0
            mid_5um = (self.area_5um_min + self.area_5um_max) / 2.0
            range_5um = (self.area_5um_max - self.area_5um_min) / 2.0

            if range_1um > 0 and range_5um > 0:
                dist_1um = abs(area_px - mid_1um) / range_1um
                dist_5um = abs(area_px - mid_5um) / range_5um
                particle.type = "1um_particle" if dist_1um <= dist_5um else "5um_particle"
            else:
                particle.type = "noise"

    def detect(self, src: np.ndarray) -> Tuple[List[Particle], np.ndarray]:
        """
        执行检测
        返回: (particles列表, 可视化图像)
        """
        t0 = time.perf_counter()

        # 1. 灰度化
        if len(src.shape) == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src.copy()

        # 2. 背景校正
        corrected = self._fast_background_correct(gray, sigma=50.0)

        # 3. MedianBlur
        blurred = cv2.medianBlur(corrected, 3)

        # 4. OTSU二值化 (自动判断前景极性)
        _, bin1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        fg_ratio1 = cv2.countNonZero(bin1) / bin1.size
        fg_ratio2 = cv2.countNonZero(bin2) / bin2.size

        use_inv = True
        if fg_ratio1 > 0.5:
            use_inv = False
        if fg_ratio2 > 0.5:
            use_inv = True
        if fg_ratio1 < 1e-4 and fg_ratio2 > 1e-4:
            use_inv = False
        if fg_ratio2 < 1e-4 and fg_ratio1 > 1e-4:
            use_inv = True

        binary = bin1 if use_inv else bin2

        # 5. ConnectedComponentsWithStats
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        h, w = binary.shape
        particles = []

        # 6. 分析每个连通域
        for label in range(1, n_labels):
            left   = stats[label, cv2.CC_STAT_LEFT]
            top    = stats[label, cv2.CC_STAT_TOP]
            width  = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area   = stats[label, cv2.CC_STAT_AREA]

            # 跳过边界噪点
            if (left <= 1 or top <= 1 or
                left + width >= w - 1 or top + height >= h - 1):
                if area < 20:
                    continue

            p = Particle(
                x=centroids[label, 0],
                y=centroids[label, 1],
                area=float(area),
                bbox=(left, top, width, height)
            )

            # 7. 分类
            self._classify(p.area, p)
            particles.append(p)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        count_1um = sum(1 for p in particles if p.type == "1um_particle")
        count_5um = sum(1 for p in particles if p.type == "5um_particle")
        count_noise = sum(1 for p in particles if p.type not in ("1um_particle", "5um_particle"))

        self.stats = {
            "total_candidates": n_labels - 1,
            "count_1um": count_1um,
            "count_5um": count_5um,
            "count_noise": count_noise,
            "elapsed_ms": elapsed_ms,
            "fps": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
        }

        # 8. 可视化
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for p in particles:
            if p.type == "1um_particle":
                color = (0, 255, 0); thickness = 1
            elif p.type == "5um_particle":
                color = (0, 0, 255); thickness = 2
            else:
                color = (128, 128, 128); thickness = 1

            x, y, w_box, h_box = p.bbox
            cv2.rectangle(vis, (x, y, x + w_box, y + h_box), color, thickness)
            cv2.circle(vis, (int(p.x), int(p.y)), 3, color, -1)

            if p.type in ("1um_particle", "5um_particle", "large_particle"):
                label = {"1um_particle": "1um", "5um_particle": "5um",
                         "large_particle": ">5um"}[p.type]
                cv2.putText(vis, label, (int(p.x) + 5, int(p.y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        info = f"FPS:{self.stats['fps']:.1f} | 1um:{count_1um} | 5um:{count_5um} | px:{self.pixel_size_um:.3f} um/px"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return particles, vis


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="显微镜颗粒检测算法验证")
    parser.add_argument("--pixel-size", type=float, default=0.1,
                        help="像素物理尺寸 (μm/pixel), default: 0.1")
    parser.add_argument("--input", type=str, default=None,
                        help="输入图像路径或目录")
    parser.add_argument("--output", type=str, default="./results",
                        help="输出目录, default: ./results")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="基准测试: 重复运行N次后输出平均FPS")
    parser.add_argument("--list", action="store_true",
                        help="输出全部检测到的颗粒信息")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存结果图像")
    args = parser.parse_args()

    detector = ParticleDetector(args.pixel_size)

    print(f"=== Microscope Particle Detector (Python Validation) ===")
    print(f"Pixel size: {args.pixel_size} μm/pixel")
    print(f"1um area range: [{detector.area_1um_min:.1f}, {detector.area_1um_max:.1f}] px^2")
    print(f"5um area range: [{detector.area_5um_min:.1f}, {detector.area_5um_max:.1f}] px^2")
    print()

    # 确定输入图像
    input_path = args.input
    if input_path is None:
        # 默认使用工作目录下的图像
        input_path = os.path.dirname(__file__) or "."

    if os.path.isdir(input_path):
        image_files = sorted(glob.glob(os.path.join(input_path, "*.bmp")) +
                             glob.glob(os.path.join(input_path, "*.png")) +
                             glob.glob(os.path.join(input_path, "*.jpg")) +
                             glob.glob(os.path.join(input_path, "*.tif")) +
                             glob.glob(os.path.join(input_path, "*.tiff")))
        # 排除子目录中的
        image_files = [f for f in image_files if os.path.dirname(f) == input_path]
    else:
        image_files = [input_path]

    print(f"Found {len(image_files)} image(s)\n")

    if not image_files:
        print("No images found!")
        return

    # 基准测试模式
    if args.benchmark > 0:
        print(f"[BENCHMARK] Running {args.benchmark} iterations...")
        src = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
        if src is None:
            print(f"ERROR: Cannot read {image_files[0]}")
            return

        times = []
        for _ in range(5):  # warmup
            detector.detect(src)

        for i in range(args.benchmark):
            _, _ = detector.detect(src)
            times.append(detector.stats["elapsed_ms"])
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.benchmark}...")

        times.sort()
        avg = np.mean(times)
        p50 = np.median(times)
        p99 = np.percentile(times, 99) if len(times) >= 100 else times[-1]

        print(f"\n  Image: {src.shape[1]}x{src.shape[0]}")
        print(f"  Avg: {avg:.2f} ms ({1000/avg:.1f} FPS)")
        print(f"  P50: {p50:.2f} ms ({1000/p50:.1f} FPS)")
        print(f"  P99: {p99:.2f} ms ({1000/p99:.1f} FPS)")
        print(f"  Min: {times[0]:.2f} ms  Max: {times[-1]:.2f} ms")
        print(f"  Target 30 FPS: {'PASS' if 1000/avg >= 30 else 'FAIL'}")
        return

    # 正常处理模式
    os.makedirs(args.output, exist_ok=True)

    for img_path in image_files:
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if src is None:
            print(f"ERROR: Cannot read {img_path}")
            continue

        particles, vis = detector.detect(src)
        stats = detector.stats

        fname = os.path.basename(img_path)
        print(f"--- {fname} ---")
        print(f"  Size: {src.shape[1]}x{src.shape[0]}  "
              f"Time: {stats['elapsed_ms']:.2f} ms  "
              f"FPS: {stats['fps']:.1f}")
        print(f"  Candidates: {stats['total_candidates']}  "
              f"1μm: {stats['count_1um']}  "
              f"5μm: {stats['count_5um']}  "
              f"Noise/Other: {stats['count_noise']}")

        if args.list:
            print("  Particles:")
            for i, p in enumerate(particles):
                if p.type == "noise":
                    continue
                print(f"    {{")
                print(f'      "x": {p.x:.2f}, "y": {p.y:.2f}, ')
                print(f'      "area": {p.area:.1f}, "diam_um": {p.equiv_diameter_um:.3f}, ')
                print(f'      "type": "{p.type}"')
                if i < len(particles) - 1:
                    print("    },")
                else:
                    print("    }")

        if not args.no_save:
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(args.output, f"{stem}_detected.png")
            cv2.imwrite(out_path, vis)
            print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()