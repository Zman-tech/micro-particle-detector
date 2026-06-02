# Microscope Particle Detection — 显微颗粒检测

## 项目概述

基于 OpenCV 的显微镜颗粒检测模块，用于从显微镜相机图像中检测 1μm 和 5μm 颗粒。

## 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 显微镜相机图像 (BMP, 8-bit 灰度, 2448×2048) |
| 已知参数 | 像素物理尺寸 (μm/pixel), 默认 0.03 |
| 输出 | 颗粒数量、中心坐标、粒径分类、圆度、可视化图像 |

## 数据

- 样本图像目录: `20260422_114158/` (136 张 BMP)
- 图像尺寸: 2448×2048, 8-bit 灰度
- 图像均值 ~99.5, 标准差 ~5.4 (低对比度)
- 伴随文件: `remote_trace.txt` — 显微镜台控制器轨迹 (x, y, z, button, u1, u2)

## 像素测量 (用户实测)

- **小颗粒 (≈1μm)**: 直径约 **30–60 px**, 面积约 700–2800 px²
- **大颗粒 (≈5μm)**: 直径约 **160 px**, 面积约 ~20,000 px²
- **推算 pixel_size**: 约 **0.025–0.033 μm/px** → 默认 0.03

## 算法管线

```
1. 灰度化 (BGR/BGRA → GRAY)
2. 背景校正: GaussianBlur(σ=50, 8×降采样加速) → corrected = gray - bg + 128
3. MedianBlur(k=3)
4. OTSU 二值化 (自动明/暗场, 双模式投票)
5. Morph Close (5×5 ELLIPSE) 弥合连通域碎片
6. findContours → 每个轮廓: area, perimeter, circularity, centroid (moments)
7. 圆度联合判定:
   - circ < 0.35 → 碎片 → noise
   - circ < 0.55 且面积 > 1μm上界 → 粘连线团 → noise (防止小颗粒簇误判为大颗粒)
   - 通过 → 按面积分类
8. 面积分类: noise / 1μm_particle / 5μm_particle / large_particle
```

## 4060Ti GPU 加速

- OpenCL UMat 加速: resize, GaussianBlur, morphologyEx
- 默认启用, `--no-gpu` 禁用
- 稳态性能: **25.9ms / 39 FPS** (2448×2048)

## 分类阈值 (pixel_size=0.03)

| 类型 | 面积范围 (px²) | 等效直径 (px) | 圆度要求 |
|------|---------------|--------------|---------|
| noise | < 349 或 > 43633 无圆度 | < 21 | — |
| 1μm (绿) | 349–1963 | 21–50 | circ ≥ 0.35 |
| 5μm (橙) | 10908–43633 | 118–236 | circ ≥ 0.35 |
| >5μm (红) | > 43633 | > 236 | circ ≥ 0.35 |
| rejected | 任何 | — | circ < 0.35 或 面积>1μm上界且circ<0.55 |

## 性能 (136帧, 4060Ti, OpenCL ON)

- Avg: 25.9ms (39 FPS), P50: 25.8ms
- 1μm 检出: 20.9/帧 (总计 2847)
- 5μm 检出: 5.3/帧 (总计 726)
- 圆度过滤丢弃: 18.4/帧 (总计 2499, 防止误分类)

## 项目结构

```
.claude/CLAUDE.md          ← 本文件
src/particle_detector.h    ← C++ 检测器 (contours + circularity)
src/particle_detector.cpp  ← C++ 实现 (OpenCL UMat)
src/main.cpp               ← CLI 工具
validate_algorithm.py       ← Python 参考实现
visualizer.py               ← 交互式可视化 (GPU加速)
build_msvc.bat              ← MSVC 构建脚本
```

## 构建 (C++)

```batch
# 在 VS Developer Command Prompt (x64) 中:
build_msvc.bat
```

依赖: CMake 3.14+, OpenCV 4.x, C++17

## Python 运行

```bash
python visualizer.py                          # 交互式可视化
python visualizer.py --no-gpu                 # CPU模式
python visualizer.py --no-circ-filter         # 关闭圆度过滤
python validate_algorithm.py --pixel-size 0.03 --input ./20260422_114158/
```
