# Microscope Particle Detection — 显微颗粒检测

## 1. 项目概述

从显微镜图像中自动检测 1μm 和 5μm 颗粒。纯 Python + OpenCV, 4060Ti GPU 加速。

## 2. 运行环境

- **Python**: Anaconda3, Python 3.10.9 (`D:\Anaconda3`)
- **依赖**: `opencv-python 4.13.0`, `numpy 2.2.6`
- **GPU**: OpenCL UMat (4060Ti), 默认开启, `--no-gpu` 关闭

```bash
pip install opencv-python numpy
```

## 3. 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 显微镜相机图像 (BMP, 8-bit 灰度, 2448×2048) |
| 参数 | pixel_size μm/pixel, 默认 0.03 |
| 输出 | 颗粒 ID、中心坐标、外接圆直径、填充率、粒径分类 |

## 4. 数据

| 数据集 | 数量 | 特点 |
|--------|------|------|
| `20260422_114158/` | 136 BMP | 1μm + 5μm 颗粒 |
| `20260415_165254/` | 79 BMP | 以 1μm 为主 |
| `20260415_165244/` | BMP | 附加数据集 |
| `20260415_170130/` | BMP | 附加数据集 |
| `20260415_170150/` | BMP | 附加数据集 |
| `20260415_153922/` | BMP | 附加数据集 |

用户像素工具实测:

| 颗粒 | 直径 (px) |
|------|----------|
| 小微粒 (≈1μm) | **44** (范围 ±3: 41–47) |
| 大微粒 (≈5μm) | **166** (范围 ±3: 163–169) |

## 5. 检测算法

### 核心思路

显微镜下的真实微粒呈现特征性的 **"黑色中心 + 白色圆环"** (衍射暗斑 + 衍射亮环) 结构。OTSU 二值化提取的常常是**圆环/弧段而非实心圆**, 统计连通域像素面积毫无意义。改用 **`minEnclosingCircle` 外接圆直径** 判定颗粒真实大小, 并用 **径向强度剖面 (RIP)** 验证环状结构。

### 管线

```
灰度图 → 背景校正(σ=50, 8×降采样) → MedianBlur(3)
       → OTSU 二值化 (双模投票) → Morph Close (5×5 ELLIPSE)
       → findContours → minEnclosingCircle → 外接圆直径
       → 填充率过滤 → 焦点过滤 (Laplacian方差)
       → 重叠轮廓合并 → 环状结构验证 (径向强度剖面)
       → 直径分类
```

### 处理参数 (calibrate.py 调优确认)

| 参数 | 值 | 说明 |
|------|-----|------|
| bg_sigma | 50 | 背景校正高斯 σ (8×降采样) |
| morph_k | 5 | 形态学闭运算核大小 (ELLIPSE) |
| median_k | 3 | 中值滤波核 |
| otsu_thresh | auto | OTSU 双模自动选择 |

### 环状结构检测 (Ring Detection)

在 3 个同心环 (0.30r, 0.75r, 1.10r) 各采样 24 个点:

| 特征 | 公式 | 含义 |
|------|------|------|
| `ring_contrast` | `(median_ring − median_center) / 255` | 环比中心亮多少 |
| `ring_consistency` | 环上采样点中 `I > median_center` 的比例 | 环的完整度 [0-1] |
| `outer_drop` | `(median_ring − median_outer) / 255` | 环比外圈亮多少 |

### 分类 (严格直径范围)

| 外接圆直径 (px) | 分类 | 附加条件 | 颜色 |
|---------------|------|---------|------|
| < 20 | noise | — | — |
| **41–47** | **1μm** | ring_consistency ≥ 0.35 或 ring_contrast > −0.01 | 🟢 绿 |
| **163–169** | **5μm** | circ ≥ 0.30 且 (ring_consistency ≥ 0.10 或 fill ≥ 0.20) | 🟠 橙 |
| > 169 | large | circ ≥ 0.50 | 🔴 红 |
| 其他 | rejected | — | — |

**关键**: 严格按外接圆直径分类。不在 1μm 也不在 5μm 范围 → 直接丢弃 (noise_rejected), 不强制归类。直径范围可通过 +/- 键实时调整。

### 填充率

`fill_ratio = contourArea / enclosingCircleArea`

- 小颗粒 (r < 25px): fill_ratio ≥ 0.06
- 其他: fill_ratio ≥ 0.10
- 衍射环通常 < 0.10 → 丢弃

### 焦点过滤

Laplacian 方差 (边缘锐度) — 离焦颗粒响应低:
- r < 20px: 阈值 8.0
- r < 30px: 阈值 12.0
- r < 60px: 阈值 6.0
- r ≥ 60px: 阈值 3.0

### 轮廓合并

单次贪心合并: 圆心距 < `max(r1, r2) × 1.2` → 同一颗粒 → 加权平均圆心 + 最大外接圆半径。

## 6. 跟踪系统 (规划中)

模板匹配跟踪器 (`tracker.py`) 尚未实现。当前每帧独立检测。

## 7. GPU 加速

`cv2.UMat` → OpenCL: resize, GaussianBlur, morphologyEx 在 GPU 执行。

## 8. 性能 (4060Ti, 2448×2048 BMP)

| 模式 | 耗时 | FPS |
|------|------|-----|
| 检测管线 (含ring validation) | ~50-70ms | 15-20 |
| 纯检测 (无ring) | ~35ms | 29 |

## 9. 项目结构

```
.claude/CLAUDE.md          ← 本文件
visualizer.py               ← 主程序 (交互式可视化 + 外接圆直径分类)
calibrate.py                ← 参数校准工具 (Trackbar 交互调参)
validate_algorithm.py       ← 旧版批量验证脚本 (连通域+面积分类, 参考)
```

## 10. 运行

```bash
python visualizer.py                          # 交互式可视化
python visualizer.py --input ./20260415_165254
python visualizer.py --no-gpu                 # CPU模式
python calibrate.py                           # 参数校准 (Trackbar 交互)
python calibrate.py --batch                   # 批量测试, 输出统计
python calibrate.py --sigma 50 --morph 5      # 指定参数启动
```

## 11. 键盘操作 (visualizer.py)

| 按键 | 功能 |
|------|------|
| Space | 播放/暂停 |
| ← → | 上一帧/下一帧 |
| ↑ ↓ | 加速/减速 |
| C | 圆度过滤 开/关 |
| P | 循环 pixel_size |
| B | bbox 叠加 |
| G | 二值图/灰度 |
| +/- | 扩大/缩小直径范围 (±1px) |
| R | 重置直径范围: 1μm [41,47] 5μm [163,169] |
| S | 保存截图 |
| E | 批量导出全部帧 |
| H | 帮助 |
| Esc/Q | 退出 |
