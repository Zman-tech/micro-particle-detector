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

用户像素工具实测:

| 颗粒 | 直径 (px) |
|------|----------|
| 小微粒 (≈1μm) | **44** (范围 ±5: 39–49) |
| 大微粒 (≈5μm) | **166** (范围 ±5: 161–171) |

## 5. 检测算法

### 核心思路

显微镜下的真实微粒呈现特征性的 **"黑色中心 + 白色圆环"** (衍射暗斑 + 衍射亮环) 结构。OTSU 二值化提取的常常是**圆环/弧段而非实心圆**, 统计连通域像素面积毫无意义。改用 **`minEnclosingCircle` 外接圆直径** 判定颗粒真实大小, 并用 **径向强度剖面 (RIP)** 验证环状结构。

### 管线

```
灰度图 → 背景校正(σ=30, 8×降采样) → MedianBlur(3)
       → OTSU 二值化 (双模投票) → Morph Close (7×7 ELLIPSE)
       → findContours → minEnclosingCircle → 外接圆直径
       → 环状结构验证 (径向强度剖面)
       → 焦点过滤 (Laplacian方差)
       → 填充率过滤 → 重叠轮廓合并 → 直径分类
```

### 处理参数 (calibrate.py 验证)

| 参数 | 值 | 说明 |
|------|-----|------|
| bg_sigma | 30 | 背景校正高斯 σ |
| morph_k | 7 | 形态学闭运算核大小 |
| median_k | 3 | 中值滤波核 |
| otus_thresh | auto | OTSU 双模自动选择 |

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
| **39–49** | **1μm** | ring_consistency ≥ 0.35 或 ring_contrast > −0.01 | 🟢 绿 |
| **161–171** | **5μm** | circ ≥ 0.30 且 (ring_consistency ≥ 0.10 或 fill ≥ 0.20) | 🟠 橙 |
| > 171 | large | circ ≥ 0.50 | 🔴 红 |
| 其他 | rejected | — | — |

**关键**: 严格按外接圆直径分类。不在 1μm 也不在 5μm 范围 → 直接丢弃, 不强制归类。

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

## 6. 跟踪系统

### 架构

```
首帧/按D → 检测管线 → bootstrap tracks (立即 active)
之后每帧 → 模板匹配 (CCORR_NORMED, 4×+8×搜索窗)
模板失败或漂移 → 连续6帧检测不到 → 删除
```

无 Kalman 滤波 — 跟丢了就杀, 不靠惯性猜。

### 模板匹配 + 防漂移机制

- `cv2.matchTemplate` with `TM_CCORR_NORMED`
- 4× 搜索窗 + 8× 兜底
- **模板更新**: 仅在检测匹配成功时更新 (EMA α=0.25); **lost 状态冻结模板**
- **原始模板校验**: 每次匹配后与 `original_template` 比对, 相关度 < 0.70 → 拒绝匹配 (模板已漂移)
- **lost 状态**: 匹配阈值提升至 0.82 (主动态为 0.75), 防止假阳性维持虚假 track
- **连续丢失上限**: `lost_streak > max_lost` (默认6帧) → 强制删除, 无论模板匹配结果

### 生命周期

```
tentative(1帧) → active(检测匹配) → lost(模板追, 最多6帧, 模板冻结) → dead
```

### 标注

| 圆圈 | 标记 | 含义 |
|------|------|------|
| 实线 + 粗 | #ID | active (检测锁住) |
| 细线 | #ID~ | lost (模板匹配追, 模板冻结) |
| 细线 | #ID. | tentative (预览) |

### 匈牙利匹配

检测 ↔ 跟踪关联: 欧氏距离 + 速度变化惩罚, 全局最优匹配。防止近距颗粒 ID 互换。

### 重复检测

两个 track 圆心距 < `min(r1, r2) × 1.2` → 保留老 track, 删除新 track。

## 7. GPU 加速

`cv2.UMat` → OpenCL: resize, GaussianBlur, morphologyEx 在 GPU 执行。

## 8. 性能 (4060Ti)

| 模式 | 耗时 | FPS |
|------|------|-----|
| 检测管线 | ~35ms | 29 |
| 跟踪模式 | ~22ms | 45 |

## 9. 项目结构

```
.claude/CLAUDE.md          ← 本文件
visualizer.py               ← 主程序 (交互式可视化 + 检测器)
tracker.py                  ← 模板匹配跟踪器
calibrate.py                ← 参数校准脚本
validate_algorithm.py       ← 批量验证脚本
src/                        ← C++ 参考实现 (保留, 未编译)
```

## 10. 运行

```bash
python visualizer.py                          # 交互式可视化
python visualizer.py --input ./20260415_165254
python calibrate.py --batch                   # 批量测试参数
python calibrate.py --sigma 50 --morph 5      # 测试不同参数
python validate_algorithm.py --pixel-size 0.03 --input ./
```

## 11. 键盘操作

| 按键 | 功能 |
|------|------|
| **D** | 强制重检测 (清空 track 从零开始) |
| Space | 播放/暂停 |
| ← → | 上下帧 |
| ↑ ↓ | 调速 |
| T | 跟踪 开/关 |
| C | 圆度过滤 开/关 |
| P | 循环 pixel_size |
| B | bbox 叠加 |
| G | 二值图/灰度 |
| +/- | 调直径范围 |
| R | 重置直径范围到 [39-49] [161-171] |
| S | 保存截图 |
| H | 帮助 |
