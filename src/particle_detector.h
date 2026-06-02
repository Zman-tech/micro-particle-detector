#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

/// @brief 颗粒检测结果
struct Particle {
    double x       = 0.0;     ///< 质心 x (像素, 轮廓矩精算)
    double y       = 0.0;     ///< 质心 y
    double area    = 0.0;     ///< 面积 (px², contourArea)
    double perimeter = 0.0;   ///< 周长 (px, arcLength)
    double circularity = 0.0; ///< 圆度 4πA/P² (1.0=正圆, <0.35=碎片)
    double radius_px = 0.0;   ///< 等效半径 = sqrt(area/pi)
    double equiv_diameter_um = 0.0; ///< 等效直径 (μm)
    std::string type;         ///< "1um_particle" | "5um_particle" | "large_particle" | "noise"

    cv::Rect bbox;
};

/// @brief 检测统计
struct DetectionStats {
    int total_candidates = 0;
    int count_1um        = 0;
    int count_5um        = 0;
    int count_large      = 0;
    int count_rejected   = 0;  ///< 被圆度过滤丢弃的
    double elapsed_ms    = 0.0;
    double fps           = 0.0;
};

/// @brief 显微镜颗粒检测器
///
/// 算法流程:
///   1. 灰度化
///   2. 背景校正: GaussianBlur(σ=50, 8×降采样)
///   3. MedianBlur(k=3)
///   4. OTSU二值化 (自动明/暗场)
///   5. Morph Close (5×5) 弥合碎片
///   6. findContours → 面积/周长/圆度/质心
///   7. 圆度联合判定: 过滤不规则碎片, 区分真颗粒 vs 粘连线团
///   8. 面积 → 物理分类: noise / 1μm / 5μm / large
///
/// 4060Ti GPU: OpenCL UMat 加速 resize / GaussianBlur / morphologyEx
class ParticleDetector {
public:
    /// @param pixel_size_um  像素物理尺寸 (μm/pixel)
    ///   calibrated: small(1μm)=30-60px → ~0.03 μm/px
    explicit ParticleDetector(double pixel_size_um = 0.03);

    void detect(cv::InputArray src,
                std::vector<Particle>& particles,
                cv::OutputArray debug_vis = cv::noArray());

    const DetectionStats& stats() const { return m_stats; }

    void setPixelSize(double um_per_pixel) {
        m_pixel_size_um = um_per_pixel;
        updateThresholds();
    }
    double pixelSize() const { return m_pixel_size_um; }

    void setAreaThreshold1um(double min_area, double max_area) {
        m_area_1um_min = min_area;
        m_area_1um_max = max_area;
    }
    void setAreaThreshold5um(double min_area, double max_area) {
        m_area_5um_min = min_area;
        m_area_5um_max = max_area;
    }

private:
    void classify(Particle& p) const;
    void updateThresholds();

    double m_pixel_size_um;
    double m_area_1um_min = 0.0;
    double m_area_1um_max = 0.0;
    double m_area_5um_min = 0.0;
    double m_area_5um_max = 0.0;

    // 内部缓冲区
    cv::Mat m_gray;
    cv::Mat m_corrected;
    cv::Mat m_blurred;
    cv::Mat m_binary;
    cv::Mat m_morph_kernel;

    DetectionStats m_stats;
};