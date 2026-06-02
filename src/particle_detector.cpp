#include "particle_detector.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std::chrono;

// ---------------------------------------------------------------------------
ParticleDetector::ParticleDetector(double pixel_size_um)
    : m_pixel_size_um(pixel_size_um)
{
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
    }
    // 5×5 椭圆核 (比3×3更能弥合较大碎片)
    m_morph_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    updateThresholds();
}

// ---------------------------------------------------------------------------
void ParticleDetector::updateThresholds()
{
    const double px2 = m_pixel_size_um * m_pixel_size_um;
    double area_1um_theory = M_PI * 0.25  / px2;
    double area_5um_theory = M_PI * 6.25  / px2;

    m_area_1um_min = area_1um_theory * 0.4;
    m_area_1um_max = area_1um_theory * 2.25;
    m_area_5um_min = area_5um_theory * 0.5;
    m_area_5um_max = area_5um_theory * 2.0;

    if (m_area_1um_max > m_area_5um_min * 0.7) {
        m_area_1um_max = m_area_5um_min * 0.7;
    }
}

// ---------------------------------------------------------------------------
// 分类: 面积 + 圆度联合判定
//   - 圆度 < 0.35 → 极度不规则 → noise
//   - 圆度 < 0.55 且面积 > 1μm上界 → 粘连线团 → noise (防止小颗粒簇误判为大颗粒)
//   - 通过圆度检 → 按面积分类
// ---------------------------------------------------------------------------
void ParticleDetector::classify(Particle& p) const
{
    double a = p.area;
    double circ = p.circularity;

    if (a < 20.0) {
        p.type = "noise"; return;
    }

    // ---- 圆度过滤 ----
    if (circ < 0.35) {
        p.type = "noise"; return;           // 碎片/噪点
    }
    if (circ < 0.55 && a > m_area_1um_max) {
        p.type = "noise"; return;           // 多个小颗粒粘在一起 → 不应归类为大颗粒
    }

    // ---- 面积分类 ----
    p.equiv_diameter_um = 2.0 * std::sqrt(a / M_PI) * m_pixel_size_um;

    if (a >= m_area_1um_min && a <= m_area_1um_max) {
        p.type = "1um_particle";
    } else if (a >= m_area_5um_min && a <= m_area_5um_max) {
        p.type = "5um_particle";
    } else if (a < m_area_1um_min) {
        p.type = "noise";
    } else if (a > m_area_5um_max) {
        p.type = "large_particle";
    } else {
        double mid1 = (m_area_1um_min + m_area_1um_max) / 2.0;
        double rng1 = std::max((m_area_1um_max - m_area_1um_min) / 2.0, 1.0);
        double mid5 = (m_area_5um_min + m_area_5um_max) / 2.0;
        double rng5 = std::max((m_area_5um_max - m_area_5um_min) / 2.0, 1.0);
        p.type = (std::abs(a - mid1) / rng1 <= std::abs(a - mid5) / rng5)
                     ? "1um_particle" : "5um_particle";
    }
}

// ---------------------------------------------------------------------------
// 背景校正 (8×降采样)
// ---------------------------------------------------------------------------
static void fastBackgroundCorrect(cv::InputArray _src, cv::OutputArray _dst, double sigma)
{
    cv::Mat src = _src.getMat();
    const int scale = 8;
    int sc = src.cols / scale;
    int sr = src.rows / scale;

    cv::Mat small;
    cv::resize(src, small, cv::Size(sc, sr), 0, 0, cv::INTER_LINEAR);
    cv::GaussianBlur(small, small, cv::Size(0, 0), sigma / scale, sigma / scale,
                     cv::BORDER_REPLICATE);
    cv::Mat bg;
    cv::resize(small, bg, cv::Size(src.cols, src.rows), 0, 0, cv::INTER_LINEAR);

    cv::Mat diff;
    cv::subtract(src, bg, diff, cv::noArray(), CV_16S);
    diff += 128;
    diff.convertTo(_dst, CV_8U);
}

// ---------------------------------------------------------------------------
// 主检测流程
// ---------------------------------------------------------------------------
void ParticleDetector::detect(cv::InputArray _src,
                               std::vector<Particle>& particles,
                               cv::OutputArray debug_vis)
{
    auto t0 = high_resolution_clock::now();
    particles.clear();

    cv::Mat src = _src.getMat();

    // --- 1. 灰度化 ---
    if (src.channels() == 3) {
        cv::cvtColor(src, m_gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, m_gray, cv::COLOR_BGRA2GRAY);
    } else {
        m_gray = src.clone();
    }

    // --- 2. 背景校正 ---
    fastBackgroundCorrect(m_gray, m_corrected, 50.0);

    // --- 3. MedianBlur ---
    cv::medianBlur(m_corrected, m_blurred, 3);

    // --- 4. OTSU + 自动极性 ---
    cv::Mat bin1, bin2;
    cv::threshold(m_blurred, bin1, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::threshold(m_blurred, bin2, 0, 255, cv::THRESH_BINARY     | cv::THRESH_OTSU);

    double r1 = cv::countNonZero(bin1) / (double)(bin1.total());
    double r2 = cv::countNonZero(bin2) / (double)(bin2.total());

    bool use_inv = true;
    if (r1 > 0.5)  use_inv = false;
    if (r2 > 0.5)  use_inv = true;
    if (r1 < 1e-4 && r2 > 1e-4) use_inv = false;
    if (r2 < 1e-4 && r1 > 1e-4) use_inv = true;

    m_binary = use_inv ? bin1 : bin2;

    // --- 5. 形态学闭运算 ---
    cv::morphologyEx(m_binary, m_binary, cv::MORPH_CLOSE, m_morph_kernel);

    // --- 6. findContours → 面积 + 周长 + 圆度 + 质心 ---
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int c1um = 0, c5um = 0, clarge = 0, creject = 0;
    particles.reserve(contours.size());

    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area < 20.0) continue;

        double perimeter = cv::arcLength(cnt, true);
        if (perimeter < 1e-6) continue;

        double circ = 4.0 * M_PI * area / (perimeter * perimeter);
        if (circ > 1.0) circ = 1.0;

        cv::Moments M = cv::moments(cnt);
        if (M.m00 <= 0) continue;
        double cx = M.m10 / M.m00;
        double cy = M.m01 / M.m00;

        cv::Rect bbox = cv::boundingRect(cnt);
        double radius = std::sqrt(area / M_PI);

        Particle p;
        p.x = cx;
        p.y = cy;
        p.area = area;
        p.perimeter = perimeter;
        p.circularity = circ;
        p.radius_px = radius;
        p.bbox = bbox;

        classify(p);

        if (p.type == "1um_particle")      { c1um++;   particles.push_back(p); }
        else if (p.type == "5um_particle") { c5um++;   particles.push_back(p); }
        else if (p.type == "large_particle"){ clarge++; particles.push_back(p); }
        else                               { creject++; }
    }

    // 统计
    m_stats.total_candidates = static_cast<int>(contours.size());
    m_stats.count_1um   = c1um;
    m_stats.count_5um   = c5um;
    m_stats.count_large = clarge;
    m_stats.count_rejected = creject;

    auto t1 = high_resolution_clock::now();
    m_stats.elapsed_ms = duration<double, std::milli>(t1 - t0).count();
    m_stats.fps = 1000.0 / m_stats.elapsed_ms;

    // --- 7. 调试可视化 ---
    if (debug_vis.needed()) {
        cv::Mat vis;
        if (src.channels() == 1) {
            cv::cvtColor(src, vis, cv::COLOR_GRAY2BGR);
        } else {
            vis = src.clone();
        }

        for (const auto& p : particles) {
            cv::Scalar color;
            int thickness;

            if (p.type == "1um_particle") {
                color = cv::Scalar(0, 255, 0); thickness = 1;
            } else if (p.type == "5um_particle") {
                color = cv::Scalar(0, 165, 255); thickness = 2;
            } else {
                color = cv::Scalar(0, 0, 255); thickness = 1;
            }

            cv::Point center(static_cast<int>(p.x), static_cast<int>(p.y));
            // 显示半径 ×1.2 补偿OTSU边缘切除
            int r = static_cast<int>(p.radius_px * 1.20);
            cv::circle(vis, center, r, color, thickness);

            int cross = 4;
            cv::line(vis, cv::Point(center.x - cross, center.y),
                     cv::Point(center.x + cross, center.y), color, 1);
            cv::line(vis, cv::Point(center.x, center.y - cross),
                     cv::Point(center.x, center.y + cross), color, 1);

            std::string label;
            if (p.type == "1um_particle")      label = "1um";
            else if (p.type == "5um_particle")  label = "5um";
            else                                label = ">5um";

            cv::putText(vis, label,
                        cv::Point(center.x + r + 3, center.y - r - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv::LINE_AA);
        }

        std::string info = cv::format("FPS:%.1f | 1um:%d | 5um:%d | >5um:%d | rej:%d",
                                       m_stats.fps, m_stats.count_1um,
                                       m_stats.count_5um, m_stats.count_large,
                                       m_stats.count_rejected);
        cv::putText(vis, info, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        vis.copyTo(debug_vis);
    }
}