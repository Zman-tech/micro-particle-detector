#include "particle_detector.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// 打印用法
// ---------------------------------------------------------------------------
static void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --pixel-size <um>   Pixel size in um/pixel (default: 0.1)\n"
              << "  --input <path>      Input image file or directory\n"
              << "  --output <path>     Output directory for result images\n"
              << "  --no-gui            Headless mode, no window display\n"
              << "  --benchmark N       Run benchmark: process N frames, report avg FPS\n"
              << "  --list              List detected particles to stdout (JSON-like)\n"
              << "  --help              Show this message\n"
              << "\n"
              << "Example:\n"
              << "  particle_detect --pixel-size 0.065 --input ./images/ --output ./results/\n"
              << "  particle_detect --pixel-size 0.1 --benchmark 100 --no-gui\n";
}

// ---------------------------------------------------------------------------
// 处理单张图像
// ---------------------------------------------------------------------------
static void processImage(ParticleDetector& detector,
                          const std::string& path,
                          const std::string& output_dir,
                          bool show_gui,
                          bool list_particles)
{
    cv::Mat src = cv::imread(path, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "[ERROR] Cannot read: " << path << "\n";
        return;
    }

    std::vector<Particle> particles;
    cv::Mat vis;
    detector.detect(src, particles, vis);

    const auto& stats = detector.stats();

    // 控制台输出
    std::cout << "--------------------------------------------------\n";
    std::cout << "File: " << fs::path(path).filename() << "\n";
    std::cout << "  Size: " << src.cols << "x" << src.rows
              << "  Time: " << std::fixed << std::setprecision(2)
              << stats.elapsed_ms << " ms"
              << "  FPS: " << std::setprecision(1) << stats.fps << "\n";
    std::cout << "  Total candidates: " << stats.total_candidates << "\n";
    std::cout << "  1um particles:    " << stats.count_1um << "\n";
    std::cout << "  5um particles:    " << stats.count_5um << "\n";
    std::cout << "  Noise/other:      " << stats.count_noise << "\n";

    if (list_particles) {
        std::cout << "  Particles:\n";
        for (size_t i = 0; i < particles.size(); ++i) {
            const auto& p = particles[i];
            if (p.type == "noise") continue;
            std::cout << "    {\n"
                      << "      \"x\": " << std::fixed << std::setprecision(2) << p.x << ",\n"
                      << "      \"y\": " << std::fixed << std::setprecision(2) << p.y << ",\n"
                      << "      \"area\": " << std::fixed << std::setprecision(1) << p.area << ",\n"
                      << "      \"diam_um\": " << std::fixed << std::setprecision(3) << p.equiv_diameter_um << ",\n"
                      << "      \"type\": \"" << p.type << "\"\n"
                      << "    }";
            if (i < particles.size() - 1) std::cout << ",";
            std::cout << "\n";
        }
    }

    // 保存结果图像
    if (!output_dir.empty()) {
        fs::create_directories(output_dir);
        std::string out_name = fs::path(path).stem().string() + "_detected.png";
        std::string out_path = (fs::path(output_dir) / out_name).string();
        cv::imwrite(out_path, vis);
        std::cout << "  Saved: " << out_path << "\n";
    }

    // GUI显示
    if (show_gui && !vis.empty()) {
        cv::namedWindow("Particle Detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Particle Detection", 1224, 1024);
        cv::imshow("Particle Detection", vis);
        int key = cv::waitKey(0);
        if (key == 27 || key == 'q') {
            cv::destroyAllWindows();
        }
    }
}

// ---------------------------------------------------------------------------
// 基准测试模式
// ---------------------------------------------------------------------------
static void runBenchmark(ParticleDetector& detector,
                          const std::string& input_path,
                          int n_frames)
{
    // 加载第一张图像
    cv::Mat src;
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.path().extension() == ".bmp" ||
                entry.path().extension() == ".png" ||
                entry.path().extension() == ".jpg" ||
                entry.path().extension() == ".tif" ||
                entry.path().extension() == ".tiff") {
                src = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                if (!src.empty()) break;
            }
        }
        if (src.empty()) {
            // 创建一个合成图像
            std::cout << "[INFO] No valid image found, using synthetic 2448x2048 image\n";
            src = cv::Mat(2048, 2448, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::randn(src, cv::Scalar(0,0,0), cv::Scalar(15,15,15));
        }
    } else {
        src = cv::imread(input_path, cv::IMREAD_COLOR);
    }

    if (src.empty()) {
        std::cerr << "[ERROR] Cannot load image for benchmark\n";
        return;
    }

    std::cout << "Benchmark: " << n_frames << " frames on "
              << src.cols << "x" << src.rows << std::endl;

    std::vector<double> times;
    times.reserve(n_frames);

    std::vector<Particle> particles;

    // 预热
    for (int i = 0; i < 5; ++i) {
        detector.detect(src, particles);
    }

    // 正式测试
    for (int i = 0; i < n_frames; ++i) {
        detector.detect(src, particles);
        times.push_back(detector.stats().elapsed_ms);
    }

    // 统计
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();
    double p50 = times[times.size() / 2];
    double p99 = times[static_cast<size_t>(times.size() * 0.99)];
    double min_t = times.front();
    double max_t = times.back();

    std::cout << "\nResults:\n";
    std::cout << "  Avg:  " << std::fixed << std::setprecision(2) << avg << " ms  ("
              << std::setprecision(1) << 1000.0/avg << " FPS)\n";
    std::cout << "  P50:  " << std::setprecision(2) << p50 << " ms  ("
              << std::setprecision(1) << 1000.0/p50 << " FPS)\n";
    std::cout << "  P99:  " << std::setprecision(2) << p99 << " ms  ("
              << std::setprecision(1) << 1000.0/p99 << " FPS)\n";
    std::cout << "  Min:  " << std::setprecision(2) << min_t << " ms\n";
    std::cout << "  Max:  " << std::setprecision(2) << max_t << " ms\n";

    bool meets_target = (1000.0 / avg) >= 30.0;
    std::cout << "\n  Target 30 FPS: " << (meets_target ? "PASS" : "FAIL") << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    double pixel_size    = 0.03;  // 默认 0.03 μm/pixel (校准: 1μm颗粒≈30-60px直径)
    std::string input    = ".";
    std::string output;
    bool show_gui        = true;
    bool list_particles  = false;
    int  benchmark       = 0;     // 0 = no benchmark

    // 解析参数
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--pixel-size" && i + 1 < argc) {
            pixel_size = std::stod(argv[++i]);
        } else if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--no-gui") {
            show_gui = false;
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark = std::stoi(argv[++i]);
        } else if (arg == "--list") {
            list_particles = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // 创建检测器
    ParticleDetector detector(pixel_size);

    std::cout << "=== Microscope Particle Detector ===\n";
    std::cout << "Pixel size: " << pixel_size << " um/pixel\n";
    std::cout << "OpenCV: " << CV_VERSION << "\n";
    std::cout << "OpenCL: " << (cv::ocl::haveOpenCL() ? "available" : "not available") << "\n\n";

    // 基准测试模式
    if (benchmark > 0) {
        show_gui = false;
        runBenchmark(detector, input, benchmark);
        return 0;
    }

    // 处理图像
    if (fs::is_directory(input)) {
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(input)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".bmp" || ext == ".png" || ext == ".jpg" ||
                ext == ".jpeg" || ext == ".tif" || ext == ".tiff") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());

        std::cout << "Found " << files.size() << " images\n";

        if (files.empty()) {
            std::cerr << "No images found in " << input << "\n";
            return 1;
        }

        for (const auto& f : files) {
            processImage(detector, f, output, show_gui, list_particles);
        }
    } else {
        processImage(detector, input, output, show_gui, list_particles);
    }

    return 0;
}