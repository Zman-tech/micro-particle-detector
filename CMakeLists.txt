cmake_minimum_required(VERSION 3.14)
project(MicroParticleDetector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV配置
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# 启用OpenCL加速(如果可用)
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    add_definitions(-DHAVE_OPENCL)
endif()

# 优化选项
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# 可执行文件
add_executable(particle_detect
    src/main.cpp
    src/particle_detector.cpp
)

target_include_directories(particle_detect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(particle_detect PRIVATE
    ${OpenCV_LIBS}
)

if(OpenCL_FOUND)
    target_link_libraries(particle_detect PRIVATE ${OpenCL_LIBRARIES})
endif()

# 启用OpenCV的并行后端
if(OpenCV_FOUND)
    set_target_properties(particle_detect PROPERTIES
        CXX_STANDARD 17
    )
endif()