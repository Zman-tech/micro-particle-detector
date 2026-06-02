@echo off
REM ===========================================================================
REM  Build script for MicroParticleDetector
REM  Run this from a Visual Studio Developer Command Prompt (x64)
REM ===========================================================================

setlocal

set "OPENCV_DIR=D:\softwares\opencv\build"
set "BUILD_DIR=%~dp0build"

if not exist "%OPENCV_DIR%\OpenCVConfig.cmake" (
    echo [ERROR] OpenCV not found at %OPENCV_DIR%
    echo Please edit this script and set OPENCV_DIR to your OpenCV build directory.
    exit /b 1
)

echo === Configuring with CMake ===
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DOpenCV_DIR="%OPENCV_DIR%" ^
    -B "%BUILD_DIR%" ^
    -S "%~dp0"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo.
echo === Building Release ===
cmake --build "%BUILD_DIR%" --config Release

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo === Build successful! ===
echo Binary: %BUILD_DIR%\Release\particle_detect.exe
echo.
echo Quick test:
echo   %BUILD_DIR%\Release\particle_detect.exe --pixel-size 0.065 --input D:\.visual\20260422_114158 --output D:\.visual\results --no-gui

endlocal