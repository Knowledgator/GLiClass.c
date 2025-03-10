# PowerShell script to clean, configure, and build the project

# Set error handling
$ErrorActionPreference = "Stop"

# Remove the 'build' directory if it exists
if (Test-Path -Path "build") {
    Remove-Item -Recurse -Force "build"
}

# Create a new 'build' directory
New-Item -ItemType Directory -Path "build" | Out-Null

# Run CMake configuration
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_TARGET=CPU -DONNXRUNTIME_PATH="./onnxruntime-win-x64-1.19.2" -S . -B build

# Build the project
cmake --build build --config Release -j

# Override default onnxruntime
Copy-Item -Path ".\onnxruntime-win-x64-1.19.2\lib\*" -Destination ".\build\Release" -Recurse