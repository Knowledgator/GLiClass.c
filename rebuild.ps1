# PowerShell script to clean, configure, and build the project

# Set error handling
$ErrorActionPreference = "Stop"

# Remove the 'build' directory if it exists
if (Test-Path -Path "build") {
    Remove-Item -Recurse -Force "build"
}

# Create a new 'build' directory
New-Item -ItemType Directory -Path "build" | Out-Null

# Navigate into the build directory
Set-Location -Path "build"

# Run CMake configuration
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_TARGET=CPU ..

# Build the project
cmake --build . --config Release

# Return to the original directory
Set-Location ..
