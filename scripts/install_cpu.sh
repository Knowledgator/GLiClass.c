#!/bin/bash

# status messages
DONE_MSG="\033[1;32m[== DONE ==]\033[0m"
ERROR_MSG="\033[1;31m[== ERROR ==]\033[0m"
INFO_MSG="\033[1;34m[== INFO ==]\033[0m"


check_command() {
    command -v "$1" >/dev/null 2>&1
}

check_jq() {
    if check_command jq; then
        echo -e "$DONE_MSG jq found: $(jq --version)"
    else
        echo -e "$ERROR_MSG jq not found."
        echo -e "$INFO_MSG Installing jq ..."
        sudo apt-get install jq -y
    fi
}

check_cmake() {
    if check_command cmake; then
        cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
        required_version="3.25.0"

        if [ "$(printf '%s\n' "$required_version" "$cmake_version" | sort -V | head -n1)" = "$required_version" ]; then
            echo -e "$DONE_MSG CMake $cmake_version found."
        else
            echo -e "$ERROR_MSG Older version of CMake ($cmake_version) detected. Updating..."
            sudo apt remove --purge cmake -y
            sudo snap install cmake --classic
        fi
    else
        echo -e "$INFO_MSG Installing CMake..."
        sudo snap install cmake --classic
    fi
}


check_rust() {
    if check_command rustc; then
        echo -e "$DONE_MSG Rust found: $(rustc --version)"
    else
        echo -e "$INFO_MSG Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
}

check_openmp() {
    if dpkg -s libomp-dev >/dev/null 2>&1; then
        echo -e "$DONE_MSG OpenMP found."
    else
        echo -e "$INFO_MSG Installing OpenMP..."
        sudo apt update
        sudo apt install -y libomp-dev
    fi
}

check_onnxruntime() {
    ONNX_DIR="onnxruntime-linux-x64-1.19.2"

    if [ -d "$ONNX_DIR" ]; then
        echo -e "$DONE_MSG ONNXRuntime already installed in directory: $ONNX_DIR"
    else
        echo -e "$INFO_MSG Downloading and installing ONNXRuntime..."
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/$ONNX_DIR.tgz
        tar -xvzf $ONNX_DIR.tgz
        rm $ONNX_DIR.tgz
        echo -e "$DONE_MSG ONNXRuntime successfully installed."
    fi
}

echo -e "$INFO_MSG Checking and installing dependencies for CPU build..."

check_jq
check_cmake
check_rust
check_openmp
check_onnxruntime

echo -e "$DONE_MSG All dependencies are installed!"
