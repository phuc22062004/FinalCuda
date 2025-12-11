#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building CUDA versions..."

nvcc -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/main.cpp" \
    "${PROJECT_ROOT}/src/data/cifar10_loader.cpp" \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -o autoencoder_cuda_basic

nvcc -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/main.cpp" \
    "${PROJECT_ROOT}/src/data/cifar10_loader.cpp" \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v1.cu" \
    -I"${PROJECT_ROOT}/include" \
    -o autoencoder_cuda_opt_v1

nvcc -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/main.cpp" \
    "${PROJECT_ROOT}/src/data/cifar10_loader.cpp" \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v2.cu" \
    -I"${PROJECT_ROOT}/include" \
    -o autoencoder_cuda_opt_v2

echo "CUDA binaries created in: ${BUILD_DIR}"


