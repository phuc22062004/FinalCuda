#!/usr/bin/env bash
set -e

echo "=== Building CUDA Autoencoder (Basic Version) ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

mkdir -p "${BUILD_DIR}"

echo "Building CUDA basic version..."
nvcc -std=c++17 -O3 \
    "${PROJECT_ROOT}/src/main_cuda.cpp" \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o "${BUILD_DIR}/autoencoder_cuda_basic"

echo ""
echo "✓ Build successful!"
echo "Binary: ${BUILD_DIR}/autoencoder_cuda_basic"
echo ""
echo "To run:"
echo "  cd ${BUILD_DIR}"
echo "  ./autoencoder_cuda_basic"
