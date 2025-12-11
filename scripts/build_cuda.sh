#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "========================================" 
echo "   Building CUDA Autoencoder Versions"
echo "========================================"
echo ""

# Build CUDA Basic version
echo "Building CUDA Basic version..."
nvcc -std=c++17 -O3 \
    "${PROJECT_ROOT}/src/main_cuda_basic.cpp" \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_cuda_basic

if [ $? -eq 0 ]; then
    echo "✓ CUDA Basic version built successfully!"
else
    echo "✗ Failed to build CUDA Basic version"
    exit 1
fi

echo ""
echo "========================================"
echo "Build complete!"
echo "Executable: ${BUILD_DIR}/autoencoder_cuda_basic"
echo "========================================"


