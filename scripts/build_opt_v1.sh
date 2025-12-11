#!/bin/bash

# Build script for GPU Optimized V1 (Shared Memory Tiling)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==================================================================="
echo "Building GPU Optimized V1 - Shared Memory Tiling"
echo "==================================================================="

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build_opt_v1"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Compile with optimizations
echo "Compiling autoencoder_opt_v1.cu..."
nvcc -O3 -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -use_fast_math \
    -maxrregcount=64 \
    -I"$PROJECT_ROOT/include" \
    -I"$PROJECT_ROOT/src/data" \
    "$PROJECT_ROOT/src/main_cuda.cpp" \
    "$PROJECT_ROOT/src/cuda/autoencoder_opt_v1.cu" \
    -o autoencoder_opt_v1

echo ""
echo "==================================================================="
echo "Build complete!"
echo "Executable: $BUILD_DIR/autoencoder_opt_v1"
echo "==================================================================="
echo ""
echo "Optimizations enabled:"
echo "  - Shared memory tiling for conv2d (TILE_SIZE=16x16)"
echo "  - Specialized kernel for Conv1 (3-channel input)"
echo "  - Coalesced memory access pattern"
echo "  - Fast math operations (-use_fast_math)"
echo "  - Register usage limit (-maxrregcount=64)"
echo ""
echo "To run:"
echo "  cd $BUILD_DIR"
echo "  ./autoencoder_opt_v1 <data_path> <model_path> <epochs> <batch_size> <lr> <num_samples>"
echo ""
