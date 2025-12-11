#!/bin/bash

# Build script for GPU Optimized V2 (Kernel Fusion + Argmax Pool)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==================================================================="
echo "Building GPU Optimized V2 - Kernel Fusion + Argmax Pool"
echo "==================================================================="

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build_opt_v2"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Compile with optimizations
echo "Compiling autoencoder_opt_v2.cu..."
nvcc -O3 -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -use_fast_math \
    -I"$PROJECT_ROOT/include" \
    -I"$PROJECT_ROOT/src/data" \
    "$PROJECT_ROOT/src/main_cuda.cpp" \
    "$PROJECT_ROOT/src/cuda/autoencoder_opt_v2.cu" \
    -o autoencoder_opt_v2

echo ""
echo "==================================================================="
echo "Build complete!"
echo "Executable: $BUILD_DIR/autoencoder_opt_v2"
echo "==================================================================="
echo ""
echo "Optimizations enabled:"
echo "  - Conv+Bias+ReLU kernel fusion (5 fused operations)"
echo "  - MaxPool with argmax indices (no atomicAdd)"
echo "  - Fast math operations (-use_fast_math)"
echo ""
echo "Expected improvements:"
echo "  - 30-40% reduction in kernel launches"
echo "  - 20-30% reduction in global memory traffic"
echo "  - No atomic contention in pool backward"
echo ""
echo "To run:"
echo "  cd $BUILD_DIR"
echo "  ./autoencoder_opt_v2 <data_path> <model_path> <epochs> <batch_size> <lr> <num_samples>"
echo ""
