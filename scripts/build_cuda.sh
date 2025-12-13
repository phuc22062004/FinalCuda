#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building CUDA versions..."

# Build basic version
echo "Building basic version..."
nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_basic.o

nvcc -std=c++17 -O2 -c \
    -DVERSION_NAME='"CUDA BASIC (Phase 2)"' \
    "${PROJECT_ROOT}/src/main_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o main_cuda_basic.o

nvcc -std=c++17 -O2 \
    main_cuda_basic.o autoencoder_basic.o \
    -o autoencoder_cuda_basic

# Build optimized version 1 (Memory Optimized)
echo "Building opt_v1 (Memory Optimized)..."
nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v1.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v1.o

nvcc -std=c++17 -O2 -c \
    -DVERSION_NAME='"CUDA OPT_V1 (Memory Optimized)"' \
    "${PROJECT_ROOT}/src/main_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o main_cuda_opt_v1.o

nvcc -std=c++17 -O2 \
    main_cuda_opt_v1.o autoencoder_opt_v1.o \
    -o autoencoder_cuda_opt_v1

# Build optimized version 2 (Speed Optimized)
echo "Building opt_v2 (Speed Optimized)..."
nvcc -std=c++17 -O3 -use_fast_math -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v2.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v2.o

nvcc -std=c++17 -O3 -c \
    -DVERSION_NAME='"CUDA OPT_V2 (Speed Optimized)"' \
    "${PROJECT_ROOT}/src/main_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o main_cuda_opt_v2.o

nvcc -std=c++17 -O3 -use_fast_math \
    main_cuda_opt_v2.o autoencoder_opt_v2.o \
    -o autoencoder_cuda_opt_v2

# Skip optimized version 2 for now (not implemented)
# # Build optimized version 2     -o main_cuda_opt_v1.o
# 
# nvcc -std=c++17 -O2 \
#     main_cuda_opt_v1.o autoencoder_opt_v1.o \
#     -o autoencoder_cuda_opt_v1
# 
# # Build optimized version 2
# nvcc -std=c++17 -O2 -c \
#     "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v2.cu" \
#     -I"${PROJECT_ROOT}/include" \
#     -I"${PROJECT_ROOT}/src/data" \
#     -o autoencoder_opt_v2.o
# 
# nvcc -std=c++17 -O2 -c \
#     "${PROJECT_ROOT}/src/main_cuda.cpp" \
#     -I"${PROJECT_ROOT}/include" \
#     -I"${PROJECT_ROOT}/src/data" \
#     -o main_cuda_opt_v2.o
# 
# nvcc -std=c++17 -O2 \
#     main_cuda_opt_v2.o autoencoder_opt_v2.o \
#     -o autoencoder_cuda_opt_v2

echo "CUDA binaries created in: ${BUILD_DIR}"


