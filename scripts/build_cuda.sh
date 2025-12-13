#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building CUDA versions..."

# Build basic version
nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_basic.o

nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/main_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o main_cuda_basic.o

nvcc -std=c++17 -O2 \
    main_cuda_basic.o autoencoder_basic.o \
    -o autoencoder_cuda_basic

# Skip optimized versions for now (not implemented)
# # Build optimized version 1
# nvcc -std=c++17 -O2 -c \
#     "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v1.cu" \
#     -I"${PROJECT_ROOT}/include" \
#     -I"${PROJECT_ROOT}/src/data" \
#     -o autoencoder_opt_v1.o
# 
# nvcc -std=c++17 -O2 -c \
#     "${PROJECT_ROOT}/src/main_cuda.cpp" \
#     -I"${PROJECT_ROOT}/include" \
#     -I"${PROJECT_ROOT}/src/data" \
#     -o main_cuda_opt_v1.o
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


