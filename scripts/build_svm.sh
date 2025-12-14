#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_svm"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building SVM Feature Extraction Tools..."

# Build CPU feature extractor
echo "Building CPU feature extractor..."
g++ -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/svm/extract_features_cpu.cpp" \
    "${PROJECT_ROOT}/src/cpu/autoencoder_cpu.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cpu

# Build CUDA feature extractor
echo "Building CUDA feature extractor..."
nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_basic_svm.o

nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda.o

nvcc -std=c++17 -O2 \
    extract_features_cuda.o autoencoder_basic_svm.o \
    -o extract_features_cuda

# Build CUDA OPT_V1 feature extractor
echo "Building CUDA OPT_V1 feature extractor..."
nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v1.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v1_svm.o

nvcc -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda_opt_v1.o

nvcc -std=c++17 -O2 \
    extract_features_cuda_opt_v1.o autoencoder_opt_v1_svm.o \
    -o extract_features_cuda_opt_v1

# Build CUDA OPT_V2 feature extractor
echo "Building CUDA OPT_V2 feature extractor..."
nvcc -std=c++17 -O3 -use_fast_math -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v2.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v2_svm.o

nvcc -std=c++17 -O3 -use_fast_math -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda_opt_v2.o

nvcc -std=c++17 -O3 -use_fast_math \
    extract_features_cuda_opt_v2.o autoencoder_opt_v2_svm.o \
    -o extract_features_cuda_opt_v2

echo ""
echo "SVM feature extractors built successfully!"
echo "  - extract_features_cpu"
echo "  - extract_features_cuda (basic)"
echo "  - extract_features_cuda_opt_v1"
echo "  - extract_features_cuda_opt_v2"


