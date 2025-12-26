#!/usr/bin/env bash
set -e

# 1. Detect GPU Architecture automatically
echo "Detecting GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    ARCH_CODE=${COMPUTE_CAP//./}
    ARCH_FLAG="-arch=sm_${ARCH_CODE}"
    echo "Detected GPU: sm_${ARCH_CODE}. Using flag: ${ARCH_FLAG}"
else
    echo "Warning: nvidia-smi not found. Defaulting to sm_75 (T4)."
    ARCH_FLAG="-arch=sm_75"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_svm"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building SVM Feature Extraction Tools..."

# === Build CPU feature extractor ===
# (No changes needed here, g++ doesn't use CUDA flags)
echo "Building CPU feature extractor..."
g++ -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/svm/extract_features_cpu.cpp" \
    "${PROJECT_ROOT}/src/cpu/autoencoder_cpu.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cpu

# === Build CUDA feature extractor (Basic) ===
echo "Building CUDA feature extractor (Basic)..."
nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_basic.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_basic_svm.o

nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda.o

nvcc ${ARCH_FLAG} -std=c++17 -O2 \
    extract_features_cuda.o autoencoder_basic_svm.o \
    -o extract_features_cuda

# === Build CUDA OPT_V1 feature extractor ===
echo "Building CUDA OPT_V1 feature extractor..."
nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v1.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v1_svm.o

nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda_opt_v1.o

nvcc ${ARCH_FLAG} -std=c++17 -O2 \
    extract_features_cuda_opt_v1.o autoencoder_opt_v1_svm.o \
    -o extract_features_cuda_opt_v1

# === Build CUDA OPT_V2 feature extractor ===
echo "Building CUDA OPT_V2 feature extractor..."
nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/cuda/autoencoder_opt_v2.cu" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_opt_v2_svm.o

nvcc ${ARCH_FLAG} -std=c++17 -O2 -c \
    "${PROJECT_ROOT}/src/svm/extract_features_cuda.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o extract_features_cuda_opt_v2.o

nvcc ${ARCH_FLAG} -std=c++17 -O2  \
    extract_features_cuda_opt_v2.o autoencoder_opt_v2_svm.o \
    -o extract_features_cuda_opt_v2

echo ""
echo "SVM feature extractors built successfully!"
echo "  - extract_features_cpu"
echo "  - extract_features_cuda (basic)"
echo "  - extract_features_cuda_opt_v1"
echo "  - extract_features_cuda_opt_v2"