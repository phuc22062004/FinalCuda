#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cpu"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Building CPU version..."
g++ -std=c++17 -O2 \
    "${PROJECT_ROOT}/src/main.cpp" \
    "${PROJECT_ROOT}/src/cpu/autoencoder_cpu.cpp" \
    -I"${PROJECT_ROOT}/include" \
    -I"${PROJECT_ROOT}/src/data" \
    -o autoencoder_cpu

echo "CPU binary created at: ${BUILD_DIR}/autoencoder_cpu"


