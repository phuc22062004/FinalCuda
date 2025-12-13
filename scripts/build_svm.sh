#!/bin/bash

# Build script for SVM Integration Pipeline

set -e

echo "Building SVM Integration Pipeline..."

# Create build directory
BUILD_DIR="build_svm"
mkdir -p $BUILD_DIR

# Compile
echo "Compiling..."
g++ -std=c++17 -O3 -Wall \
    -I./include \
    -I./src \
    src/main_svm.cpp \
    src/svm/svm_integration.cpp \
    src/cpu/autoencoder_cpu.cpp \
    -o $BUILD_DIR/autoencoder_svm \
    -pthread

if [ $? -eq 0 ]; then
    echo "Build successful! Binary created at: $BUILD_DIR/autoencoder_svm"
else
    echo "Build failed!"
    exit 1
fi
