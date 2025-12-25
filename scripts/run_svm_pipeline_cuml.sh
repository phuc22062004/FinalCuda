#!/usr/bin/env bash
# Complete pipeline script for SVM classification using cuML

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================="
echo "CIFAR-10 Autoencoder + cuML SVM Pipeline"
echo "========================================="
echo ""

# Configuration
CIFAR_DIR="./cifar-10-binary/cifar-10-batches-bin"
USE_CUDA=true  # Set to false to use CPU version

if [ "$USE_CUDA" = true ]; then
    echo "Using CUDA version..."
    WEIGHTS_FILE="autoencoder_cuda_opt_v1_weights.bin"
    FEATURE_EXTRACTOR="./build_svm/extract_features_cuda_opt_v1"
    TRAIN_FEATURES="train_features_cuda.libsvm"
    TEST_FEATURES="test_features_cuda.libsvm"
    CM_OUTPUT="confusion_matrix_cuml.png"
else
    echo "Using CPU version..."
    WEIGHTS_FILE="autoencoder_weights.bin"
    FEATURE_EXTRACTOR="./build_svm/extract_features_cpu"
    TRAIN_FEATURES="train_features_cpu.libsvm"
    TEST_FEATURES="test_features_cpu.libsvm"
    CM_OUTPUT="confusion_matrix_cuml_cpu.png"
fi

# Step 1: Extract features
echo ""
echo "Step 1: Extracting features from trained autoencoder..."
echo "========================================="
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ERROR: Weights file not found: $WEIGHTS_FILE"
    echo "Please train the autoencoder first!"
    exit 1
fi

$FEATURE_EXTRACTOR \
    "$CIFAR_DIR" \
    "$WEIGHTS_FILE" \
    "$TRAIN_FEATURES" \
    "$TEST_FEATURES"

echo ""
echo "Features extracted successfully!"
echo "  Train: $TRAIN_FEATURES"
echo "  Test:  $TEST_FEATURES"

# Step 2: Train and test SVM using cuML
echo ""
echo "Step 2: Training and testing SVM classifier with cuML..."
echo "========================================="

python3 scripts/svm_train_test_cuml.py \
    --train "$TRAIN_FEATURES" \
    --test "$TEST_FEATURES" \
    --C 10.0 \
    --gamma scale \
    --kernel rbf \
    --cm-output "$CM_OUTPUT" \
    --predictions "predictions_cuml.txt"

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Confusion matrix: $CM_OUTPUT"
echo "  - Predictions: predictions_cuml.txt"
echo ""
