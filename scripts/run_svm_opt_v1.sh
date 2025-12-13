#!/usr/bin/env bash
# Run SVM Pipeline for CUDA OPT_V1

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WEIGHTS_FILE="autoencoder_cuda_opt_v1_weights.bin"
TRAIN_FEATURES="train_features_opt_v1.libsvm"
TEST_FEATURES="test_features_opt_v1.libsvm"

echo "========================================"
echo "SVM Pipeline for CUDA OPT_V1"
echo "========================================"
echo ""

# Check if weights exist
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "❌ Error: Weights file not found: $WEIGHTS_FILE"
    echo "Please train opt_v1 first:"
    echo "  ./build_cuda/autoencoder_cuda_opt_v1 ./cifar-10-binary/cifar-10-batches-bin $WEIGHTS_FILE 200"
    exit 1
fi

echo "Step 1: Extracting features from OPT_V1 model..."
echo "-------------------------------------------"
./build_svm/extract_features_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    $WEIGHTS_FILE \
    $TRAIN_FEATURES \
    $TEST_FEATURES

echo ""
echo "✅ Features extracted:"
echo "   Train: $TRAIN_FEATURES ($(du -h $TRAIN_FEATURES | cut -f1))"
echo "   Test:  $TEST_FEATURES ($(du -h $TEST_FEATURES | cut -f1))"
echo ""

echo "Step 2: Training SVM classifier..."
echo "-------------------------------------------"
./scripts/svm_train_test_cpp.sh \
    --train $TRAIN_FEATURES \
    --test $TEST_FEATURES

echo ""
echo "========================================"
echo "✅ SVM Pipeline for OPT_V1 completed!"
echo "========================================"
