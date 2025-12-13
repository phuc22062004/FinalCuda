# SVM Integration Guide

This directory contains tools for Phase 4: SVM Integration.

## Overview

The SVM integration pipeline:
1. Extract features from trained autoencoder (encoder bottleneck: 8192 features)
2. Train SVM classifier on extracted features
3. Evaluate classification performance on CIFAR-10 test set

## Components

### 1. Feature Extraction
- `extract_features_cpu.cpp`: Extract features using CPU-trained autoencoder
- `extract_features_cuda.cpp`: Extract features using CUDA-trained autoencoder

### 2. SVM Training/Testing
- `svm_train_test.py`: Train and test SVM using ThunderSVM (supports both CPU and GPU)

## Installation

### ThunderSVM (Recommended - supports both CPU and GPU)

```bash
# Install via pip
pip install thundersvm

# Or install from source for latest version
git clone https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
mkdir build && cd build
cmake ..
make -j
sudo make install
cd python
pip install .
```

### Fallback: sklearn (CPU only)

If ThunderSVM is not available, the script will automatically fall back to sklearn:

```bash
pip install scikit-learn numpy matplotlib seaborn
```

## Build

Build the feature extraction tools:

```bash
chmod +x scripts/build_svm.sh
./scripts/build_svm.sh
```

This creates:
- `build_svm/extract_features_cpu`: CPU feature extractor
- `build_svm/extract_features_cuda`: CUDA feature extractor

## Usage

### Option 1: Complete Pipeline (Recommended)

Run the complete pipeline with one command:

```bash
# Make script executable
chmod +x scripts/run_svm_pipeline.sh

# Run with CUDA features
./scripts/run_svm_pipeline.sh

# Run with CUDA features + GPU SVM
./scripts/run_svm_pipeline.sh --svm-gpu
```

Edit the script to use CPU version by changing `USE_CUDA=false`.

### Option 2: Manual Steps

#### Step 1: Extract Features

For CUDA version:
```bash
./build_svm/extract_features_cuda \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    train_features_cuda.libsvm \
    test_features_cuda.libsvm
```

For CPU version:
```bash
./build_svm/extract_features_cpu \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_weights.bin \
    train_features_cpu.libsvm \
    test_features_cpu.libsvm
```

#### Step 2: Train and Test SVM

CPU version:
```bash
python3 src/svm/svm_train_test.py \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 10.0 \
    --gamma auto \
    --output confusion_matrix.png
```

GPU version (requires ThunderSVM):
```bash
python3 src/svm/svm_train_test.py \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 10.0 \
    --gamma auto \
    --output confusion_matrix.png \
    --gpu
```

## SVM Parameters

- `--C`: SVM regularization parameter (default: 10.0)
  - Higher C: Less regularization (may overfit)
  - Lower C: More regularization (may underfit)

- `--gamma`: RBF kernel parameter (default: auto)
  - 'auto': Uses 1 / n_features
  - Can specify value like 0.001, 0.01, etc.

- `--gpu`: Use GPU for SVM training (requires ThunderSVM)

## Expected Results

| Metric | Target |
|--------|--------|
| Feature extraction time | < 20 seconds for 60K images |
| SVM training time | 1-5 minutes |
| Test accuracy | 60-65% |

## Output Files

- `train_features_*.libsvm`: Training features in LibSVM format
- `test_features_*.libsvm`: Test features in LibSVM format
- `confusion_matrix*.png`: Confusion matrix visualization

## Troubleshooting

### ThunderSVM not found
If you get "ThunderSVM not available", the script will automatically use sklearn's SVM (CPU only). Install ThunderSVM for GPU support.

### CUDA out of memory during feature extraction
Reduce batch processing or extract features one by one. The current implementation processes images sequentially to minimize memory usage.

### Low accuracy (<50%)
- Check that autoencoder weights are properly trained
- Verify feature extraction is working correctly
- Try different SVM hyperparameters (C, gamma)
- Consider using more training data

## Performance Comparison

Run both CPU and GPU versions to compare:

```bash
# Extract features with CUDA
./build_svm/extract_features_cuda ... 

# Train with CPU SVM
python3 src/svm/svm_train_test.py --train ... --test ...

# Train with GPU SVM
python3 src/svm/svm_train_test.py --train ... --test ... --gpu
```

The GPU version (ThunderSVM) should be significantly faster for large datasets.

## Integration with Other Optimized Versions

To use features from optimized CUDA versions (opt_v1, opt_v2), you'll need to:

1. Implement `extract_features()` method in those versions
2. Build feature extractors for those versions
3. Run the pipeline with corresponding weights

The current implementation works with:
- `autoencoder_cpu` (CPU version)
- `autoencoder_cuda_basic` (basic CUDA version)

## References

- ThunderSVM: https://github.com/Xtra-Computing/thundersvm
- LibSVM format: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
