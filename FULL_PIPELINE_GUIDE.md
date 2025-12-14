# HƯỚNG DẪN CHẠY ĐẦY ĐỦ - CUDA AUTOENCODER + SVM PIPELINE

> **CSC14120 Final Project - Complete Step-by-Step Guide**  
> Từ build → train → extract features → SVM classification

---

## 📋 MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Cấu trúc project](#2-cấu-trúc-project)
3. [Chuẩn bị môi trường](#3-chuẩn-bị-môi-trường)
4. [Build tất cả phiên bản](#4-build-tất-cả-phiên-bản)
5. [Train Autoencoder](#5-train-autoencoder)
6. [Extract Features với Scaling](#6-extract-features-với-scaling)
7. [Train SVM](#7-train-svm)
8. [So sánh kết quả](#8-so-sánh-kết-quả)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. TỔNG QUAN

Pipeline gồm 2 giai đoạn:

```
[CIFAR-10 Images]
      ↓
[Stage 1: Train Autoencoder] → weights.bin
      ↓
[Stage 2: Extract Features] → features.libsvm (with Z-score scaling)
      ↓
[Stage 3: Train SVM] → svm_model
      ↓
[Predict + Accuracy]
```

### Các phiên bản Autoencoder

| Phiên bản | Mô tả | Thời gian (200 epochs) | Optimization |
|-----------|-------|------------------------|--------------|
| **CPU** | Baseline C++ | ~40-50 phút | Không có GPU |
| **CUDA Basic** | Naive CUDA | ~8-10 phút | Basic parallelization |
| **CUDA OPT_V1** | Memory optimized | ~6-7 phút | Coalescing + Constant memory |
| **CUDA OPT_V2** | Speed optimized | ~4-5 phút | Kernel fusion + Vectorization |

---

## 2. CẤU TRÚC PROJECT

```
FinalCuda/
├── src/
│   ├── main_cpu.cpp              # CPU entry point
│   ├── main_cuda.cpp             # CUDA entry point
│   ├── cpu/
│   │   └── autoencoder_cpu.cpp   # CPU implementation
│   ├── cuda/
│   │   ├── autoencoder_basic.cu  # CUDA basic
│   │   ├── autoencoder_opt_v1.cu # Memory optimized
│   │   └── autoencoder_opt_v2.cu # Speed optimized
│   ├── data/
│   │   └── cifar10_loader.h      # CIFAR-10 loader
│   └── svm/
│       └── extract_features_cuda.cpp  # Feature extraction + Z-score scaling
├── include/
│   ├── autoencoder.hpp           # CPU header
│   ├── autoencoder_cuda.h        # CUDA header
│   └── config.h                  # Hyperparameters
├── scripts/
│   ├── build_cpu.sh              # Build CPU
│   ├── build_cuda.sh             # Build CUDA versions
│   └── build_svm.sh              # Build feature extractors
├── cifar-10-binary/
│   └── cifar-10-batches-bin/     # Dataset
├── build_cpu/                    # CPU executables
├── build_cuda/                   # CUDA executables
└── build_svm/                    # SVM tools
```

---

## 3. CHUẨN BỊ MÔI TRƯỜNG

### 3.1. Yêu cầu hệ thống

- **OS**: Linux (Ubuntu 20.04+) hoặc WSL2
- **CUDA**: 11.0+ (tested with CUDA 12.0)
- **GPU**: NVIDIA GPU với compute capability 6.0+ (RTX 3050 hoặc cao hơn)
- **Compiler**: GCC 9+ và nvcc
- **RAM**: 8GB+ (16GB recommended)
- **Disk**: ~10GB free space

### 3.2. Kiểm tra CUDA

```bash
nvcc --version
nvidia-smi
```

### 3.3. Download CIFAR-10 dataset

Đảm bảo folder `cifar-10-binary/cifar-10-batches-bin/` có các file:
- `data_batch_1.bin` → `data_batch_5.bin`
- `test_batch.bin`
- `batches.meta.txt`

---

## 4. BUILD TẤT CẢ PHIÊN BẢN

### 4.1. Build CPU baseline

```bash
cd /home/senyamiku/LTSS/FinalCuda
chmod +x scripts/*.sh
./scripts/build_cpu.sh
```

**Output**: `build_cpu/autoencoder_cpu`

### 4.2. Build CUDA versions (Basic + OPT_V1 + OPT_V2)

```bash
./scripts/build_cuda.sh
```

**Output**:
- `build_cuda/autoencoder_cuda_basic`
- `build_cuda/autoencoder_cuda_opt_v1`
- `build_cuda/autoencoder_cuda_opt_v2`

### 4.3. Build SVM feature extractors

```bash
./scripts/build_svm.sh
```

**Output**:
- `build_svm/extract_features_cpu`
- `build_svm/extract_features_cuda` (basic)
- `build_svm/extract_features_cuda_opt_v1`

---

## 5. TRAIN AUTOENCODER

### 5.1. Cú pháp lệnh

```bash
./build_<version>/autoencoder_<version> \
    <cifar_dir> \
    <weights_output> \
    <num_epochs> \
    <batch_size> \
    <learning_rate> \
    [num_images]
```

### 5.2. Train CPU (kiểm tra baseline)

```bash
# Test với 1000 images, 10 epochs
./build_cpu/autoencoder_cpu \
    cifar-10-binary/cifar-10-batches-bin \
    weights_cpu_test.bin \
    10 32 0.001 1000
```

**Thời gian**: ~2-3 phút cho 1000 images

### 5.3. Train CUDA Basic (full training)

```bash
# Full training: 50000 images, 200 epochs
./build_cuda/autoencoder_cuda_basic \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    200 32 0.001
```

**Thời gian**: ~8-10 phút  
**Loss cuối**: ~0.02-0.03

### 5.4. Train CUDA OPT_V1 (memory optimized)

```bash
# Memory optimized với constant memory + coalescing
./build_cuda/autoencoder_cuda_opt_v1 \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    200 32 0.001
```

**Thời gian**: ~6-7 phút  
**Optimization**: 
- Memory coalescing (threadIdx.x cho width)
- Constant memory cho conv1/conv5 weights
- Gradient clipping [-1.0, 1.0]
- Removed redundant memset

### 5.5. Train CUDA OPT_V2 (speed optimized) ⚡

```bash
# Speed optimized với kernel fusion + vectorization
./build_cuda/autoencoder_cuda_opt_v2 \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v2_weights.bin \
    200 32 0.001
```

**Thời gian**: ~4-5 phút (~30% faster than OPT_V1)  
**Optimization**:
- Kernel fusion (conv+bias+relu in one kernel)
- Vectorized float4 for SGD updates
- Vectorized float4 for MSE loss
- Tuned block dimensions per layer
- Specialized hardcoded 3x3 kernels

**Test nhanh** (3 epochs, 1000 images):
```bash
time ./build_cuda/autoencoder_cuda_opt_v2 \
    cifar-10-binary/cifar-10-batches-bin \
    test_opt_v2.bin \
    3 32 0.001 1000
```

**Expected**: ~26 seconds for 3 epochs

---

## 6. EXTRACT FEATURES VỚI SCALING

### 6.1. Tại sao cần Z-score scaling?

Features từ autoencoder (ReLU outputs) có:
- Phân phối không chuẩn (toàn giá trị dương)
- Variance không đồng đều giữa các chiều
- SVM RBF kernel hoạt động kém với unscaled features

**Kết quả**:
- ❌ **Không scale**: ~46% accuracy
- ✅ **Có scale**: ~60-65% accuracy

### 6.2. Z-Score Scaling Pipeline (2-Pass)

```
PASS 1: Extract train → Compute mean/std → Cache to disk
        ↓ (finalize statistics)
        Save scaler_z.bin
PASS 2: Read cache → Scale → Write LibSVM
TEST:   Extract → Scale (with loaded scaler) → Write LibSVM
```

### 6.3. Extract với CUDA Basic

```bash
cd /home/senyamiku/LTSS/FinalCuda

# Cú pháp: ./build_svm/extract_features_cuda <cifar_dir> <weights> [output_train] [output_test]
./build_svm/extract_features_cuda \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    train_features_cuda.libsvm \
    test_features_cuda.libsvm
```

**Output**:
- `train_features_cuda.libsvm` (~5.7GB, 50K samples, 8192 dims, scaled)
- `test_features_cuda.libsvm` (~1.2GB, 10K samples, scaled)
- `scaler_z.bin` (~97KB, mean/std statistics)
- `train_cache.bin` (~1.6GB, binary cache)

**Thời gian**: ~7-8 phút cho 60K images

**Kiểm tra output**:
```bash
# Check format (should see negative and positive values)
head -2 train_features_cuda.libsvm
```

Expected:
```
6 1:-1.348 2:-0.584 3:-0.453 ...
3 11:-1.224 12:-0.826 ...
```

### 6.4. Extract với OPT_V1

```bash
./build_svm/extract_features_cuda_opt_v1 \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    train_features_opt_v1.libsvm \
    test_features_opt_v1.libsvm
```

---

## 7. TRAIN SVM

### 7.1. Cài đặt ThunderSVM (recommended)

```bash
# Clone và build
git clone https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
mkdir build && cd build
cmake ..
make -j
```

Executables: `thundersvm-train`, `thundersvm-predict`

### 7.2. Train SVM với RBF kernel

```bash
# Default parameters (C=1, gamma=auto)
./thundersvm-train \
    -s 0 \
    -t 2 \
    -c 1.0 \
    -g 0.0001220703125 \
    train_features_cuda.libsvm \
    svm_model_cuda.txt
```

**Parameters**:
- `-s 0`: C-SVC classification
- `-t 2`: RBF kernel
- `-c 1.0`: Cost parameter
- `-g`: Gamma (1 / num_features = 1/8192 ≈ 0.000122)

**Thời gian**: ~5-15 phút tùy hardware

### 7.3. Grid search tối ưu (optional)

```bash
# Tìm C và gamma tối ưu
for c in 0.1 1 10 100; do
  for g in 0.00001 0.0001 0.001; do
    echo "Testing C=$c, gamma=$g"
    ./thundersvm-train -s 0 -t 2 -c $c -g $g \
      train_features_cuda.libsvm model_c${c}_g${g}.txt
  done
done
```

### 7.4. Predict và đánh giá

```bash
# Predict trên test set
./thundersvm-predict \
    test_features_cuda.libsvm \
    svm_model_cuda.txt \
    predictions.txt

# Accuracy sẽ được in ra console
```

**Expected accuracy**:
- Basic features (scaled): **~60-62%**
- OPT_V1 features (scaled): **~60-63%**
- Grid search optimized: **~63-65%**

---

## 8. SO SÁNH KẾT QUẢ

### 8.1. Training time comparison

| Version | 200 epochs | 3 epochs (1K images) | Speedup vs CPU |
|---------|------------|----------------------|----------------|
| CPU | ~45 min | ~2.5 min | 1x |
| CUDA Basic | ~9 min | ~45s | 5x |
| CUDA OPT_V1 | ~6.5 min | ~36s | 7x |
| CUDA OPT_V2 | ~4.5 min | ~26s | 10x |

### 8.2. Feature extraction time

| Version | 60K images | Features per second |
|---------|------------|---------------------|
| CPU | ~15-20 min | ~50-60 img/s |
| CUDA Basic | ~7-8 min | ~120-140 img/s |
| CUDA OPT_V1 | ~6-7 min | ~140-160 img/s |

### 8.3. SVM accuracy (CIFAR-10)

| Features | Scaling | Accuracy |
|----------|---------|----------|
| Basic | ❌ No | ~46% |
| Basic | ✅ Z-score | **~60-62%** |
| OPT_V1 | ✅ Z-score | **~60-63%** |
| Grid search | ✅ Z-score | **~63-65%** |

### 8.4. Optimization summary

**OPT_V1** (Memory focused):
- ✅ Memory coalescing (threadIdx.x for width dimension)
- ✅ Constant memory for conv1/conv5 (54KB)
- ✅ Removed redundant cudaMemset (2ms saved)
- ✅ Gradient clipping to [-1.0, 1.0]
- ✅ `__restrict__` pointers + `#pragma unroll`

**OPT_V2** (Speed focused):
- ✅ All OPT_V1 optimizations
- ✅ Kernel fusion (conv+bias+relu → 1 kernel)
- ✅ Vectorized float4 for SGD updates
- ✅ Vectorized float4 for MSE loss
- ✅ Tuned block dimensions (32×8, 16×16, 8×8)
- ✅ Specialized hardcoded 3×3 kernels
- ✅ `-O3 -use_fast_math` compiler flags

---

## 9. TROUBLESHOOTING

### 9.1. Gradient explosion

**Triệu chứng**: Loss tăng đột ngột từ ~0.3 → 50+

**Nguyên nhân**: 
- Missing cudaMemset cho gradient buffers
- Gradient clipping quá lỏng

**Giải pháp**:
```cpp
// Initialize gradient buffers
cudaMemset(d_grad_relu1, 0, size);
cudaMemset(d_grad_relu2, 0, size);

// Tighten gradient clipping
float val = fminf(fmaxf(grad, -1.0f), 1.0f);  // [-1.0, 1.0]
```

### 9.2. Out of memory

**Giải pháp**:
- Giảm batch size: 32 → 16
- Train với subset nhỏ hơn
- Sử dụng GPU có VRAM lớn hơn

### 9.3. SVM accuracy thấp (~46%)

**Nguyên nhân**: Features không được scale

**Giải pháp**: Đảm bảo đã chạy extract_features_cuda mới (có Z-score scaling)

```bash
# Kiểm tra scaler_z.bin có tồn tại
ls -lh scaler_z.bin

# Kiểm tra features có giá trị âm (đã scale)
head -2 train_features_cuda.libsvm
```

### 9.4. Build errors

**nvcc not found**:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Missing headers**:
```bash
# Kiểm tra include paths
ls include/
ls src/data/
```

---

## 📝 QUICK START SCRIPT

```bash
#!/bin/bash
# Full pipeline from scratch

cd /home/senyamiku/LTSS/FinalCuda

# 1. Build tất cả
./scripts/build_cuda.sh
./scripts/build_svm.sh

# 2. Train OPT_V2 (fastest)
./build_cuda/autoencoder_cuda_opt_v2 \
    cifar-10-binary/cifar-10-batches-bin \
    weights_opt_v2.bin \
    200 32 0.001

# 3. Extract features with Z-score scaling
./build_svm/extract_features_cuda \
    cifar-10-binary/cifar-10-batches-bin \
    weights_opt_v2.bin \
    train_scaled.libsvm \
    test_scaled.libsvm

# 4. Train SVM
thundersvm-train -s 0 -t 2 -c 1.0 -g 0.000122 \
    train_scaled.libsvm svm_model.txt

# 5. Predict
thundersvm-predict test_scaled.libsvm svm_model.txt predictions.txt

echo "Pipeline complete! Check accuracy above."
```

**Total time**: ~15-20 phút

---

## 📊 EXPECTED RESULTS

### Loss convergence (OPT_V2, 200 epochs)

```
Epoch 1/200:   loss 0.228 | time: 1.2s
Epoch 10/200:  loss 0.087 | time: 1.1s
Epoch 50/200:  loss 0.043 | time: 1.1s
Epoch 100/200: loss 0.028 | time: 1.1s
Epoch 200/200: loss 0.019 | time: 1.1s
Total: ~4 min 30s
```

### Feature extraction output

```
=== CUDA Feature Extraction for SVM (With Z-Score Scaling) ===
PASS 1: Extracting train features + computing statistics...
  Completed 50000/50000
  Statistics computed (mean/std for 8192 dims)
PASS 2: Scaling and writing train features...
  Completed: 50000 samples
Extracting and scaling test features...
  Completed: 10000 samples
Total time: 429s

Scaling statistics:
  Samples:  50000
  Features: 8192
  Example mean[0]:   0.160194
  Example stddev[0]: 0.118832
```

### SVM training output

```
*
optimization finished, #iter = 12543
obj = -8234.567, rho = -0.123
nSV = 18234, nBSV = 15123
Total nSV = 18234
*
Accuracy = 62.34% (6234/10000)
```

---

## 🎯 KẾT LUẬN

Pipeline đã được optimize qua 3 giai đoạn:

1. **CUDA Basic** → **OPT_V1**: ~30% faster (memory optimization)
2. **OPT_V1** → **OPT_V2**: ~30% faster (speed optimization)
3. **Overall speedup**: **~10x faster than CPU**

Feature scaling (Z-score) cải thiện SVM accuracy từ **46% → 62%** (+16%).

Kết hợp cả 2 optimizations đạt được:
- ⚡ Training time: **4-5 phút** (vs 45 phút CPU)
- 📈 SVM accuracy: **~62%** với CIFAR-10
- 💾 Memory efficient với 2-pass pipeline

---

**Author**: CSC14120 Final Project  
**Date**: December 2025  
**GPU**: NVIDIA RTX 3050 Laptop  
**CUDA**: 12.0
