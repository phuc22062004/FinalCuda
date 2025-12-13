# Hướng dẫn: Train Autoencoder OPT_V1 và SVM Classification

## Tổng quan
Pipeline hoàn chỉnh để train autoencoder opt_v1 (memory optimized) và sử dụng nó cho SVM classification trên CIFAR-10.

---

## Bước 1: Train Autoencoder OPT_V1

### 1.1. Build executable (nếu chưa build)
```bash
cd /home/senyamiku/LTSS/FinalCuda
./scripts/build_cuda.sh
```

**Output mong đợi:**
- `build_cuda/autoencoder_cuda_opt_v1` executable được tạo

### 1.2. Train model với tham số đầy đủ

**Cú pháp:**
```bash
./build_cuda/autoencoder_cuda_opt_v1 <cifar_dir> <weights_file> <epochs> [batch_size] [learning_rate] [max_images]
```

**Các tham số:**
- `cifar_dir`: Đường dẫn đến CIFAR-10 dataset
- `weights_file`: Tên file để lưu weights
- `epochs`: Số epochs (khuyến nghị: 200)
- `batch_size`: Kích thước batch (default: 64)
- `learning_rate`: Learning rate (default: 0.001)
- `max_images`: Số ảnh training (default: 50000)

**Ví dụ training đầy đủ (200 epochs, toàn bộ dataset):**
```bash
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    200 64 0.001 50000
```

**Ví dụ training nhanh (test):**
```bash
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_test.bin \
    3 32 0.001 1000
```

**Output mong đợi:**
```
=== CUDA OPT_V1 (Memory Optimized) ===
CIFAR dir: ./cifar-10-binary/cifar-10-batches-bin
Weights:   autoencoder_cuda_opt_v1_weights.bin
Epochs:    200
Batch:     64
LR:        0.001
Max train: 50000

Epoch 1/200 - shuffling...
  Batch 1 loss: 0.293899
  ...
Epoch avg loss: 0.0123 | time: 10s
...
Total training time: 2000s
Saving weights to autoencoder_cuda_opt_v1_weights.bin
Done.
```

**Thời gian ước tính:**
- 200 epochs, 50K images: ~30-40 phút
- 3 epochs, 1K images: ~30 giây

---

## Bước 2: Build SVM Feature Extraction Tools

### 2.1. Build feature extractor
```bash
./scripts/build_svm.sh
```

**Output mong đợi:**
- `build_svm/extract_features_cuda_opt_v1` executable được tạo

### 2.2. Verify executable
```bash
ls -lh build_svm/extract_features_cuda_opt_v1
```

---

## Bước 3: Extract Features từ Autoencoder

### 3.1. Extract features cho SVM

**Cú pháp:**
```bash
./build_svm/extract_features_cuda_opt_v1 \
    <cifar_dir> \
    <weights_file> \
    <output_train_features> \
    <output_test_features>
```

**Ví dụ:**
```bash
./build_svm/extract_features_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    train_features_opt_v1.libsvm \
    test_features_opt_v1.libsvm
```

**Output mong đợi:**
```
=== CUDA Feature Extraction for SVM ===
CIFAR-10 dir: ./cifar-10-binary/cifar-10-batches-bin
Weights:      autoencoder_cuda_opt_v1_weights.bin
Output train: train_features_opt_v1.libsvm
Output test:  test_features_opt_v1.libsvm

Loaded train images: 50000
Loaded test images:  10000
Loaded autoencoder weights

Extracting training features...
  Processed 0/50000
  Processed 5000/50000
  ...
  Processed 50000/50000

Extracting test features...
  Processed 0/10000
  ...
  Processed 10000/10000

Feature extraction completed!
Time: 123s
```

**Thời gian ước tính:** ~2-3 phút cho 60K images

**Verify output files:**
```bash
ls -lh train_features_opt_v1.libsvm test_features_opt_v1.libsvm
```

Kích thước mong đợi:
- `train_features_opt_v1.libsvm`: ~3.6 GB (50K samples × 8192 features)
- `test_features_opt_v1.libsvm`: ~733 MB (10K samples × 8192 features)

---

## Bước 4: Train và Test SVM

### 4.1. Train SVM với ThunderSVM (CPU mode)

**Cú pháp:**
```bash
./scripts/svm_train_test_cpp.sh \
    --train <train_features> \
    --test <test_features> \
    [--C <value>] \
    [--gamma <value>]
```

**Ví dụ với default parameters:**
```bash
./scripts/svm_train_test_cpp.sh \
    --train train_features_opt_v1.libsvm \
    --test test_features_opt_v1.libsvm
```

**Ví dụ với custom parameters:**
```bash
./scripts/svm_train_test_cpp.sh \
    --train train_features_opt_v1.libsvm \
    --test test_features_opt_v1.libsvm \
    --C 100.0 \
    --gamma 0.001
```

**Output mong đợi:**
```
==========================================
ThunderSVM C++ - SVM Training & Testing
==========================================
Training file:    train_features_opt_v1.libsvm
Test file:        test_features_opt_v1.libsvm
Model file:       svm_model.txt
Prediction file:  predictions.txt
C parameter:      10.0
Gamma:            auto
GPU enabled:      0
==========================================

Step 1: Training SVM...
...
Training completed!

Step 2: Testing SVM...
...
Testing completed!

Step 3: Calculating accuracy...

==========================================
RESULTS
==========================================
Test Accuracy: 62.45%

Per-class Accuracy:
  Class 0 (airplane):   65.2%
  Class 1 (automobile): 70.8%
  Class 2 (bird):       52.3%
  ...
==========================================
```

**Thời gian ước tính:**
- CPU mode: 5-10 phút
- GPU mode (nếu hoạt động): 1-2 phút

---

## Bước 5: Chạy Toàn Bộ Pipeline Tự Động (Khuyến nghị)

### 5.1. Sử dụng script tự động

Thay vì chạy từng bước, bạn có thể dùng script tự động:

```bash
./scripts/run_svm_opt_v1.sh
```

Script này sẽ:
1. ✅ Kiểm tra weights file có tồn tại không
2. ✅ Extract features từ opt_v1
3. ✅ Train SVM
4. ✅ Test và hiển thị kết quả

**Nếu chưa có weights, train trước:**
```bash
# Step 1: Train autoencoder
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    200

# Step 2: Run SVM pipeline
./scripts/run_svm_opt_v1.sh
```

---

## Tổng kết Commands Đầy Đủ

### Quick Start (Training nhanh để test)
```bash
# 1. Train autoencoder (3 epochs, 1000 images)
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_test.bin \
    3 32 0.001 1000

# 2. Extract features
./build_svm/extract_features_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_test.bin \
    train_features_opt_v1_test.libsvm \
    test_features_opt_v1_test.libsvm

# 3. Train SVM
./scripts/svm_train_test_cpp.sh \
    --train train_features_opt_v1_test.libsvm \
    --test test_features_opt_v1_test.libsvm
```

### Full Pipeline (Production)
```bash
# 1. Train autoencoder (200 epochs, 50K images)
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    200 64 0.001 50000

# 2. Run SVM pipeline (tự động extract + train + test)
./scripts/run_svm_opt_v1.sh
```

---

## Kết quả Mong Đợi

### Autoencoder Training
- **Final Loss**: < 0.015 (sau 200 epochs)
- **Training time**: ~30-40 phút (200 epochs, 50K images)

### Feature Extraction
- **Time**: ~2-3 phút (60K images)
- **Feature dimension**: 8192 (128 × 8 × 8)

### SVM Classification
- **Test Accuracy**: 60-65%
- **Training time**: 5-10 phút (CPU)

---

## Troubleshooting

### Lỗi: Weights file not found
```bash
# Kiểm tra file có tồn tại không
ls -lh autoencoder_cuda_opt_v1_weights.bin

# Nếu không có, train lại
./build_cuda/autoencoder_cuda_opt_v1 ./cifar-10-binary/cifar-10-batches-bin autoencoder_cuda_opt_v1_weights.bin 200
```

### Lỗi: CUDA out of memory
```bash
# Giảm batch size
./build_cuda/autoencoder_cuda_opt_v1 ... 32 ...  # thay vì 64
```

### Lỗi: ThunderSVM GPU detection failed
```bash
# Dùng CPU mode (bỏ --gpu flag)
./scripts/svm_train_test_cpp.sh --train ... --test ...
```

### Lỗi: Feature extraction quá chậm
```bash
# Check GPU có đang được sử dụng không
nvidia-smi

# Nếu GPU không hoạt động, file vẫn sẽ được tạo nhưng chậm hơn
```

---

## Memory Optimizations trong OPT_V1

OPT_V1 giảm memory usage bằng cách:
1. **In-place ReLU**: ReLU không cần output buffer riêng
2. **Gradient buffer reuse**: Dùng lại buffer cho nhiều layer
3. **Reduced allocations**: Giảm số lượng cudaMalloc calls

**So sánh với Basic:**
- Memory saved: ~20-30%
- Speed: Tương đương hoặc nhanh hơn 5-10%

---

## Next Steps

Sau khi hoàn thành pipeline OPT_V1:
1. ✅ So sánh accuracy với Basic version
2. ✅ Document kết quả trong Report.ipynb
3. 🎯 (Optional) Implement OPT_V2 (speed optimization)
4. 🎯 (Optional) Fine-tune SVM hyperparameters

---

**Good luck! 🚀**
