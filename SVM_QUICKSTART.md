# Phase 4: SVM Integration - Quick Start Guide

Bạn đã hoàn thành Phase 4: SVM Integration! Đây là hướng dẫn nhanh để sử dụng.

## ⚡ Yêu cầu trước khi chạy

- ✅ ThunderSVM đã cài đặt (C++ executables: `thundersvm-train`, `thundersvm-predict`)
- ✅ Autoencoder đã train xong (`autoencoder_cuda_basic_weights.bin`)
- ✅ CIFAR-10 dataset tại `./cifar-10-binary/cifar-10-batches-bin`

## 📋 Tổng quan

Pipeline SVM bao gồm 2 bước chính:
1. **Trích xuất features** từ autoencoder đã train (bottleneck layer: 128×8×8 = 8192 features)
2. **Train SVM classifier** trên features đã trích xuất bằng ThunderSVM (C++)

## 🔧 Cài đặt ThunderSVM (nếu chưa có)

ThunderSVM hỗ trợ cả CPU và GPU, nhanh hơn nhiều so với sklearn's SVM:

```bash
# Build từ source
git clone https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
mkdir build && cd build

# Build với GCC-11 (tương thích với CUDA 12.0)
cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
make -j

# Install (cần sudo)
sudo make install

# Update library cache
sudo ldconfig
```

Verify cài đặt:
```bash
which thundersvm-train thundersvm-predict
# Phải thấy: /usr/local/bin/thundersvm-train và /usr/local/bin/thundersvm-predict
```

## 🚀 Cách sử dụng

### ⭐ Option 1: Chạy toàn bộ pipeline tự động (KHUYẾN NGHỊ)

```bash
# Cho executable quyền thực thi (chỉ cần 1 lần)
chmod +x scripts/run_svm_pipeline.sh scripts/svm_train_test_cpp.sh

# Chạy với CPU SVM (an toàn, hoạt động trên mọi môi trường)
./scripts/run_svm_pipeline.sh

# Chạy với GPU SVM (nhanh hơn, nhưng có thể không hoạt động trên WSL2)
./scripts/run_svm_pipeline.sh --svm-gpu
```

Pipeline tự động sẽ:
1. ✅ Extract features từ CUDA autoencoder
2. ✅ Train SVM classifier
3. ✅ Predict trên test set
4. ✅ Tính accuracy và hiển thị kết quả

### Option 2: Chạy từng bước (chi tiết)

#### Bước 0: Build feature extraction tools (nếu chưa build)

```bash
chmod +x scripts/build_svm.sh
./scripts/build_svm.sh
```

Kiểm tra:
```bash
ls -lh build_svm/
# Phải thấy: extract_features_cuda và extract_features_cpu
```

#### Bước 1: Trích xuất features

**Dùng CUDA-trained model (KHUYẾN NGHỊ)**:
```bash
./build_svm/extract_features_cuda \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    train_features_cuda.libsvm \
    test_features_cuda.libsvm
```

Thời gian: ~2-3 phút cho 60,000 ảnh
Output: 
- `train_features_cuda.libsvm` (~3.6 GB, 50,000 samples)
- `test_features_cuda.libsvm` (~733 MB, 10,000 samples)

**Dùng CPU-trained model**:
```bash
./build_svm/extract_features_cpu \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_weights.bin \
    train_features_cpu.libsvm \
    test_features_cpu.libsvm
```

#### Bước 2: Train và test SVM

**CPU SVM (KHUYẾN NGHỊ cho WSL2)**:
```bash
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm
```

**GPU SVM (nhanh hơn, nhưng WSL2 có thể không hỗ trợ)**:
```bash
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --gpu
```

**Tuning parameters**:
```bash
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 100.0 \
    --gamma 0.001
```

## 📊 Kết quả mong đợi

| Metric | CPU | GPU | Note |
|--------|-----|-----|------|
| Feature extraction time | ~2-3 phút | ~2-3 phút | 60K ảnh, 8192 dims |
| SVM training time | 3-10 phút | 30s - 2 phút | Tùy dataset size |
| Test accuracy | 60-65% | 60-65% | CIFAR-10, 10 classes |

## 📁 Các file output

- `train_features_cuda.libsvm`: Training features (~3.6 GB, 50K samples × 8192 dims)
- `test_features_cuda.libsvm`: Test features (~733 MB, 10K samples × 8192 dims)
- `svm_model.txt`: Trained SVM model
- `predictions.txt`: Predictions trên test set
- `svm_results_*.log`: Kết quả training và testing

## 🎯 Tuning SVM hyperparameters

Thử các giá trị khác nhau để cải thiện accuracy:

```bash
# C parameter (regularization strength)
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 1.0      # Weak regularization
    
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 100.0    # Strong regularization

# Gamma parameter (RBF kernel width)
./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --gamma 0.001

./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --gamma 0.01
```

## ⚡ So sánh CPU vs GPU

Để đo thời gian chính xác:

```bash
# Feature extraction (CUDA)
time ./build_svm/extract_features_cuda \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    train_features_cuda.libsvm \
    test_features_cuda.libsvm

# Train với CPU SVM
time ./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm

# Train với GPU SVM (nếu hoạt động)
time ./scripts/svm_train_test_cpp.sh \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --gpu
```

**⚠️ Lưu ý WSL2**: GPU SVM có thể không hoạt động trên WSL2 do ThunderSVM không detect được CUDA device. Dùng CPU mode thay thế.

## 🔍 Troubleshooting

### ThunderSVM không tìm thấy
```bash
# Kiểm tra cài đặt
which thundersvm-train thundersvm-predict

# Nếu không có, build và install lại:
cd thundersvm/build
sudo make install
sudo ldconfig
```

### GPU detection failed (WSL2)
```
FATAL: no CUDA-capable device is detected
```
**Giải pháp**: Dùng CPU mode thay vì `--gpu` flag:
```bash
./scripts/run_svm_pipeline.sh  # Không có --svm-gpu
```

### CUDA out of memory khi extract features
- Feature extraction đã xử lý từng ảnh một nên không nên bị vấn đề này
- Nếu vẫn bị, kiểm tra GPU memory: `nvidia-smi`

### Accuracy thấp (<50%)
1. Kiểm tra autoencoder đã train đủ epochs chưa (target loss < 0.01)
2. Thử các SVM hyperparameters khác (C=1.0, 10.0, 100.0)
3. Kiểm tra feature extraction có đúng không: `head -n 5 train_features_cuda.libsvm`

### File quá lớn
- `train_features_cuda.libsvm`: ~3.6 GB (bình thường)
- `test_features_cuda.libsvm`: ~733 MB (bình thường)
- Cần ~5GB free disk space

## 📝 Tích hợp vào Report

Khi viết báo cáo Phase 4, nhớ bao gồm:

1. **Feature Extraction**
   - Thời gian trích xuất features
   - Số chiều features (8192)
   - Kích thước file output

2. **SVM Training**
   - SVM hyperparameters (C, gamma, kernel type)
   - Thời gian training
   - CPU vs GPU (nếu có)

3. **Results**
   - Test accuracy (%)
   - Per-class accuracy
   - Confusion matrix (nếu có)

4. **Comparison**
   - So sánh với baseline (random: 10%, simple classifier: ~50%)
   - So sánh CPU vs CUDA autoencoder
   - So sánh CPU vs GPU SVM

## 🔧 Sử dụng với Optimized Versions

Hiện tại chỉ hỗ trợ `autoencoder_basic`. Để dùng với opt_v1/v2:

1. Implement method `extract_features()` trong `autoencoder_opt_v1.cu` và `autoencoder_opt_v2.cu`
2. Build feature extractors cho các version đó trong [scripts/build_svm.sh](scripts/build_svm.sh)
3. Chạy pipeline với weights tương ứng:
   ```bash
   ./build_svm/extract_features_cuda \
       ./cifar-10-binary/cifar-10-batches-bin \
       autoencoder_cuda_opt_v1_weights.bin \
       train_features_opt_v1.libsvm \
       test_features_opt_v1.libsvm
   ```

## ✅ Next Steps

Sau khi hoàn thành Phase 4:
1. ✅ Analyze results (accuracy, per-class metrics)
2. ✅ So sánh với baseline và CPU version
3. ✅ Document findings trong [Report.ipynb](Report.ipynb)
4. 🎯 (Optional) Implement opt_v1 và opt_v2 để so sánh speedup

## 📚 Tham khảo thêm

- ThunderSVM Documentation: https://github.com/Xtra-Computing/thundersvm
- LibSVM Format: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- Chi tiết implementation: [src/svm/README.md](src/svm/README.md)

---

**Chúc bạn thành công! 🚀**

Nếu cần hỗ trợ, xem [PHASE4_COMPLETE.md](PHASE4_COMPLETE.md) để biết thêm chi tiết.
