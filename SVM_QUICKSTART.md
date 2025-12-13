# Phase 4: SVM Integration - Quick Start Guide

Bạn đã hoàn thành Phase 4: SVM Integration! Đây là hướng dẫn nhanh để sử dụng.

## Tổng quan

Pipeline SVM bao gồm 2 bước chính:
1. **Trích xuất features** từ autoencoder đã train (bottleneck layer: 128×8×8 = 8192 features)
2. **Train SVM classifier** trên features đã trích xuất

## Cài đặt ThunderSVM (tùy chọn nhưng được khuyến nghị)

ThunderSVM hỗ trợ cả CPU và GPU, nhanh hơn nhiều so với sklearn's SVM:

```bash
# Cài đặt qua pip (đơn giản nhất)
pip install thundersvm

# HOẶC từ source (cho phiên bản mới nhất)
git clone https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
mkdir build && cd build
cmake ..
make -j
sudo make install
cd python
pip install .
```

Nếu không cài ThunderSVM, script sẽ tự động dùng sklearn (chỉ CPU).

## Cách sử dụng nhanh

### Option 1: Chạy toàn bộ pipeline (khuyến nghị)

```bash
# Cho executable quyền thực thi
chmod +x scripts/run_svm_pipeline.sh

# Chạy với CUDA features + CPU SVM
./scripts/run_svm_pipeline.sh

# Chạy với CUDA features + GPU SVM (nhanh hơn, cần ThunderSVM)
./scripts/run_svm_pipeline.sh --svm-gpu
```

### Option 2: Chạy từng bước

#### Bước 1: Trích xuất features

**Dùng CUDA-trained model (khuyến nghị)**:
```bash
./build_svm/extract_features_cuda \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_basic_weights.bin \
    train_features_cuda.libsvm \
    test_features_cuda.libsvm
```

**Dùng CPU-trained model**:
```bash
./build_svm/extract_features_cpu \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_weights.bin \
    train_features_cpu.libsvm \
    test_features_cpu.libsvm
```

#### Bước 2: Train và test SVM

**CPU SVM**:
```bash
python3 src/svm/svm_train_test.py \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 10.0 \
    --gamma auto \
    --output confusion_matrix.png
```

**GPU SVM (nhanh hơn, cần ThunderSVM)**:
```bash
python3 src/svm/svm_train_test.py \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 10.0 \
    --gamma auto \
    --output confusion_matrix.png \
    --gpu
```

## Kết quả mong đợi

| Metric | Target |
|--------|--------|
| Feature extraction time | < 20 giây cho 60K ảnh |
| SVM training time | 1-5 phút |
| Test accuracy | 60-65% |

## Các file output

- `train_features_*.libsvm`: Training features (LibSVM format)
- `test_features_*.libsvm`: Test features (LibSVM format)
- `confusion_matrix*.png`: Confusion matrix visualization

## Tuning SVM parameters

Thử các giá trị khác nhau để cải thiện accuracy:

```bash
# C parameter (regularization)
python3 src/svm/svm_train_test.py ... --C 1.0    # Ít regularization hơn
python3 src/svm/svm_train_test.py ... --C 100.0  # Nhiều regularization hơn

# Gamma parameter (RBF kernel)
python3 src/svm/svm_train_test.py ... --gamma 0.001
python3 src/svm/svm_train_test.py ... --gamma 0.01
```

## So sánh CPU vs GPU

Để so sánh hiệu năng:

```bash
# Extract features với CUDA
./build_svm/extract_features_cuda ... (ghi nhận thời gian)

# Train với CPU SVM
time python3 src/svm/svm_train_test.py ... 

# Train với GPU SVM  
time python3 src/svm/svm_train_test.py ... --gpu
```

## Sử dụng với optimized versions (opt_v1, opt_v2)

Hiện tại chỉ hỗ trợ `autoencoder_basic`. Để dùng với opt_v1/v2:

1. Implement method `extract_features()` trong opt_v1.cu và opt_v2.cu
2. Build feature extractors cho các version đó
3. Chạy pipeline với weights tương ứng

## Troubleshooting

**ThunderSVM không tìm thấy**
- Script sẽ tự động fallback sang sklearn (CPU only)
- Cài ThunderSVM để dùng GPU

**CUDA out of memory khi extract features**
- Hiện tại đã xử lý từng ảnh một nên không nên bị vấn đề này
- Nếu vẫn bị, kiểm tra GPU memory

**Accuracy thấp (<50%)**
- Kiểm tra autoencoder đã train đủ chưa
- Thử các SVM parameters khác (C, gamma)
- Kiểm tra feature extraction có đúng không

## Tích hợp vào Report

Khi viết báo cáo Phase 4, nhớ bao gồm:
1. Feature extraction time
2. SVM training time  
3. Test accuracy
4. Confusion matrix
5. Per-class accuracy
6. So sánh CPU vs GPU (cả autoencoder và SVM)

Xem thêm chi tiết trong `src/svm/README.md`.

## Next Steps

Sau khi hoàn thành Phase 4:
1. ✅ Analyze results (confusion matrix, per-class accuracy)
2. ✅ Compare với baseline methods
3. ✅ Document findings trong Report.ipynb
4. 🎯 (Optional) Implement opt_v1 và opt_v2 để so sánh speedup

Chúc bạn thành công! 🚀
