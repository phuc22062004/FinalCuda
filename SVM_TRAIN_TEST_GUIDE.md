# SVM Training and Testing with Model Persistence

## Overview

Bây giờ bạn có 2 file riêng biệt:
1. **training_svm.py** - Train model và lưu model file
2. **testing_svm.py** - Load model đã train và test (KHÔNG cần train lại)

## Workflow

### Bước 1: Train và lưu model

```bash
python src/svm/training_svm.py \
    --train train_features_opt_v1.libsvm \
    --test test_features_opt_v1.libsvm \
    --C 10.0 \
    --gamma scale \
    --save-model svm_model_cuml.pkl \
    --predictions predictions_train.txt \
    --cm-output confusion_matrix_train.png
```

**Output:**
- `svm_model_cuml.pkl` - Model đã train (file này quan trọng!)
- `predictions_train.txt` - Predictions trên test set
- `confusion_matrix_train.png` - Confusion matrix

### Bước 2: Test với model đã lưu (không cần train lại)

```bash
python src/svm/testing_svm.py \
    --model svm_model_cuml.pkl \
    --test test_features_opt_v1.libsvm \
    --predictions predictions_test.txt \
    --cm-output confusion_matrix_test.png
```

**Lợi ích:**
- ✅ Không cần train lại (tiết kiệm thời gian)
- ✅ Test trên nhiều test set khác nhau
- ✅ Dùng model để inference sau này
- ✅ Share model với người khác

## Use Cases

### 1. Train một lần, test nhiều lần

```bash
# Train 1 lần
python src/svm/training_svm.py \
    --train train_features.libsvm \
    --test test_features.libsvm \
    --save-model my_model.pkl

# Test nhiều lần với datasets khác nhau
python src/svm/testing_svm.py --model my_model.pkl --test test1.libsvm
python src/svm/testing_svm.py --model my_model.pkl --test test2.libsvm
python src/svm/testing_svm.py --model my_model.pkl --test test3.libsvm
```

### 2. Train với hyperparameters khác nhau

```bash
# Train với C=1.0
python src/svm/training_svm.py \
    --train train.libsvm --test test.libsvm \
    --C 1.0 --save-model model_C1.pkl

# Train với C=10.0
python src/svm/training_svm.py \
    --train train.libsvm --test test.libsvm \
    --C 10.0 --save-model model_C10.pkl

# Train với C=100.0
python src/svm/training_svm.py \
    --train train.libsvm --test test.libsvm \
    --C 100.0 --save-model model_C100.pkl

# So sánh các models
python src/svm/testing_svm.py --model model_C1.pkl --test test.libsvm
python src/svm/testing_svm.py --model model_C10.pkl --test test.libsvm
python src/svm/testing_svm.py --model model_C100.pkl --test test.libsvm
```

### 3. Production deployment

```bash
# Train trên toàn bộ data
python src/svm/training_svm.py \
    --train all_train_features.libsvm \
    --test validation_features.libsvm \
    --save-model production_model.pkl

# Deploy: chỉ cần file production_model.pkl
# Testing script rất nhanh (chỉ inference, không train)
python src/svm/testing_svm.py \
    --model production_model.pkl \
    --test new_data.libsvm
```

## Parameters

### training_svm.py

**Required:**
- `--train`: Training features file
- `--test`: Test features file

**Optional:**
- `--C`: SVM C parameter (default: 10.0)
- `--gamma`: Gamma parameter (default: scale)
- `--kernel`: Kernel type: rbf/linear/poly/sigmoid (default: rbf)
- `--max-iter`: Maximum iterations (default: -1)
- `--tol`: Tolerance (default: 1e-3)
- `--cache-size`: Cache size MB (default: 2000)
- `--save-model`: Model output file (default: svm_model_cuml.pkl)
- `--predictions`: Predictions output (default: predictions_cuml.txt)
- `--cm-output`: Confusion matrix image (default: confusion_matrix_cuml.png)
- `--no-plot`: Skip plotting confusion matrix

### testing_svm.py

**Required:**
- `--model`: Pre-trained model file (.pkl)
- `--test`: Test features file

**Optional:**
- `--predictions`: Predictions output (default: predictions_test.txt)
- `--cm-output`: Confusion matrix image (default: confusion_matrix_test.png)
- `--no-plot`: Skip plotting confusion matrix

## Example Complete Workflow

```bash
# 1. Extract features (nếu chưa có)
./build_svm/extract_features_cuda_opt_v1 \
    cifar-10-binary/cifar-10-batches-bin \
    autoencoder_cuda_opt_v1_weights.bin \
    train_features_opt_v1.libsvm \
    test_features_opt_v1.libsvm

# 2. Train SVM và lưu model
python src/svm/training_svm.py \
    --train train_features_opt_v1.libsvm \
    --test test_features_opt_v1.libsvm \
    --C 10.0 \
    --gamma scale \
    --kernel rbf \
    --save-model svm_cifar10_best.pkl \
    --predictions predictions_initial.txt \
    --cm-output cm_initial.png

# 3. Test lại với model đã lưu (rất nhanh!)
python src/svm/testing_svm.py \
    --model svm_cifar10_best.pkl \
    --test test_features_opt_v1.libsvm \
    --predictions predictions_final.txt \
    --cm-output cm_final.png
```

## Model File Info

**Model file (`.pkl`):**
- Chứa toàn bộ trained SVM model
- Bao gồm support vectors, coefficients, hyperparameters
- Kích thước: thường 50-500 MB tùy vào số support vectors
- Format: Python pickle (binary)

**Lưu ý:**
- Model file chỉ hoạt động với cuML installed
- Nếu chuyển sang máy khác, cần cài cuML phiên bản tương thích
- Model file có thể lớn nếu dataset phức tạp

## Performance Comparison

| Operation | training_svm.py | testing_svm.py |
|-----------|-----------------|----------------|
| Load data | ✓ | ✓ |
| Train model | ✓ (slow) | ✗ |
| Save model | ✓ | ✗ |
| Load model | ✗ | ✓ (fast) |
| Predict | ✓ | ✓ |
| Total time | ~15-60s | ~2-5s |

**Testing nhanh hơn 10-20x so với training!**

## Troubleshooting

### Model file too large
```bash
# Nếu model file quá lớn (>500MB), thử:
python src/svm/training_svm.py \
    --kernel linear \  # Linear kernel nhỏ hơn RBF
    --C 1.0           # C nhỏ hơn → ít support vectors hơn
```

### Cannot load model on different machine
```bash
# Đảm bảo cài đúng phiên bản cuML
conda install -c rapidsai -c conda-forge cuml=24.12
```

### Model accuracy different after loading
- Không nên xảy ra! Model được lưu hoàn chỉnh
- Kiểm tra test data có giống nhau không
- Kiểm tra cuML version

## Tips

1. **Lưu model với tên có ý nghĩa:**
   ```bash
   --save-model cifar10_C10_rbf_acc65.pkl
   ```

2. **Backup model files:**
   ```bash
   cp svm_model_cuml.pkl models/backup/
   ```

3. **Test nhanh nhiều lần:**
   ```bash
   # Một khi có model, test rất nhanh!
   for i in {1..10}; do
       python src/svm/testing_svm.py --model model.pkl --test test$i.libsvm
   done
   ```

4. **Version control:**
   - Không commit model files vào git (quá lớn)
   - Lưu model trên cloud storage hoặc model registry
   - Ghi lại hyperparameters trong tên file
