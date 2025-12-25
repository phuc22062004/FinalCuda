# Using cuML SVM for CIFAR-10 Classification

## Overview

Thay vì sử dụng ThunderSVM, bạn có thể dùng cuML (RAPIDS AI) để train SVM trên GPU. cuML có nhiều ưu điểm:

- ✅ Dễ cài đặt hơn (chỉ cần pip/conda)
- ✅ Tích hợp tốt với Python ecosystem
- ✅ API tương tự sklearn
- ✅ GPU acceleration mạnh mẽ
- ✅ Hỗ trợ tốt hơn cho các tác vụ ML khác

## Installation

### Option 1: Conda (Recommended)

```bash
# Create new environment
conda create -n rapids-24.12 -c rapidsai -c conda-forge -c nvidia \
    cuml=24.12 python=3.11 cuda-version=12.2

# Activate environment
conda activate rapids-24.12

# Install additional packages
pip install matplotlib seaborn scikit-learn
```

### Option 2: Pip (For CUDA 12.x)

```bash
pip install cuml-cu12 matplotlib seaborn scikit-learn
```

### Option 3: Docker

```bash
docker pull rapidsai/rapidsai:24.12-cuda12.2-runtime-ubuntu22.04-py3.11

docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    rapidsai/rapidsai:24.12-cuda12.2-runtime-ubuntu22.04-py3.11
```

## Usage

### Complete Pipeline (Extract Features + Train SVM)

```bash
# Make script executable
chmod +x scripts/run_svm_pipeline_cuml.sh

# Run complete pipeline
./scripts/run_svm_pipeline_cuml.sh
```

### Train SVM Only (If features already extracted)

```bash
python3 scripts/svm_train_test_cuml.py \
    --train train_features_cuda.libsvm \
    --test test_features_cuda.libsvm \
    --C 10.0 \
    --gamma scale \
    --kernel rbf \
    --cm-output confusion_matrix_cuml.png \
    --predictions predictions_cuml.txt
```

### Available Options

```bash
python3 scripts/svm_train_test_cuml.py --help
```

Options:
- `--train`: Training features file (LibSVM format)
- `--test`: Test features file (LibSVM format)
- `--C`: SVM C parameter (default: 10.0)
- `--gamma`: Gamma parameter - 'scale', 'auto', or float (default: scale)
- `--kernel`: Kernel type - 'rbf', 'linear', 'poly', 'sigmoid' (default: rbf)
- `--cm-output`: Confusion matrix output file (default: confusion_matrix_cuml.png)
- `--predictions`: Predictions output file (default: predictions_cuml.txt)

## Comparison: cuML vs ThunderSVM

| Feature | cuML | ThunderSVM |
|---------|------|------------|
| Installation | ✅ Easy (pip/conda) | ⚠️ Complex (build from source) |
| API | ✅ sklearn-like | ⚠️ Custom interface |
| Multi-class | ✅ Built-in | ✅ Built-in |
| Kernels | RBF, Linear, Poly, Sigmoid | RBF, Linear, Poly, Sigmoid |
| Python integration | ✅ Excellent | ⚠️ Limited |
| Speed | ⚠️ Fast (GPU) | ✅ Very fast (GPU) |
| Memory | ✅ Efficient | ✅ Efficient |
| Ecosystem | ✅ RAPIDS AI suite | ⚠️ Standalone |

## Example Output

```
==================================================
cuML SVM - Training & Testing Pipeline
==================================================
Training file:    train_features_cuda.libsvm
Test file:        test_features_cuda.libsvm
C parameter:      10.0
Gamma:            scale
Kernel:           rbf
==================================================

[1/4] Loading data...
✓ cuML available - using GPU acceleration
Loading features from: train_features_cuda.libsvm
Loading features from: test_features_cuda.libsvm

[2/4] Training SVM...
==================================================
Training SVM Classifier
==================================================
Training samples: 50000
Feature dimension: 8192
Number of classes: 10
C parameter: 10.0
Gamma: scale
Kernel: rbf
Backend: cuML (GPU)
==================================================

Training started...
✓ Training completed in 15.32 seconds

[3/4] Testing SVM...
==================================================
Testing SVM Classifier
==================================================
Test samples: 10000
Predicting...
✓ Prediction completed in 2.15 seconds

[4/4] Computing metrics...
==================================================
CLASSIFICATION RESULTS
==================================================
Total samples:     10000
Correct:           6543
Incorrect:         3457
Accuracy:          65.43%
==================================================

Per-class Accuracy:
--------------------------------------------------
  airplane     (class 0): 70.20%
  automobile   (class 1): 75.80%
  bird         (class 2): 52.30%
  cat          (class 3): 48.90%
  deer         (class 4): 60.10%
  dog          (class 5): 55.70%
  frog         (class 6): 73.20%
  horse        (class 7): 68.40%
  ship         (class 8): 76.50%
  truck        (class 9): 73.20%

✓ Predictions saved to: predictions_cuml.txt
✓ Confusion matrix saved to: confusion_matrix_cuml.png

==================================================
SUMMARY
==================================================
Training time:     15.32 seconds
Testing time:      2.15 seconds
Total time:        17.47 seconds
Final accuracy:    65.43%
Backend:           cuML (GPU)
==================================================
```

## Troubleshooting

### cuML not available

If you see "cuML not available", the script will automatically fall back to sklearn (CPU only):

```bash
⚠ cuML not available - falling back to sklearn (CPU)
```

To fix, install cuML properly:

```bash
# Check CUDA version
nvcc --version

# Install matching cuML version
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml cuda-version=12.2
```

### GPU Memory Error

If you get GPU out of memory error, try:

1. Reduce batch size in feature extraction
2. Use smaller C parameter
3. Use linear kernel instead of RBF

### Slow Performance

If cuML is slow:

1. Make sure CUDA drivers are up to date
2. Check GPU is being used: `nvidia-smi`
3. Try different gamma values
4. Consider using linear kernel for faster training

## Hyperparameter Tuning

For better accuracy, try grid search:

```python
# Example grid search with cuML
from cuml.model_selection import GridSearchCV

param_grid = {
    'C': [1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.001, 0.01]
}

grid = GridSearchCV(cumlSVC(kernel='rbf'), param_grid, cv=3)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

## Performance Tips

1. **Feature Scaling**: Features are already z-score normalized by extract_features_cuda
2. **Gamma**: Try 'scale' (default) or 'auto' first
3. **C Parameter**: Start with 10.0, increase if underfitting
4. **Kernel**: RBF usually works best for CIFAR-10

## References

- cuML Documentation: https://docs.rapids.ai/api/cuml/stable/
- RAPIDS AI: https://rapids.ai/
- cuML GitHub: https://github.com/rapidsai/cuml
