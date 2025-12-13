#!/usr/bin/env bash
# SVM Training and Testing using ThunderSVM C++ executables

set -e

# Default parameters
TRAIN_FILE="train_features_cuda.libsvm"
TEST_FILE="test_features_cuda.libsvm"
MODEL_FILE="svm_model.txt"
PREDICT_FILE="predictions.txt"
C_PARAM=10.0
GAMMA="auto"
USE_GPU=0  # 0=CPU, 1=GPU

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --test)
            TEST_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --output)
            PREDICT_FILE="$2"
            shift 2
            ;;
        --C)
            C_PARAM="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ThunderSVM C++ - SVM Training & Testing"
echo "=========================================="
echo "Training file:    $TRAIN_FILE"
echo "Test file:        $TEST_FILE"
echo "Model file:       $MODEL_FILE"
echo "Prediction file:  $PREDICT_FILE"
echo "C parameter:      $C_PARAM"
echo "Gamma:            $GAMMA"
echo "GPU enabled:      $USE_GPU"
echo "=========================================="
echo ""

# Check if files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

# Build thundersvm-train command
TRAIN_CMD="thundersvm-train -s 0 -t 2 -c $C_PARAM -u $USE_GPU"

# Add gamma if not auto
if [ "$GAMMA" != "auto" ]; then
    TRAIN_CMD="$TRAIN_CMD -g $GAMMA"
fi

TRAIN_CMD="$TRAIN_CMD $TRAIN_FILE $MODEL_FILE"

# Train SVM
echo "Step 1: Training SVM..."
echo "Command: $TRAIN_CMD"
echo ""
time $TRAIN_CMD

echo ""
echo "Training completed!"
echo ""

# Predict
echo "Step 2: Testing SVM..."
echo "Command: thundersvm-predict $TEST_FILE $MODEL_FILE $PREDICT_FILE"
echo ""
time thundersvm-predict "$TEST_FILE" "$MODEL_FILE" "$PREDICT_FILE"

echo ""
echo "Testing completed!"
echo ""

# Calculate accuracy
echo "Step 3: Calculating accuracy..."
python3 - <<EOF
import sys

# Read predictions
with open('$PREDICT_FILE', 'r') as f:
    predictions = [int(float(line.strip())) for line in f]

# Read true labels from test file
with open('$TEST_FILE', 'r') as f:
    true_labels = []
    for line in f:
        label = int(line.strip().split()[0])
        true_labels.append(label)

# Calculate accuracy
correct = sum(1 for i in range(len(predictions)) if predictions[i] == true_labels[i])
total = len(predictions)
accuracy = correct / total * 100

print(f"========================================")
print(f"RESULTS")
print(f"========================================")
print(f"Total samples:     {total}")
print(f"Correct:           {correct}")
print(f"Incorrect:         {total - correct}")
print(f"Accuracy:          {accuracy:.2f}%")
print(f"========================================")

# Per-class accuracy
from collections import defaultdict
class_correct = defaultdict(int)
class_total = defaultdict(int)

for i in range(len(predictions)):
    true_label = true_labels[i]
    pred_label = predictions[i]
    class_total[true_label] += 1
    if true_label == pred_label:
        class_correct[true_label] += 1

cifar10_classes = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

print("\nPer-class Accuracy:")
for cls in sorted(class_total.keys()):
    cls_acc = class_correct[cls] / class_total[cls] * 100 if class_total[cls] > 0 else 0
    cls_name = cifar10_classes.get(cls, f'Class {cls}')
    print(f"  {cls_name:12s}: {cls_acc:5.2f}% ({class_correct[cls]}/{class_total[cls]})")

# Confusion matrix
print("\nConfusion Matrix:")
n_classes = len(cifar10_classes)
cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]

for i in range(len(predictions)):
    true_label = true_labels[i]
    pred_label = predictions[i]
    cm[true_label][pred_label] += 1

# Print header
print("     ", end="")
for i in range(n_classes):
    print(f"{i:5d}", end="")
print()

# Print matrix
for i in range(n_classes):
    print(f"{i:3d}: ", end="")
    for j in range(n_classes):
        print(f"{cm[i][j]:5d}", end="")
    print()

EOF

echo ""
echo "Results saved to:"
echo "  Model:       $MODEL_FILE"
echo "  Predictions: $PREDICT_FILE"
echo ""
