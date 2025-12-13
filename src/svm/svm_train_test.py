#!/usr/bin/env python3
"""
SVM Training and Testing using ThunderSVM
Supports both CPU and GPU execution
"""

import argparse
import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import thundersvm, fall back to sklearn if not available
try:
    from thundersvm import SVC as ThunderSVC
    THUNDER_AVAILABLE = True
    print("ThunderSVM is available!")
except ImportError:
    from sklearn.svm import SVC
    THUNDER_AVAILABLE = False
    print("ThunderSVM not available, using sklearn SVM (CPU only)")


def load_libsvm_data(filepath):
    """Load data in LibSVM format"""
    labels = []
    features = []
    
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            
            # Parse features
            feature_dict = {}
            for item in parts[1:]:
                idx, val = item.split(':')
                feature_dict[int(idx) - 1] = float(val)  # Convert to 0-indexed
            
            # Create feature vector (fill missing indices with 0)
            if features:
                feat_vec = np.zeros(len(features[0]))
            else:
                # First sample, determine size
                max_idx = max(feature_dict.keys())
                feat_vec = np.zeros(max_idx + 1)
            
            for idx, val in feature_dict.items():
                feat_vec[idx] = val
            
            features.append(feat_vec)
    
    labels = np.array(labels)
    features = np.array(features)
    
    print(f"  Loaded {len(labels)} samples with {features.shape[1]} features")
    return features, labels


def train_svm(X_train, y_train, use_gpu=False, C=10.0, gamma='auto'):
    """Train SVM classifier"""
    print(f"\n{'='*60}")
    print(f"Training SVM (GPU={use_gpu})...")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  C: {C}, gamma: {gamma}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    if THUNDER_AVAILABLE and use_gpu:
        # Use ThunderSVM with GPU
        clf = ThunderSVC(
            C=C,
            kernel='rbf',
            gamma=gamma,
            verbose=True,
            gpu_id=0  # Use first GPU
        )
    elif THUNDER_AVAILABLE:
        # Use ThunderSVM with CPU
        clf = ThunderSVC(
            C=C,
            kernel='rbf',
            gamma=gamma,
            verbose=True,
            n_jobs=-1  # Use all CPU cores
        )
    else:
        # Fall back to sklearn SVM (CPU only)
        clf = SVC(
            C=C,
            kernel='rbf',
            gamma=gamma,
            verbose=True
        )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    return clf, train_time


def evaluate_svm(clf, X_test, y_test, class_names=None):
    """Evaluate SVM classifier"""
    print(f"\n{'='*60}")
    print("Evaluating SVM...")
    print(f"  Test samples: {len(y_test)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Testing time: {test_time:.2f} seconds")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() * 100
        print(f"  {class_name:12s}: {class_acc:5.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return accuracy, cm, test_time


def plot_confusion_matrix(cm, class_names, output_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='SVM Training and Testing with ThunderSVM')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training features (LibSVM format)')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test features (LibSVM format)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for SVM training (requires ThunderSVM)')
    parser.add_argument('--C', type=float, default=10.0,
                       help='SVM C parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='auto',
                       help='SVM gamma parameter (default: auto)')
    parser.add_argument('--output', type=str, default='confusion_matrix.png',
                       help='Output path for confusion matrix plot')
    
    args = parser.parse_args()
    
    # CIFAR-10 class names
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Load data
    X_train, y_train = load_libsvm_data(args.train)
    X_test, y_test = load_libsvm_data(args.test)
    
    # Check if GPU is requested but not available
    if args.gpu and not THUNDER_AVAILABLE:
        print("\nWARNING: GPU requested but ThunderSVM not available. Using CPU instead.")
        args.gpu = False
    
    # Train SVM
    clf, train_time = train_svm(X_train, y_train, use_gpu=args.gpu,
                                C=args.C, gamma=args.gamma)
    
    # Evaluate SVM
    accuracy, cm, test_time = evaluate_svm(clf, X_test, y_test, cifar10_classes)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, cifar10_classes, args.output)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training samples:  {len(y_train)}")
    print(f"Test samples:      {len(y_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"SVM training time: {train_time:.2f}s")
    print(f"SVM testing time:  {test_time:.2f}s")
    print(f"Test accuracy:     {accuracy*100:.2f}%")
    print(f"GPU used:          {args.gpu}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
