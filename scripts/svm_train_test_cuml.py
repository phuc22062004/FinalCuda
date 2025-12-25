#!/usr/bin/env python3
"""
SVM Training and Testing using cuML (RAPIDS AI)
Replaces ThunderSVM with cuML's GPU-accelerated SVM
"""

import argparse
import time
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import cuML, fallback to sklearn if not available
try:
    from cuml.svm import SVC as cumlSVC
    CUML_AVAILABLE = True
    print("✓ cuML available - using GPU acceleration")
except ImportError:
    from sklearn.svm import SVC as cumlSVC
    CUML_AVAILABLE = False
    print("⚠ cuML not available - falling back to sklearn (CPU)")

def load_features(file_path):
    """Load features from LibSVM format file"""
    print(f"Loading features from: {file_path}")
    X, y = load_svmlight_file(file_path)
    X = X.toarray()  # Convert sparse to dense
    return X, y.astype(int)

def train_svm(X_train, y_train, C=10.0, gamma='scale', kernel='rbf'):
    """Train SVM classifier using cuML or sklearn"""
    print("\n" + "="*50)
    print("Training SVM Classifier")
    print("="*50)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"C parameter: {C}")
    print(f"Gamma: {gamma}")
    print(f"Kernel: {kernel}")
    print(f"Backend: {'cuML (GPU)' if CUML_AVAILABLE else 'sklearn (CPU)'}")
    print("="*50)
    
    # Create SVM classifier
    if CUML_AVAILABLE:
        # cuML SVC
        clf = cumlSVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            cache_size=2000,  # MB
            max_iter=-1,      # No limit
            tol=1e-3
        )
    else:
        # sklearn SVC
        clf = cumlSVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            cache_size=2000,
            max_iter=-1,
            tol=1e-3
        )
    
    # Train
    print("\nTraining started...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f} seconds")
    return clf, training_time

def test_svm(clf, X_test, y_test):
    """Test SVM classifier and compute metrics"""
    print("\n" + "="*50)
    print("Testing SVM Classifier")
    print("="*50)
    print(f"Test samples: {X_test.shape[0]}")
    
    # Predict
    print("Predicting...")
    start_time = time.time()
    y_pred = clf.predict(X_test)
    
    # Convert cuML predictions to numpy if needed
    if CUML_AVAILABLE:
        try:
            y_pred = y_pred.to_numpy()
        except:
            y_pred = np.array(y_pred)
    
    prediction_time = time.time() - start_time
    print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, prediction_time

def print_results(y_test, y_pred, accuracy):
    """Print detailed classification results"""
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Total samples:     {len(y_test)}")
    print(f"Correct:           {np.sum(y_test == y_pred)}")
    print(f"Incorrect:         {np.sum(y_test != y_pred)}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print("="*50)
    
    # CIFAR-10 class names
    cifar10_classes = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 50)
    for label in sorted(np.unique(y_test)):
        mask = y_test == label
        class_acc = np.mean(y_pred[mask] == label)
        class_name = cifar10_classes.get(label, f'class_{label}')
        print(f"  {class_name:12s} (class {label}): {class_acc*100:5.2f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return cm

def plot_confusion_matrix(cm, output_file='confusion_matrix_cuml.png'):
    """Plot and save confusion matrix"""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cifar10_classes,
                yticklabels=cifar10_classes)
    plt.title('Confusion Matrix - CIFAR-10 Classification (cuML SVM)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and test SVM using cuML')
    parser.add_argument('--train', type=str, required=True,
                       help='Training features file (LibSVM format)')
    parser.add_argument('--test', type=str, required=True,
                       help='Test features file (LibSVM format)')
    parser.add_argument('--C', type=float, default=10.0,
                       help='SVM C parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='scale',
                       help='Gamma parameter (default: scale)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       help='Kernel type (default: rbf)')
    parser.add_argument('--cm-output', type=str, default='confusion_matrix_cuml.png',
                       help='Confusion matrix output file')
    parser.add_argument('--predictions', type=str, default='predictions_cuml.txt',
                       help='Predictions output file')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("cuML SVM - Training & Testing Pipeline")
    print("="*50)
    print(f"Training file:    {args.train}")
    print(f"Test file:        {args.test}")
    print(f"C parameter:      {args.C}")
    print(f"Gamma:            {args.gamma}")
    print(f"Kernel:           {args.kernel}")
    print("="*50)
    
    # Load data
    print("\n[1/4] Loading data...")
    X_train, y_train = load_features(args.train)
    X_test, y_test = load_features(args.test)
    
    # Train SVM
    print("\n[2/4] Training SVM...")
    clf, train_time = train_svm(X_train, y_train, 
                                C=args.C, gamma=args.gamma, kernel=args.kernel)
    
    # Test SVM
    print("\n[3/4] Testing SVM...")
    y_pred, accuracy, test_time = test_svm(clf, X_test, y_test)
    
    # Print results
    print("\n[4/4] Computing metrics...")
    cm = print_results(y_test, y_pred, accuracy)
    
    # Save predictions
    np.savetxt(args.predictions, y_pred, fmt='%d')
    print(f"\n✓ Predictions saved to: {args.predictions}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, args.cm_output)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Training time:     {train_time:.2f} seconds")
    print(f"Testing time:      {test_time:.2f} seconds")
    print(f"Total time:        {train_time + test_time:.2f} seconds")
    print(f"Final accuracy:    {accuracy*100:.2f}%")
    print(f"Backend:           {'cuML (GPU)' if CUML_AVAILABLE else 'sklearn (CPU)'}")
    print("="*50)

if __name__ == '__main__':
    main()
