#!/usr/bin/env python3
"""
Train and test SVM using cuML with CIFAR-10 features
Usage: python training_svm.py --train <train_file> --test <test_file> [options]
"""

import argparse
import time
import numpy as np
import cudf
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from cuml.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test SVM using cuML for CIFAR-10 classification'
    )
    parser.add_argument('--train', type=str, required=True,
                       help='Training features file (LibSVM format)')
    parser.add_argument('--test', type=str, required=True,
                       help='Test features file (LibSVM format)')
    parser.add_argument('--C', type=float, default=10.0,
                       help='SVM C parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='scale',
                       help='Gamma parameter: scale/auto/float (default: scale)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['rbf', 'linear', 'poly', 'sigmoid'],
                       help='Kernel type (default: rbf)')
    parser.add_argument('--max-iter', type=int, default=-1,
                       help='Maximum iterations (default: -1 = no limit)')
    parser.add_argument('--tol', type=float, default=1e-3,
                       help='Tolerance for stopping criterion (default: 1e-3)')
    parser.add_argument('--cache-size', type=int, default=2000,
                       help='Cache size in MB (default: 2000)')
    parser.add_argument('--predictions', type=str, default='predictions_cuml.txt',
                       help='Output file for predictions (default: predictions_cuml.txt)')
    parser.add_argument('--cm-output', type=str, default='confusion_matrix_cuml.png',
                       help='Confusion matrix output file (default: confusion_matrix_cuml.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting confusion matrix')
    return parser.parse_args()

def load_data(file_path):
    """Load features from LibSVM format file"""
    print(f"Loading data from: {file_path}")
    start_time = time.time()
    X, y = load_svmlight_file(file_path)
    X = X.toarray()  # Convert sparse to dense
    load_time = time.time() - start_time
    print(f"  Shape: {X.shape}, Labels: {y.shape}")
    print(f"  Load time: {load_time:.2f}s")
    return X, y.astype(int)

def train_svm(X_train, y_train, args):
    """Train SVM classifier using cuML"""
    print("\n" + "="*60)
    print("Training SVM Classifier with cuML")
    print("="*60)
    print(f"Training samples:    {X_train.shape[0]}")
    print(f"Feature dimension:   {X_train.shape[1]}")
    print(f"Number of classes:   {len(np.unique(y_train))}")
    print(f"C parameter:         {args.C}")
    print(f"Gamma:               {args.gamma}")
    print(f"Kernel:              {args.kernel}")
    print(f"Max iterations:      {args.max_iter}")
    print(f"Tolerance:           {args.tol}")
    print(f"Cache size:          {args.cache_size} MB")
    print("="*60)
    
    # Convert to cuDF
    print("\nConverting to cuDF format...")
    convert_start = time.time()
    X_train_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    y_train_cudf = cudf.Series(y_train)
    convert_time = time.time() - convert_start
    print(f"  Conversion time: {convert_time:.2f}s")
    
    # Create SVM classifier
    print("\nInitializing SVM classifier...")
    clf = SVC(
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
        cache_size=args.cache_size,
        max_iter=args.max_iter,
        tol=args.tol
    )
    
    # Train
    print("\nTraining started...")
    train_start = time.time()
    clf.fit(X_train_cudf, y_train_cudf)
    train_time = time.time() - train_start
    
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    return clf, train_time

def test_svm(clf, X_test, y_test):
    """Test SVM classifier and compute predictions"""
    print("\n" + "="*60)
    print("Testing SVM Classifier")
    print("="*60)
    print(f"Test samples: {X_test.shape[0]}")
    
    # Convert to cuDF
    print("\nConverting test data to cuDF format...")
    convert_start = time.time()
    X_test_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
    y_test_cudf = cudf.Series(y_test)
    convert_time = time.time() - convert_start
    print(f"  Conversion time: {convert_time:.2f}s")
    
    # Predict
    print("\nPredicting...")
    predict_start = time.time()
    y_pred = clf.predict(X_test_cudf)
    y_pred_np = y_pred.to_numpy()
    predict_time = time.time() - predict_start
    
    print(f"✓ Prediction completed in {predict_time:.2f} seconds")
    
    return y_pred_np, predict_time

def print_results(y_test, y_pred):
    """Print detailed classification results"""
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Total samples:       {len(y_test)}")
    print(f"Correct predictions: {np.sum(y_test == y_pred)}")
    print(f"Wrong predictions:   {np.sum(y_test != y_pred)}")
    print(f"Accuracy:            {accuracy*100:.2f}%")
    print("="*60)
    
    # CIFAR-10 class names
    cifar10_classes = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 60)
    for label in sorted(np.unique(y_test)):
        mask = y_test == label
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        class_name = cifar10_classes.get(label, f'class_{label}')
        total = np.sum(mask)
        correct = np.sum((y_test[mask] == y_pred[mask]))
        print(f"  {class_name:12s} (class {label}): {class_acc*100:5.2f}% ({correct}/{total})")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    target_names = [cifar10_classes[i] for i in range(10)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return cm, accuracy

def plot_confusion_matrix(cm, output_file):
    """Plot and save confusion matrix"""
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cifar10_classes,
                yticklabels=cifar10_classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CIFAR-10 Classification (cuML SVM)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_file}")
    plt.close()

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("cuML SVM Training Pipeline for CIFAR-10")
    print("="*60)
    print(f"Training file:       {args.train}")
    print(f"Test file:           {args.test}")
    print(f"Predictions output:  {args.predictions}")
    print(f"Confusion matrix:    {args.cm_output}")
    print("="*60)
    
    # Load training data
    print("\n[1/5] Loading training data...")
    X_train, y_train = load_data(args.train)
    
    # Load test data
    print("\n[2/5] Loading test data...")
    X_test, y_test = load_data(args.test)
    
    # Train SVM
    print("\n[3/5] Training SVM...")
    clf, train_time = train_svm(X_train, y_train, args)
    
    # Test SVM
    print("\n[4/5] Testing SVM...")
    y_pred, test_time = test_svm(clf, X_test, y_test)
    
    # Evaluate and print results
    print("\n[5/5] Computing metrics...")
    cm, accuracy = print_results(y_test, y_pred)
    
    # Save predictions
    np.savetxt(args.predictions, y_pred, fmt='%d')
    print(f"\n✓ Predictions saved to: {args.predictions}")
    
    # Plot confusion matrix
    if not args.no_plot:
        plot_confusion_matrix(cm, args.cm_output)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Training time:       {train_time:.2f} seconds")
    print(f"Testing time:        {test_time:.2f} seconds")
    print(f"Total time:          {train_time + test_time:.2f} seconds")
    print(f"Final accuracy:      {accuracy*100:.2f}%")
    print(f"Backend:             cuML (GPU)")
    print("="*60)
    print("\n✓ Pipeline completed successfully!\n")

if __name__ == '__main__':
    main()