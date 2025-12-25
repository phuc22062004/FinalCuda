#!/usr/bin/env python3
"""
Test SVM using pre-trained cuML model (no training required)
Usage: python testing_svm.py --model <model_file> --test <test_file>
"""

import argparse
import time
import numpy as np
import cudf
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test pre-trained SVM model for CIFAR-10 classification'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Trained model file (.pkl)')
    parser.add_argument('--test', type=str, required=True,
                       help='Test features file (LibSVM format)')
    parser.add_argument('--predictions', type=str, default='predictions_test.txt',
                       help='Output file for predictions (default: predictions_test.txt)')
    parser.add_argument('--cm-output', type=str, default='confusion_matrix_test.png',
                       help='Confusion matrix output file (default: confusion_matrix_test.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting confusion matrix')
    return parser.parse_args()

def load_model(model_path):
    """Load pre-trained SVM model"""
    print(f"Loading pre-trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    start_time = time.time()
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    load_time = time.time() - start_time
    
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Model file size: {model_size:.2f} MB")
    print(f"  Load time: {load_time:.2f}s")
    print(f"✓ Model loaded successfully")
    
    return clf

def load_data(file_path):
    """Load features from LibSVM format file"""
    print(f"\nLoading test data from: {file_path}")
    start_time = time.time()
    X, y = load_svmlight_file(file_path)
    X = X.toarray()  # Convert sparse to dense
    load_time = time.time() - start_time
    print(f"  Shape: {X.shape}, Labels: {y.shape}")
    print(f"  Load time: {load_time:.2f}s")
    return X, y.astype(int)

def test_svm(clf, X_test, y_test):
    """Test SVM classifier and compute predictions"""
    print("\n" + "="*60)
    print("Testing SVM Classifier")
    print("="*60)
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_test.shape[1]}")
    
    # Convert to cuDF
    print("\nConverting test data to cuDF format...")
    convert_start = time.time()
    X_test_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
    convert_time = time.time() - convert_start
    print(f"  Conversion time: {convert_time:.2f}s")
    
    # Predict
    print("\nPredicting...")
    predict_start = time.time()
    y_pred = clf.predict(X_test_cudf)
    
    # Convert cuML predictions to numpy
    try:
        y_pred_np = y_pred.to_numpy()
    except:
        y_pred_np = np.array(y_pred)
    
    predict_time = time.time() - predict_start
    
    print(f"✓ Prediction completed in {predict_time:.2f} seconds")
    print(f"  Throughput: {len(y_test) / predict_time:.0f} samples/second")
    
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
    plt.title('Confusion Matrix - CIFAR-10 Classification (cuML SVM - Testing)', 
              fontsize=14, pad=20)
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
    print("cuML SVM Testing Pipeline for CIFAR-10")
    print("="*60)
    print(f"Model file:          {args.model}")
    print(f"Test file:           {args.test}")
    print(f"Predictions output:  {args.predictions}")
    print(f"Confusion matrix:    {args.cm_output}")
    print("="*60)
    
    # Load pre-trained model
    print("\n[1/4] Loading pre-trained model...")
    clf = load_model(args.model)
    
    # Load test data
    print("\n[2/4] Loading test data...")
    X_test, y_test = load_data(args.test)
    
    # Test SVM
    print("\n[3/4] Testing SVM...")
    y_pred, test_time = test_svm(clf, X_test, y_test)
    
    # Evaluate and print results
    print("\n[4/4] Computing metrics...")
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
    print(f"Testing time:        {test_time:.2f} seconds")
    print(f"Throughput:          {len(y_test) / test_time:.0f} samples/second")
    print(f"Final accuracy:      {accuracy*100:.2f}%")
    print(f"Backend:             cuML (GPU)")
    print("="*60)
    print("\n✓ Testing completed successfully!\n")

if __name__ == '__main__':
    main()
