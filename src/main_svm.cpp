#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include "autoencoder.hpp"
#include "cifar10_loader.h"
#include "svm/svm_integration.h"
#include "config.h"

// CIFAR-10 class names
const std::vector<std::string> CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Simple SVM prediction using nearest centroid (for demonstration)
// In production, use LIBSVM library
std::vector<int> predict_svm_simple(const FeatureDataset& train_features, 
                                    const FeatureDataset& test_features) {
    std::cout << "Training simple nearest-centroid classifier...\n";
    
    // Calculate class centroids from training data
    int num_classes = 10;
    std::vector<std::vector<float>> centroids(num_classes);
    std::vector<int> class_counts(num_classes, 0);
    
    // Initialize centroids
    for (int c = 0; c < num_classes; c++) {
        centroids[c].resize(train_features.feature_dim, 0.0f);
    }
    
    // Sum features for each class
    for (size_t i = 0; i < train_features.features.size(); i++) {
        int label = train_features.labels[i];
        if (label >= 0 && label < num_classes) {
            class_counts[label]++;
            for (int j = 0; j < train_features.feature_dim; j++) {
                centroids[label][j] += train_features.features[i][j];
            }
        }
    }
    
    // Average to get centroids
    for (int c = 0; c < num_classes; c++) {
        if (class_counts[c] > 0) {
            for (int j = 0; j < train_features.feature_dim; j++) {
                centroids[c][j] /= class_counts[c];
            }
        }
    }
    
    std::cout << "Predicting on test set...\n";
    
    // Predict using nearest centroid
    std::vector<int> predictions;
    for (size_t i = 0; i < test_features.features.size(); i++) {
        int best_class = 0;
        float best_dist = std::numeric_limits<float>::max();
        
        for (int c = 0; c < num_classes; c++) {
            float dist = 0.0f;
            for (int j = 0; j < test_features.feature_dim; j++) {
                float diff = test_features.features[i][j] - centroids[c][j];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            
            if (dist < best_dist) {
                best_dist = dist;
                best_class = c;
            }
        }
        
        predictions.push_back(best_class);
    }
    
    return predictions;
}

int main(int argc, char** argv) {
    std::string cifar_dir = "./cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "autoencoder_weights.bin";
    std::string train_features_path = "train_features.libsvm";
    std::string test_features_path = "test_features.libsvm";
    
    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) train_features_path = argv[3];
    if (argc > 4) test_features_path = argv[4];
    
    std::cout << "========================================\n";
    std::cout << "  Phase 4: SVM Integration Pipeline\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  CIFAR-10 directory: " << cifar_dir << "\n";
    std::cout << "  Model weights: " << model_path << "\n";
    std::cout << "  Train features output: " << train_features_path << "\n";
    std::cout << "  Test features output: " << test_features_path << "\n\n";
    
    // Step 1: Load trained autoencoder
    std::cout << "=== Step 1: Loading Trained Autoencoder ===\n";
    AutoencoderCPU autoencoder;
    
    if (!autoencoder.load_weights(model_path)) {
        std::cerr << "Error: Failed to load model weights from " << model_path << "\n";
        std::cerr << "Please train the autoencoder first using main.cpp\n";
        return 1;
    }
    std::cout << "Model loaded successfully!\n\n";
    
    // Step 2: Load CIFAR-10 datasets
    std::cout << "=== Step 2: Loading CIFAR-10 Datasets ===\n";
    CIFAR10Dataset train_dataset;
    if (!train_dataset.load_train(cifar_dir)) {
        std::cerr << "Error: Failed to load training data\n";
        return 1;
    }
    std::cout << "Loaded " << train_dataset.num_images << " training images\n";
    
    CIFAR10Dataset test_dataset;
    if (!test_dataset.load_test(cifar_dir)) {
        std::cerr << "Error: Failed to load test data\n";
        return 1;
    }
    std::cout << "Loaded " << test_dataset.num_images << " test images\n\n";
    
    // Step 3: Extract features from training set
    std::cout << "=== Step 3: Extracting Training Features ===\n";
    auto train_start = std::chrono::high_resolution_clock::now();
    FeatureDataset train_features = extract_all_features(autoencoder, train_dataset);
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
    
    std::cout << "Training feature extraction time: " << train_duration.count() / 1000.0 << " seconds\n";
    std::cout << "Training feature extraction speed: " 
              << train_features.num_samples / (train_duration.count() / 1000.0) << " images/second\n\n";
    
    // Step 4: Extract features from test set
    std::cout << "=== Step 4: Extracting Test Features ===\n";
    auto test_start = std::chrono::high_resolution_clock::now();
    FeatureDataset test_features = extract_all_features(autoencoder, test_dataset);
    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);
    
    std::cout << "Test feature extraction time: " << test_duration.count() / 1000.0 << " seconds\n";
    std::cout << "Test feature extraction speed: " 
              << test_features.num_samples / (test_duration.count() / 1000.0) << " images/second\n\n";
    
    // Step 5: Save features to LIBSVM format
    std::cout << "=== Step 5: Saving Features ===\n";
    if (!save_features_libsvm(train_features, train_features_path)) {
        std::cerr << "Warning: Failed to save training features\n";
    }
    
    if (!save_features_libsvm(test_features, test_features_path)) {
        std::cerr << "Warning: Failed to save test features\n";
    }
    
    // Step 6: Train and evaluate classifier
    std::cout << "=== Step 6: Training Classifier ===\n";
    auto svm_start = std::chrono::high_resolution_clock::now();
    std::vector<int> predictions = predict_svm_simple(train_features, test_features);
    auto svm_end = std::chrono::high_resolution_clock::now();
    auto svm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(svm_end - svm_start);
    
    std::cout << "Classifier training and prediction time: " << svm_duration.count() / 1000.0 << " seconds\n\n";
    
    // Step 7: Evaluate performance
    std::cout << "=== Step 7: Evaluation Results ===\n";
    
    double accuracy = calculate_accuracy(test_features.labels, predictions);
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    
    ConfusionMatrix cm = calculate_confusion_matrix(test_features.labels, predictions, 10);
    print_confusion_matrix(cm, CIFAR10_CLASSES);
    print_class_metrics(cm, CIFAR10_CLASSES);
    
    // Summary statistics
    std::cout << "=== Summary ===\n";
    std::cout << "Total training images: " << train_features.num_samples << "\n";
    std::cout << "Total test images: " << test_features.num_samples << "\n";
    std::cout << "Feature dimension: " << train_features.feature_dim << "\n";
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    std::cout << "Total feature extraction time: " 
              << (train_duration.count() + test_duration.count()) / 1000.0 << " seconds\n";
    std::cout << "Average extraction speed: " 
              << (train_features.num_samples + test_features.num_samples) / 
                 ((train_duration.count() + test_duration.count()) / 1000.0) 
              << " images/second\n";
    
    std::cout << "\n=== Pipeline Completed Successfully ===\n";
    
    return 0;
}
