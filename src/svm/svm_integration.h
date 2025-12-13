#ifndef SVM_INTEGRATION_H
#define SVM_INTEGRATION_H

#include <vector>
#include <string>
#include "autoencoder.hpp"
#include "cifar10_loader.h"

// Structure to hold extracted features
struct FeatureDataset {
    std::vector<std::vector<float>> features;  // Each vector is 8192-dim
    std::vector<int> labels;
    int num_samples;
    int feature_dim;
};

// Confusion matrix structure
struct ConfusionMatrix {
    int num_classes;
    std::vector<std::vector<int>> matrix;  // matrix[true_label][predicted_label]
};

// Function declarations
std::vector<float> extract_features(const Tensor3D& latent);
Tensor3D image_to_tensor3d(const std::vector<float>& img);
FeatureDataset extract_all_features(AutoencoderCPU& autoencoder, CIFAR10Dataset& dataset);
bool save_features_libsvm(const FeatureDataset& features, const std::string& filepath);
double calculate_accuracy(const std::vector<int>& true_labels, const std::vector<int>& pred_labels);
ConfusionMatrix calculate_confusion_matrix(const std::vector<int>& true_labels, 
                                          const std::vector<int>& pred_labels, 
                                          int num_classes);
void print_confusion_matrix(const ConfusionMatrix& cm, const std::vector<std::string>& class_names);
void print_class_metrics(const ConfusionMatrix& cm, const std::vector<std::string>& class_names);

#endif // SVM_INTEGRATION_H
