// Extract features using CPU-trained autoencoder for SVM training
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "cifar10_loader.h"
#include "config.h"
#include "autoencoder.hpp"

void save_features_libsvm_format(
    const std::string& filepath,
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels
) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << "\n";
        return;
    }

    for (size_t i = 0; i < features.size(); i++) {
        // Write label
        file << labels[i];
        
        // Write features in LibSVM format: label index:value index:value ...
        for (size_t j = 0; j < features[i].size(); j++) {
            file << " " << (j + 1) << ":" << features[i][j];
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Saved " << features.size() << " samples to " << filepath << "\n";
}

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string weights_path = "autoencoder_weights.bin";
    std::string output_train = "train_features.libsvm";
    std::string output_test = "test_features.libsvm";

    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) weights_path = argv[2];
    if (argc > 3) output_train = argv[3];
    if (argc > 4) output_test = argv[4];

    std::cout << "=== CPU Feature Extraction for SVM ===\n";
    std::cout << "CIFAR-10 dir: " << cifar_dir << "\n";
    std::cout << "Weights:      " << weights_path << "\n";
    std::cout << "Output train: " << output_train << "\n";
    std::cout << "Output test:  " << output_test << "\n\n";

    // Load datasets
    CIFAR10Dataset train_dataset;
    if (!train_dataset.load_train(cifar_dir)) {
        std::cerr << "Failed to load train data\n";
        return 1;
    }
    std::cout << "Loaded train images: " << train_dataset.num_images << "\n";

    CIFAR10Dataset test_dataset;
    if (!test_dataset.load_test(cifar_dir)) {
        std::cerr << "Failed to load test data\n";
        return 1;
    }
    std::cout << "Loaded test images:  " << test_dataset.num_images << "\n\n";

    // Load trained autoencoder
    AutoencoderCPU ae;
    if (!ae.load_weights(weights_path)) {
        std::cerr << "Failed to load weights from: " << weights_path << "\n";
        return 1;
    }
    std::cout << "Loaded autoencoder weights\n\n";

    // Extract training features
    std::cout << "Extracting training features...\n";
    std::vector<std::vector<float>> train_features;
    std::vector<int> train_labels;
    
    for (int i = 0; i < train_dataset.num_images; i++) {
        if (i % 5000 == 0) {
            std::cout << "  Processed " << i << "/" << train_dataset.num_images << "\n";
        }
        
        // Convert to Tensor3D format
        Tensor3D input(CIFAR_IMAGE_CHANNELS, CIFAR_IMAGE_HEIGHT, CIFAR_IMAGE_WIDTH);
        const int img_size = CIFAR_IMAGE_CHANNELS * CIFAR_IMAGE_HEIGHT * CIFAR_IMAGE_WIDTH;
        for (int j = 0; j < img_size; j++) {
            input.data[j] = train_dataset.images[i][j];
        }
        
        // Run encoder to get features (bottleneck layer)
        Tensor3D encoded = ae.forward(input);
        
        // Convert to vector
        std::vector<float> features(encoded.data.begin(), encoded.data.end());
        
        train_features.push_back(features);
        train_labels.push_back(train_dataset.labels[i]);
    }
    std::cout << "  Completed " << train_dataset.num_images << "/" << train_dataset.num_images << "\n";
    std::cout << "Feature dimension: " << train_features[0].size() << "\n\n";

    // Extract test features
    std::cout << "Extracting test features...\n";
    std::vector<std::vector<float>> test_features;
    std::vector<int> test_labels;
    
    for (int i = 0; i < test_dataset.num_images; i++) {
        if (i % 1000 == 0) {
            std::cout << "  Processed " << i << "/" << test_dataset.num_images << "\n";
        }
        
        // Convert to Tensor3D format
        Tensor3D input(CIFAR_IMAGE_CHANNELS, CIFAR_IMAGE_HEIGHT, CIFAR_IMAGE_WIDTH);
        const int img_size = CIFAR_IMAGE_CHANNELS * CIFAR_IMAGE_HEIGHT * CIFAR_IMAGE_WIDTH;
        for (int j = 0; j < img_size; j++) {
            input.data[j] = test_dataset.images[i][j];
        }
        
        // Run encoder to get features
        Tensor3D encoded = ae.forward(input);
        
        // Convert to vector
        std::vector<float> features(encoded.data.begin(), encoded.data.end());
        
        test_features.push_back(features);
        test_labels.push_back(test_dataset.labels[i]);
    }
    std::cout << "  Completed " << test_dataset.num_images << "/" << test_dataset.num_images << "\n\n";

    // Save in LibSVM format
    std::cout << "Saving features...\n";
    save_features_libsvm_format(output_train, train_features, train_labels);
    save_features_libsvm_format(output_test, test_features, test_labels);

    std::cout << "\nFeature extraction complete!\n";
    std::cout << "Train features: " << output_train << "\n";
    std::cout << "Test features:  " << output_test << "\n";

    return 0;
}
