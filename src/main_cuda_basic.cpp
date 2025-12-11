#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "cifar10_loader.h"
#include "config.h"

// Forward declaration of GPU class
class AutoencoderGPU {
public:
    AutoencoderGPU();
    ~AutoencoderGPU();
    float train_step(const std::vector<float>& image, float learning_rate);
    void forward(const std::vector<float>& image);
    void get_output(std::vector<float>& output);
    bool save_weights(const std::string& filename) const;
private:
    // GPU pointers will be defined in .cu file
    void* impl;
};

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "model_basic.bin";
    int epochs = 5;
    int batch_size = 32;
    float learning_rate = 0.001f;
    int max_train_images = 1000;
    
    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) learning_rate = std::stof(argv[5]);
    if (argc > 6) max_train_images = std::stoi(argv[6]);
    
    std::cout << "=================================================================\n";
    std::cout << "GPU BASIC - Naive Implementation\n";
    std::cout << "=================================================================\n";
    std::cout << "Hyperparameters:\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Model save path: " << model_path << "\n";
    std::cout << "  Max training images: " << max_train_images << "\n\n";
    
    std::cout << "Loading CIFAR-10 training data from: " << cifar_dir << "\n";
    
    CIFAR10Dataset train_dataset;
    if (!train_dataset.load_train(cifar_dir)) {
        std::cerr << "Failed to load CIFAR-10 training data\n";
        return 1;
    }
    
    std::cout << "Loaded " << train_dataset.num_images << " training images\n";
    
    int num_train_images = train_dataset.num_images;
    if (max_train_images > 0 && max_train_images < train_dataset.num_images) {
        num_train_images = max_train_images;
        std::cout << "Using " << num_train_images << " images for training (limited from " 
                  << train_dataset.num_images << ")\n";
    }
    
    CIFAR10Dataset test_dataset;
    if (!test_dataset.load_test(cifar_dir)) {
        std::cerr << "Failed to load CIFAR-10 test data\n";
        return 1;
    }
    
    std::cout << "Loaded " << test_dataset.num_images << " test images\n\n";
    
    AutoencoderGPU autoencoder;

    std::cout << "\n=== Training ===\n";
    std::cout.flush();
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " - Shuffling data...\n";
        std::cout.flush();
        train_dataset.shuffle_data();
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int total_batches = (num_train_images + batch_size - 1) / batch_size;
        
        std::cout << "  Processing " << total_batches << " batches...\n";
        std::cout.flush();
        
        for (int start = 0; start < num_train_images; start += batch_size) {
            std::vector<std::vector<float>> batch_images;
            std::vector<int> batch_labels;
            train_dataset.get_batch(start, batch_size, batch_images, batch_labels);
            
            float batch_loss = 0.0f;
            
            int batch_num = (start / batch_size) + 1;
            if (batch_num % 10 == 0 || batch_num == 1) {
                std::cout << "  Batch " << batch_num << "/" << total_batches << " ... ";
                std::cout.flush();
            }
            
            for (size_t i = 0; i < batch_images.size(); i++) {
                float loss = autoencoder.train_step(batch_images[i], learning_rate);
                batch_loss += loss;
            }
            
            batch_loss /= batch_images.size();
            epoch_loss += batch_loss;
            num_batches++;
            
            if (batch_num % 10 == 0 || batch_num == 1) {
                std::cout << "Loss: " << batch_loss << "\n";
                std::cout.flush();
            }
        }
        
        epoch_loss /= num_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs;
        std::cout << " - Average Loss: " << epoch_loss;
        std::cout << " - Time: " << epoch_duration.count() << "s\n";
        std::cout.flush();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    
    std::cout << "\n=== Training Completed ===\n";
    std::cout << "Total training time: " << total_duration.count() << "s\n";
    
    std::cout << "\nSaving model weights to: " << model_path << "\n";
    if (autoencoder.save_weights(model_path)) {
        std::cout << "Model saved successfully!\n";
    } else {
        std::cerr << "Failed to save model weights\n";
    }
    
    return 0;
}
