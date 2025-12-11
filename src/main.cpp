#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "autoencoder.hpp"
#include "cifar10_loader.h"
#include "config.h"

Tensor3D image_to_tensor3d(const std::vector<float>& img) {
    Tensor3D tensor(CIFAR_IMAGE_CHANNELS, CIFAR_IMAGE_HEIGHT, CIFAR_IMAGE_WIDTH);
    
    for (int c = 0; c < CIFAR_IMAGE_CHANNELS; c++) {
        for (int h = 0; h < CIFAR_IMAGE_HEIGHT; h++) {
            for (int w = 0; w < CIFAR_IMAGE_WIDTH; w++) {
                int idx = c * (CIFAR_IMAGE_HEIGHT * CIFAR_IMAGE_WIDTH) + h * CIFAR_IMAGE_WIDTH + w;
                tensor.at(c, h, w) = img[idx];
            }
        }
    }
    
    return tensor;
}

float mse_loss(const Tensor3D& pred, const Tensor3D& target) {
    if (pred.C != target.C || pred.H != target.H || pred.W != target.W) {
        std::cerr << "Error: Tensor dimensions don't match for MSE calculation\n";
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < pred.data.size(); i++) {
        float diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    
    return sum / pred.data.size();
}

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "autoencoder_weights.bin";
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
    
    AutoencoderCPU autoencoder;

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
                Tensor3D input = image_to_tensor3d(batch_images[i]);
                float loss = autoencoder.train_step(input, learning_rate);
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
        // Limit training images for CPU (default 10k)
        epoch_loss /= num_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs;
        std::cout << " - Average Loss: " << epoch_loss;
        std::cout << " - Time: " << epoch_duration.count() << "s\n";
        std::cout.flush();
        
        if ((epoch + 1) % 5 == 0) {
            float test_loss = 0.0f;
            int num_test_samples = std::min(1000, test_dataset.num_images);
            
            for (int i = 0; i < num_test_samples; i++) {
                Tensor3D input = image_to_tensor3d(test_dataset.images[i]);
                Tensor3D latent = autoencoder.forward(input);
                Tensor3D reconstructed = autoencoder.decode(latent);
                test_loss += mse_loss(reconstructed, input);
            }
            
            test_loss /= num_test_samples;
            std::cout << "  Test Loss: " << test_loss << "\n";
        }
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
    
    std::cout << "\n=== Final Evaluation ===\n";
    float test_loss = 0.0f;
    int num_test = std::min(100, test_dataset.num_images);
    
    for (int i = 0; i < num_test; i++) {
        Tensor3D input = image_to_tensor3d(test_dataset.images[i]);
        Tensor3D latent = autoencoder.forward(input);
        Tensor3D reconstructed = autoencoder.decode(latent);
        test_loss += mse_loss(reconstructed, input);
    }
    
    test_loss /= num_test;
    std::cout << "Average test loss (on " << num_test << " images): " << test_loss << "\n";
    
    return 0;
}
