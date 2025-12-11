#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "cifar10_loader.h"
#include "config.h"

// Forward declarations từ CUDA file
class AutoencoderGPU;

extern "C" {
    AutoencoderGPU* create_autoencoder_cuda();
    void destroy_autoencoder_cuda(AutoencoderGPU* ae);
    float train_step_cuda(AutoencoderGPU* ae, const float* input, float lr);
    void forward_cuda(AutoencoderGPU* ae, const float* input, float* output);
    bool save_weights_cuda(AutoencoderGPU* ae, const std::string& path);
}

float mse_loss(const std::vector<float>& pred, const std::vector<float>& target) {
    if (pred.size() != target.size()) {
        std::cerr << "Error: Size mismatch in MSE calculation\n";
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < pred.size(); i++) {
        float diff = pred.data()[i] - target.data()[i];
        sum += diff * diff;
    }
    
    return sum / pred.size();
}

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "autoencoder_cuda_weights.bin";
    int epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.001f;
    int max_train_images = 10000;  // Tăng lên cho GPU
    
    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) learning_rate = std::stof(argv[5]);
    if (argc > 6) max_train_images = std::stoi(argv[6]);
    
    std::cout << "=== CUDA Autoencoder Training ===\n";
    std::cout << "Hyperparameters:\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Max training images: " << max_train_images << "\n";
    std::cout << "  Model save path: " << model_path << "\n\n";
    
    // Load CIFAR-10 data
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
        std::cout << "Using " << num_train_images << " images for training\n";
    }
    
    CIFAR10Dataset test_dataset;
    if (!test_dataset.load_test(cifar_dir)) {
        std::cerr << "Failed to load CIFAR-10 test data\n";
        return 1;
    }
    
    std::cout << "Loaded " << test_dataset.num_images << " test images\n\n";
    
    // Khởi tạo autoencoder
    AutoencoderGPU* autoencoder = create_autoencoder_cuda();
    
    std::cout << "\n=== Training ===\n";
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
        train_dataset.shuffle_data();
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int total_batches = (num_train_images + batch_size - 1) / batch_size;
        
        for (int start = 0; start < num_train_images; start += batch_size) {
            std::vector<std::vector<float>> batch_images;
            std::vector<int> batch_labels;
            train_dataset.get_batch(start, batch_size, batch_images, batch_labels);
            
            float batch_loss = 0.0f;
            
            for (size_t i = 0; i < batch_images.size(); i++) {
                float loss = train_step_cuda(autoencoder, batch_images[i].data(), learning_rate);
                batch_loss += loss;
            }
            
            batch_loss /= batch_images.size();
            epoch_loss += batch_loss;
            num_batches++;
            
            int batch_num = (start / batch_size) + 1;
            if (batch_num % 10 == 0 || batch_num == 1) {
                std::cout << "  Batch " << batch_num << "/" << total_batches 
                          << " - Loss: " << batch_loss << "\n";
            }
        }
        
        epoch_loss /= num_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        std::cout << "Epoch " << (epoch + 1) << " completed - Avg Loss: " << epoch_loss 
                  << " - Time: " << epoch_duration.count() / 1000.0 << "s\n\n";
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\n=== Training Completed ===\n";
    std::cout << "Total training time: " << total_duration.count() / 1000.0 << "s\n";
    std::cout << "Average time per epoch: " << (total_duration.count() / 1000.0) / epochs << "s\n\n";
    
    std::cout << "Saving model weights to: " << model_path << "\n";
    if (save_weights_cuda(autoencoder, model_path)) {
        std::cout << "Model saved successfully!\n";
    } else {
        std::cerr << "Failed to save model weights\n";
    }
    
    // Cleanup
    destroy_autoencoder_cuda(autoencoder);
    
    return 0;
}
