#include <iostream>
#include <vector>
#include <string>
#include <chrono>
   
#include "cifar10_loader.h"
#include "config.h"
#include "autoencoder_cuda.h"

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "autoencoder_cuda_opt_v1_weights.bin";
    int epochs = 1;
    int batch_size = 64;       // TRUE BATCH SIZE - processes 64 images simultaneously on GPU
    float learning_rate = 0.001f;
    int max_train_images = 1000;

    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) learning_rate = std::stof(argv[5]);
    if (argc > 6) max_train_images = std::stoi(argv[6]);

    std::cout << "=== CUDA OPTIMIZED V1 - TRUE BATCH PROCESSING ===\n";
    std::cout << "CIFAR dir: " << cifar_dir << "\n";
    std::cout << "Weights:   " << model_path << "\n";
    std::cout << "Epochs:    " << epochs << "\n";
    std::cout << "Batch:     " << batch_size << " (GPU processes all simultaneously!)\n";
    std::cout << "LR:        " << learning_rate << "\n";
    std::cout << "Max train: " << max_train_images << "\n\n";

    CIFAR10Dataset train_dataset;
    if (!train_dataset.load_train(cifar_dir)) {
        std::cerr << "Failed to load train data\n";
        return 1;
    }
    std::cout << "Loaded train images: " << train_dataset.num_images << "\n";

    int num_train_images = train_dataset.num_images;
    if (max_train_images > 0 && max_train_images < num_train_images) {
        num_train_images = max_train_images;
        std::cout << "Using " << num_train_images << " images (debug limit)\n";
    }

    AutoencoderCUDA ae;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << " - shuffling...\n";
        train_dataset.shuffle_data();

        float epoch_loss = 0.0f;
        int batch_count = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Process in TRUE BATCHES - all images in batch processed simultaneously on GPU
        for (int start = 0; start < num_train_images; start += batch_size) {
            int end = std::min(start + batch_size, num_train_images);
            int current_batch_size = end - start;
            
            // Prepare batch buffer (contiguous memory)
            std::vector<float> batch_buffer(current_batch_size * 3072);
            for (int i = 0; i < current_batch_size; i++) {
                std::copy(train_dataset.images[start + i].begin(),
                         train_dataset.images[start + i].end(),
                         batch_buffer.begin() + i * 3072);
            }
            
            // Process ENTIRE BATCH on GPU simultaneously
            float batch_loss = ae.train_step_batch(batch_buffer.data(), current_batch_size, learning_rate);
            
            epoch_loss += batch_loss;
            batch_count++;

            if (batch_count == 1 || batch_count % 10 == 0) {
                std::cout << "  Batch " << batch_count << " (" << current_batch_size << " images) loss: " 
                         << batch_loss << "\n";
            }
        }

        epoch_loss /= batch_count;

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count();
        float imgs_per_sec = (float)num_train_images / (epoch_ms / 1000.0f);

        std::cout << "Epoch avg loss: " << epoch_loss 
                 << " | time: " << epoch_ms << "ms"
                 << " | throughput: " << imgs_per_sec << " imgs/sec\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::cout << "\nTotal training time: " << total_ms << "ms\n";
    std::cout << "Average throughput: " << (float)(num_train_images * epochs) / (total_ms / 1000.0f) << " imgs/sec\n";

    std::cout << "Saving weights to " << model_path << "\n";
    if (!ae.save_weights(model_path)) {
        std::cerr << "Save weights failed\n";
        return 1;
    }
    std::cout << "Done.\n";
    return 0;
}