#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "cifar10_loader.h"
#include "config.h"
#include "autoencoder_cuda.h"

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string model_path = "autoencoder_cuda_basic_weights.bin";
    int epochs = 1;
    int batch_size = 64;       // GPU basic: cứ để 64 theo đề
    float learning_rate = 0.001f;
    int max_train_images = 1000; // debug nhanh

    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);
    if (argc > 5) learning_rate = std::stof(argv[5]);
    if (argc > 6) max_train_images = std::stoi(argv[6]);

    std::cout << "=== CUDA BASIC (Phase 2) ===\n";
    std::cout << "CIFAR dir: " << cifar_dir << "\n";
    std::cout << "Weights:   " << model_path << "\n";
    std::cout << "Epochs:    " << epochs << "\n";
    std::cout << "Batch:     " << batch_size << "\n";
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
        int seen = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int start = 0; start < num_train_images; start += batch_size) {
            int end = std::min(start + batch_size, num_train_images);
            float batch_loss = 0.0f;

            for (int i = start; i < end; i++) {
                // dataset.images[i] đã là vector<float> size 3072 theo CHW và [0,1]
                batch_loss += ae.train_step(train_dataset.images[i].data(), learning_rate);
            }

            batch_loss /= (end - start);
            epoch_loss += batch_loss;
            seen++;

            int batch_num = start / batch_size + 1;
            if (batch_num == 1 || batch_num % 10 == 0) {
                std::cout << "  Batch " << batch_num << " loss: " << batch_loss << "\n";
            }
        }

        epoch_loss /= seen;

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_s = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

        std::cout << "Epoch avg loss: " << epoch_loss << " | time: " << epoch_s << "s\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    std::cout << "\nTotal training time: " << total_s << "s\n";

    std::cout << "Saving weights to " << model_path << "\n";
    if (!ae.save_weights(model_path)) {
        std::cerr << "Save weights failed\n";
        return 1;
    }
    std::cout << "Done.\n";
    return 0;
}
