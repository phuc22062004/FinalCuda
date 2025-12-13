#pragma once
#include <string>
#include <vector>

// features: N x 8192 float
class AutoencoderCUDA {
public:
    AutoencoderCUDA();
    ~AutoencoderCUDA();

    // train autoencoder on images (each image size = 3072 float, normalized [0,1])
    // return average loss
    float train(int epochs, int batch_size, float lr,
                const std::vector<std::vector<float>>& train_images,
                int max_train_images);

    // extract features (encoder output flatten 8192)
    void extract_features(const std::vector<std::vector<float>>& images,
                          int batch_size,
                          std::vector<std::vector<float>>& out_features);

    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);
};
