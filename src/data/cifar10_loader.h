#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

class CIFAR10Dataset {
public:
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    int num_images = 0;

    CIFAR10Dataset() {}

    bool load_batch(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open CIFAR batch: " << path << "\n";
            return false;
        }

        const int IMG_SIZE = 3072;
        const int RECORD_SIZE = 1 + IMG_SIZE;

        std::vector<unsigned char> buffer(RECORD_SIZE);

        while (file.read((char*)buffer.data(), RECORD_SIZE)) {
            unsigned char label = buffer[0];
            labels.push_back((int)label);

            std::vector<float> img(IMG_SIZE);

            for (int i = 0; i < IMG_SIZE; i++) {
                img[i] = buffer[i + 1] / 255.0f;
            }
            images.push_back(img);
        }

        file.close();
        num_images = images.size();
        return true;
    }

    void shuffle_data() {
        std::vector<int> idx(num_images);
        for (int i = 0; i < num_images; i++) idx[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(idx.begin(), idx.end(), g);

        auto images_copy = images;
        auto labels_copy = labels;

        for (int i = 0; i < num_images; i++) {
            images[i] = images_copy[idx[i]];
            labels[i] = labels_copy[idx[i]];
        }
    }

    void get_batch(int start, int batch_size,
                   std::vector<std::vector<float>> &batch_images,
                   std::vector<int> &batch_labels) 
    {
        batch_images.clear();
        batch_labels.clear();
        int end = std::min(start + batch_size, num_images);

        for (int i = start; i < end; i++) {
            batch_images.push_back(images[i]);
            batch_labels.push_back(labels[i]);
        }
    }
};

#endif
