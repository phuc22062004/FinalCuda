#include "cifar10_loader.h"

int main() {
    CIFAR10Dataset train_data;

    train_data.load_batch("data_batch_1.bin");
    train_data.load_batch("data_batch_2.bin");
    train_data.load_batch("data_batch_3.bin");
    train_data.load_batch("data_batch_4.bin");
    train_data.load_batch("data_batch_5.bin");

    std::cout << "Loaded training images: " << train_data.num_images << "\n";

    train_data.shuffle_data();

    std::vector<std::vector<float>> batch_img;
    std::vector<int> batch_lab;

    train_data.get_batch(0, 32, batch_img, batch_lab);

    std::cout << "Batch size: " << batch_img.size() << "\n";
    std::cout << "Label[0]: " << batch_lab[0] << "\n";

    return 0;
}
