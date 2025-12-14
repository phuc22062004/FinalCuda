// Extract features using CUDA-trained autoencoder for SVM training
// WITH Z-SCORE SCALING for better SVM performance
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include "cifar10_loader.h"
#include "config.h"
#include "autoencoder_cuda.h"

// ============================================================================
// Z-Score Scaler using Welford's online algorithm
// Computes mean and std for each feature dimension efficiently
// ============================================================================
struct ZScaler {
    int D;                          // Feature dimension (8192)
    long long n = 0;                // Number of samples seen
    std::vector<double> mean, m2;   // Mean and M2 (for variance calculation)
    std::vector<float> stdv;        // Standard deviation per dimension
    
    explicit ZScaler(int dim) : D(dim), mean(dim, 0.0), m2(dim, 0.0), stdv(dim, 1.0f) {}
    
    // Update statistics with new sample (online)
    void update(const float* x) {
        n++;
        for (int d = 0; d < D; d++) {
            double xd = x[d];
            double delta = xd - mean[d];
            mean[d] += delta / (double)n;
            double delta2 = xd - mean[d];
            m2[d] += delta * delta2;
        }
    }
    
    // Finalize: compute std from m2
    void finalize(float eps = 1e-6f) {
        for (int d = 0; d < D; d++) {
            double var = (n > 1) ? (m2[d] / (double)(n - 1)) : 0.0;
            double s = std::sqrt(var);
            if (s < eps) s = eps;  // Prevent division by zero
            stdv[d] = (float)s;
        }
    }
    
    // Transform a single value using computed statistics
    inline float transform(float x, int d) const {
        return (x - (float)mean[d]) / stdv[d];
    }
};

// Save scaler to binary file
static bool save_scaler(const std::string& path, const ZScaler& sc) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write((char*)&sc.D, sizeof(int));
    f.write((char*)&sc.n, sizeof(long long));
    f.write((char*)sc.mean.data(), sc.D * sizeof(double));
    f.write((char*)sc.stdv.data(), sc.D * sizeof(float));
    return true;
}

// Load scaler from binary file
static bool load_scaler(const std::string& path, ZScaler& sc) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    int D = 0; 
    long long n = 0;
    f.read((char*)&D, sizeof(int));
    f.read((char*)&n, sizeof(long long));
    if (D != sc.D) return false;
    sc.n = n;
    f.read((char*)sc.mean.data(), sc.D * sizeof(double));
    f.read((char*)sc.stdv.data(), sc.D * sizeof(float));
    return true;
}

// Write one sample to LibSVM format with scaling
static void write_one_libsvm(std::ofstream& out, int label, const float* feat, const ZScaler& sc) {
    out << label;
    for (int j = 0; j < sc.D; j++) {
        float v = sc.transform(feat[j], j);
        out << " " << (j + 1) << ":" << v;
    }
    out << "\n";
}

// Extract features from encoder (after pool2: 128x8x8 = 8192 features)
std::vector<float> extract_features_from_encoder(AutoencoderCUDA& ae, const float* input_chw) {
    std::vector<float> features(8192); // 128 * 8 * 8
    ae.extract_features(input_chw, features.data());
    return features;
}

// ============================================================================
// 2-Pass Pipeline Helper: Write float* to binary cache
// ============================================================================
static bool write_cache_sample(std::ofstream& cache, int label, const float* feat, int D) {
    cache.write((char*)&label, sizeof(int));
    cache.write((char*)feat, D * sizeof(float));
    return cache.good();
}

// ============================================================================
// 2-Pass Pipeline Helper: Read one sample from binary cache
// ============================================================================
static bool read_cache_sample(std::ifstream& cache, int& label, float* feat, int D) {
    cache.read((char*)&label, sizeof(int));
    if (!cache.good()) return false;
    cache.read((char*)feat, D * sizeof(float));
    return cache.good();
}

int main(int argc, char** argv) {
    std::string cifar_dir = "../cifar-10-binary/cifar-10-batches-bin";
    std::string weights_path = "autoencoder_cuda_basic_weights.bin";
    std::string output_train = "train_features_cuda.libsvm";
    std::string output_test = "test_features_cuda.libsvm";
    std::string scaler_path = "scaler_z.bin";
    std::string cache_path = "train_cache.bin";

    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) weights_path = argv[2];
    if (argc > 3) output_train = argv[3];
    if (argc > 4) output_test = argv[4];

    std::cout << "=== CUDA Feature Extraction for SVM (With Z-Score Scaling) ===\n";
    std::cout << "CIFAR-10 dir: " << cifar_dir << "\n";
    std::cout << "Weights:      " << weights_path << "\n";
    std::cout << "Output train: " << output_train << "\n";
    std::cout << "Output test:  " << output_test << "\n";
    std::cout << "Scaler:       " << scaler_path << "\n\n";

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
    AutoencoderCUDA ae;
    if (!ae.load_weights(weights_path)) {
        std::cerr << "Failed to load weights from: " << weights_path << "\n";
        return 1;
    }
    std::cout << "Loaded autoencoder weights\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    
    const int D = 8192;  // Feature dimension
    ZScaler scaler(D);
    
    // ========================================================================
    // PASS 1: Extract train features + compute statistics + cache
    // ========================================================================
    std::cout << "PASS 1: Extracting train features + computing statistics...\n";
    auto pass1_start = std::chrono::high_resolution_clock::now();
    
    long long extract_time_us = 0;
    long long welford_time_us = 0;
    long long cache_time_us = 0;
    
    {
        std::ofstream cache_file(cache_path, std::ios::binary);
        if (!cache_file.is_open()) {
            std::cerr << "Failed to open cache file: " << cache_path << "\n";
            return 1;
        }
        
        for (int i = 0; i < train_dataset.num_images; i++) {
            if (i % 5000 == 0) {
                std::cout << "  Processed " << i << "/" << train_dataset.num_images << "\n";
            }
            
            float* image_chw = train_dataset.images[i].data();
            
            // TIME: Feature extraction
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<float> features = extract_features_from_encoder(ae, image_chw);
            auto t2 = std::chrono::high_resolution_clock::now();
            extract_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            
            int label = train_dataset.labels[i];
            
            // TIME: Update scaler statistics
            auto t3 = std::chrono::high_resolution_clock::now();
            scaler.update(features.data());
            auto t4 = std::chrono::high_resolution_clock::now();
            welford_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
            
            // TIME: Cache features to disk
            auto t5 = std::chrono::high_resolution_clock::now();
            write_cache_sample(cache_file, label, features.data(), D);
            auto t6 = std::chrono::high_resolution_clock::now();
            cache_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        }
        std::cout << "  Completed " << train_dataset.num_images << "/" << train_dataset.num_images << "\n";
    }
    
    // Finalize scaler (compute std from accumulated M2)
    auto finalize_start = std::chrono::high_resolution_clock::now();
    scaler.finalize();
    auto finalize_end = std::chrono::high_resolution_clock::now();
    long long finalize_time_us = std::chrono::duration_cast<std::chrono::microseconds>(finalize_end - finalize_start).count();
    
    std::cout << "  Statistics computed (mean/std for " << D << " dims)\n";
    
    // Save scaler for test set
    if (!save_scaler(scaler_path, scaler)) {
        std::cerr << "Failed to save scaler\n";
        return 1;
    }
    std::cout << "  Scaler saved to: " << scaler_path << "\n";
    
    auto pass1_end = std::chrono::high_resolution_clock::now();
    auto pass1_total = std::chrono::duration_cast<std::chrono::seconds>(pass1_end - pass1_start);
    
    std::cout << "  PASS 1 Breakdown:\n";
    std::cout << "    - GPU extraction:     " << (extract_time_us / 1000000) << "s\n";
    std::cout << "    - Welford update:     " << (welford_time_us / 1000000) << "s\n";
    std::cout << "    - Cache write:        " << (cache_time_us / 1000000) << "s\n";
    std::cout << "    - Finalize stats:     " << (finalize_time_us / 1000000) << "s\n";
    std::cout << "  PASS 1 Total: " << pass1_total.count() << "s\n\n";
    
    // ========================================================================
    // PASS 2: Read cache + scale + write LibSVM
    // ========================================================================
    std::cout << "PASS 2: Scaling and writing train features...\n";
    auto pass2_start = std::chrono::high_resolution_clock::now();
    
    long long cache_read_time_us = 0;
    long long scaling_time_us = 0;
    long long libsvm_write_time_us = 0;
    
    {
        std::ifstream cache_file(cache_path, std::ios::binary);
        std::ofstream libsvm_file(output_train);
        
        if (!cache_file.is_open() || !libsvm_file.is_open()) {
            std::cerr << "Failed to open files\n";
            return 1;
        }
        
        std::vector<float> feat(D);
        int label;
        int count = 0;
        
        while (true) {
            // TIME: Read from cache
            auto t1 = std::chrono::high_resolution_clock::now();
            bool read_ok = read_cache_sample(cache_file, label, feat.data(), D);
            auto t2 = std::chrono::high_resolution_clock::now();
            cache_read_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            
            if (!read_ok) break;
            
            // TIME: Scale (transform is inline, so measure the whole libsvm write which includes scaling)
            auto t3 = std::chrono::high_resolution_clock::now();
            write_one_libsvm(libsvm_file, label, feat.data(), scaler);
            auto t4 = std::chrono::high_resolution_clock::now();
            libsvm_write_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
            
            count++;
            if (count % 5000 == 0) {
                std::cout << "  Written " << count << " samples\n";
            }
        }
        std::cout << "  Completed: " << count << " samples to " << output_train << "\n";
    }
    
    auto pass2_end = std::chrono::high_resolution_clock::now();
    auto pass2_total = std::chrono::duration_cast<std::chrono::seconds>(pass2_end - pass2_start);
    
    std::cout << "  PASS 2 Breakdown:\n";
    std::cout << "    - Cache read:         " << (cache_read_time_us / 1000000) << "s\n";
    std::cout << "    - Scale + LibSVM:     " << (libsvm_write_time_us / 1000000) << "s\n";
    std::cout << "  PASS 2 Total: " << pass2_total.count() << "s\n\n";
    
    // ========================================================================
    // TEST SET: Extract + scale immediately (no cache needed)
    // ========================================================================
    std::cout << "Extracting and scaling test features...\n";
    auto test_start = std::chrono::high_resolution_clock::now();
    
    long long test_extract_time_us = 0;
    long long test_write_time_us = 0;
    
    {
        std::ofstream libsvm_file(output_test);
        if (!libsvm_file.is_open()) {
            std::cerr << "Failed to open test output\n";
            return 1;
        }
        
        for (int i = 0; i < test_dataset.num_images; i++) {
            if (i % 2000 == 0) {
                std::cout << "  Processed " << i << "/" << test_dataset.num_images << "\n";
            }
            
            float* image_chw = test_dataset.images[i].data();
            
            // TIME: Feature extraction
            auto t1 = std::chrono::high_resolution_clock::now();
            std::vector<float> features = extract_features_from_encoder(ae, image_chw);
            auto t2 = std::chrono::high_resolution_clock::now();
            test_extract_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            
            int label = test_dataset.labels[i];
            
            // TIME: Scale and write directly
            auto t3 = std::chrono::high_resolution_clock::now();
            write_one_libsvm(libsvm_file, label, features.data(), scaler);
            auto t4 = std::chrono::high_resolution_clock::now();
            test_write_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        }
        std::cout << "  Completed: " << test_dataset.num_images << " samples to " << output_test << "\n";
    }
    
    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_total = std::chrono::duration_cast<std::chrono::seconds>(test_end - test_start);
    
    std::cout << "  TEST Breakdown:\n";
    std::cout << "    - GPU extraction:     " << (test_extract_time_us / 1000000) << "s\n";
    std::cout << "    - Scale + LibSVM:     " << (test_write_time_us / 1000000) << "s\n";
    std::cout << "  TEST Total: " << test_total.count() << "s\n\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "=============================================================\n";
    std::cout << "               PERFORMANCE SUMMARY\n";
    std::cout << "=============================================================\n";
    std::cout << "PASS 1 (Train extraction + statistics):\n";
    std::cout << "  - GPU extraction:     " << (extract_time_us / 1000000) << "s\n";
    std::cout << "  - Welford update:     " << (welford_time_us / 1000000) << "s\n";
    std::cout << "  - Cache write:        " << (cache_time_us / 1000000) << "s\n";
    std::cout << "  - Finalize stats:     " << (finalize_time_us / 1000000) << "s\n\n";
    
    std::cout << "PASS 2 (Train scaling + LibSVM):\n";
    std::cout << "  - Cache read:         " << (cache_read_time_us / 1000000) << "s\n";
    std::cout << "  - Scale + LibSVM:     " << (libsvm_write_time_us / 1000000) << "s\n\n";
    
    std::cout << "TEST (Extract + scale + LibSVM):\n";
    std::cout << "  - GPU extraction:     " << (test_extract_time_us / 1000000) << "s\n";
    std::cout << "  - Scale + LibSVM:     " << (test_write_time_us / 1000000) << "s\n\n";
    
    long long total_gpu_us = extract_time_us + test_extract_time_us;
    long long total_io_us = cache_time_us + cache_read_time_us + libsvm_write_time_us + test_write_time_us;
    
    std::cout << "🔥 TOTAL GPU EXTRACTION:  " << (total_gpu_us / 1000000) << "s\n";
    std::cout << "📊 TOTAL I/O + SCALING:   " << (total_io_us / 1000000) << "s\n";
    std::cout << "⏱️  TOTAL TIME:            " << duration.count() << "s\n";
    std::cout << "=============================================================\n\n";

    std::cout << "=== Feature extraction complete! ===\n";
    std::cout << "Train features (scaled): " << output_train << "\n";
    std::cout << "Test features (scaled):  " << output_test << "\n";
    std::cout << "Scaler saved to:         " << scaler_path << "\n";
    std::cout << "\nScaling statistics:\n";
    std::cout << "  Samples:  " << scaler.n << "\n";
    std::cout << "  Features: " << D << "\n";
    std::cout << "  Example mean[0]:   " << scaler.mean[0] << "\n";
    std::cout << "  Example stddev[0]: " << scaler.stdv[0] << "\n";

    return 0;
}