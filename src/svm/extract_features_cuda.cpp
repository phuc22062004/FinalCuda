// Extract features using CUDA-trained autoencoder for SVM training
// WITH Z-SCORE SCALING for better SVM performance
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

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

// Fast binary writer (for quick extraction)
static void write_features_binary(const std::string& feat_path, const std::string& label_path,
                                   const std::string& cache_path, const ZScaler& scaler, int N, int D) {
    std::ifstream cache_file(cache_path, std::ios::binary);
    std::ofstream feat_file(feat_path, std::ios::binary);
    std::ofstream label_file(label_path, std::ios::binary);
    
    std::vector<float> feat(D);
    std::vector<float> scaled_feat(D);
    int label;
    
    for (int i = 0; i < N; i++) {
        cache_file.read((char*)&label, sizeof(int));
        cache_file.read((char*)feat.data(), D * sizeof(float));
        
        // Scale features
        for (int d = 0; d < D; d++) {
            scaled_feat[d] = scaler.transform(feat[d], d);
        }
        
        // Write binary
        feat_file.write((char*)scaled_feat.data(), D * sizeof(float));
        label_file.write((char*)&label, sizeof(int));
    }
}

// Extract features from encoder (after pool2: 128x8x8 = 8192 features)
// REUSE BUFFER - không return vector nữa
static inline void extract_features_from_encoder(AutoencoderCUDA& ae, const float* input_chw, float* out_feat) {
    ae.extract_features(input_chw, out_feat);
}

// ============================================================================
// OPTIMIZED: Extract features with CUDA streams (TRUE async batching)
// Safe: AutoencoderCUDA::extract_features_async() doesn't affect training
// ============================================================================
static void extract_features_chunked(
    AutoencoderCUDA& ae,
    CIFAR10Dataset& dataset,
    const std::string& cache_path,
    int D,
    long long& total_extract_us,
    long long& total_cache_us)
{
    const int CHUNK_SIZE = 256;  // Process 256 images at a time
    const int num_images = dataset.num_images;
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Allocate PINNED memory for async H2D/D2H transfers
    float* pinned_input = nullptr;   // For input images
    float* pinned_output = nullptr;  // For extracted features
    cudaMallocHost(&pinned_input, CHUNK_SIZE * 3 * 32 * 32 * sizeof(float));
    cudaMallocHost(&pinned_output, CHUNK_SIZE * D * sizeof(float));
    
    std::ofstream cache_file(cache_path, std::ios::binary);
    static char cache_buffer[1 << 24]; // 16MB
    cache_file.rdbuf()->pubsetbuf(cache_buffer, sizeof(cache_buffer));
    
    for (int start = 0; start < num_images; start += CHUNK_SIZE) {
        int end = std::min(start + CHUNK_SIZE, num_images);
        int chunk_size = end - start;
        
        if (start % 5000 == 0) {
            std::cout << "  Processed " << start << "/" << num_images << "\n";
        }
        
        auto extract_start = std::chrono::high_resolution_clock::now();
        
        // Submit all images in chunk to stream (pipelined)
        for (int i = 0; i < chunk_size; i++) {
            int idx = start + i;
            float* image_chw = dataset.images[idx].data();
            
            // Copy to pinned buffer first (for better cache locality)
            std::memcpy(pinned_input + i * 3 * 32 * 32, image_chw, 3 * 32 * 32 * sizeof(float));
            
            // Launch async extraction (no sync inside)
            ae.extract_features_async(pinned_input + i * 3 * 32 * 32,
                                     pinned_output + i * D,
                                     stream);
        }
        
        // Sync stream ONCE per chunk (not per image)
        cudaStreamSynchronize(stream);
        
        auto extract_end = std::chrono::high_resolution_clock::now();
        total_extract_us += std::chrono::duration_cast<std::chrono::microseconds>(extract_end - extract_start).count();
        
        // Write chunk to cache
        auto cache_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < chunk_size; i++) {
            int idx = start + i;
            int label = dataset.labels[idx];
            cache_file.write((char*)&label, sizeof(int));
            cache_file.write((char*)(pinned_output + i * D), D * sizeof(float));
        }
        auto cache_end = std::chrono::high_resolution_clock::now();
        total_cache_us += std::chrono::duration_cast<std::chrono::microseconds>(cache_end - cache_start).count();
    }
    
    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    cudaStreamDestroy(stream);
    std::cout << "  Completed " << num_images << "/" << num_images << "\n";
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
    bool fast_mode = false;  // Skip LibSVM text generation

    if (argc > 1) cifar_dir = argv[1];
    if (argc > 2) weights_path = argv[2];
    if (argc > 3) output_train = argv[3];
    if (argc > 4) output_test = argv[4];
    
    // Check for --fast flag
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--fast") {
            fast_mode = true;
            std::cout << "*** FAST MODE: Skipping LibSVM text generation ***\n";
        }
    }

    std::cout << "=== CUDA Feature Extraction for SVM (With Z-Score Scaling) ===\n";
    std::cout << "CIFAR-10 dir: " << cifar_dir << "\n";
    std::cout << "Weights:      " << weights_path << "\n";
    std::cout << "Output train: " << output_train << "\n";
    std::cout << "Output test:  " << output_test << "\n";
    std::cout << "Scaler:       " << scaler_path << "\n";
    std::cout << "Fast mode:    " << (fast_mode ? "YES (binary only)" : "NO (binary + libsvm)") << "\n\n";

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
    
    // Reuse buffer - không cấp phát mỗi ảnh
    std::vector<float> feat_buffer(D);
    
    // ========================================================================
    // PASS 1: Extract train features + cache (OPTIMIZED with chunking)
    // ========================================================================
    std::cout << "PASS 1: Extracting train features to cache (chunked + pinned memory)...\n";
    auto pass1_start = std::chrono::high_resolution_clock::now();
    
    // Track pure extraction time separately
    long long total_extract_us = 0;  // microseconds
    long long total_cache_us = 0;
    
    // Use optimized chunked extraction
    extract_features_chunked(ae, train_dataset, cache_path, D, total_extract_us, total_cache_us);
    
    auto pass1_end = std::chrono::high_resolution_clock::now();
    auto pass1_time = std::chrono::duration_cast<std::chrono::seconds>(pass1_end - pass1_start);
    auto extract_time = std::chrono::seconds(total_extract_us / 1000000);
    auto cache_write_time = std::chrono::seconds(total_cache_us / 1000000);
    
    std::cout << "  PASS 1 breakdown:\n";
    std::cout << "    - Pure GPU extraction:  " << extract_time.count() << "s\n";
    std::cout << "    - Cache write I/O:      " << cache_write_time.count() << "s\n";
    std::cout << "  PASS 1 total time:        " << pass1_time.count() << "s\n\n";
    
    // ========================================================================
    // PASS 1.5: Compute mean and std from cache (2-pass: faster than Welford)
    // ========================================================================
    std::cout << "Computing statistics (mean/std) from cache...\n";
    auto stats_start = std::chrono::high_resolution_clock::now();
    
    ZScaler scaler(D);
    std::vector<double> sum(D, 0.0);
    std::vector<double> sumsq(D, 0.0);
    
    // First pass: compute mean
    {
        std::ifstream cache_file(cache_path, std::ios::binary);
        static char read_buffer[1 << 24]; // 16MB
        cache_file.rdbuf()->pubsetbuf(read_buffer, sizeof(read_buffer));
        
        std::vector<float> feat(D);
        int label;
        long long count = 0;
        
        while (read_cache_sample(cache_file, label, feat.data(), D)) {
            for (int d = 0; d < D; d++) {
                sum[d] += feat[d];
            }
            count++;
        }
        
        scaler.n = count;
        for (int d = 0; d < D; d++) {
            scaler.mean[d] = sum[d] / count;
        }
    }
    
    // Second pass: compute variance
    {
        std::ifstream cache_file(cache_path, std::ios::binary);
        static char read_buffer2[1 << 24]; // 16MB
        cache_file.rdbuf()->pubsetbuf(read_buffer2, sizeof(read_buffer2));
        
        std::vector<float> feat(D);
        int label;
        
        while (read_cache_sample(cache_file, label, feat.data(), D)) {
            for (int d = 0; d < D; d++) {
                double diff = feat[d] - scaler.mean[d];
                sumsq[d] += diff * diff;
            }
        }
    }
    
    // Compute std
    float eps = 1e-6f;
    for (int d = 0; d < D; d++) {
        double var = sumsq[d] / (scaler.n - 1);
        double s = std::sqrt(var);
        if (s < eps) s = eps;
        scaler.stdv[d] = (float)s;
    }
    
    auto stats_end = std::chrono::high_resolution_clock::now();
    auto stats_time = std::chrono::duration_cast<std::chrono::seconds>(stats_end - stats_start);
    std::cout << "  Statistics computed (mean/std for " << D << " dims)\n";
    std::cout << "  Stats computation time: " << stats_time.count() << "s\n";
    
    
    // Save scaler for test set
    if (!save_scaler(scaler_path, scaler)) {
        std::cerr << "Failed to save scaler\n";
        return 1;
    }
    std::cout << "  Scaler saved to: " << scaler_path << "\n\n";
    
    // ========================================================================
    // PASS 2: Read cache + scale + write BINARY (fast)
    // ========================================================================
    std::cout << "PASS 2: Scaling and writing binary features...\n";
    auto pass2_start = std::chrono::high_resolution_clock::now();
    
    std::string train_bin = output_train + ".bin";
    std::string train_labels_bin = output_train + ".labels.bin";
    
    write_features_binary(train_bin, train_labels_bin, cache_path, scaler, 
                         train_dataset.num_images, D);
    
    std::cout << "  Binary features written to:\n";
    std::cout << "    " << train_bin << "\n";
    std::cout << "    " << train_labels_bin << "\n";
    
    auto pass2_end = std::chrono::high_resolution_clock::now();
    auto binary_time = std::chrono::duration_cast<std::chrono::seconds>(pass2_end - pass2_start);
    std::cout << "  Binary write time: " << binary_time.count() << "s\n\n";
    
    // ========================================================================
    // OPTIONAL: Write LibSVM text (SLOW - only if not in fast mode)
    // ========================================================================
    auto libsvm_time = std::chrono::seconds(0);
    if (!fast_mode) {
        std::cout << "Converting to LibSVM text format...\n";
        auto libsvm_start = std::chrono::high_resolution_clock::now();
        {
            std::ifstream cache_file(cache_path, std::ios::binary);
            std::ofstream libsvm_file(output_train);
            
            if (!cache_file.is_open() || !libsvm_file.is_open()) {
                std::cerr << "Failed to open files\n";
                return 1;
            }
            
            // Large buffer for faster I/O
            static char write_buffer[1 << 26]; // 64MB
            libsvm_file.rdbuf()->pubsetbuf(write_buffer, sizeof(write_buffer));
            
            std::vector<float> feat(D);
            int label;
            int count = 0;
            
            while (read_cache_sample(cache_file, label, feat.data(), D)) {
                write_one_libsvm(libsvm_file, label, feat.data(), scaler);
                count++;
                if (count % 5000 == 0) {
                    std::cout << "  Written " << count << " samples\n";
                }
            }
            std::cout << "  Completed: " << count << " samples to " << output_train << "\n";
        }
        
        auto libsvm_end = std::chrono::high_resolution_clock::now();
        libsvm_time = std::chrono::duration_cast<std::chrono::seconds>(libsvm_end - libsvm_start);
        std::cout << "  LibSVM text write time: " << libsvm_time.count() << "s\n\n";
    } else {
        std::cout << "*** Skipping LibSVM text (fast mode) ***\n\n";
    }
    
    // ========================================================================
    // TEST SET: Extract + scale + write binary (async with stream)
    // ========================================================================
    std::cout << "Extracting and scaling test features (async)...\n";
    auto test_start = std::chrono::high_resolution_clock::now();
    
    // Track test extraction time separately
    long long test_extract_us = 0;
    long long test_write_us = 0;
    
    // Write binary first (fast)
    std::string test_bin = output_test + ".bin";
    std::string test_labels_bin = output_test + ".labels.bin";
    {
        std::ofstream feat_file(test_bin, std::ios::binary);
        std::ofstream label_file(test_labels_bin, std::ios::binary);
        
        // Large buffers
        static char feat_buf[1 << 24], label_buf[1 << 20];
        feat_file.rdbuf()->pubsetbuf(feat_buf, sizeof(feat_buf));
        label_file.rdbuf()->pubsetbuf(label_buf, sizeof(label_buf));
        
        // Create stream for async operations
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // Pinned memory for chunk
        const int CHUNK_SIZE = 256;
        float* pinned_input = nullptr;
        float* pinned_output = nullptr;
        cudaMallocHost(&pinned_input, CHUNK_SIZE * 3 * 32 * 32 * sizeof(float));
        cudaMallocHost(&pinned_output, CHUNK_SIZE * D * sizeof(float));
        std::vector<float> scaled_chunk(CHUNK_SIZE * D);
        
        for (int start = 0; start < test_dataset.num_images; start += CHUNK_SIZE) {
            int end = std::min(start + CHUNK_SIZE, test_dataset.num_images);
            int chunk_size = end - start;
            
            if (start % 2000 == 0) {
                std::cout << "  Processed " << start << "/" << test_dataset.num_images << "\n";
            }
            
            // Extract chunk with async stream
            auto ext_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < chunk_size; i++) {
                int idx = start + i;
                float* image_chw = test_dataset.images[idx].data();
                std::memcpy(pinned_input + i * 3 * 32 * 32, image_chw, 3 * 32 * 32 * sizeof(float));
                ae.extract_features_async(pinned_input + i * 3 * 32 * 32,
                                         pinned_output + i * D,
                                         stream);
            }
            cudaStreamSynchronize(stream);  // Sync once per chunk
            auto ext_end = std::chrono::high_resolution_clock::now();
            test_extract_us += std::chrono::duration_cast<std::chrono::microseconds>(ext_end - ext_start).count();
            
            // Scale and write chunk
            auto write_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < chunk_size; i++) {
                int idx = start + i;
                int label = test_dataset.labels[idx];
                
                // Scale
                for (int d = 0; d < D; d++) {
                    scaled_chunk[i * D + d] = scaler.transform(pinned_output[i * D + d], d);
                }
                
                // Write
                feat_file.write((char*)(scaled_chunk.data() + i * D), D * sizeof(float));
                label_file.write((char*)&label, sizeof(int));
            }
            auto write_end = std::chrono::high_resolution_clock::now();
            test_write_us += std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
        }
        
        cudaFreeHost(pinned_input);
        cudaFreeHost(pinned_output);
        cudaStreamDestroy(stream);
        std::cout << "  Binary test features written\n";
    }
    
    auto test_mid = std::chrono::high_resolution_clock::now();
    auto test_total_time = std::chrono::duration_cast<std::chrono::seconds>(test_mid - test_start);
    auto test_gpu_time = std::chrono::seconds(test_extract_us / 1000000);
    auto test_io_time = std::chrono::seconds(test_write_us / 1000000);
    
    std::cout << "  TEST breakdown:\n";
    std::cout << "    - Pure GPU extraction:  " << test_gpu_time.count() << "s\n";
    std::cout << "    - Binary write I/O:     " << test_io_time.count() << "s\n";
    std::cout << "  TEST total time:          " << test_total_time.count() << "s\n";
    
    // Write LibSVM text (optional)
    auto test_libsvm_time = std::chrono::seconds(0);
    if (!fast_mode) {
        std::cout << "  Converting test to LibSVM text...\n";
        auto test_lib_start = std::chrono::high_resolution_clock::now();
        {
            std::ifstream feat_file(test_bin, std::ios::binary);
            std::ifstream label_file(test_labels_bin, std::ios::binary);
            std::ofstream libsvm_file(output_test);
            
            // Large buffer
            static char test_write_buffer[1 << 26]; // 64MB
            libsvm_file.rdbuf()->pubsetbuf(test_write_buffer, sizeof(test_write_buffer));
            
            std::vector<float> feat(D);
            int label;
            int count = 0;
            
            while (label_file.read((char*)&label, sizeof(int))) {
                feat_file.read((char*)feat.data(), D * sizeof(float));
                
                // Write LibSVM (features already scaled)
                libsvm_file << label;
                for (int j = 0; j < D; j++) {
                    libsvm_file << " " << (j + 1) << ":" << feat[j];
                }
                libsvm_file << "\n";
                count++;
            }
            std::cout << "  Completed: " << count << " samples to " << output_test << "\n";
        }
        auto test_lib_end = std::chrono::high_resolution_clock::now();
        test_libsvm_time = std::chrono::duration_cast<std::chrono::seconds>(test_lib_end - test_lib_start);
        std::cout << "  Test LibSVM text write: " << test_libsvm_time.count() << "s\n\n";
    } else {
        std::cout << "*** Skipping test LibSVM text (fast mode) ***\n\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "=============================================================\n";
    std::cout << "               PERFORMANCE BREAKDOWN\n";
    std::cout << "=============================================================\n";
    std::cout << "Total time:                      " << duration.count() << "s\n\n";
    
    std::cout << "🔥 PURE GPU EXTRACTION (what matters):\n";
    std::cout << "  - Train extraction (50K imgs):  " << extract_time.count() << "s\n";
    std::cout << "  - Test extraction (10K imgs):   " << test_gpu_time.count() << "s\n";
    std::cout << "  >>> TOTAL GPU COMPUTE:          " << (extract_time.count() + test_gpu_time.count()) << "s <<<\n\n";
    
    std::cout << "📊 POST-PROCESSING (I/O overhead):\n";
    std::cout << "  - PASS 1 cache write:           " << cache_write_time.count() << "s\n";
    std::cout << "  - Statistics (mean/std):        " << stats_time.count() << "s\n";
    std::cout << "  - PASS 2 (binary):              " << binary_time.count() << "s\n";
    std::cout << "  - TEST binary write:            " << test_io_time.count() << "s\n";
    
    if (!fast_mode) {
        std::cout << "  - TRAIN LibSVM text:            " << libsvm_time.count() << "s\n";
        std::cout << "  - TEST LibSVM text:             " << test_libsvm_time.count() << "s\n";
        std::cout << "\n>>> I/O overhead (text):          " << (libsvm_time.count() + test_libsvm_time.count()) << "s\n";
    }
    std::cout << "=============================================================\n\n";

    std::cout << "=== Feature extraction complete! ===\n";
    std::cout << "\nBinary features (FAST, for quick reload):\n";
    std::cout << "  Train: " << train_bin << " + .labels.bin\n";
    std::cout << "  Test:  " << test_bin << " + .labels.bin\n";
    
    if (!fast_mode) {
        std::cout << "\nLibSVM text (for traditional SVM tools):\n";
        std::cout << "  Train: " << output_train << "\n";
        std::cout << "  Test:  " << output_test << "\n";
    }
    
    std::cout << "\nScaler: " << scaler_path << "\n";
    std::cout << "\nScaling statistics:\n";
    std::cout << "  Samples:  " << scaler.n << "\n";
    std::cout << "  Features: " << D << "\n";
    std::cout << "  Example mean[0]:   " << scaler.mean[0] << "\n";
    std::cout << "  Example stddev[0]: " << scaler.stdv[0] << "\n";
    
    if (fast_mode) {
        std::cout << "\n💡 To convert binary to LibSVM later, run without --fast flag\n";
        std::cout << "💡 Or use Python: np.fromfile('" << train_bin << "', dtype=np.float32).reshape(-1, " << D << ")\n";
    }

    return 0;
}
