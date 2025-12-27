## CSC14120 Final Project - CUDA Autoencoder + SVM (C/C++)

This repository implements the **two-stage pipeline** described in `CSC14120_2025_Final Project.pdf`:

- **Stage 1**: Convolutional autoencoder (CPU baseline + CUDA basic + CUDA optimized versions)
- **Stage 2**: SVM classifier (using LIBSVM or similar) on learned encoder features

### Requirements

**System Requirements:**
- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- CUDA Toolkit 11.0+ (or 12.0+)
- GCC/G++ compiler with C++17 support
- Python 3.7+ (for SVM training/testing scripts)

**Dependencies:**
- CUDA Toolkit (`nvcc` compiler)
- Standard C++ libraries
- Python packages: `numpy`, `scikit-learn` (for SVM)

### Data Preparation

1. **Download CIFAR-10 Dataset:**
   ```bash
   # Download CIFAR-10 binary version
   wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
   tar -xzf cifar-10-binary.tar.gz
   mv cifar-10-batches-bin cifar-10-binary/
   ```

2. **Verify data structure:**
   ```
   cifar-10-binary/
   └── cifar-10-batches-bin/
       ├── data_batch_1.bin
       ├── data_batch_2.bin
       ├── data_batch_3.bin
       ├── data_batch_4.bin
       ├── data_batch_5.bin
       └── test_batch.bin
   ```

### Folder Structure

- **`src/`**
  - **`main_cpu.cpp`**: CPU baseline entry point
  - **`main_cuda.cpp`**: CUDA implementations entry point
  - **`data/`**: CIFAR-10 loading and preprocessing (`cifar10_loader.h`)
  - **`cpu/`**: CPU baseline autoencoder (`autoencoder_cpu.cpp`)
  - **`cuda/`**: GPU implementations
    - `autoencoder_basic.cu` – Phase 2.2 (basic CUDA)
    - `autoencoder_opt_v1.cu` – Phase 2.3 (optimized v1, shared memory, etc.)
    - `autoencoder_opt_v2.cu` – Phase 2.4 (kernel fusion, streams, etc.)
  - **`svm/`**: SVM integration
    - `extract_features_cpu.cpp` – Extract features using CPU model
    - `extract_features_cuda.cpp` – Extract features using CUDA models
    - `training_svm.py` – Train SVM classifier
    - `testing_svm.py` – Test SVM classifier

- **`include/`**
  - `autoencoder.hpp` – Base autoencoder interface
  - `autoencoder_cuda.h` – CUDA autoencoder headers
  - `config.h` – Configuration constants

- **`cifar-10-binary/`**
  - Downloaded CIFAR-10 binary files

- **`scripts/`**
  - `build_cpu.sh` – Build CPU baseline
  - `build_cuda.sh` – Build all CUDA versions (auto-detects GPU architecture)
  - `build_svm.sh` – Build feature extraction binaries

- **`build_cpu/`** – CPU binary output directory
- **`build_cuda/`** – CUDA binaries output directory
- **`build_svm/`** – SVM feature extraction binaries
- **`weights/`** – Saved model weights

- **Documentation:**
  - `PHASE_2.1_CPU_BASELINE.md` – CPU implementation details
  - `PHASE_2.2_GPU_BASIC.md` – GPU basic implementation
  - `PHASE_2.3_GPU_OPT_V1.md` – GPU optimized v1 details
  - `PHASE_2.4_GPU_OPT_V2.md` – GPU optimized v2 details
  - `PHASE_2.5_SVM_INTEGRATION.md` – SVM integration guide
  - `SECTION_3_4_5_ANALYSIS.md` – Performance analysis
  - `Report_Final.ipynb` – Complete Jupyter notebook report
  - `Instruction.md` – Project instructions

### Build Instructions

**1. Build CPU Baseline:**
```bash
chmod +x scripts/*.sh
./scripts/build_cpu.sh
```

**2. Build CUDA Versions:**
```bash
./scripts/build_cuda.sh
```
This script automatically detects your GPU architecture and builds:
- `autoencoder_cuda_basic` (Phase 2.2)
- `autoencoder_cuda_opt_v1` (Phase 2.3)
- `autoencoder_cuda_opt_v2` (Phase 2.4)

**3. Build SVM Feature Extraction:**
```bash
./scripts/build_svm.sh
```

### Usage

#### CPU Baseline Training

```bash
./build_cpu/autoencoder_cpu \
    <cifar_dir> \
    [weights_path] \
    [epochs] \
    [batch_size] \
    [learning_rate] \
    [max_train_images]
```

**Example:**
```bash
./build_cpu/autoencoder_cpu \
    ./cifar-10-binary/cifar-10-batches-bin \
    autoencoder_weights.bin \
    5 \
    32 \
    0.001 \
    1000
```

#### CUDA Training

**Basic Version (Phase 2.2):**
```bash
./build_cuda/autoencoder_cuda_basic \
    <cifar_dir> \
    [weights_path] \
    [epochs] \
    [batch_size] \
    [learning_rate] \
    [max_train_images]
```

**Optimized V1 (Phase 2.3):**
```bash
./build_cuda/autoencoder_cuda_opt_v1 \
    ./cifar-10-binary/cifar-10-batches-bin \
    weights/autoencoder_cuda_opt_v1_weights.bin \
    3 64 0.001 50000
```

**Optimized V2 (Phase 2.4):**
```bash
./build_cuda/autoencoder_cuda_opt_v2 \
    ./cifar-10-binary/cifar-10-batches-bin \
    weights/autoencoder_cuda_opt_v2_weights.bin \
    3 64 0.001 50000
```

**Default Arguments:**
- `cifar_dir`: `../cifar-10-binary/cifar-10-batches-bin`
- `weights_path`: Auto-detected based on version
- `epochs`: 1 (CUDA) / 5 (CPU)
- `batch_size`: 64 (CUDA) / 32 (CPU)
- `learning_rate`: 0.001
- `max_train_images`: 1000

#### Feature Extraction for SVM

**Using CPU model:**
```bash
./build_svm/extract_features_cpu \
    <cifar_dir> \
    <weights_path> \
    [train_features_output] \
    [test_features_output]
```

**Using CUDA model:**
```bash
./build_svm/extract_features_cuda \
    <cifar_dir> \
    <weights_path> \
    [train_features_output] \
    [test_features_output]
```

#### SVM Training and Testing

**1. Train SVM:**
```bash
cd src/svm
python training_svm.py \
    --train <train_features_file> \
    --test <test_features_file> \
    [--output <model_file>]
```

**2. Test SVM:**
```bash
python testing_svm.py \
    --model <model_file> \
    --test <test_features_file>
```

### Performance Results

**Speedup Comparison (1000 images, 3 epochs):**

| Version | Time/epoch | Total Time | Speedup vs CPU |
|---------|------------|------------|----------------|
| CPU Baseline | 750s | 2250s | 1× |
| GPU Basic | 3.18s | 9.5s | **236×** |
| GPU Opt V1 | 2.06s | 6.18s | **364×** |
| GPU Opt V2 | 2.75s | 8.25s | **273×** |

**Note:** GPU Opt V1 achieves the best performance with shared memory optimizations.

### Project Phases

1. **Phase 2.1**: CPU Baseline Implementation
2. **Phase 2.2**: GPU Basic Implementation (naive parallelization)
3. **Phase 2.3**: GPU Optimized V1 (shared memory, memory coalescing)
4. **Phase 2.4**: GPU Optimized V2 (kernel fusion, vectorization)
5. **Phase 2.5**: SVM Integration (feature extraction + classification)

See individual phase documentation files for detailed implementation notes.

### Troubleshooting

**Build Issues:**
- Ensure CUDA Toolkit is properly installed: `nvcc --version`
- Check GPU compatibility: `nvidia-smi`
- Verify C++17 support: `g++ --version`

**Runtime Issues:**
- GPU out of memory: Reduce `batch_size` or `max_train_images`
- CUDA errors: Check GPU compute capability matches build flags
- Data loading errors: Verify CIFAR-10 binary files are in correct location

### License

This project is part of CSC14120 Final Project coursework.

