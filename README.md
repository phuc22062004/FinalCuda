# CUDA Autoencoder - Final Project

## 📁 Cấu Trúc Project

```
FinalCuda/
├── src/
│   ├── main.cpp              # CPU version
│   ├── main_cuda.cpp         # GPU version  
│   ├── cpu/
│   │   └── autoencoder_cpu.cpp
│   ├── cuda/
│   │   ├── autoencoder_basic.cu    # GPU Basic (Phase 2)
│   │   ├── autoencoder_opt_v1.cu   # GPU Optimized v1 (Phase 3)
│   │   └── autoencoder_opt_v2.cu   # GPU Optimized v2 (Phase 3)
│   ├── data/
│   │   └── cifar10_loader.h
│   └── svm/
│       └── svm_integration.cpp
├── include/
│   ├── autoencoder.hpp
│   └── config.h
├── scripts/
│   ├── build_cpu.sh
│   ├── build_cuda.sh
│   └── run_examples.sh
├── cifar-10-binary/          # CIFAR-10 dataset
├── build_gpu_basic.sh        # Quick build GPU basic
├── Report.ipynb              # Main report
├── README.md                 # This file
└── Problem_statement.md      # Project requirements

```

## 🚀 Quick Start

### 1. Build

**CPU Version:**
```bash
mkdir -p build_cpu
cd build_cpu
g++ -std=c++17 -O2 ../src/main.cpp ../src/cpu/autoencoder_cpu.cpp \
    -I../include -I../src/data -o autoencoder_cpu
```

**GPU Basic Version:**
```bash
./build_gpu_basic.sh
```

### 2. Run

**CPU (Phase 1 - Baseline):**
```bash
cd build_cpu
./autoencoder_cpu ../cifar-10-binary/cifar-10-batches-bin model_cpu.bin 5 32 0.001 1000
```

**GPU Basic (Phase 2):**
```bash
cd build_cuda
./autoencoder_cuda_basic ../cifar-10-binary/cifar-10-batches-bin model_gpu.bin 10 32 0.001 10000
```

**Parameters:**
1. CIFAR-10 directory
2. Model save path
3. Epochs
4. Batch size
5. Learning rate
6. Max training images

## 📊 Implementation Phases

### ✅ Phase 1: CPU Baseline
- **Status:** Complete
- **Files:** `src/main.cpp`, `src/cpu/autoencoder_cpu.cpp`
- **Features:** 
  - Full autoencoder with forward/backward pass
  - Conv2D, ReLU, MaxPool, Upsample layers
  - MSE loss, SGD optimizer
  - Save/load weights

### ✅ Phase 2: GPU Basic (Naive Implementation)
- **Status:** Complete & Working
- **Files:** `src/main_cuda.cpp`, `src/cuda/autoencoder_basic.cu`
- **Features:**
  - All layers ported to CUDA kernels
  - Basic parallelization
  - Memory management (cudaMalloc/cudaFree)
  - Verified correctness vs CPU
- **Performance:** ~25x speedup vs CPU

### 🔄 Phase 3: GPU Optimized
- **Status:** In Progress
- **Files:** 
  - `autoencoder_opt_v1.cu` - Shared memory, tiling
  - `autoencoder_opt_v2.cu` - Kernel fusion, streams
- **Target:** 50-100x speedup vs CPU

### ⏳ Phase 4: SVM Integration
- **Status:** Planned
- **File:** `src/svm/svm_integration.cpp`
- **Goal:** 60-65% classification accuracy

## 📈 Performance Results

| Phase | Time/Epoch (10K images) | Speedup | Loss |
|-------|------------------------|---------|------|
| CPU Baseline | ~300s | 1x | ~0.26 |
| GPU Basic | ~12s | 25x | ~0.65 |
| GPU Opt v1 | TBD | TBD | TBD |
| GPU Opt v2 | TBD | TBD | TBD |

## 🎯 Target Metrics (from Problem Statement)

- ✅ Autoencoder training time: < 10 minutes
- ⏳ Feature extraction time: < 20 seconds (60K images)
- ✅ GPU speedup over CPU: > 20x
- ⏳ Classification accuracy: 60-65%

## 📝 Documentation

- **`Report.ipynb`** - Main project report (Jupyter Notebook)
- **`Problem_statement.md`** - Full project requirements and guidelines
- **`README.md`** - This file

## 🛠️ Dependencies

- CUDA Toolkit (11.0+)
- g++ with C++17 support
- CIFAR-10 dataset (binary format)

## 📖 References

- Problem Statement: See `Problem_statement.md`
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/

## 👥 Team

[Your team information here]

---

**Last Updated:** December 11, 2025  
**Status:** Phase 2 Complete, Phase 3 In Progress
