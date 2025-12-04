## CSC14120 Final Project - CUDA Autoencoder + SVM (C/C++)

This repository implements the **two-stage pipeline** described in `CSC14120_2025_Final Project.pdf`:

- **Stage 1**: Convolutional autoencoder (CPU baseline + CUDA basic + CUDA optimized versions)
- **Stage 2**: SVM classifier (using LIBSVM or similar) on learned encoder features

### Folder Structure

- **`src/`**
  - **`main.cpp`**: Entry point, orchestrates the pipeline.
  - **`data/`**: CIFAR-10 loading and preprocessing (`cifar10_loader.*`).
  - **`cpu/`**: CPU baseline autoencoder (`autoencoder_cpu.cpp`).
  - **`cuda/`**: GPU implementations
    - `autoencoder_basic.cu` – Phase 2.2 (basic CUDA)
    - `autoencoder_opt_v1.cu` – Phase 2.3 (optimized v1, shared memory, etc.)
    - `autoencoder_opt_v2.cu` – Phase 2.4 (kernel fusion, streams, etc.)
  - **`svm/`**: SVM integration (`svm_integration.cpp`).

- **`include/`**
  - Shared headers and configuration (e.g., `config.h`).

- **`cifar-10-binary/`**
  - Downloaded CIFAR-10 binary files as specified in the project PDF.

- **`scripts/`**
  - `build_cpu.sh` – build CPU baseline.
  - `build_cuda.sh` – build CUDA versions.
  - `run_examples.sh` – example commands / paths.

- **`report/`**
  - Jupyter notebook report for Google Colab (to be added later).

### Build (Example)

```bash
cd /home/phuc-nguyen22/DOCD/CUDA/Final
chmod +x scripts/*.sh
./scripts/build_cpu.sh
./scripts/build_cuda.sh
```

Then run the binaries from `build_cpu/` and `build_cuda/` as needed.

