
**CSC14120 - PARALLEL PROGRAMMING**
**FINAL PROJECT**

-----

## 1\. Introduction

### 1.1 Problem Statement

Feature engineering is a fundamental challenge in machine learning: how do we automatically discover good representations of data that capture its underlying structure? 

In this project, you will implement an Autoencoder-based unsupervised feature learning system for image classification on the CIFAR-10 dataset. 

Traditional supervised learning trains a model end-to-end using labelled examples.However, autoencoders take a different approach: they learn meaningful representations by attempting to reconstruct the input data itself, without requiring any labels during the feature learning phase.
This unsupervised pre-training can discover features that are often more robust and generalizable than those learned through direct supervision alone. 

**Your task is to build and optimize a complete two-stage pipeline:** 

**Stage 1 - Unsupervised Feature Learning:**

  * Train a convolutional autoencoder to reconstruct CIFAR-10 images. 
  * The autoencoder learns to encode $32\times32\times3$ images into an 8,192-dimensional feature representation that captures meaningful visual patterns. 
  * No labels are used during this training phase.
  * The network learns features that capture important visual patterns (edges, textures, shapes).

**Stage 2 - Supervised Classification:**

  * Extract features from the trained encoder for all images. 
  * Train an SVM classifier on these learned features with class labels. 
  * Evaluate classification performance on the test set. 

The primary focus is on implementing and optimizing the autoencoder training and inference in CUDA, while using existing libraries (LIBSVM) for the SVM classifier. Through systematic optimization, you will accelerate the autoencoder from taking hours on CPU to completing in seconds on GPU. 

### 1.2 CIFAR-10 Dataset

CIFAR-10 is one of the most widely used benchmark datasets in computer vision and machine learning research. It provides a challenging yet computationally manageable image classification task. [cite: 28, 29]

**Dataset Specifications:**

  * **Image size:** $32\times32$ pixels (RGB) 
  * **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck 
  * **Training set:** 50,000 images (5,000 per class) 
  * **Test set:** 10,000 images (1,000 per class)
  * **Total images:** 60,000 
  * **Format:** Binary files with `uint8` pixel values 

**Dataset Organization for This Project:**

  * [cite_start]**Autoencoder training:** All 50,000 training images (labels ignored)
  * [cite_start]**SVM training:** Same 50,000 images with labels (using extracted features)
  * [cite_start]**Evaluation:** 10,000 test images with labels 

[cite_start]**Dataset Link:** [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) 

### 1.3 Expected Performance Targets

| Metric | Target |
| :--- | :--- |
| Autoencoder training time | \< 10 minutes |
| Feature extraction time | \< 20 seconds for all 60K images |
| Test classification accuracy | 60-65% |
| GPU speedup over CPU | \> 20x |

[cite_start]

-----

## 2\. Background Knowledge

### 2.1 Autoencoders for Unsupervised Feature Learning

**What is an Autoencoder?**
[cite_start]An autoencoder is a neural network trained to reconstruct its input, forcing the network to learn a compressed, meaningful representation in the process. 

It consists of two parts:

1.  [cite_start]**Encoder:** Compresses the input into a low-dimensional latent representation (feature vector). 
2.  [cite_start]**Decoder:** Reconstructs the original input from the latent representation. 

**Training Objective:**
$$Loss = MSE(Input, Reconstructed\_Output)$$
$$= (1/N) * \Sigma (x - decoder(encoder(x)))^2$$
[cite_start]The network learns to minimize reconstruction error, forcing the encoder to capture essential information about the input. 

**Key Concepts:**

1.  **Bottleneck Layer (Latent Space):**
      * The smallest layer in the middle.
      * Forces compression and feature extraction.
      * In this project: $(6, 6, 128) = 4,608 dimensions$
2.  **Symmetric Architecture:**
      * Decoder mirrors the encoder.
      * Up sampling operations reverse the down sampling. 
      * Helps in reconstruction quality
3.  **Feature Extraction:**
      * After training, we discard the decoder.
      * Use only the encoder to extract features.
      * These features feed into the SVM classifier. 

**Recommended Reading Materials:**

  * **Foundational Papers:** Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the Dimensionality of Data with Neural Networks." Science. [cite_start][Link](https://www.cs.toronto.edu/~hinton/science.pdf) [cite: 95, 97]
  * **Book Chapters:** Goodfellow, I., Bengio, Y., & Courville, A. "Deep Learning" (2016) Chapter 14: Autoencoders. [cite_start][Link](https://www.deeplearningbook.org/contents/autoencoders.html) [cite: 99, 100]
  * **Video guide:** Autoencoders and Representation Learning. [cite_start][Link](https://www.youtube.com/watch?v=R3DNKE3zKFk) [cite: 101, 102]
  * **Reference source code:**
      * [cite_start][https://github.com/turkdogan/autoencoder](https://github.com/turkdogan/autoencoder) [cite: 104]
      * [cite_start][https://github.com/tbennun/cudnn-training](https://github.com/tbennun/cudnn-training) [cite: 105]

**CNN References:**
Since your autoencoder uses convolutional layers, knowledge of CNNs is required.

  * [cite_start]Video: "How Convolutional Neural Networks work". [cite: 109]
  * [cite_start]Stanford Cheatsheet on CNNs. [cite: 111]
  * [cite_start]"Dive into Deep Learning" book on CNNs. [cite: 111]

### 2.2 Support Vector Machine (SVM) for Classification

**Your Role with SVM:** You do NOT need to implement SVM from scratch. [cite_start]Use existing optimized libraries. 

**Recommended SVM Libraries:**

1.  **LIBSVM (Primary Recommendation):**
      * Most popular SVM library.
      * Easy to use C/C++ interface.
      * Supports multi-class classification.
      * RBF kernel built-in.
      * [cite_start]Link: [https://www.csie.ntu.edu.tw/\~cjlin/libsvm/](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 
2.  **LIBLINEAR (Alternative for Linear SVM):**
      * Faster for linear kernels.
      * Good if you want to experiment.
      * [cite_start]Link: [https://www.csie.ntu.edu.tw/\~cjlin/liblinear/](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) 
3.  **ThunderSVM (GPU-Accelerated):**
      * GPU implementation of SVM.
      * Optional: Could explore for extra credit.
      * [cite_start]Link: [https://github.com/Xtra-Computing/thundersvm](https://github.com/Xtra-Computing/thundersvm) 

-----

## 3\. Project Pipeline

The project follows a clear 5-step pipeline:

**Step 1: Load CIFAR-10 Images**

  * 50,000 training images (for autoencoder)
  * 10,000 test images
  * Preprocess: normalize to [0,1]

**Step 2: Train Autoencoder (Your CUDA Code)**

  * Use all 50k images (unsupervised training)
  * Ignore labels during autoencoder training
  * CNN-based autoencoder architecture (see Section 4)
  * Train to minimize reconstruction loss
  * Save encoder weights 
**Step 3: Extract Features (Your CUDA Code)**

  * Load trained encoder weights
  * Run encoder forward pass (no decoder)
  * train\_features: (50000, 8192)
  * test\_features: (10000, 8192) 

**Step 4: Train SVM (Library)**

  * Input: train\_features + labels
  * Kernel: RBF (Radial Basis Function)
  * Hyperparameters: $C=10$, gamma auto
  * Output: trained SVM model
  * (Can use other SVM setup as you see fit)

**Step 5: Evaluate**

  * Predict on test\_features using SVM
  * Calculate accuracy, confusion matrix
  * Expected accuracy: 60-65%
  * Compare with baseline methods
**Key Points:**

  * Your focus: Steps 2 and 3 (autoencoder implementation in CUDA)
  * Using libraries: Step 4 (SVM training)
  * Evaluation: Step 5 (measure success) 

-----

## 4\. Network Architecture

### 4.1 Architecture Overview

INPUT: (32, 32, 3) $\rightarrow$ ENCODER (compress) $\rightarrow$ LATENT: (8, 8, 128) = 8,192 features $\rightarrow$ DECODER (reconstruct) $\rightarrow$ OUTPUT: (32, 32, 3) 

### 4.2 Detailed Architecture Specification

**ENCODER (Down sampling Path)**

1.  **INPUT:** (32, 32, 3)
2.  **Conv2D** (256 filters, $3\times3$ kernel, padding=1, stride=1) + **ReLU** $\rightarrow$ (32, 32, 256)
3.  **MaxPool2D** (2x2, stride=2) $\rightarrow$ (16, 16, 256)
4.  **Conv2D** (128 filters, $3\times3$ kernel, padding=1, stride=1) + **ReLU** $\rightarrow$ (16, 16, 128)
5.  **MaxPool2D** (2x2, stride=2) $\rightarrow$ (8, 8, 128)
6.  **LATENT REPRESENTATION:** (8, 8, 128) $\rightarrow$ 8192 dimensions [cite: 174-191]

**DECODER (Upsampling Path - Mirror of Encoder)**

1.  **LATENT:** (8, 8, 128)
2.  **Conv2D** (128 filters, $3\times3$ kernel, padding=1, stride=1) + **ReLU** $\rightarrow$ (8, 8, 128)
3.  **UpSample2D** (2x2) [Nearest neighbor or bilinear] $\rightarrow$ (16, 16, 128)
4.  **Conv2D** (256 filters, $3\times3$ kernel, padding=1, stride=1) + **ReLU** $\rightarrow$ (16, 16, 256)
5.  **UpSample2D** (2x2) [Nearest neighbor or bilinear] $\rightarrow$ (32, 32, 256)
6.  **Conv2D** (3 filters, $3\times3$ kernel, padding=1, stride=1) [No activation] $\rightarrow$ (32, 32, 3)
7.  **OUTPUT:** (32, 32, 3) 

**Keras setup for reference:**

```python
input_size = (32, 32, 3)
input_image = Input(shape=input_size)
# Encoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_image)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), padding='same')(x)
```

[cite_start][cite: 209-227]

**Parameter Summary:**

| Layer (type) | Output Shape | Param \# |
| :--- | :--- | :--- |
| input\_1 (InputLayer) | (None, 32, 32, 3) | 0 |
| conv2d\_1 (Conv2D) | (None, 32, 32, 256) | 7,168 |
| max\_pooling2d\_1 (MaxPooling2D) | (None, 16, 16, 256) | 0 |
| conv2d\_2 (Conv2D) | (None, 16, 16, 128) | 295,040 |
| encoded\_layer (MaxPooling2D) | (None, 8, 8, 128) | 0 |
| conv2d\_3 (Conv2D) | (None, 8, 8, 128) | 147,584 |
| up\_sampling2d\_1 (UpSampling2D) | (None, 16, 16, 128) | 0 |
| conv2d\_4 (Conv2D) | (None, 16, 16, 256) | 295,168 |
| up\_sampling2d\_2 (UpSampling2D) | (None, 32, 32, 256) | 0 |
| conv2d\_5 (Conv2D) | (None, 32, 32, 3) | 6,915 |
| **Total params:** | | **751,875** |


-----

Dưới đây là nội dung phần "Implementation Guideline" được chuyển sang định dạng Markdown đầy đủ theo yêu cầu của bạn:

## 5. Implementation Guideline
[cite_start]You will implement this project in at least 4 progressive phases, focusing on CUDA optimization for the autoencoder training and inference[cite: 233].

### Phase 1: CPU Baseline & Data Pipeline
[cite_start]**Objective:** Set up the project infrastructure and create a working CPU baseline[cite: 234, 235].

**What to Implement:**

[cite_start]**1.1 Data Loading and Preprocessing** [cite: 237]
* [cite_start]Create a `CIFAR10 Dataset` class to handle data loading[cite: 238].
* [cite_start]Read CIFAR-10 binary files (5 training batches + 1 test batch)[cite: 239].
* [cite_start]Parse the binary format: 1 byte label + 3,072 bytes image per record[cite: 240].
* [cite_start]Convert `uint8` pixel values [0, 255] to `float` [0, 1] for normalization[cite: 240].
* [cite_start]Implement batch generation for training[cite: 240].
* [cite_start]Add data shuffling capability[cite: 244].
* [cite_start]Organize train images (50,000), test images (10,000), and their labels in memory[cite: 245].

[cite_start]**1.2 CPU Neural Network Layers** [cite: 246]
[cite_start]Implement CPU versions of all necessary operations[cite: 247]:
* [cite_start]**Convolution (Conv2D):** Apply $3\times3$ kernels with padding and stride[cite: 248].
* [cite_start]**ReLU Activation:** Element-wise $max(0, x)$ operation[cite: 249].
* [cite_start]**Max Pooling:** $2\times2$ pooling to downsample by half[cite: 250].
* [cite_start]**Upsampling:** Nearest neighbor interpolation to double spatial dimensions[cite: 251].
* [cite_start]**MSE Loss:** Mean squared error between output and target for reconstruction[cite: 252].

[cite_start]**1.3 Autoencoder** [cite: 253]
* [cite_start]Create an `Autoencoder` class to encapsulate the network[cite: 254].
* [cite_start]Allocate memory for all weight matrices and biases (5 conv layers total)[cite: 255].
* [cite_start]Implement weight initialization[cite: 255].
* [cite_start]Allocate memory for intermediate activations between layers[cite: 256].
* [cite_start]Implement forward pass: sequentially apply encoder layers, then decoder layers[cite: 257].
* [cite_start]Implement backward pass[cite: 258].
* [cite_start]Implement feature extraction method: run encoder only and return latent representation[cite: 259].
* [cite_start]Add weight saving/loading functionality for model persistence[cite: 260].

[cite_start]**1.4 Training** [cite: 261]
* [cite_start]Set hyperparameters: batch size (32), epochs (20), learning rate (0.001)[cite: 262].
* [cite_start]Loop over epochs and batches[cite: 263].
* [cite_start]For each batch: forward pass $\rightarrow$ compute loss $\rightarrow$ backward pass $\rightarrow$ update weights[cite: 264, 265].
* [cite_start]Track and display training loss[cite: 266].
* [cite_start]Measure time per epoch[cite: 267].
* [cite_start]Save trained model weights after completion[cite: 268].

**Deliverables:**
* [cite_start]Working data loading and preprocessing pipeline[cite: 270].
* [cite_start]Complete CPU implementation of all layers[cite: 274].
* [cite_start]Baseline performance measurements[cite: 275].

---

### Phase 2: Naive GPU Implementation
[cite_start]**Objective:** Port all operations to GPU with basic parallelization[cite: 276, 277].

**What to Implement:**

[cite_start]**2.1 GPU Memory Management** [cite: 279]
* [cite_start]Create `GPUAutoencoder` class for GPU operations[cite: 280].
* [cite_start]Allocate device memory for all weight matrices[cite: 281].
* [cite_start]Allocate device memory for all activation buffers[cite: 282].
* [cite_start]Allocate device memory for gradients[cite: 283].
* [cite_start]Implement functions to copy weights between host and device[cite: 284].
* [cite_start]Implement proper memory cleanup (`cudaFree`)[cite: 285].

[cite_start]**2.2 Naive GPU Kernels** [cite: 286]
Design basic GPU kernels where each thread handles simple work:

* **Convolution Kernel:**
    * [cite_start]Each thread computes one output pixel[cite: 288].
    * [cite_start]Thread performs nested loops over kernel and input channels[cite: 289].
    * [cite_start]Uses global memory for all reads/writes[cite: 290].
    * [cite_start]Handles boundary conditions with padding[cite: 291].
* **ReLU Kernel:**
    * [cite_start]Each thread processes one element[cite: 293].
    * [cite_start]Simple in-place operation: $x = max(0, x)$[cite: 294].
* **MaxPooling Kernel:**
    * [cite_start]Each thread computes one output element[cite: 296].
    * [cite_start]Finds maximum value in $2\times2$ input window[cite: 297].
    * [cite_start]Writes result to output[cite: 298].
* **Upsampling Kernel:**
    * [cite_start]Each thread computes one output pixel[cite: 300].
    * [cite_start]Maps output coordinates back to input coordinates (divide by 2)[cite: 304].
    * [cite_start]Copies input value to output (nearest neighbour)[cite: 305].
* **MSE Loss Kernel:**
    * [cite_start]Parallel reduction to sum squared differences[cite: 307].
    * [cite_start]Use shared memory for partial sums[cite: 308].
    * [cite_start]Use `atomicAdd` for final accumulation[cite: 309].

[cite_start]**2.3 GPU Forward Pass** [cite: 310]
* [cite_start]Copy input batch from host to device[cite: 311].
* [cite_start]Launch kernels sequentially for each layer[cite: 312].
* [cite_start]Synchronize after kernel launches[cite: 313].
* [cite_start]Copy final output back to host[cite: 314].
* [cite_start]Compute loss value[cite: 315].

[cite_start]**2.4 GPU Backward Pass** [cite: 316]
Design gradient computation kernels:
* [cite_start]Implement gradient of each layer type (conv, relu, maxpool, upsample)[cite: 318].
* [cite_start]Backpropagate errors from output to input[cite: 319].
* [cite_start]Compute gradients with respect to weights[cite: 320].
* [cite_start]Implement simple SGD weight update: `weight -= learning_rate × gradient`[cite: 321].

[cite_start]**2.5 GPU Training Loop** [cite: 322]
* [cite_start]Set larger batch size for GPU (64 instead of 32)[cite: 323].
* [cite_start]For each batch: copy to device $\rightarrow$ forward $\rightarrow$ compute loss $\rightarrow$ backward $\rightarrow$ update[cite: 324, 325].
* [cite_start]Use GPU timer for accurate timing[cite: 326].
* [cite_start]Display progress and loss values[cite: 327].
* [cite_start]Save trained weights[cite: 328].

**Deliverables:**
* [cite_start]All layers successfully ported to GPU[cite: 330].
* [cite_start]Working GPU training loop[cite: 331].
* [cite_start]Forward and backward passes implemented[cite: 335].
* [cite_start]Correctness verification against CPU results[cite: 336].

---

### Phase 3: Advanced Optimization
[cite_start]**Objective:** Optimize memory access patterns and apply advanced techniques for maximum performance (May have several versions here)[cite: 337, 338].

[cite_start]**Optimization Ideas** (You don’t need to implement all; this list is just a suggestion)[cite: 339]:

[cite_start]**Category 1: Memory Optimization** [cite: 340]
1.  **Shared Memory Tiling for Convolution:** Load input tiles cooperatively into shared memory to reduce global memory accesses. [cite_start]Each thread block processes a tile, reusing loaded data across multiple threads [cite: 341-343].
2.  [cite_start]**Convert to Matrix Multiplication:** Transform convolution into matrix multiplication, then leverage optimized via matrix multiplication[cite: 344, 345].
3.  **Memory Coalescing Optimization:** Ensure threads in a warp access consecutive memory address for maximum bandwidth. [cite_start]Reorganize data access patterns and consider changing memory layout [cite: 346-348].
4.  **Constant Memory for Small Weights:** Store small, frequently accessed data (biases, small kernel weights) in constant memory for fast broadcast access. [cite_start]All threads in a warp can read the same value in a single instruction [cite: 349-351].
5.  **Pinned (Page-Locked) Memory:** Use `cudaMallocHost` for pinned memory on CPU side to enable faster asynchronous transfers. [cite_start]Reduces H2D and D2H transfer time compared to pageable memory [cite: 352-354].
6.  **Unified Memory Management:** Use CUDA Unified Memory to simplify memory management and allow on-demand data migration. [cite_start]System automatically handles data movement between CPU and GPU [cite: 355-357].
7.  **Memory Pool/Reuse Strategy:** Reallocate all necessary buffers once and reuse them across batches/epochs. [cite_start]Eliminates repeated `cudaMalloc`/`cudaFree` overhead which can be significant[cite: 358].

[cite_start]**Category 2: Kernel-Level Optimization** [cite: 359]
8.  **Kernel Fusion (Conv + ReLU + Bias):** Combine multiple operations into single kernel to eliminate intermediate global memory writes/reads. [cite_start]Reduces memory bandwidth requirements and improves arithmetic intensity[cite: 360, 364].
9.  **Block-Level Fusion (Entire Encoder/Decoder):** Fuse entire encoder or decoder paths into single mega-kernel using shared memory for intermediates. [cite_start]Dramatically reduces global memory traffic but requires careful shared memory management [cite: 365-367].
10. **Loop Unrolling:** Unroll loops in convolution, pooling, and other kernels to reduce loop overhead. [cite_start]Compiler can often optimize better with unrolled loops, especially for small kernel sizes [cite: 368-370].
11. **Vectorized Memory Access (float4):** Load/store 4 floats at once using `float4` for better memory bandwidth utilization. [cite_start]Particularly effective when processing channels or batch dimensions[cite: 371, 372].
12. **Optimized Thread Block Dimensions:** Tune block sizes (e.g., 16×16, 32×8) based on profiling to maximize occupancy and minimize wasted threads. [cite_start]Different layers may benefit from different configurations [cite: 373-375].
13. **Mixed Precision Training (FP16/FP32):** Use FP16 for forward pass and most computations, FP32 for weight updates and accumulators. [cite_start]Reduces memory bandwidth[cite: 376, 377].

[cite_start]**Category 3: Parallelism & Concurrency** [cite: 378]
14. **Gradient Checkpointing:** Store only subset of activations during forward pass, recompute others during backward pass. [cite_start]Trades computation for memory, enabling larger batch sizes [cite: 379-381].
15. **Multi-Stream Pipeline:** Use multiple CUDA streams to overlap H2D transfer, computation, and D2H transfer. [cite_start]Can hide transfer latency by processing next batch while current batch trains [cite: 382-384].
16. **Batched Operations:** Process multiple images in parallel rather than one at a time. [cite_start]Amortizes kernel launch overhead and improves GPU occupancy[cite: 385, 386].

---

### Phase 4: SVM Integration and analysed result
[cite_start]**Objective:** Complete the pipeline and thoroughly evaluate performance[cite: 387, 388].

Dưới đây là nội dung phần "Project Report" và "Project Deliverable" được chuyển sang định dạng Markdown đầy đủ:

## 6. Project Report

### Report Format and Structure
**Submission Format:**
[cite_start]Your project report must be submitted as a Notebook (`.ipynb`) that can run on Google Colab [cite: 394-395].

### Section 1: Problem Description
**What to Include:**

1.  **Problem Statement:**
    * Clearly define the image classification task.
    * [cite_start]Describe the motivation for GPU acceleration [cite: 398-401].
2.  **CIFAR-10 Dataset Overview:**
    * Dataset specifications (size, classes, split).
    * Show sample images from each class (use visualization).
    * [cite_start]Explain data preprocessing steps (normalization, format) [cite: 402-405].
3.  **Autoencoder Architecture:**
    * Describe the network architecture with a diagram.
    * Specify layer dimensions and transformations.
    * Explain the encoder-decoder structure and latent representation.
    * [cite_start]Include architecture visualization [cite: 406-410].
4.  **Project Objectives:**
    * Performance goals (training time, speedup targets, accuracy).
    * Technical learning objectives.
    * [cite_start]Success criteria [cite: 411-414].

### Section 2: Implementation Phases
For each phase, provide a structured description following this template:

#### Phase 2.1: CPU Baseline Implementation
**Objectives:**
* What you aimed to achieve in this phase.
* [cite_start]Why this phase is necessary [cite: 417-423].

**Implementation Details:**
* **Data Pipeline:** How you loaded and pre-processed CIFAR-10 data.
* **Layer Implementations:** Brief description of each layer (Conv2D, ReLU, MaxPool, Upsample).
* **Training Loop:** How you structured the training process.
* [cite_start]**Key Code Snippets:** Show 2-3 critical functions (e.g., convolution function signature and main loop structure) [cite: 424-428].

**Results:**
* Training time per epoch and total training time.
* Final reconstruction loss.
* Sample reconstructed images (show original vs reconstructed).
* [cite_start]Memory usage [cite: 429-433].

**Key Takeaways:**
* What did you learn about the algorithm?
* [cite_start]What insights guided your GPU implementation? [cite: 434-436]

#### Phase 2.2: GPU Basic Implementation
**Objectives:**
* Port CPU code to GPU with basic parallelization.
* Verify correctness of GPU kernels.
* [cite_start]Establish baseline GPU performance [cite: 437-441].

**Implementation Details:**
* **Parallelization Strategy:** How you mapped operations to GPU threads.
* **Kernel Designs:**
    * Convolution kernel: thread-to-output mapping.
    * Pooling kernel: how threads handle $2\times2$ windows.
    * Other kernels (ReLU, upsampling).
* **Memory Management:** Device memory allocation strategy.
* [cite_start]**Key Code Snippets:** Show kernel signatures and launch configurations [cite: 442-453].

**Results:**
* Training time per epoch and total training time.
* Speedup over CPU baseline (include table and chart).
* GPU memory usage.
* [cite_start]Verification that outputs match CPU (show error metrics) [cite: 454-458].

**Profiling Analysis:**
* Basic profiling results (time spent in each kernel type).
* Memory bandwidth utilization (if measured).
* [cite_start]Initial bottleneck identification [cite: 459-462].

**Key Takeaways:**
* What was surprisingly fast or slow?
* [cite_start]Where do you see optimization opportunities? [cite: 463-465]

#### Phase 2.3: GPU Optimized Implementation - Version 1
**Optimization Focus:** (e.g., Memory Optimization)

**Objectives:**
* What specific optimization(s) you targeted.
* [cite_start]Expected performance improvement [cite: 466-470].

**Implementation Details:**
* **Optimization Technique(s) Applied:**
    * Detailed explanation of the optimization (e.g., shared memory tiling).
    * Why this optimization should help.
    * Implementation approach.
* [cite_start]**Key Code Snippets:** Show the optimized kernel or key changes [cite: 471-477].

**Results:**
* Training time comparison with previous version.
* Speedup over previous phase (incremental and cumulative).
* Performance metrics (bandwidth utilization, occupancy).
* [cite_start]Profiling comparison: before vs after [cite: 478-485].

**Analysis:**
* Why did this optimization work (or not work as expected)?
* What did profiling reveal?
* [cite_start]What's the next bottleneck? [cite: 486-489]

**Key Takeaways:**
* Lessons learned from this optimization.
* [cite_start]Applicability to other problems [cite: 490-492].

#### Phase 2.4: GPU Optimized Implementation - Version 2 (if applicable)
**Optimization Focus:** (e.g., Kernel Fusion and Advanced Techniques)
Follow the same structure as Version 1:
* Objectives
* Implementation Details
* Results
* Analysis
* Key Takeaways

*Note: You may have multiple optimization versions. [cite_start]Create a separate subsection for each major optimization iteration.* [cite: 493-501]

#### Phase 2.5: SVM Integration
**Objectives:**
* Extract features using trained encoder.
* Train SVM classifier on learned features.
* [cite_start]Evaluate end-to-end classification performance [cite: 502-506].

**Implementation Details:**
* **Feature Extraction:** How you extracted features from encoder (Note: Check consistency with architecture, text says 1,024-dim but architecture in section 4 says 8,192).
* **LIBSVM Integration:** How you interfaced with LIBSVM.
* **Hyperparameter Selection:** SVM parameters chosen (C, gamma, kernel).
* [cite_start]**Key Code Snippets:** Feature extraction and SVM training code [cite: 507-513].

**Results:**
* Feature extraction time (50K train + 10K test).
* SVM training time.
* Classification accuracy on test set.
* Per-class accuracy breakdown (table).
* Confusion matrix (visualization).
* [cite_start]Comparison with baseline methods (if available) [cite: 514-520].

**Analysis:**
* Which classes are easiest/hardest to classify?
* What does the confusion matrix reveal?
* [cite_start]How does accuracy compare to expectations? [cite: 521-524]

**Key Takeaways:**
* Quality of learned features.
* [cite_start]Effectiveness of two-stage approach [cite: 525-527].

### Section 3: Comprehensive Performance Analysis

**3.1 Performance Comparison Across All Phases**
Create comprehensive comparison including:

**Table Format:**

| Phase | Training Time | Speedup (vs CPU) | Incremental Speedup | Memory Usage | Key Optimization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU Baseline** | 1800s | 1.0× | - | - | - |
| **GPU Basic** | 180s | 10.0× | 10.0× | 2.1 GB | Parallelization |
| **GPU Opt v1** | 45s | 40.0× | 4.0× | 2.3 GB | Shared memory |
| **GPU Opt v2** | 25s | 72.0× | 1.8× | 2.5 GB | Kernel Fusion + Streams |

[cite_start][cite: 531-535]

**Visualization Requirements:**
* Bar chart showing training time across phases.
* [cite_start]Line graph showing cumulative speedup [cite: 536-538].

### Section 4: Lessons Learned and Challenges Overcome
**4.1 Key Technical Insights**
[cite_start]What you have learn about: CUDA Programming, Deep Learning, Performance Optimization [cite: 540-541].

**4.2 Major Challenges and Solutions**
Present 2-3 significant challenges using this format:
* **Challenge 1: [Brief Title]**
    * **Problem:** One sentence describing the issue.
    * **Solution:** 1-2 sentences explaining how you solved it.
    * [cite_start]**Lesson:** One sentence on what you learned [cite: 542-547].

### Section 5: Conclusion and Future Work
**5.1 Project Summary**
* Recap of what was accomplished.
* Final performance metrics summary table.
* [cite_start]Achievement of original objectives [cite: 549-552].

**5.2 Key Achievements**
Highlight your best results:
* Maximum speedup achieved.
* Classification accuracy.
* Most successful optimization.
* [cite_start]Technical skills mastered [cite: 553-559].

**5.3 Limitations**
Honestly discuss:
* Current performance bottlenecks.
* Accuracy limitations.
* [cite_start]Implementation constraints [cite: 560-565].

**5.4 Future Improvements**
[cite_start][cite: 566]

---

## 7. Project Deliverable

### Submission Requirements
You must submit the following items:

**1. Team Plan and Work Distribution**
* Document detailing each team member's responsibilities.
* Task breakdown and timeline.
* [cite_start]Contribution percentage for each member [cite: 571-574].

**2. Project Report**
* Jupyter Notebook (`.ipynb`) as specified in Report Requirements.
* Must be executable on Google Colab.
* [cite_start]Export to PDF for archival purposes [cite: 575-578].

**3. Source Code Package**
* All source code files (`.cpp`, `.cu`, `.h`, header files, etc.).
* **README.md** with:
    * Setup instructions (dependencies, CUDA version, libraries).
    * Compilation commands.
    * Execution instructions with example commands.
    * Hardware requirements (GPU model, memory).
    * Expected outputs.
* [cite_start]Trained model weights (if file size permits, otherwise provide download link) [cite: 579-591].

**4. Presentation Video (15-20 minutes)**
* Upload to YouTube with **Unlisted** visibility.
* Include link in your notebook report and README file.
* [cite_start]**DO NOT** upload video to Moodle (file size restrictions) [cite: 592-595].

**Video Content Should Cover:**
* Problem overview and approach (2-3 min).
* Live demonstration of running code (5-7 min).
* Performance results and analysis (5-7 min).
* [cite_start]Key optimizations and lessons learned (3-5 min) [cite: 596-600].