#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// GPU OPTIMIZED V1 - SHARED MEMORY TILING
// ============================================================================
// Optimization Focus: Memory optimization for Conv2D using shared memory
// Technique: Tile-based cooperative loading with coalesced access
// Expected: 1.5-2x speedup from reduced global memory traffic
// ============================================================================

// ============================================================================
// GPU OPTIMIZED V1 - MEMORY OPTIMIZATION
// Optimization Focus: Shared memory tiling for Conv2D forward
// Technique: Cooperative loading of input tiles to reduce global memory access
// Expected: 1.5-2x speedup from reduced memory bandwidth usage
// ============================================================================

#define TILE_H 8
#define TILE_W 8

// V1: Conv2D Forward with Shared Memory Tiling
// Grid: (C_out, ceil(H_out/TILE_H), ceil(W_out/TILE_W))
// Block: (1, TILE_H, TILE_W) - each block handles 1 output channel, 1 tile
__global__ void conv2d_kernel_v1(
    const float* __restrict__ input,   // [C_in, H_in, W_in]
    const float* __restrict__ weight,  // [C_out, C_in, K, K]
    const float* __restrict__ bias,    // [C_out]
    float* __restrict__ output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    // Shared memory tile: TILE + halo for K=3, pad=1
    __shared__ float s_tile[TILE_H + 2][TILE_W + 2];
    
    int oc = blockIdx.x;           // Output channel (1 per block)
    int tile_h = blockIdx.y;       // Tile index in H
    int tile_w = blockIdx.z;       // Tile index in W
    
    int ty = threadIdx.y;          // Thread row in tile [0, TILE_H)
    int tz = threadIdx.z;          // Thread col in tile [0, TILE_W)
    
    int oh = tile_h * TILE_H + ty; // Global output position
    int ow = tile_w * TILE_W + tz;
    
    float sum = 0.0f;
    
    // Process each input channel
    for (int ic = 0; ic < C_in; ++ic) {
        // Cooperatively load tile into shared memory
        // Each thread loads multiple elements to cover halo region
        int tile_size_h = TILE_H + K - 1;  // 8 + 2 = 10 for K=3
        int tile_size_w = TILE_W + K - 1;  // 8 + 2 = 10
        
        for (int i = ty; i < tile_size_h; i += TILE_H) {
            for (int j = tz; j < tile_size_w; j += TILE_W) {
                int ih = tile_h * TILE_H + i - pad;
                int iw = tile_w * TILE_W + j - pad;
                
                float val = 0.0f;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    val = input[ic * H_in * W_in + ih * W_in + iw];
                }
                s_tile[i][j] = val;
            }
        }
        __syncthreads();
        
        // Compute convolution from shared memory (reuse!)
        if (oh < H_out && ow < W_out) {
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    float val = s_tile[ty + kh][tz + kw];
                    int w_idx = ((oc * C_in + ic) * K + kh) * K + kw;
                    sum += val * weight[w_idx];
                }
            }
        }
        __syncthreads();
    }
    
    // Write output with bias
    if (oh < H_out && ow < W_out) {
        output[oc * H_out * W_out + oh * W_out + ow] = sum + bias[oc];
    }
}

// ============================================================================
// CUDA KERNELS - FORWARD PASS (Original Basic Kernels)
// ============================================================================

// Conv2D kernel: Basic implementation
__global__ void conv2d_kernel(
    const float* input,   // [C_in, H, W]
    const float* weight,  // [C_out, C_in, K, K]
    const float* bias,    // [C_out]
    float* output,        // [C_out, H_out, W_out]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int oc = blockIdx.x;  // Output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Output height
    int ow = blockIdx.z * blockDim.z + threadIdx.z;  // Output width
    
    if (oc >= C_out || oh >= H_out || ow >= W_out) return;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;
                
                float input_val = 0.0f;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    input_val = input[ic * H_in * W_in + ih * W_in + iw];
                }
                
                int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
                sum += input_val * weight[weight_idx];
            }
        }
    }
    
    sum += bias[oc];
    output[oc * H_out * W_out + oh * W_out + ow] = sum;
}

// ReLU activation kernel
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// MaxPool2D kernel (2x2, stride 2)
__global__ void maxpool_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H/2, W/2]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H / 2;
    int W_out = W / 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    float max_val = input[c * H * W + ih * W + iw];
    max_val = fmaxf(max_val, input[c * H * W + ih * W + (iw + 1)]);
    max_val = fmaxf(max_val, input[c * H * W + (ih + 1) * W + iw]);
    max_val = fmaxf(max_val, input[c * H * W + (ih + 1) * W + (iw + 1)]);
    
    output[c * H_out * W_out + oh * W_out + ow] = max_val;
}

// Upsample2x kernel (nearest neighbor)
__global__ void upsample_kernel(
    const float* input,   // [C, H, W]
    float* output,        // [C, H*2, W*2]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh / 2;
    int iw = ow / 2;
    
    float val = input[c * H * W + ih * W + iw];
    output[c * H_out * W_out + oh * W_out + ow] = val;
}

// ============================================================================
// CUDA KERNELS - BACKWARD PASS
// ============================================================================

// ReLU backward kernel
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// MaxPool backward kernel
__global__ void maxpool_backward_kernel(
    const float* grad_output,  // [C, H/2, W/2]
    const float* input,        // [C, H, W]
    float* grad_input,         // [C, H, W]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    
    int H_out = H / 2;
    int W_out = W / 2;
    
    if (c >= C || oh >= H_out || ow >= W_out) return;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    // Find which position had the max value
    float max_val = input[c * H * W + ih * W + iw];
    int max_i = ih, max_j = iw;
    
    float val = input[c * H * W + ih * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_i = ih; max_j = iw + 1; }
    
    val = input[c * H * W + (ih + 1) * W + iw];
    if (val > max_val) { max_val = val; max_i = ih + 1; max_j = iw; }
    
    val = input[c * H * W + (ih + 1) * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_i = ih + 1; max_j = iw + 1; }
    
    // Only pass gradient to the max position
    float grad = grad_output[c * H_out * W_out + oh * W_out + ow];
    atomicAdd(&grad_input[c * H * W + max_i * W + max_j], grad);
}

// Upsample backward kernel
__global__ void upsample_backward_kernel(
    const float* grad_output,  // [C, H*2, W*2]
    float* grad_input,         // [C, H, W]
    int C, int H, int W)
{
    int c = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c >= C || ih >= H || iw >= W) return;
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    // Sum gradients from all 4 upsampled positions
    float sum = 0.0f;
    sum += grad_output[c * H_out * W_out + (ih * 2) * W_out + (iw * 2)];
    sum += grad_output[c * H_out * W_out + (ih * 2) * W_out + (iw * 2 + 1)];
    sum += grad_output[c * H_out * W_out + (ih * 2 + 1) * W_out + (iw * 2)];
    sum += grad_output[c * H_out * W_out + (ih * 2 + 1) * W_out + (iw * 2 + 1)];
    
    grad_input[c * H * W + ih * W + iw] = sum;
}

// Conv2D weight gradient kernel
__global__ void conv2d_weight_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* input,        // [C_in, H_in, W_in]
    float* weight_grad,        // [C_out, C_in, K, K]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int k_idx = threadIdx.x;  // Flattened kernel index (k*K + k)
    
    if (oc >= C_out || ic >= C_in || k_idx >= K * K) return;
    
    int kh = k_idx / K;
    int kw = k_idx % K;
    
    float sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int ih = oh + kh - pad;
            int iw = ow + kw - pad;
            
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                float inp = input[ic * H_in * W_in + ih * W_in + iw];
                sum += grad * inp;
            }
        }
    }
    
    int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
    weight_grad[weight_idx] = sum;
}

// Conv2D bias gradient kernel
__global__ void conv2d_bias_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    float* bias_grad,          // [C_out]
    int C_out, int H_out, int W_out)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= C_out) return;
    
    float sum = 0.0f;
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            sum += grad_output[oc * H_out * W_out + oh * W_out + ow];
        }
    }
    bias_grad[oc] = sum;
}

// Conv2D input gradient kernel
__global__ void conv2d_input_grad_kernel(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* weight,       // [C_out, C_in, K, K]
    float* grad_input,         // [C_in, H_in, W_in]
    int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K, int pad)
{
    int ic = blockIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int iw = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ic >= C_in || ih >= H_in || iw >= W_in) return;
    
    float sum = 0.0f;
    for (int oc = 0; oc < C_out; oc++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int oh = ih - kh + pad;
                int ow = iw - kw + pad;
                
                if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                    float grad = grad_output[oc * H_out * W_out + oh * W_out + ow];
                    int weight_idx = ((oc * C_in + ic) * K + kh) * K + kw;
                    sum += grad * weight[weight_idx];
                }
            }
        }
    }
    
    grad_input[ic * H_in * W_in + ih * W_in + iw] = sum;
}

// MSE Loss and gradient kernel (same as CPU logic)
__global__ void mse_loss_kernel(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        grad[idx] = 2.0f * diff / size;
        atomicAdd(loss, diff * diff / size);
    }
}

// Weight update kernel (Simple SGD with gradient clipping)
__global__ void sgd_update_kernel(
    float* weight,
    const float* grad,
    float learning_rate,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        // Check for NaN/Inf first
        if (isnan(g) || isinf(g)) {
            g = 0.0f;
        } else {
            // Clip gradient to prevent explosion
            if (g > 5.0f) g = 5.0f;
            if (g < -5.0f) g = -5.0f;
        }
        weight[idx] -= learning_rate * g;
    }
}

// ============================================================================
// AUTOENCODER CLASS
// ============================================================================

class AutoencoderGPU {
public:
    // Host weight pointers
    std::vector<float> h_conv1_w, h_conv1_b;
    std::vector<float> h_conv2_w, h_conv2_b;
    std::vector<float> h_conv3_w, h_conv3_b;
    std::vector<float> h_conv4_w, h_conv4_b;
    std::vector<float> h_conv5_w, h_conv5_b;
    
    // Device weight pointers
    float *d_conv1_w, *d_conv1_b;
    float *d_conv2_w, *d_conv2_b;
    float *d_conv3_w, *d_conv3_b;
    float *d_conv4_w, *d_conv4_b;
    float *d_conv5_w, *d_conv5_b;
    
    // Device gradient pointers
    float *d_conv1_w_grad, *d_conv1_b_grad;
    float *d_conv2_w_grad, *d_conv2_b_grad;
    float *d_conv3_w_grad, *d_conv3_b_grad;
    float *d_conv4_w_grad, *d_conv4_b_grad;
    float *d_conv5_w_grad, *d_conv5_b_grad;
    
    // Device activation buffers
    float *d_input;
    float *d_conv1_out, *d_relu1_out, *d_pool1_out;
    float *d_conv2_out, *d_relu2_out, *d_pool2_out;
    float *d_conv3_out, *d_relu3_out, *d_up1_out;
    float *d_conv4_out, *d_relu4_out, *d_up2_out;
    float *d_conv5_out;
    
    // Device gradient buffers
    float *d_grad_conv5, *d_grad_up2, *d_grad_relu4, *d_grad_conv4;
    float *d_grad_up1, *d_grad_relu3, *d_grad_conv3;
    float *d_grad_pool2, *d_grad_relu2, *d_grad_conv2;
    float *d_grad_pool1, *d_grad_relu1, *d_grad_conv1;
    
    float *d_loss;
    float last_loss = 0.0f;
    
    AutoencoderGPU() {
        // Initialize weights on host
        h_conv1_w.resize(256 * 3 * 3 * 3);
        h_conv1_b.resize(256);
        h_conv2_w.resize(128 * 256 * 3 * 3);
        h_conv2_b.resize(128);
        h_conv3_w.resize(128 * 128 * 3 * 3);
        h_conv3_b.resize(128);
        h_conv4_w.resize(256 * 128 * 3 * 3);
        h_conv4_b.resize(256);
        h_conv5_w.resize(3 * 256 * 3 * 3);
        h_conv5_b.resize(3);
        
        // Simple random initialization (same as CPU)
        srand(time(NULL));
        for (auto& w : h_conv1_w) w = ((rand() % 100) / 500.0f - 0.1f);
        for (auto& w : h_conv2_w) w = ((rand() % 100) / 500.0f - 0.1f);
        for (auto& w : h_conv3_w) w = ((rand() % 100) / 500.0f - 0.1f);
        for (auto& w : h_conv4_w) w = ((rand() % 100) / 500.0f - 0.1f);
        for (auto& w : h_conv5_w) w = ((rand() % 100) / 500.0f - 0.1f);
        
        std::fill(h_conv1_b.begin(), h_conv1_b.end(), 0.0f);
        std::fill(h_conv2_b.begin(), h_conv2_b.end(), 0.0f);
        std::fill(h_conv3_b.begin(), h_conv3_b.end(), 0.0f);
        std::fill(h_conv4_b.begin(), h_conv4_b.end(), 0.0f);
        std::fill(h_conv5_b.begin(), h_conv5_b.end(), 0.0f);
        
        // Allocate device memory for weights
        CUDA_CHECK(cudaMalloc(&d_conv1_w, h_conv1_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_b, h_conv1_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_w, h_conv2_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_b, h_conv2_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3_w, h_conv3_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3_b, h_conv3_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv4_w, h_conv4_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv4_b, h_conv4_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv5_w, h_conv5_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv5_b, h_conv5_b.size() * sizeof(float)));
        
        // Copy weights to device
        CUDA_CHECK(cudaMemcpy(d_conv1_w, h_conv1_w.data(), h_conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv1_b, h_conv1_b.data(), h_conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv2_w, h_conv2_w.data(), h_conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv2_b, h_conv2_b.data(), h_conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv3_w, h_conv3_w.data(), h_conv3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv3_b, h_conv3_b.data(), h_conv3_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv4_w, h_conv4_w.data(), h_conv4_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv4_b, h_conv4_b.data(), h_conv4_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv5_w, h_conv5_w.data(), h_conv5_w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conv5_b, h_conv5_b.data(), h_conv5_b.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Allocate device memory for gradients
        CUDA_CHECK(cudaMalloc(&d_conv1_w_grad, h_conv1_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_b_grad, h_conv1_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_w_grad, h_conv2_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_b_grad, h_conv2_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3_w_grad, h_conv3_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3_b_grad, h_conv3_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv4_w_grad, h_conv4_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv4_b_grad, h_conv4_b.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv5_w_grad, h_conv5_w.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv5_b_grad, h_conv5_b.size() * sizeof(float)));
        
        // Allocate device memory for activations
        CUDA_CHECK(cudaMalloc(&d_input, 3 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_out, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_relu1_out, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool1_out, 256 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_out, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_relu2_out, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2_out, 128 * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv3_out, 128 * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_relu3_out, 128 * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up1_out, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv4_out, 256 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_relu4_out, 256 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up2_out, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv5_out, 3 * 32 * 32 * sizeof(float)));
        
        // Allocate device memory for gradients
        CUDA_CHECK(cudaMalloc(&d_grad_conv5, 3 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_up2, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_relu4, 256 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_conv4, 256 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_up1, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_relu3, 128 * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_conv3, 128 * 8 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_pool2, 128 * 16 * 16 * sizeof(float)));  // grad wrt pool2 input (128,16,16)
        CUDA_CHECK(cudaMalloc(&d_grad_relu2, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_conv2, 128 * 16 * 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_pool1, 256 * 32 * 32 * sizeof(float)));  // grad wrt pool1 input (256,32,32)
        CUDA_CHECK(cudaMalloc(&d_grad_relu1, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_conv1, 256 * 32 * 32 * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    }
    
    ~AutoencoderGPU() {
        // Free all device memory
        cudaFree(d_conv1_w); cudaFree(d_conv1_b);
        cudaFree(d_conv2_w); cudaFree(d_conv2_b);
        cudaFree(d_conv3_w); cudaFree(d_conv3_b);
        cudaFree(d_conv4_w); cudaFree(d_conv4_b);
        cudaFree(d_conv5_w); cudaFree(d_conv5_b);
        
        cudaFree(d_conv1_w_grad); cudaFree(d_conv1_b_grad);
        cudaFree(d_conv2_w_grad); cudaFree(d_conv2_b_grad);
        cudaFree(d_conv3_w_grad); cudaFree(d_conv3_b_grad);
        cudaFree(d_conv4_w_grad); cudaFree(d_conv4_b_grad);
        cudaFree(d_conv5_w_grad); cudaFree(d_conv5_b_grad);
        
        cudaFree(d_input);
        cudaFree(d_conv1_out); cudaFree(d_relu1_out); cudaFree(d_pool1_out);
        cudaFree(d_conv2_out); cudaFree(d_relu2_out); cudaFree(d_pool2_out);
        cudaFree(d_conv3_out); cudaFree(d_relu3_out); cudaFree(d_up1_out);
        cudaFree(d_conv4_out); cudaFree(d_relu4_out); cudaFree(d_up2_out);
        cudaFree(d_conv5_out);
        
        cudaFree(d_grad_conv5); cudaFree(d_grad_up2); cudaFree(d_grad_relu4);
        cudaFree(d_grad_conv4); cudaFree(d_grad_up1); cudaFree(d_grad_relu3);
        cudaFree(d_grad_conv3); cudaFree(d_grad_pool2); cudaFree(d_grad_relu2);
        cudaFree(d_grad_conv2); cudaFree(d_grad_pool1); cudaFree(d_grad_relu1);
        cudaFree(d_grad_conv1);
        
        cudaFree(d_loss);
    }
    
    float train_step(const std::vector<float>& image, float learning_rate) {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, image.data(), 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward();
        
        // Compute loss and gradient
        float h_loss = 0.0f;
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        
        int size = 3 * 32 * 32;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        mse_loss_kernel<<<blocks, threads>>>(d_conv5_out, d_input, d_loss, d_grad_conv5, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        last_loss = h_loss;
        
        // Backward pass
        backward();
        
        // Update weights with gradient clipping
        update_weights(learning_rate);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return h_loss;
    }
    
    void forward() {
        // V1: Use tiling grid for shared memory optimization
        dim3 block(1, TILE_H, TILE_W);
        
        // Conv1: (3, 32, 32) -> (256, 32, 32)
        dim3 grid1(256, (32 + TILE_H - 1) / TILE_H, (32 + TILE_W - 1) / TILE_W);
        conv2d_kernel_v1<<<grid1, block>>>(d_input, d_conv1_w, d_conv1_b, d_conv1_out,
                                            3, 32, 32, 256, 32, 32, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU1
        relu_kernel<<<(256*32*32 + 255)/256, 256>>>(d_conv1_out, d_relu1_out, 256*32*32);
        CUDA_CHECK(cudaGetLastError());
        
        // MaxPool1: (256, 32, 32) -> (256, 16, 16)
        dim3 blocks_pool1(256, 1, 1);
        dim3 threads_pool1(1, 16, 16);
        maxpool_kernel<<<blocks_pool1, threads_pool1>>>(d_relu1_out, d_pool1_out, 256, 32, 32);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv2: (256, 16, 16) -> (128, 16, 16)
        dim3 grid2(128, (16 + TILE_H - 1) / TILE_H, (16 + TILE_W - 1) / TILE_W);
        conv2d_kernel_v1<<<grid2, block>>>(d_pool1_out, d_conv2_w, d_conv2_b, d_conv2_out,
                                            256, 16, 16, 128, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU2
        relu_kernel<<<(128*16*16 + 255)/256, 256>>>(d_conv2_out, d_relu2_out, 128*16*16);
        CUDA_CHECK(cudaGetLastError());
        
        // MaxPool2: (128, 16, 16) -> (128, 8, 8)
        dim3 blocks_pool2(128, 1, 1);
        dim3 threads_pool2(1, 8, 8);
        maxpool_kernel<<<blocks_pool2, threads_pool2>>>(d_relu2_out, d_pool2_out, 128, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        // === DECODER ===
        
        // Conv3: (128, 8, 8) -> (128, 8, 8)
        dim3 grid3(128, (8 + TILE_H - 1) / TILE_H, (8 + TILE_W - 1) / TILE_W);
        conv2d_kernel_v1<<<grid3, block>>>(d_pool2_out, d_conv3_w, d_conv3_b, d_conv3_out,
                                             128, 8, 8, 128, 8, 8, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU3
        relu_kernel<<<(128*8*8 + 255)/256, 256>>>(d_conv3_out, d_relu3_out, 128*8*8);
        CUDA_CHECK(cudaGetLastError());
        
        // Upsample1: (128, 8, 8) -> (128, 16, 16)
        dim3 blocks_up1(128, 1, 1);
        dim3 threads_up1(1, 16, 16);
        upsample_kernel<<<blocks_up1, threads_up1>>>(d_relu3_out, d_up1_out, 128, 8, 8);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv4: (128, 16, 16) -> (256, 16, 16)
        dim3 grid4(256, (16 + TILE_H - 1) / TILE_H, (16 + TILE_W - 1) / TILE_W);
        conv2d_kernel_v1<<<grid4, block>>>(d_up1_out, d_conv4_w, d_conv4_b, d_conv4_out,
                                            128, 16, 16, 256, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU4
        relu_kernel<<<(256*16*16 + 255)/256, 256>>>(d_conv4_out, d_relu4_out, 256*16*16);
        CUDA_CHECK(cudaGetLastError());
        
        // Upsample2: (256, 16, 16) -> (256, 32, 32)
        dim3 blocks_up2(256, 2, 2);
        dim3 threads_up2(1, 16, 16);
        upsample_kernel<<<blocks_up2, threads_up2>>>(d_relu4_out, d_up2_out, 256, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv5: (256, 32, 32) -> (3, 32, 32)
        dim3 grid5(3, (32 + TILE_H - 1) / TILE_H, (32 + TILE_W - 1) / TILE_W);
        conv2d_kernel_v1<<<grid5, block>>>(d_up2_out, d_conv5_w, d_conv5_b, d_conv5_out,
                                            256, 32, 32, 3, 32, 32, 3, 1);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void backward() {
        CUDA_CHECK(cudaMemset(d_grad_relu1, 0, 256 * 32 * 32 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_relu2, 0, 128 * 16 * 16 * sizeof(float)));

        
        // Conv5 backward
        dim3 blocks5_w(3, 256);
        conv2d_weight_grad_kernel<<<blocks5_w, 9>>>(d_grad_conv5, d_up2_out, d_conv5_w_grad,
                                                    256, 32, 32, 3, 32, 32, 3, 1);
        CUDA_CHECK(cudaGetLastError());

        conv2d_bias_grad_kernel<<<1, 3>>>(d_grad_conv5, d_conv5_b_grad, 3, 32, 32);
        CUDA_CHECK(cudaGetLastError());

        dim3 blocks5_in(256, 2, 2);
        dim3 threads5(1, 16, 16);
        conv2d_input_grad_kernel<<<blocks5_in, threads5>>>(d_grad_conv5, d_conv5_w, d_grad_up2,
                                                        256, 32, 32, 3, 32, 32, 3, 1);
        CUDA_CHECK(cudaGetLastError());
            
        // Upsample2 backward
        dim3 blocks_up2(256, 1, 1);
        dim3 threads_up2(1, 16, 16);
        upsample_backward_kernel<<<blocks_up2, threads_up2>>>(d_grad_up2, d_grad_relu4, 256, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU4 backward
        relu_backward_kernel<<<(256*16*16 + 255)/256, 256>>>(d_grad_relu4, d_conv4_out, d_grad_conv4, 256*16*16);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv4 backward
        dim3 blocks4_w(256, 128);
        conv2d_weight_grad_kernel<<<blocks4_w, 9>>>(d_grad_conv4, d_up1_out, d_conv4_w_grad,
                                                     128, 16, 16, 256, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        conv2d_bias_grad_kernel<<<1, 256>>>(d_grad_conv4, d_conv4_b_grad, 256, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        dim3 blocks4_in(128, 1, 1);
        conv2d_input_grad_kernel<<<blocks4_in, threads_up2>>>(d_grad_conv4, d_conv4_w, d_grad_up1,
                                                               128, 16, 16, 256, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // Upsample1 backward
        dim3 blocks_up1(128, 1, 1);
        dim3 threads_up1(1, 8, 8);
        upsample_backward_kernel<<<blocks_up1, threads_up1>>>(d_grad_up1, d_grad_relu3, 128, 8, 8);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU3 backward
        relu_backward_kernel<<<(128*8*8 + 255)/256, 256>>>(d_grad_relu3, d_conv3_out, d_grad_conv3, 128*8*8);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv3 backward
        dim3 blocks3_w(128, 128);
        conv2d_weight_grad_kernel<<<blocks3_w, 9>>>(d_grad_conv3, d_pool2_out, d_conv3_w_grad,
                                                     128, 8, 8, 128, 8, 8, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        conv2d_bias_grad_kernel<<<1, 128>>>(d_grad_conv3, d_conv3_b_grad, 128, 8, 8);
        CUDA_CHECK(cudaGetLastError());
        
        dim3 blocks3_in(128, 1, 1);
        conv2d_input_grad_kernel<<<blocks3_in, threads_up1>>>(d_grad_conv3, d_conv3_w, d_grad_pool2,
                                                               128, 8, 8, 128, 8, 8, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // MaxPool2 backward
        dim3 blocks_pool2(128, 1, 1);
        maxpool_backward_kernel<<<blocks_pool2, threads_up2>>>(d_grad_pool2, d_relu2_out, d_grad_relu2, 128, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU2 backward
        relu_backward_kernel<<<(128*16*16 + 255)/256, 256>>>(d_grad_relu2, d_conv2_out, d_grad_conv2, 128*16*16);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv2 backward
        dim3 blocks2_w(128, 256);
        conv2d_weight_grad_kernel<<<blocks2_w, 9>>>(d_grad_conv2, d_pool1_out, d_conv2_w_grad,
                                                     256, 16, 16, 128, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        conv2d_bias_grad_kernel<<<1, 128>>>(d_grad_conv2, d_conv2_b_grad, 128, 16, 16);
        CUDA_CHECK(cudaGetLastError());
        
        dim3 blocks2_in(256, 1, 1);
        conv2d_input_grad_kernel<<<blocks2_in, threads_up2>>>(d_grad_conv2, d_conv2_w, d_grad_pool1,
                                                               256, 16, 16, 128, 16, 16, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // MaxPool1 backward
        dim3 blocks_pool1(256, 2, 2);
        dim3 threads_pool1(1, 16, 16);
        maxpool_backward_kernel<<<blocks_pool1, threads_pool1>>>(d_grad_pool1, d_relu1_out, d_grad_relu1, 256, 32, 32);
        CUDA_CHECK(cudaGetLastError());
        
        // ReLU1 backward
        relu_backward_kernel<<<(256*32*32 + 255)/256, 256>>>(d_grad_relu1, d_conv1_out, d_grad_conv1, 256*32*32);
        CUDA_CHECK(cudaGetLastError());
        
        // Conv1 backward
        dim3 blocks1_w(256, 3);
        conv2d_weight_grad_kernel<<<blocks1_w, 9>>>(d_grad_conv1, d_input, d_conv1_w_grad,
                                                     3, 32, 32, 256, 32, 32, 3, 1);
        CUDA_CHECK(cudaGetLastError());
        
        conv2d_bias_grad_kernel<<<1, 256>>>(d_grad_conv1, d_conv1_b_grad, 256, 32, 32);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void update_weights(float learning_rate) {
        int threads = 256;
        
        sgd_update_kernel<<<(h_conv1_w.size() + threads - 1) / threads, threads>>>(
            d_conv1_w, d_conv1_w_grad, learning_rate, h_conv1_w.size());
        sgd_update_kernel<<<(h_conv1_b.size() + threads - 1) / threads, threads>>>(
            d_conv1_b, d_conv1_b_grad, learning_rate, h_conv1_b.size());
        
        sgd_update_kernel<<<(h_conv2_w.size() + threads - 1) / threads, threads>>>(
            d_conv2_w, d_conv2_w_grad, learning_rate, h_conv2_w.size());
        sgd_update_kernel<<<(h_conv2_b.size() + threads - 1) / threads, threads>>>(
            d_conv2_b, d_conv2_b_grad, learning_rate, h_conv2_b.size());
        
        sgd_update_kernel<<<(h_conv3_w.size() + threads - 1) / threads, threads>>>(
            d_conv3_w, d_conv3_w_grad, learning_rate, h_conv3_w.size());
        sgd_update_kernel<<<(h_conv3_b.size() + threads - 1) / threads, threads>>>(
            d_conv3_b, d_conv3_b_grad, learning_rate, h_conv3_b.size());
        
        sgd_update_kernel<<<(h_conv4_w.size() + threads - 1) / threads, threads>>>(
            d_conv4_w, d_conv4_w_grad, learning_rate, h_conv4_w.size());
        sgd_update_kernel<<<(h_conv4_b.size() + threads - 1) / threads, threads>>>(
            d_conv4_b, d_conv4_b_grad, learning_rate, h_conv4_b.size());
        
        sgd_update_kernel<<<(h_conv5_w.size() + threads - 1) / threads, threads>>>(
            d_conv5_w, d_conv5_w_grad, learning_rate, h_conv5_w.size());
        sgd_update_kernel<<<(h_conv5_b.size() + threads - 1) / threads, threads>>>(
            d_conv5_b, d_conv5_b_grad, learning_rate, h_conv5_b.size());
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    bool save_weights(const std::string& filepath) {
        // Copy weights from device to host
        CUDA_CHECK(cudaMemcpy(h_conv1_w.data(), d_conv1_w, h_conv1_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv1_b.data(), d_conv1_b, h_conv1_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv2_w.data(), d_conv2_w, h_conv2_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv2_b.data(), d_conv2_b, h_conv2_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv3_w.data(), d_conv3_w, h_conv3_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv3_b.data(), d_conv3_b, h_conv3_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv4_w.data(), d_conv4_w, h_conv4_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv4_b.data(), d_conv4_b, h_conv4_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv5_w.data(), d_conv5_w, h_conv5_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conv5_b.data(), d_conv5_b, h_conv5_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) return false;
        
        file.write((const char*)h_conv1_w.data(), h_conv1_w.size() * sizeof(float));
        file.write((const char*)h_conv1_b.data(), h_conv1_b.size() * sizeof(float));
        file.write((const char*)h_conv2_w.data(), h_conv2_w.size() * sizeof(float));
        file.write((const char*)h_conv2_b.data(), h_conv2_b.size() * sizeof(float));
        file.write((const char*)h_conv3_w.data(), h_conv3_w.size() * sizeof(float));
        file.write((const char*)h_conv3_b.data(), h_conv3_b.size() * sizeof(float));
        file.write((const char*)h_conv4_w.data(), h_conv4_w.size() * sizeof(float));
        file.write((const char*)h_conv4_b.data(), h_conv4_b.size() * sizeof(float));
        file.write((const char*)h_conv5_w.data(), h_conv5_w.size() * sizeof(float));
        file.write((const char*)h_conv5_b.data(), h_conv5_b.size() * sizeof(float));
        
        file.close();
        return true;
    }
    
    float get_loss() const { return last_loss; }
};

// ============================================================================
// C++ Wrapper Functions for main.cpp
// ============================================================================

extern "C" {
    AutoencoderGPU* create_autoencoder_cuda() {
        return new AutoencoderGPU();
    }
    
    void destroy_autoencoder_cuda(AutoencoderGPU* ae) {
        delete ae;
    }
    
    float train_step_cuda(AutoencoderGPU* ae, const float* input, float lr) {
        std::vector<float> img(input, input + 3*32*32);
        return ae->train_step(img, lr);
    }
    
    void forward_cuda(AutoencoderGPU* ae, const float* input, float* output) {
        // Not used in training, but can be implemented if needed
    }
    
    bool save_weights_cuda(AutoencoderGPU* ae, const std::string& path) {
        return ae->save_weights(path);
    }
}
