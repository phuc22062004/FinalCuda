// Optimized CUDA Autoencoder V1: Memory Optimizations
// Key optimizations:
// 1. In-place ReLU activations (no separate output buffer)
// 2. Reuse gradient buffers across layers
// 3. Reduced temporary storage  
// 4. Optimized memory allocation pattern

#include "config.h"
#include "autoencoder_cuda.h"
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
// CUDA KERNELS - FORWARD PASS
// ============================================================================

// Conv2D kernel: Same as basic version
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

// OPTIMIZATION 1: In-place ReLU activation kernel
__global__ void relu_inplace_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
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

// ReLU backward kernel (checks original input for gradient masking)
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* forward_output,  // After ReLU (already in-place modified)
    float* grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (forward_output[idx] > 0.0f) ? grad_output[idx] : 0.0f;
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
    
    // Find which position had max value
    float max_val = input[c * H * W + ih * W + iw];
    int max_ih = ih, max_iw = iw;
    
    float val;
    val = input[c * H * W + ih * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_ih = ih; max_iw = iw + 1; }
    
    val = input[c * H * W + (ih + 1) * W + iw];
    if (val > max_val) { max_val = val; max_ih = ih + 1; max_iw = iw; }
    
    val = input[c * H * W + (ih + 1) * W + (iw + 1)];
    if (val > max_val) { max_val = val; max_ih = ih + 1; max_iw = iw + 1; }
    
    float grad = grad_output[c * H_out * W_out + oh * W_out + ow];
    
    // Zero out all positions first
    grad_input[c * H * W + ih * W + iw] = 0.0f;
    grad_input[c * H * W + ih * W + (iw + 1)] = 0.0f;
    grad_input[c * H * W + (ih + 1) * W + iw] = 0.0f;
    grad_input[c * H * W + (ih + 1) * W + (iw + 1)] = 0.0f;
    
    // Only pass gradient to max position
    grad_input[c * H * W + max_ih * W + max_iw] = grad;
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
    
    // Sum gradients from 4 upsampled positions
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
    int k_idx = threadIdx.x;  // Flattened kernel index
    
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

// MSE Loss and gradient kernel
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
        if (isnan(g) || isinf(g)) {
            g = 0.0f;
        } else {
            if (g > 5.0f) g = 5.0f;
            if (g < -5.0f) g = -5.0f;
        }
        weight[idx] -= learning_rate * g;
    }
}


// ============================================================================
// AUTOENCODER CLASS IMPLEMENTATION - MEMORY OPTIMIZED
// ============================================================================

AutoencoderCUDA::AutoencoderCUDA() : last_loss(0.0f) {
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
    
    // Simple random initialization
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
    
    // OPTIMIZATION: Allocate device memory for activations (forward pass)
    // Note: relu outputs are IN-PLACE (same as conv outputs)
    CUDA_CHECK(cudaMalloc(&d_input, 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, 256 * 32 * 32 * sizeof(float)));  // Also used as relu1_out
    CUDA_CHECK(cudaMalloc(&d_pool1_out, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, 128 * 16 * 16 * sizeof(float)));  // Also used as relu2_out
    CUDA_CHECK(cudaMalloc(&d_pool2_out, 128 * 8 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv3_out, 128 * 8 * 8 * sizeof(float)));    // Also used as relu3_out
    CUDA_CHECK(cudaMalloc(&d_up1_out, 128 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv4_out, 256 * 16 * 16 * sizeof(float)));  // Also used as relu4_out
    CUDA_CHECK(cudaMalloc(&d_up2_out, 256 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv5_out, 3 * 32 * 32 * sizeof(float)));
    
    // Point relu outputs to conv outputs (in-place)
    d_relu1_out = d_conv1_out;
    d_relu2_out = d_conv2_out;
    d_relu3_out = d_conv3_out;
    d_relu4_out = d_conv4_out;
    
    // OPTIMIZATION 2: Reuse gradient buffers
    // Use larger buffers for multiple purposes to save memory
    CUDA_CHECK(cudaMalloc(&d_grad_conv5, 256 * 32 * 32 * sizeof(float)));  // Max size buffer
    CUDA_CHECK(cudaMalloc(&d_grad_up2, 256 * 32 * 32 * sizeof(float)));    // Another large buffer
    CUDA_CHECK(cudaMalloc(&d_grad_relu4, 256 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_conv4, 256 * 16 * 16 * sizeof(float)));
    
    // Reuse gradient buffers for earlier layers
    d_grad_up1 = d_grad_up2;       // Reuse (different sizes but same max)
    d_grad_relu3 = d_grad_relu4;   // Reuse
    d_grad_conv3 = d_grad_conv4;   // Reuse
    d_grad_pool2 = d_grad_relu4;   // Reuse
    d_grad_relu2 = d_grad_conv4;   // Reuse
    d_grad_conv2 = d_grad_relu4;   // Reuse
    d_grad_pool1 = d_grad_up2;     // Reuse large buffer
    d_grad_relu1 = d_grad_conv5;   // Reuse large buffer
    d_grad_conv1 = d_grad_up2;     // Reuse large buffer
    
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
}

AutoencoderCUDA::~AutoencoderCUDA() {
    // Free weights
    cudaFree(d_conv1_w); cudaFree(d_conv1_b);
    cudaFree(d_conv2_w); cudaFree(d_conv2_b);
    cudaFree(d_conv3_w); cudaFree(d_conv3_b);
    cudaFree(d_conv4_w); cudaFree(d_conv4_b);
    cudaFree(d_conv5_w); cudaFree(d_conv5_b);
    
    // Free weight gradients
    cudaFree(d_conv1_w_grad); cudaFree(d_conv1_b_grad);
    cudaFree(d_conv2_w_grad); cudaFree(d_conv2_b_grad);
    cudaFree(d_conv3_w_grad); cudaFree(d_conv3_b_grad);
    cudaFree(d_conv4_w_grad); cudaFree(d_conv4_b_grad);
    cudaFree(d_conv5_w_grad); cudaFree(d_conv5_b_grad);
    
    // Free activations
    cudaFree(d_input);
    cudaFree(d_conv1_out);  // relu1_out points to this
    cudaFree(d_pool1_out);
    cudaFree(d_conv2_out);  // relu2_out points to this
    cudaFree(d_pool2_out);
    cudaFree(d_conv3_out);  // relu3_out points to this
    cudaFree(d_up1_out);
    cudaFree(d_conv4_out);  // relu4_out points to this
    cudaFree(d_up2_out);
    cudaFree(d_conv5_out);
    
    // Free only the actually allocated gradient buffers
    cudaFree(d_grad_conv5);
    cudaFree(d_grad_up2);
    cudaFree(d_grad_relu4);
    cudaFree(d_grad_conv4);
    // Don't free reused pointers!
    
    cudaFree(d_loss);
}

void AutoencoderCUDA::forward() {
    dim3 block(16, 16);
    
    // Conv1: 3 -> 256, 32x32 (pad=1)
    dim3 grid1(256, (32 + 15) / 16, (32 + 15) / 16);
    conv2d_kernel<<<grid1, block>>>(d_input, d_conv1_w, d_conv1_b, d_conv1_out,
                                     3, 32, 32, 256, 32, 32, 3, 1);
    
    // ReLU1 - IN-PLACE
    int size1 = 256 * 32 * 32;
    relu_inplace_kernel<<<(size1 + 255) / 256, 256>>>(d_conv1_out, size1);
    
    // Pool1: 32x32 -> 16x16
    dim3 grid_pool1(256, (16 + 15) / 16, (16 + 15) / 16);
    maxpool_kernel<<<grid_pool1, block>>>(d_relu1_out, d_pool1_out, 256, 32, 32);
    
    // Conv2: 256 -> 128, 16x16 (pad=1)
    dim3 grid2(128, (16 + 15) / 16, (16 + 15) / 16);
    conv2d_kernel<<<grid2, block>>>(d_pool1_out, d_conv2_w, d_conv2_b, d_conv2_out,
                                     256, 16, 16, 128, 16, 16, 3, 1);
    
    // ReLU2 - IN-PLACE
    int size2 = 128 * 16 * 16;
    relu_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_conv2_out, size2);
    
    // Pool2: 16x16 -> 8x8 (bottleneck)
    dim3 grid_pool2(128, (8 + 15) / 16, (8 + 15) / 16);
    maxpool_kernel<<<grid_pool2, block>>>(d_relu2_out, d_pool2_out, 128, 16, 16);
    
    // Conv3: 128 -> 128, 8x8 (pad=1)
    dim3 grid3(128, (8 + 15) / 16, (8 + 15) / 16);
    conv2d_kernel<<<grid3, block>>>(d_pool2_out, d_conv3_w, d_conv3_b, d_conv3_out,
                                     128, 8, 8, 128, 8, 8, 3, 1);
    
    // ReLU3 - IN-PLACE
    int size3 = 128 * 8 * 8;
    relu_inplace_kernel<<<(size3 + 255) / 256, 256>>>(d_conv3_out, size3);
    
    // Upsample1: 8x8 -> 16x16
    dim3 grid_up1(128, (16 + 15) / 16, (16 + 15) / 16);
    upsample_kernel<<<grid_up1, block>>>(d_relu3_out, d_up1_out, 128, 8, 8);
    
    // Conv4: 128 -> 256, 16x16 (pad=1)
    dim3 grid4(256, (16 + 15) / 16, (16 + 15) / 16);
    conv2d_kernel<<<grid4, block>>>(d_up1_out, d_conv4_w, d_conv4_b, d_conv4_out,
                                     128, 16, 16, 256, 16, 16, 3, 1);
    
    // ReLU4 - IN-PLACE
    int size4 = 256 * 16 * 16;
    relu_inplace_kernel<<<(size4 + 255) / 256, 256>>>(d_conv4_out, size4);
    
    // Upsample2: 16x16 -> 32x32
    dim3 grid_up2(256, (32 + 15) / 16, (32 + 15) / 16);
    upsample_kernel<<<grid_up2, block>>>(d_relu4_out, d_up2_out, 256, 16, 16);
    
    // Conv5: 256 -> 3, 32x32 (pad=1)
    dim3 grid5(3, (32 + 15) / 16, (32 + 15) / 16);
    conv2d_kernel<<<grid5, block>>>(d_up2_out, d_conv5_w, d_conv5_b, d_conv5_out,
                                     256, 32, 32, 3, 32, 32, 3, 1);
    
    CUDA_CHECK(cudaGetLastError());
}

void AutoencoderCUDA::backward() {
    dim3 block(16, 16);
    int output_size = 3 * 32 * 32;
    
    // Compute loss and gradient
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    mse_loss_kernel<<<(output_size + 255) / 256, 256>>>(
        d_conv5_out, d_input, d_loss, d_grad_conv5, output_size);
    CUDA_CHECK(cudaMemcpy(&last_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Conv5 backward
    dim3 grid5_w(3, 256, 1);
    conv2d_weight_grad_kernel<<<grid5_w, 9>>>(
        d_grad_conv5, d_up2_out, d_conv5_w_grad, 256, 32, 32, 3, 32, 32, 3, 1);
    
    conv2d_bias_grad_kernel<<<1, 3>>>(d_grad_conv5, d_conv5_b_grad, 3, 32, 32);
    
    dim3 grid5_i(256, (32 + 15) / 16, (32 + 15) / 16);
    conv2d_input_grad_kernel<<<grid5_i, block>>>(
        d_grad_conv5, d_conv5_w, d_grad_up2, 256, 32, 32, 3, 32, 32, 3, 1);
    
    // Upsample2 backward
    dim3 grid_up2_b(256, (16 + 15) / 16, (16 + 15) / 16);
    upsample_backward_kernel<<<grid_up2_b, block>>>(d_grad_up2, d_grad_relu4, 256, 16, 16);
    
    // ReLU4 backward
    int size4 = 256 * 16 * 16;
    relu_backward_kernel<<<(size4 + 255) / 256, 256>>>(
        d_grad_relu4, d_conv4_out, d_grad_conv4, size4);
    
    // Conv4 backward
    dim3 grid4_w(256, 128, 1);
    conv2d_weight_grad_kernel<<<grid4_w, 9>>>(
        d_grad_conv4, d_up1_out, d_conv4_w_grad, 128, 16, 16, 256, 16, 16, 3, 1);
    
    conv2d_bias_grad_kernel<<<1, 256>>>(d_grad_conv4, d_conv4_b_grad, 256, 16, 16);
    
    dim3 grid4_i(128, (16 + 15) / 16, (16 + 15) / 16);
    conv2d_input_grad_kernel<<<grid4_i, block>>>(
        d_grad_conv4, d_conv4_w, d_grad_up1, 128, 16, 16, 256, 16, 16, 3, 1);
    
    // Upsample1 backward
    dim3 grid_up1_b(128, (8 + 15) / 16, (8 + 15) / 16);
    upsample_backward_kernel<<<grid_up1_b, block>>>(d_grad_up1, d_grad_relu3, 128, 8, 8);
    
    // ReLU3 backward
    int size3 = 128 * 8 * 8;
    relu_backward_kernel<<<(size3 + 255) / 256, 256>>>(
        d_grad_relu3, d_conv3_out, d_grad_conv3, size3);
    
    // Conv3 backward
    dim3 grid3_w(128, 128, 1);
    conv2d_weight_grad_kernel<<<grid3_w, 9>>>(
        d_grad_conv3, d_pool2_out, d_conv3_w_grad, 128, 8, 8, 128, 8, 8, 3, 1);
    
    conv2d_bias_grad_kernel<<<1, 128>>>(d_grad_conv3, d_conv3_b_grad, 128, 8, 8);
    
    dim3 grid3_i(128, (8 + 15) / 16, (8 + 15) / 16);
    conv2d_input_grad_kernel<<<grid3_i, block>>>(
        d_grad_conv3, d_conv3_w, d_grad_pool2, 128, 8, 8, 128, 8, 8, 3, 1);
    
    // Pool2 backward
    dim3 grid_pool2_b(128, (16 + 15) / 16, (16 + 15) / 16);
    maxpool_backward_kernel<<<grid_pool2_b, block>>>(d_grad_pool2, d_conv2_out, d_grad_relu2, 128, 16, 16);
    
    // ReLU2 backward
    int size2 = 128 * 16 * 16;
    relu_backward_kernel<<<(size2 + 255) / 256, 256>>>(
        d_grad_relu2, d_conv2_out, d_grad_conv2, size2);
    
    // Conv2 backward
    dim3 grid2_w(128, 256, 1);
    conv2d_weight_grad_kernel<<<grid2_w, 9>>>(
        d_grad_conv2, d_pool1_out, d_conv2_w_grad, 256, 16, 16, 128, 16, 16, 3, 1);
    
    conv2d_bias_grad_kernel<<<1, 128>>>(d_grad_conv2, d_conv2_b_grad, 128, 16, 16);
    
    dim3 grid2_i(256, (16 + 15) / 16, (16 + 15) / 16);
    conv2d_input_grad_kernel<<<grid2_i, block>>>(
        d_grad_conv2, d_conv2_w, d_grad_pool1, 256, 16, 16, 128, 16, 16, 3, 1);
    
    // Pool1 backward
    dim3 grid_pool1_b(256, (32 + 15) / 16, (32 + 15) / 16);
    maxpool_backward_kernel<<<grid_pool1_b, block>>>(d_grad_pool1, d_conv1_out, d_grad_relu1, 256, 32, 32);
    
    // ReLU1 backward
    int size1 = 256 * 32 * 32;
    relu_backward_kernel<<<(size1 + 255) / 256, 256>>>(
        d_grad_relu1, d_conv1_out, d_grad_conv1, size1);
    
    // Conv1 backward
    dim3 grid1_w(256, 3, 1);
    conv2d_weight_grad_kernel<<<grid1_w, 9>>>(
        d_grad_conv1, d_input, d_conv1_w_grad, 3, 32, 32, 256, 32, 32, 3, 1);
    
    conv2d_bias_grad_kernel<<<1, 256>>>(d_grad_conv1, d_conv1_b_grad, 256, 32, 32);
    
    CUDA_CHECK(cudaGetLastError());
}

void AutoencoderCUDA::update_weights(float learning_rate) {
    // Update all weights with gradient descent
    sgd_update_kernel<<<(h_conv1_w.size() + 255) / 256, 256>>>(
        d_conv1_w, d_conv1_w_grad, learning_rate, h_conv1_w.size());
    sgd_update_kernel<<<(h_conv1_b.size() + 255) / 256, 256>>>(
        d_conv1_b, d_conv1_b_grad, learning_rate, h_conv1_b.size());
    
    sgd_update_kernel<<<(h_conv2_w.size() + 255) / 256, 256>>>(
        d_conv2_w, d_conv2_w_grad, learning_rate, h_conv2_w.size());
    sgd_update_kernel<<<(h_conv2_b.size() + 255) / 256, 256>>>(
        d_conv2_b, d_conv2_b_grad, learning_rate, h_conv2_b.size());
    
    sgd_update_kernel<<<(h_conv3_w.size() + 255) / 256, 256>>>(
        d_conv3_w, d_conv3_w_grad, learning_rate, h_conv3_w.size());
    sgd_update_kernel<<<(h_conv3_b.size() + 255) / 256, 256>>>(
        d_conv3_b, d_conv3_b_grad, learning_rate, h_conv3_b.size());
    
    sgd_update_kernel<<<(h_conv4_w.size() + 255) / 256, 256>>>(
        d_conv4_w, d_conv4_w_grad, learning_rate, h_conv4_w.size());
    sgd_update_kernel<<<(h_conv4_b.size() + 255) / 256, 256>>>(
        d_conv4_b, d_conv4_b_grad, learning_rate, h_conv4_b.size());
    
    sgd_update_kernel<<<(h_conv5_w.size() + 255) / 256, 256>>>(
        d_conv5_w, d_conv5_w_grad, learning_rate, h_conv5_w.size());
    sgd_update_kernel<<<(h_conv5_b.size() + 255) / 256, 256>>>(
        d_conv5_b, d_conv5_b_grad, learning_rate, h_conv5_b.size());
    
    CUDA_CHECK(cudaGetLastError());
}

float AutoencoderCUDA::train_step(const float* input_chw, float learning_rate) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass
    forward();
    
    // Backward pass
    backward();
    
    // Update weights
    update_weights(learning_rate);
    
    return last_loss;
}

bool AutoencoderCUDA::save_weights(const std::string& filepath) const {
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
    
    // Save to file
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(h_conv1_w.data()), h_conv1_w.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv1_b.data()), h_conv1_b.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv2_w.data()), h_conv2_w.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv2_b.data()), h_conv2_b.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv3_w.data()), h_conv3_w.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv3_b.data()), h_conv3_b.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv4_w.data()), h_conv4_w.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv4_b.data()), h_conv4_b.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv5_w.data()), h_conv5_w.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_conv5_b.data()), h_conv5_b.size() * sizeof(float));
    
    file.close();
    return true;
}

bool AutoencoderCUDA::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(h_conv1_w.data()), h_conv1_w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv1_b.data()), h_conv1_b.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv2_w.data()), h_conv2_w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv2_b.data()), h_conv2_b.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv3_w.data()), h_conv3_w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv3_b.data()), h_conv3_b.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv4_w.data()), h_conv4_w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv4_b.data()), h_conv4_b.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv5_w.data()), h_conv5_w.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_conv5_b.data()), h_conv5_b.size() * sizeof(float));
    
    file.close();
    
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
    
    return true;
}

float AutoencoderCUDA::get_loss() const {
    return last_loss;
}

// Extract features from encoder (bottleneck layer: 128*8*8 = 8192 features)
void AutoencoderCUDA::extract_features(const float* input_chw, float* output_features) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input_chw, 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run encoder only (up to pool2_out which is the bottleneck)
    dim3 block(16, 16);
    
    // Conv1
    dim3 grid1(256, (32 + 15) / 16, (32 + 15) / 16);
    conv2d_kernel<<<grid1, block>>>(d_input, d_conv1_w, d_conv1_b, d_conv1_out,
                                     3, 32, 32, 256, 32, 32, 3, 1);
    
    // ReLU1 - IN-PLACE
    int size1 = 256 * 32 * 32;
    relu_inplace_kernel<<<(size1 + 255) / 256, 256>>>(d_conv1_out, size1);
    
    // Pool1
    dim3 grid_pool1(256, (16 + 15) / 16, (16 + 15) / 16);
    maxpool_kernel<<<grid_pool1, block>>>(d_relu1_out, d_pool1_out, 256, 32, 32);
    
    // Conv2
    dim3 grid2(128, (16 + 15) / 16, (16 + 15) / 16);
    conv2d_kernel<<<grid2, block>>>(d_pool1_out, d_conv2_w, d_conv2_b, d_conv2_out,
                                     256, 16, 16, 128, 16, 16, 3, 1);
    
    // ReLU2 - IN-PLACE
    int size2 = 128 * 16 * 16;
    relu_inplace_kernel<<<(size2 + 255) / 256, 256>>>(d_conv2_out, size2);
    
    // Pool2 - BOTTLENECK (128*8*8 = 8192 features)
    dim3 grid_pool2(128, (8 + 15) / 16, (8 + 15) / 16);
    maxpool_kernel<<<grid_pool2, block>>>(d_relu2_out, d_pool2_out, 128, 16, 16);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Copy features to host
    CUDA_CHECK(cudaMemcpy(output_features, d_pool2_out, 128 * 8 * 8 * sizeof(float), cudaMemcpyDeviceToHost));
}
