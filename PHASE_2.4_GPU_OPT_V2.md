# Phase 2.4: GPU Optimized Implementation - Version 2

## Optimization Focus: Kernel Fusion & Vectorized Operations

### Objectives

**Những Gì Chúng Ta Muốn Đạt Được:**
- **Kernel Fusion**: Merge Conv + Bias + ReLU thành 1 kernel duy nhất
- **Vectorized Memory Operations**: Sử dụng float4 cho SGD updates và loss computation
- **Reduce kernel launch overhead**: Giảm số lượng kernel calls
- **Target: 1.5-2× speedup** so với GPU Opt V1

**Tại Sao Optimizations Này:**
- Phase 2.3 profiling cho thấy:
  - Conv kernels chiếm 98% kernel time
  - cudaLaunchKernel overhead: 24.7% API time
  - Nhiều small kernels (ReLU, bias) có overhead cao
- Kernel fusion giảm:
  - Intermediate memory writes/reads
  - Kernel launch overhead
  - Global memory bandwidth usage

---

## Chi Tiết Triển Khai (Implementation Details)

### Optimization Techniques Applied

#### 1. Kernel Fusion: Conv + Bias + ReLU (Ưu tiên #1)

**Vấn đề:**
```cpp
// OLD (V1): 3 separate operations
conv2d_tiled_kernel<<<...>>>(input, weight, conv_out, ...);  // Write conv_out
// ReLU in-place: read conv_out, write back
relu_inplace_kernel<<<...>>>(conv_out, size);                // Read + Write
```

**Giải pháp:**
```cpp
// NEW (V2): Single fused kernel
__global__ void conv2d_bias_relu_fused_kernel(
    const float* input, const float* weight, const float* bias, 
    float* output, ...)
{
    int oc = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.x + threadIdx.x;
    
    // Compute convolution
    float sum = 0.0f;
    #pragma unroll
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += input[...] * weight[...];
            }
        }
    }
    
    // Add bias and apply ReLU in one step - NO intermediate writes!
    sum += bias[oc];
    output[...] = (sum > 0.0f) ? sum : 0.0f;  // Fused!
}
```

**Lợi ích:**
- Eliminate intermediate buffer writes (conv_out)
- Single kernel launch thay vì 2
- Giảm global memory accesses: 2 writes → 1 write

**Trade-off:**
- ⚠️ V2 implementation không dùng shared memory tiling (từ V1)
- ⚠️ Fused kernel đọc từ global memory mỗi lần (no caching)

#### 2. Vectorized SGD Update (float4)

**Vấn đề:**
```cpp
// OLD (V1): Scalar processing - 1 float per thread
__global__ void sgd_update_kernel(float* weight, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weight[idx] -= lr * grad[idx];  // 1 float
    }
}
```

**Giải pháp:**
```cpp
// NEW (V2): Process 4 floats per thread
__global__ void sgd_update_vec4(float* weight, const float* grad, float lr, int size) {
    int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i4 + 3 < size) {
        float4* W = (float4*)(weight + i4);
        const float4* G = (const float4*)(grad + i4);
        
        float4 w = *W;
        float4 g = *G;
        
        // Update 4 elements at once
        w.x -= lr * g.x;
        w.y -= lr * g.y;
        w.z -= lr * g.z;
        w.w -= lr * g.w;
        
        *W = w;  // Single 128-bit write
    }
}
```

**Lợi ích:**
- 4× more elements per thread
- Better memory bandwidth utilization (128-bit transactions)
- Fewer kernel launches (4× fewer threads needed)

#### 3. Vectorized Loss Computation (float4)

**Tương tự SGD, áp dụng cho MSE loss:**
```cpp
__global__ void mse_loss_vec4(
    const float* pred, const float* target,
    float* loss, float* grad, int size)
{
    int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i4 + 3 < size) {
        float4 p = *((const float4*)(pred + i4));
        float4 t = *((const float4*)(target + i4));
        
        // Compute 4 losses at once
        float4 diff;
        diff.x = p.x - t.x;
        diff.y = p.y - t.y;
        diff.z = p.z - t.z;
        diff.w = p.w - t.w;
        
        // Write gradients
        *((float4*)(grad + i4)) = diff;
        
        // Accumulate loss
        float local_loss = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z + diff.w*diff.w;
        atomicAdd(loss, local_loss);
    }
}
```

#### 4. Strategic Loop Unrolling

**Compiler hints for 3×3 convolutions:**
```cpp
#pragma unroll
for (int ic = 0; ic < C_in; ic++) {
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            sum += input[...] * weight[...];
        }
    }
}
```

**Lợi ích:**
- Compiler unrolls inner loops (3×3 = 9 iterations)
- Reduces loop overhead
- Better instruction pipelining

---

## Cách Chạy GPU Optimized V2

### Build Code
```bash
cd /home/senyamiku/LTSS/FinalCuda
bash scripts/build_cuda.sh
```

### Training
```bash
# Test với 1000 ảnh
./build_cuda/autoencoder_cuda_opt_v2 cifar-10-binary/cifar-10-batches-bin weights/opt_v2.bin 3 32 0.001 1000

# Full training (50,000 ảnh)
./build_cuda/autoencoder_cuda_opt_v2 cifar-10-binary/cifar-10-batches-bin weights/opt_v2.bin 3 64 0.001 50000
```

### Profiling
```bash
# Profile với 1000 ảnh
nsys profile --stats=true -o report_opt2 ./build_cuda/autoencoder_cuda_opt_v2 cifar-10-binary/cifar-10-batches-bin weights/opt_v2.bin 3 64 0.001 1000

# Xem report
nsys stats report_opt2.nsys-rep
```

---

## Kết Quả (Results)

### Cấu Hình

- **Hardware**: NVIDIA A100-SXM4-40GB (40 GB VRAM)
- **Dataset**: CIFAR-10 (1,000 test / 50,000 full training)
- **Hyperparameters**:
  - Epochs: 3
  - Batch size: 32-64
  - Learning rate: 0.001
  - Optimizer: SGD with gradient clipping (vectorized)

### Training Performance

#### Test Run (1000 images, 3 epochs, batch=32)

```
Epoch 1/3 - Average Loss: 0.169965 - Time: 2273ms - Throughput: 439.947 imgs/sec
Epoch 2/3 - Average Loss: 0.0935318 - Time: 2223ms - Throughput: 449.843 imgs/sec
Epoch 3/3 - Average Loss: 0.0999917 - Time: 2224ms - Throughput: 449.64 imgs/sec

Total training time: 8254ms (8.25s)
Average throughput: 363.46 imgs/sec
```

**So sánh với các phiên bản trước:**
- GPU Basic (Phase 2.2, 1K, 3 epochs): 9530ms → 3.18s/epoch
- GPU Opt V1 (Phase 2.3, 1K, 3 epochs): 6184ms → 2.06s/epoch
- GPU Opt V2 (1K, 3 epochs): 8254ms → 2.75s/epoch

**Speedup vs Basic: 1.15× (V2 nhanh hơn Basic 15%)**
**Slowdown vs V1: 0.75× (V2 chậm hơn V1 33%!)**

#### Full Training (50,000 images, 3 epochs, batch=64)

```
Epoch 1/3 - Average Loss: 0.0527183 - Time: 111135ms - Throughput: 449.903 imgs/sec
Epoch 2/3 - Average Loss: 0.0307476 - Time: 111121ms - Throughput: 449.96 imgs/sec
Epoch 3/3 - Average Loss: 0.0252823 - Time: 111109ms - Throughput: 450.009 imgs/sec

Total training time: 334923ms (335s = 5.58 minutes)
Average throughput: 447.864 imgs/sec
```

**So sánh với các phiên bản trước (50K, batch=64):**
- GPU Basic (Phase 2.2): ~317s, ~472 imgs/sec
- GPU Opt V1 (Phase 2.3): 231s, 648.699 imgs/sec
- GPU Opt V2 (Phase 2.4): 335s, 447.864 imgs/sec

**Speedup vs Basic: 0.95× (V2 chậm hơn Basic 5%!) ⚠️**
**Slowdown vs V1: 0.69× (V2 chậm hơn V1 45%!)**

### Bảng So Sánh Performance

| Metric | GPU Basic (2.2) | GPU Opt V1 (2.3) | GPU Opt V2 (2.4) | vs Basic | vs V1 |
|--------|-----------------|------------------|------------------|----------|-------|
| **Time/epoch (1K, batch=32)** | 3.18s | 2.06s | 2.75s | **1.15× faster** | **0.75× (slower)** |
| **Total time (1K, 3 epochs)** | 9.53s | 6.18s | 8.25s | **1.15× faster** | **0.75× (slower)** |
| **Time/epoch (50K, batch=64)** |  ~133s | ~76s | ~111s | **1.19× faster** | **0.68× (slower)** |
| **Throughput (batch=32)** | 315 imgs/sec | 485 imgs/sec | 363 imgs/sec | **1.15× faster** | **0.75× (slower)** |
| **Throughput (batch=64, 50K)** | ~472 imgs/sec | 649 imgs/sec | 448 imgs/sec | **~0.95× (similar)** | **0.69× (slower)** |
| **Memory usage (batch=64)** | 441 MiB | 617 MiB | 437 MiB | **Similar** | **-29%** |

**❌ UNEXPECTED RESULT**: V2 chậm hơn V1 thay vì nhanh hơn!

**Cumulative Speedup vs CPU:**
- CPU Baseline: 750s/epoch
- GPU Basic: 3.18s/epoch → **236× speedup**
- GPU Opt V1: 2.06s/epoch → **364× speedup**
- GPU Opt V2: 2.75s/epoch → **273× speedup**

**Ranking: V1 (364×) > V2 (273×) > Basic (236×) > CPU (1×)**

### GPU Memory Usage

**Từ nvidia-smi (1000 images training):**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15    CUDA: 12.4 |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Memory-Usage         | GPU-Util  Power      |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB    | 437MiB / 40960MiB    | 99%      127W/400W   |
+-----------------------------------------------------------------------------+
```

**Phân tích:**
- **Memory Used**: 437 MiB (vs 617 MiB ở V1)
- **Giảm 29%** do:
  - Không có batch processing buffers (chạy 1000 images, batch nhỏ)
  - Kernel fusion eliminates intermediate buffers
  - ⚠️ Nhưng không có shared memory tiles
- **GPU Utilization**: 99% (excellent)
- **Power**: 127W/400W (32% - lower than V1's 149W)

**Memory efficiency tốt hơn nhưng throughput thấp hơn!**

### Profiling Analysis (nsys, 1000 images)

#### Kernel Time Breakdown

```
Kernel Name                          | Time(%)  | Total Time | Avg Time  | Calls
-------------------------------------|----------|------------|-----------|-------
conv2d_weight_grad_kernel            | 25.6%    | 1.67s      | 111.6 μs  | 15,000
conv2d_bias_relu_fused_kernel        | 23.4%    | 1.53s      | 127.3 μs  | 12,000 ← FUSED!
conv2d_input_grad_kernel             | 20.9%    | 1.36s      | 113.6 μs  | 12,000
conv2d_kernel (no ReLU, for Conv5)   | 15.0%    | 0.98s      | 325.6 μs  | 3,000
conv2d_bias_grad_kernel              | 13.5%    | 0.88s      | 58.9 μs   | 15,000
relu_backward_kernel                 | 0.5%     | 29.8ms     | 2.5 μs    | 12,000
maxpool_backward_kernel              | 0.3%     | 16.7ms     | 2.8 μs    | 6,000
mse_loss_vec4                        | 0.2%     | 14.9ms     | 4.9 μs    | 3,048 ← VECTORIZED!
upsample_kernel                      | 0.2%     | 14.9ms     | 2.5 μs    | 6,000
maxpool_kernel                       | 0.2%     | 14.1ms     | 2.4 μs    | 6,000
upsample_backward_kernel             | 0.2%     | 14.0ms     | 2.3 μs    | 6,000
sgd_update_vec4                      | 0.0%     | 1.2ms      | 2.3 μs    | 528 ← VECTORIZED!

Total kernel time: 6.54s (out of 8.31s total)
```

**So sánh với Basic và V1:**

| Kernel Category | Basic (2.2) | V1 (2.3) | V2 (2.4) | vs Basic | vs V1 |
|----------------|-------------|----------|----------|----------|-------|
| **Conv forward** | ~4.5s (93%) | 0.44s (9.8%) | 1.53s (23.4%) | **2.9× faster** ✅ | **3.5× slower** 🔴 |
| **Conv backward** | N/A | 3.94s (87.6%) | 4.89s (74.8%) | N/A | 1.24× slower |
| **Loss (vec4)** | ~0.3s | N/A | 14.9ms (0.2%) | **20× faster** ✅ | New |
| **SGD (vec4)** | ~0.1s | N/A | 1.2ms (0.0%) | **83× faster** ✅ | New |
| **Total kernel** | ~7.3s | 4.50s | 6.54s | **1.12× faster** | **1.45× slower** 🔴 |

**❗ CRITICAL FINDING**: 
- **Conv forward chậm hơn 3.5×** so với V1's tiled version!
- Fused kernel không có shared memory → read all data from global memory
- V1's shared memory tiling >> kernel fusion benefit

#### CUDA API Time

```
Operation                | Time(%)  | Total Time | Calls
-------------------------|----------|------------|---------
cudaMemcpy               | 92.3%    | 6.30s      | 15,116
cudaLaunchKernel         | 6.0%     | 410ms      | 96,816
cudaMalloc               | 1.5%     | 99.9ms     | 37
cudaMemset               | 0.2%     | 12.9ms     | 3,336
```

**So sánh với V1:**

| API | V1 (2.3) | V2 (2.4) | Change |
|-----|----------|----------|--------|
| **cudaMemcpy** | 4.33s (90.9%) | 6.30s (92.3%) | 1.45× slower |
| **cudaLaunchKernel** | 313ms (6.6%) | 410ms (6.0%) | 1.31× slower |
| **Total API** | 4.76s | 6.82s | **1.43× slower** |

**Observations:**
- cudaMemcpy vẫn là bottleneck (92.3%)
- Launch overhead giảm nhẹ (6.6% → 6.0%) nhờ kernel fusion
- Nhưng total time tăng do kernels chậm hơn

#### Memory Operations

```
Operation          | Time(%)  | Total Time | Count  | Avg Size
-------------------|----------|------------|--------|----------
memcpy DtoD        | 64.5%    | 23.7ms     | 12,000 | 12 KB
memcpy DtoH        | 16.2%    | 5.9ms      | 3,058  | 1 KB
memset             | 9.9%     | 3.7ms      | 3,336  | 42 KB
memcpy HtoD        | 9.4%     | 3.4ms      | 58     | 687 KB
```

---

## Analysis

### Performance Position: Better Than Basic, Worse Than V1

**V2 sits in the middle:**
- ✅ **Nhanh hơn Basic 15%** nhờ kernel fusion + vectorization
- ❌ **Chậm hơn V1 33%** do mất shared memory optimization
- Result: V2 là improvement over Basic, nhưng regression from V1

**So sánh chi tiết:**
```
Basic (9.53s) ←[15% faster]→ V2 (8.25s) ←[33% slower]→ V1 (6.18s)
    ↑                            ↑                           ↑
  Naive                   Kernel Fusion              Shared Memory
```

### Why Did V2 Perform WORSE Than V1?

#### Root Cause: Lost Shared Memory Optimization

**V1 Implementation:**
```cpp
// V1: Shared memory tiling
__shared__ float s_input[BATCH_SIZE][SHARED_TILE_HEIGHT][SHARED_TILE_WIDTH];

// Load tile once, reuse nhiều lần
for (batch in input channels) {
    load_tile_to_shared_memory();
    __syncthreads();
    
    // All threads compute from shared memory (FAST!)
    compute_convolution_from_shared();
}
```

**V2 Implementation:**
```cpp
// V2: Fused but no shared memory
__global__ void conv2d_bias_relu_fused_kernel(...) {
    // Read directly from global memory (SLOW!)
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += input[...] * weight[...];  // Global memory read!
            }
        }
    }
}
```

**Performance Impact:**
- V1 conv forward: 0.44s (with shared memory)
- V2 conv forward: 1.53s (without shared memory)
- **3.5× regression!**

#### Kernel Fusion Benefits (Minor)

**Pros:**
- Eliminates 1 intermediate write per conv layer
- Reduces kernel launches (12K → fewer)
- Slight reduction in memory usage

**Cons:**
- Lost shared memory reuse (major)
- Global memory bandwidth becomes bottleneck
- Arithmetic intensity decreased

**Net result:** Small fusion benefit << Large shared memory loss

#### Vectorization Benefits (Minimal Impact)

**SGD vectorization:**
- Before: scalar updates
- After: float4 (4× elements per thread)
- Time saved: ~1-2ms (negligible in 8s total)

**Loss vectorization:**
- Similar minimal impact
- MSE computation is tiny fraction of total time

**Conclusion:** Vectorization helped but impact too small

### What Went Wrong?

**Design Decision Error:**
1. ✅ Kernel fusion is good idea
2. ❌ But shouldn't replace shared memory tiling
3. ❌ Should have: **Fused kernel WITH shared memory**

**Correct approach (not implemented):**
```cpp
// IDEAL: Conv + Bias + ReLU với shared memory tiling
__global__ void conv2d_bias_relu_tiled_fused(...) {
    __shared__ float s_input[...];  // Keep tiling!
    
    // Load to shared memory
    // Compute from shared memory
    // Apply bias + ReLU at end
    
    // Best of both worlds!
}
```

### Performance Comparison Summary

| Version | Key Optimization | Time (1K, 3 epochs) | vs Basic | vs V1 |
|---------|-----------------|---------------------|----------|-------|
| **Basic (Phase 2.2)** | Naive GPU parallelization | 9.53s | Baseline | -35% |
| **V2 (Phase 2.4)** | Kernel fusion + vectorization | 8.25s | **+15%** ✅ | **-25%** ❌ |
| **V1 (Phase 2.3)** | Shared memory tiling | 6.18s | **+54%** ⭐ | Baseline |

**Lesson learned:** Không phải mọi optimization đều tốt hơn. Trade-offs matter!

---

## Key Takeaways

### Lessons Learned

1. **Shared Memory >> Kernel Fusion**
   - Shared memory tiling: 3.5× speedup
   - Kernel fusion alone: Not enough to compensate
   - **Priority matters**: Optimize biggest bottleneck first

2. **Optimization Trade-offs Are Real**
   - V2 giảm memory usage (29%)
   - Nhưng tăng compute time (33%)
   - Trade-off không xứng đáng

3. **Incremental Optimization Is Safer**
   - V1 → V2 should have kept V1's optimizations
   - Add kernel fusion ON TOP of shared memory
   - Don't replace working optimizations

4. **Vectorization Impact Is Small**
   - float4 SGD: ~1ms saved (negligible)
   - Good for polish, not main optimization
   - Focus on kernel compute first

5. **Profiling Is Essential**
   - Without profiling, V2 looks good (fusion + vectorization)
   - With profiling: clearly worse than V1
   - **Always measure, don't assume!**

### What Should Have Been Done

**Correct V2 Implementation:**
```cpp
// IDEAL: Combine ALL optimizations
__global__ void conv2d_bias_relu_tiled_fused_kernel(...) {
    // ✅ Shared memory tiling (from V1)
    __shared__ float s_input[...];
    
    // ✅ Kernel fusion (V2 idea)
    float sum = compute_from_shared_memory();
    sum += bias[oc];
    output[...] = fmaxf(0.0f, sum);  // Fused bias + ReLU
    
    // ✅ Memory coalescing (from V1)
    // ✅ Loop unrolling
}
```

**Expected performance:**
- Keep V1's conv forward speed: 0.44s
- Add fusion benefits: ~5-10% improvement
- **Target: 5.5-6s total (vs 6.18s V1, vs 8.25s actual V2)**

### Recommendations for Real-World Projects

1. **Always Keep Working Optimizations**
   - Build incrementally
   - Don't remove what works
   - A + B > A or B alone

2. **Profile Before and After**
   - Measure every optimization
   - Compare against baseline
   - Reject if regression > 5%

3. **Understand Root Causes**
   - Know WHY optimization works
   - Shared memory: data reuse
   - Kernel fusion: eliminate writes
   - Choose based on bottleneck

4. **Document Trade-offs**
   - Memory vs speed
   - Complexity vs maintainability
   - Make conscious decisions

### Final Verdict

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Speedup vs Basic | ✅ Better | ✅ +15% | ✅ Success |
| Speedup vs V1 | 1.5-2× | 0.75× | ❌ Failed |
| Memory efficiency | Improved | ✅ -29% vs V1 | ✅ Success |
| Code complexity | Similar | ✅ Similar | ✅ OK |
| Overall | Better than V1 | **Middle: Basic < V2 < V1** | ⚠️ **PARTIAL SUCCESS** |

**Conclusion:** 
- ✅ Phase 2.4 **is an improvement over Basic** (15% faster)
- ❌ But **regression from V1** (25% slower)
- V2 demonstrates kernel fusion works, but not enough to beat shared memory
- Lesson: Đôi khi, ít tối ưu hơn lại tốt hơn nếu trade-offs sai

---

## What's Next?

**If Continuing Optimization:**

1. **V3 (Hypothetical): Fused + Tiled**
   - Combine V1's shared memory + V2's fusion
   - Expected: 5-6s (vs 6.18s V1)
   - True improvement over both

2. **CUDA Streams (Real next step)**
   - Async execution
   - Overlap compute + memory transfer
   - Target: 2-3× speedup by hiding latency

3. **Better Memory Management**
   - Pinned memory for faster transfers
   - Batch prefetching
   - Double buffering

---

## File Structure Reference

```
src/
├── main_cuda.cpp
├── cuda/
│   ├── autoencoder_basic.cu         # Phase 2.2: Naive GPU
│   ├── autoencoder_opt_v1.cu        # Phase 2.3: Shared Memory + Coalescing ✅ BEST
│   └── autoencoder_opt_v2.cu        # Phase 2.4: Kernel Fusion (regression) ❌
include/
└── autoencoder_cuda.h
scripts/
└── build_cuda.sh
```

**Key Changes in autoencoder_opt_v2.cu:**
- `conv2d_bias_relu_fused_kernel()`: Fused Conv+Bias+ReLU (lines 51-89)
- `sgd_update_vec4()`: Vectorized SGD with float4 (lines 476-512)
- `mse_loss_vec4()`: Vectorized loss computation (lines 539-586)
- ❌ **Missing**: Shared memory tiling from V1
- ❌ **Result**: 33% slower than V1

**Total lines**: ~1,200 (vs ~1,300 in V1)

**Performance Ranking (1000 images, 3 epochs):**
1. 🥇 **GPU Opt V1 (2.3)**: 6.18s (364× vs CPU) - BEST ⭐
2. 🥈 **GPU Opt V2 (2.4)**: 8.25s (273× vs CPU) - MIDDLE GROUND
3. 🥉 **GPU Basic (2.2)**: 9.53s (236× vs CPU) - BASELINE GPU
4. ⏱️ **CPU Baseline**: 2250s (1×) - SLOWEST

**Gap analysis:**
- V1 → V2: +33% time (regression due to no shared memory)
- V2 → Basic: +15% time (improvement from kernel fusion)
- Basic → CPU: +23,500% time (massive GPU speedup)
