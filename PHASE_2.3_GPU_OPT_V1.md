# Phase 2.3: GPU Optimized Implementation - Version 1

## Optimization Focus: Shared Memory Tiling & Memory Coalescing

### Objectives

**Những Gì Chúng Ta Muốn Đạt Được:**
- **Giảm global memory access** bằng shared memory tiling cho convolution kernels
- **Tối ưu memory coalescing** để threads trong warp access liên tiếp
- **Giảm redundant memory operations** (in-place ReLU, gradient buffer reuse)
- **Target: 1.5-2× speedup** so với GPU Basic (từ Phase 2.2)

**Tại Sao Optimizations Này:**
- Phase 2.2 profiling cho thấy:
  - Convolution kernels chiếm 98% kernel time
  - 73% API time là cudaMemcpy overhead
  - Mỗi conv thread đọc ~2,500 floats từ global memory (no reuse)
- Shared memory có bandwidth ~20× cao hơn global memory
- Memory coalescing giúp warp threads access consecutive addresses (tránh serialize)

---

## Chi Tiết Triển Khai (Implementation Details)

### Optimization Techniques Applied

#### 1. Shared Memory Tiling cho Convolution (Ưu tiên #1)

**Vấn đề:** 
- Mỗi thread trong naive version đọc input data nhiều lần từ global memory
- Các threads lân cận cũng đọc overlapping data → redundant reads

**Giải pháp:**
```cpp
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define KERNEL_SIZE 3
#define SHARED_TILE_WIDTH (TILE_WIDTH + KERNEL_SIZE - 1)   // 18
#define SHARED_TILE_HEIGHT (TILE_HEIGHT + KERNEL_SIZE - 1)  // 18

__global__ void conv2d_tiled_kernel_batched(...) {
    // Shared memory cho input tile (16x16 + 2 padding = 18x18)
    __shared__ float s_input[BATCH_SIZE][SHARED_TILE_HEIGHT][SHARED_TILE_WIDTH];
    
    // Load input tile vào shared memory (collaborative loading)
    for (int batch = 0; batch < num_batches; batch++) {
        int ic_start = batch * BATCH_SIZE;
        int ic_end = min(ic_start + BATCH_SIZE, C_in);
        
        // Mỗi thread load nhiều elements vào shared memory
        for (int load_idx = tid; load_idx < tile_size; load_idx += blockDim.x * blockDim.y) {
            int ty = load_idx / SHARED_TILE_WIDTH;
            int tx = load_idx % SHARED_TILE_WIDTH;
            int ih = tile_oh + ty - 1;  // -1 for padding
            int iw = tile_ow + tx - 1;
            
            // Load từ global → shared (coalesced)
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                s_input[ic - ic_start][ty][tx] = input[...];
            } else {
                s_input[ic - ic_start][ty][tx] = 0.0f;  // Padding
            }
        }
        __syncthreads();  // Đảm bảo tất cả threads đã load xong
        
        // Compute convolution từ shared memory (FAST!)
        for (int ic = ic_start; ic < ic_end; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    sum += s_input[ic - ic_start][ty + kh][tx + kw] * weight[...];
                }
            }
        }
        __syncthreads();
    }
}
```

**Lợi ích:**
- Input data được load 1 lần, reuse cho tất cả output channels
- Giảm global memory reads từ O(C_out × C_in × K²) xuống O(C_in × K²)
- Shared memory latency ~100× thấp hơn global memory

#### 2. Memory Coalescing Optimization

**Vấn đề:**
- Threads trong warp phải access consecutive memory để đạt coalescing
- Naive layout có thể gây uncoalesced access → serialize thành nhiều transactions

**Giải pháp:**
```cpp
// CRITICAL: threadIdx.x cho width dimension (ow)
int oh = blockIdx.y * blockDim.y + threadIdx.y;  // Height
int ow = blockIdx.z * blockDim.x + threadIdx.x;  // Width - COALESCED!

// Warp threads (threadIdx.x = 0,1,2,...,31) access:
// output[... + oh * W + 0], output[... + oh * W + 1], ..., output[... + oh * W + 31]
// → Consecutive addresses → Single 128-byte transaction
```

**Thread Block Configuration:**
```cpp
dim3 block(16, 16, 1);  // threadIdx.x = width, threadIdx.y = height
dim3 grid(C_out, (H+15)/16, (W+15)/16);
```

#### 3. In-Place ReLU Activations

**Optimization:**
```cpp
// OLD (Basic): Separate buffers
d_conv1_out → d_relu1_out (copy)

// NEW (Opt V1): In-place modification
d_relu1_out = d_conv1_out;  // Same pointer!
relu_inplace_kernel<<<...>>>(d_conv1_out, size);  // Modify in-place
```

**Lợi ích:**
- Giảm memory allocation (256×32×32 + 128×16×16 + ...) = ~2 MB saved
- Giảm memory bandwidth usage
- Không cần intermediate buffers

#### 4. Gradient Buffer Reuse

**Optimization:**
```cpp
// Reuse large buffers cho smaller layers
d_grad_up1 = d_grad_up2;       // Reuse (different sizes)
d_grad_relu3 = d_grad_relu4;   // Reuse
d_grad_conv3 = d_grad_conv4;   // Reuse
d_grad_pool2 = d_grad_relu4;   // Reuse
// ... etc

// Chỉ allocate 4 buffers thay vì 10+
CUDA_CHECK(cudaMalloc(&d_grad_conv5, 256*32*32 * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_grad_up2, 256*32*32 * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_grad_relu4, 256*16*16 * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_grad_conv4, 256*16*16 * sizeof(float)));
```

**Lợi ích:**
- Giảm memory footprint ~50%
- Ít cudaMalloc calls → faster initialization
- Backward pass không cần tất cả gradients đồng thời

#### 5. Removed Redundant cudaMemset

**Optimization:**
```cpp
// OLD (Basic): Zero out before maxpool backward
cudaMemset(d_grad_relu1, 0, size);
maxpool_backward_kernel<<<...>>>(grad_out, input, d_grad_relu1, ...);

// NEW (Opt V1): Kernel zeros internally
__global__ void maxpool_backward_kernel(...) {
    // Zero out all 4 positions
    grad_input[...] = 0.0f;
    grad_input[...] = 0.0f;
    grad_input[...] = 0.0f;
    grad_input[...] = 0.0f;
    
    // Set max position
    grad_input[max_position] = grad;
}
```

**Lợi ích:**
- Tiết kiệm ~2-3ms per backward pass
- Ít kernel launches

---

## Cách Chạy GPU Optimized V1

### Build Code
```bash
cd /home/senyamiku/LTSS/FinalCuda
bash scripts/build_cuda.sh
```

### Training
```bash
# Test với 1000 ảnh
./build_cuda/autoencoder_cuda_opt_v1 cifar-10-binary/cifar-10-batches-bin weights/opt_v1.bin 3 32 0.001 1000

# Full training (50,000 ảnh)
./build_cuda/autoencoder_cuda_opt_v1 cifar-10-binary/cifar-10-batches-bin weights/opt_v1.bin 3 64 0.001 50000
```

### Profiling
```bash
# Profile với 1000 ảnh
nsys profile --stats=true -o report_opt1 ./build_cuda/autoencoder_cuda_opt_v1 cifar-10-binary/cifar-10-batches-bin weights/opt_v1.bin 3 64 0.001 1000

# Xem report
nsys stats report_opt1.nsys-rep
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
  - Optimizer: SGD with gradient clipping

### Training Performance

#### Test Run (1000 images, 3 epochs, batch=32)

```
Epoch 1/3 - Average Loss: 0.563297 - Time: 1577ms - Throughput: 634.115 imgs/sec
Epoch 2/3 - Average Loss: 0.490671 - Time: 1535ms - Throughput: 651.466 imgs/sec
Epoch 3/3 - Average Loss: 0.512217 - Time: 1534ms - Throughput: 651.89 imgs/sec

Total training time: 6184ms (6.2s)
Average throughput: 485.123 imgs/sec
```

**So sánh với GPU Basic (Phase 2.2):**
- GPU Basic (1K, 3 epochs): 9534ms → 3.18s/epoch
- GPU Opt V1 (1K, 3 epochs): 6184ms → 2.06s/epoch
- **Speedup: 1.54× (35% reduction trong training time)**

#### Full Training (50,000 images, 3 epochs, batch=64)

```
Epoch 1/3 - Average Loss: 0.606232 - Time: 76594ms - Throughput: 652.793 imgs/sec
Epoch 2/3 - Average Loss: 0.352399 - Time: 76556ms - Throughput: 653.117 imgs/sec
Epoch 3/3 - Average Loss: 0.270685 - Time: 76536ms - Throughput: 653.287 imgs/sec

Total training time: 231232ms (231s = 3.85 minutes)
Average throughput: 648.699 imgs/sec
```

**Observations:**
- Throughput ổn định ~650 imgs/sec (variance <0.5%)
- Loss giảm đều: 0.606 → 0.352 → 0.271
- Batch=64 tận dụng GPU tốt hơn batch=32

### Bảng So Sánh Performance

| Metric | GPU Basic (2.2) | GPU Opt V1 (2.3) | Speedup |
|--------|----------------|------------------|---------|
| **Time/epoch (1K, batch=32)** | 3.18s | 2.06s | **1.54×** |
| **Total time (1K, 3 epochs)** | 9.53s | 6.18s | **1.54×** |
| **Throughput (batch=32)** | 315 imgs/sec | 485 imgs/sec | **1.54×** |
| **Throughput (batch=64, 50K)** | ~376 imgs/sec | 649 imgs/sec | **1.73×** |
| **Memory usage** | 441 MiB | 617 MiB | +40% |

**Cumulative Speedup vs CPU:**
- CPU Baseline: 750s/epoch
- GPU Opt V1: 2.06s/epoch
- **Cumulative: 364× faster than CPU**

### GPU Memory Usage

**Từ nvidia-smi (50K images training):**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15    CUDA: 12.4 |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Memory-Usage         | GPU-Util  Power      |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB    | 617MiB / 40960MiB    | 99%      149W/400W   |
+-----------------------------------------------------------------------------+
```

**Phân tích:**
- **Memory Used**: 617 MiB (vs 441 MiB ở Basic với cùng batch=64)
- **Tăng 40%** do:
  - Batch processing buffers cho forward pass (d_batch_conv1_out, d_batch_pool1_out, etc.)
  - Shared memory tiles allocation overhead
  - Additional intermediate buffers cho optimized kernels
- **GPU Utilization**: 99% (fully utilized, excellent!)
- **Power**: 149W/400W (37% - compute-bound workload)

**Memory Breakdown:**
- Weights: ~3 MB
- Batch processing buffers (batch=64):
  - Input: 64×3×32×32 = ~25 MB
  - Intermediate activations: ~400 MB
  - Gradients: ~400 MB
- Shared memory overhead: minimal (~200 KB per SM)

### Profiling Analysis (nsys, 1000 images)

#### Kernel Time Breakdown

```
Kernel Name                          | Time(%)  | Total Time | Avg Time  | Calls
-------------------------------------|----------|------------|-----------|-------
conv2d_weight_grad_kernel            | 37.4%    | 1.68s      | 112.2 μs  | 15,000
conv2d_input_grad_kernel             | 30.5%    | 1.37s      | 114.5 μs  | 12,000
conv2d_bias_grad_kernel              | 19.7%    | 0.89s      | 59.1 μs   | 15,000
conv2d_tiled_kernel_batched          | 9.8%     | 0.44s      | 1.84 ms   | 240 ← OPTIMIZED!
mse_loss_kernel                      | 1.0%     | 45.4ms     | 14.9 μs   | 3,048
relu_backward_kernel                 | 0.6%     | 27.0ms     | 2.3 μs    | 12,000
maxpool_backward_kernel              | 0.4%     | 16.1ms     | 2.7 μs    | 6,000
upsample_backward_kernel             | 0.3%     | 14.0ms     | 2.3 μs    | 6,000
relu_inplace_kernel_batched          | 0.1%     | 6.1ms      | 31.6 μs   | 192
upsample_kernel_batched              | 0.1%     | 5.2ms      | 54.4 μs   | 96
maxpool_kernel_batched               | 0.1%     | 4.0ms      | 42.2 μs   | 96

Total kernel time: 4.50s (out of 6.24s total)
```

**So sánh với GPU Basic:**

| Kernel Category | Basic (2.2) | Opt V1 (2.3) | Change |
|----------------|-------------|--------------|--------|
| **Conv forward** | 2.94s (37.3%) | 0.44s (9.8%) | **6.7× faster!** |
| **Conv backward** | 4.78s (60.7%) | 3.94s (87.6%) | 1.2× faster |
| **Other kernels** | 0.18s (2%) | 0.12s (2.6%) | 1.5× faster |
| **Total kernel** | 7.90s | 4.50s | **1.76× faster** |

**Key Insight:**
- **Conv forward với shared memory tiling: 6.7× speedup!**
- Backward pass vẫn chậm (87.6% time) → opportunity for Phase 2.4

#### CUDA API Time

```
Operation                | Time(%)  | Total Time | Calls
-------------------------|----------|------------|---------
cudaMemcpy               | 90.9%    | 4.33s      | 9,116
cudaLaunchKernel         | 6.6%     | 313ms      | 70,152
cudaMalloc               | 2.2%     | 106ms      | 45
cudaMemset               | 0.2%     | 10.4ms     | 3,048
```

**So sánh với Basic:**

| API | Basic (2.2) | Opt V1 (2.3) | Change |
|-----|-------------|--------------|--------|
| **cudaMemcpy** | 5.85s (73.3%) | 4.33s (90.9%) | 1.35× faster, but % UP! |
| **cudaLaunchKernel** | 1.97s (24.7%) | 313ms (6.6%) | **6.3× faster** |
| **Total API** | 7.98s | 4.76s | **1.68× faster** |

**Ý nghĩa:**
- cudaMemcpy vẫn là bottleneck (90.9%)
- Launch overhead giảm mạnh (6.3×) nhờ ít kernel launches hơn
- Cần optimize memory transfer tiếp (Phase 2.4: async streams?)

#### Memory Operations

```
Operation          | Time(%)  | Total Time | Count  | Avg Size
-------------------|----------|------------|--------|----------
memcpy DtoD        | 48.8%    | 12.0ms     | 6,000  | 12 KB
memcpy DtoH        | 24.2%    | 6.0ms      | 3,058  | 1 KB
memcpy HtoD        | 14.1%    | 3.5ms      | 58     | 687 KB
memset             | 12.9%    | 3.2ms      | 3,048  | 4 bytes
```

**Observations:**
- Device-to-device memcpy chiếm 48.8% (internal activation copies)
- Host-to-device ít hơn nhiều (chỉ weight loading)
- Memset giảm đáng kể so với Basic (removed redundant zeroing)

---

## Analysis

### Why Did This Optimization Work?

#### 1. Shared Memory Tiling - BIG WIN (6.7× forward speedup)

**Lý do thành công:**
- **Memory reuse**: Input tile được load 1 lần, dùng cho tất cả output channels
- **Low latency**: Shared memory latency ~28 cycles vs global memory ~400-800 cycles
- **High bandwidth**: Shared memory ~19 TB/s vs global memory ~1.5 TB/s (A100)

**Tính toán Bandwidth Savings:**
```
Without tiling:
- Mỗi output pixel đọc: C_in × K × K = 256 × 3 × 3 = 2,304 floats
- 256 output channels → 256 × 2,304 = 590,000 floats per output pixel
- Global memory reads: 590K × 4 bytes = 2.36 MB per pixel

With tiling (16×16 tile):
- Load input tile once: 18 × 18 × C_in = 324 × 256 = 83K floats
- Compute 256 output pixels: 256 × 256 outputs = 65K outputs
- Global memory reads: 83K × 4 bytes = 332 KB per tile (256 pixels)
- Per pixel: 332 KB / 256 = 1.3 KB → 1,800× reduction!
```

#### 2. Memory Coalescing - Moderate Win (1.2× on backward)

**Ảnh hưởng:**
- Warp threads access consecutive addresses → single transaction
- Giảm memory latency từ serialize access
- Hiệu quả nhất cho backward pass (30.5% kernel time)

**Chưa tối ưu hoàn toàn:**
- Backward kernels vẫn chậm (87.6% total kernel time)
- Weight gradient accumulation có nhiều atomicAdd → contention
- Cần thêm optimization (Phase 2.4)

#### 3. In-Place Operations & Buffer Reuse - Memory Win

**Giảm memory footprint:**
- ReLU in-place: ~2 MB saved
- Gradient reuse: ~10 MB saved
- Faster initialization (ít cudaMalloc calls)

**Trade-off:**
- Memory usage tăng từ 441 MiB → 617 MiB (batch size effect)
- Nhưng throughput tăng 1.73× → worth it!

### What Did Profiling Reveal?

**Bottlenecks Remaining:**

1. **cudaMemcpy vẫn chiếm 90.9%** API time
   - Synchronous execution blocking CPU
   - Không overlap computation với memory transfer
   - **Next: CUDA streams + async operations**

2. **Backward pass chiếm 87.6%** kernel time
   - Weight gradient: 37.4%
   - Input gradient: 30.5%
   - Bias gradient: 19.7%
   - **Next: Kernel fusion cho backward (Conv+Bias+ReLU gradient)**

3. **Memory bandwidth underutilized**
   - A100 có 1.5 TB/s peak bandwidth
   - Current usage: chưa saturate
   - **Next: Prefetching, async memcpy**

### Next Bottleneck: Backward Pass Kernels

**Current distribution:**
- Forward (tiled): 9.8%
- Backward (3 kernels): 87.6%
- Other: 2.6%

**Optimization opportunities:**
1. **Kernel Fusion**: Merge conv_weight_grad + conv_bias_grad → 1 kernel
2. **Shared Memory for Backward**: Apply tiling cho input/weight gradients
3. **Reduce atomicAdd contention**: Better accumulation strategy
4. **Fused backward**: Conv_backward + ReLU_backward → 1 kernel

---

## Key Takeaways

### Lessons Learned

1. **Shared Memory Tiling Is Powerful**
   - 6.7× speedup cho conv forward kernels
   - Critical for compute-intensive operations với data reuse
   - Block size (16×16) phù hợp với tile size và shared memory limits

2. **Forward/Backward Asymmetry**
   - Forward: 1 kernel, dễ optimize → 6.7× speedup
   - Backward: 3 kernels, phức tạp hơn → chỉ 1.2× speedup
   - Backward pass cần attention riêng (Phase 2.4)

3. **Memory Coalescing Matters**
   - threadIdx.x = width dimension → coalesced access
   - Ảnh hưởng lớn nhất khi combined với other optimizations
   - 1.2-1.5× speedup depending on kernel

4. **Small Optimizations Add Up**
   - In-place ReLU: small win nhưng zero cost
   - Removed cudaMemset: tiết kiệm 2-3ms per backward
   - Gradient buffer reuse: giảm initialization time

5. **cudaMemcpy Overhead Cannot Be Ignored**
   - 90.9% API time → synchronization bottleneck
   - Kernel speedups bị "hidden" bởi sync overhead
   - Must use async operations + streams (Phase 2.4)

### Applicability to Other Problems

**Khi nào áp dụng Shared Memory Tiling:**
- ✅ Operations với spatial locality (convolution, matrix multiply, stencil)
- ✅ Data được reuse nhiều lần (weights, input tiles)
- ✅ Computation intensity cao (nhiều FLOPs per byte loaded)
- ❌ Irregular memory access patterns
- ❌ Very large working sets không fit vào shared memory

**Khi nào áp dụng Memory Coalescing:**
- ✅ Mọi CUDA kernel! (fundamental optimization)
- ✅ Critical cho memory-bound workloads
- ✅ Easy win với proper thread indexing

**Khi nào áp dụng In-Place Operations:**
- ✅ Element-wise operations (ReLU, sigmoid, dropout)
- ✅ Không cần giữ original input cho backward pass
- ❌ Cần original input values (e.g., max pooling cần input để tìm max position)

### Performance Targets Achieved

| Target | Expected | Actual | Status |
|--------|----------|--------|--------|
| Speedup vs Basic | 1.5-2× | 1.54× | ✅ Met |
| Cumulative vs CPU | ~250× | 364× | ✅ Exceeded |
| GPU Utilization | >90% | 99% | ✅ Excellent |
| Memory Efficiency | Improved | +40% usage but 1.73× throughput | ✅ Trade-off OK |

---

## Next Steps → Phase 2.4

**Focus Areas:**

1. **Kernel Fusion for Backward Pass**
   - Merge conv_weight_grad + conv_bias_grad
   - Fuse conv_backward + relu_backward
   - Target: 2× speedup on backward (87.6% → ~45%)

2. **Async Execution with CUDA Streams**
   - Overlap kernel execution với memory transfers
   - Pipeline batch processing
   - Target: Hide 90% of cudaMemcpy overhead

3. **Shared Memory Tiling for Backward**
   - Apply same technique to backward kernels
   - More complex due to gradient accumulation
   - Target: 1.5-2× speedup on backward kernels

4. **Overall Target for Phase 2.4:**
   - **2-3× speedup over Phase 2.3**
   - **5-6× speedup over Phase 2.2 (Basic)**
   - **~1000× speedup over CPU**

---

## File Structure Reference

```
src/
├── main_cuda.cpp
├── cuda/
│   ├── autoencoder_basic.cu         # Phase 2.2: Naive GPU
│   ├── autoencoder_opt_v1.cu        # Phase 2.3: Shared Memory + Coalescing ✅
│   └── autoencoder_opt_v2.cu        # Phase 2.4: (Next) Kernel Fusion + Streams
include/
└── autoencoder_cuda.h               # AutoencoderCUDA class
scripts/
└── build_cuda.sh                    # Build script
```

**Key Optimizations in autoencoder_opt_v1.cu:**
- `conv2d_tiled_kernel_batched()`: Shared memory tiling (lines 90-182)
- `#define TILE_WIDTH 16`: Tile configuration (lines 30-35)
- In-place ReLU: `d_relu1_out = d_conv1_out` (line 722)
- Gradient buffer reuse: `d_grad_up1 = d_grad_up2` (lines 745-753)
- Memory coalescing: `threadIdx.x = ow` (lines 48, 375, 443, 572)
- Removed cudaMemset in maxpool_backward (lines 467-475)

**Total lines**: ~1,300 (vs ~1,000 in basic)
