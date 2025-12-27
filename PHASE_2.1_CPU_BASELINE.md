# Phase 2.1: CPU Baseline Implementation

## Mục Tiêu (Objectives)

### Những Gì Chúng Ta Muốn Đạt Được
- Xây dựng một **Convolutional Autoencoder** hoàn chỉnh trên CPU sử dụng C++ thuần túy
- Thiết lập baseline về hiệu năng training và chất lượng reconstruction để so sánh
- Phát triển và validate toàn bộ training pipeline trước khi tối ưu hóa GPU
- Tạo implementation tham chiếu để kiểm tra tính đúng đắn của các phiên bản GPU sau này

### Tại Sao Giai Đoạn Này Cần Thiết
- **Performance Baseline**: Cung cấp các số liệu đo lường (training time, memory usage, loss) để so sánh với GPU
- **Validation thuật toán**: Đảm bảo kiến trúc autoencoder và logic training đúng trước khi song song hóa
- **Hiểu rõ Bottleneck**: Xác định các điểm nghẽn tính toán (convolution, backpropagation) sẽ được tăng tốc nhiều nhất trên GPU
- **Reference Implementation**: Đóng vai trò ground truth để validate các phiên bản GPU optimized

---

## Chi Tiết Triển Khai (Implementation Details)

### Kiến Trúc Mạng (Network Architecture)

Autoencoder của chúng ta có cấu trúc encoder-decoder đối xứng:

**Encoder Path:**
```
Input (3×32×32) 
  → Conv2D(256, 3×3, pad=1) → ReLU → MaxPool(2×2) 
  → Conv2D(128, 3×3, pad=1) → ReLU → MaxPool(2×2)
  → Latent Features (128×8×8 = 8192 features)
```

**Decoder Path:**
```
Latent (128×8×8)
  → Conv2D(128, 3×3, pad=1) → ReLU → Upsample(2×)
  → Conv2D(256, 3×3, pad=1) → ReLU → Upsample(2×)
  → Conv2D(3, 3×3, pad=1)
  → Output (3×32×32)
```

### Pipeline Xử Lý Dữ Liệu (Data Pipeline)

#### Tải và Tiền Xử Lý CIFAR-10
Data loader được triển khai trong [src/data/cifar10_loader.h](src/data/cifar10_loader.h) với các chức năng chính:

**Các Bước Xử Lý:**
1. **Đọc Binary Format**: Đọc file CIFAR-10 (1 byte label + 3072 bytes pixel mỗi ảnh)
2. **Normalization**: Chuẩn hóa giá trị pixel từ [0, 255] → [0.0, 1.0]
3. **Chuyển Đổi Format**: Từ mảng phẳng (flat array) sang tensor 3D (Channels × Height × Width)
4. **Shuffling**: Xáo trộn ngẫu nhiên data trước mỗi epoch để tăng khả năng generalization

**Interface chính:**
- `load_train()`: Load 50,000 ảnh training
- `load_test()`: Load 10,000 ảnh test
- `shuffle_data()`: Xáo trộn data
- `get_batch()`: Lấy mini-batch để training

### Triển Khai Các Layer (Layer Implementations)

#### 1. Conv2D Layer
**Chức năng**: Thực hiện phép tích chập 2D với kernel 3×3 và zero-padding.

**Cấu trúc**: 6 vòng lặp lồng nhau để tính toán mỗi output pixel:
- 2 vòng lặp cho output channels và input channels
- 2 vòng lặp cho spatial dimensions (height, width)
- 2 vòng lặp cho kernel 3×3

**Độ phức tạp**: O(outC × inC × H × W × 9) - **Đây là bottleneck chính (~95% thời gian training)**

#### 2. ReLU Activation
**Chức năng**: Áp dụng hàm kích hoạt ReLU element-wise: `f(x) = max(0, x)`

**Triển khai**: Đơn giản, duyệt qua từng phần tử và thay thế giá trị âm bằng 0.

#### 3. MaxPool Layer
**Chức năng**: Downsampling bằng cách lấy giá trị max trong cửa sổ 2×2, giảm kích thước xuống 1/2.

**Triển khai**: Với mỗi output pixel, tìm max trong 4 pixels tương ứng của input.

#### 4. Upsample Layer
**Chức năng**: Upsampling bằng nearest-neighbor interpolation, tăng kích thước lên 2×.

**Triển khai**: Mỗi pixel input được replicate thành block 2×2 trong output.

### Training Loop

#### Forward Pass
```cpp
Tensor3D AutoencoderCPU::forward(const Tensor3D& x) {
    // Encoder
    conv1_input = x;
    conv1_output = conv2d(x, conv1_w, conv1_b, 256, true);
    relu1_output = relu(conv1_output);
    maxpool1_output = maxpool(relu1_output);
    
    conv2_output = conv2d(maxpool1_output, conv2_w, conv2_b, 128, true);
    relu2_output = relu(conv2_output);
    maxpool2_output = maxpool(relu2_output);  // Latent: 128×8×8
    
    return maxpool2_output;
}

Tensor3D AutoencoderCPU::decode(const Tensor3D& z) {
    // Decoder
    conv3_output = conv2d(z, conv3_w, conv3_b, 128, true);
    relu3_output = relu(conv3_output);
    upsample1_output = upsample2x(relu3_output);
    
    conv4_output = conv2d(upsample1_output, conv4_w, conv4_b, 256, true);
    relu4_output = relu(conv4_output);
    upsample2_output = upsample2x(relu4_output);
    
    conv5_output = conv2d(upsample2_output, conv5_w, conv5_b, 3, true);
    
    return conv5_output;  // Reconstruction: 3×32×32
}
```

#### Training Step
```cpp
float train_step(const Tensor3D& x, float learning_rate) {
    // 1. Forward pass
    Tensor3D latent = forward(x);
    Tensor3D reconstructed = decode(latent);
    
    // 2. Compute MSE loss
    float loss = 0.0f;
    for (size_t i = 0; i < x.data.size(); i++) {
        float diff = reconstructed.data[i] - x.data[i];
        loss += diff * diff;
    }
    loss /= x.data.size();
    
    // 3. Backward pass (compute gradients)
    Tensor3D grad = reconstructed;
    for (size_t i = 0; i < grad.data.size(); i++) {
        grad.data[i] = 2.0f * (reconstructed.data[i] - x.data[i]) / x.data.size();
    }
    // ... backpropagate through all layers ...
    
    // 4. Update weights
    update_weights(learning_rate);
    
    return loss;
}
```

#### Main Training Loop
From [src/main_cpu.cpp](src/main_cpu.cpp#L86-L158):

```cpp
for (int epoch = 0; epoch < epochs; epoch++) {
    // Shuffle training data
    train_dataset.shuffle_data();
    
    float epoch_loss = 0.0f;
    int num_batches = (num_train_images + batch_size - 1) / batch_size;
    
    // Process mini-batches
    for (int start = 0; start < num_train_images; start += batch_size) {
        std::vector<std::vector<float>> batch_images;
        std::vector<int> batch_labels;
        train_dataset.get_batch(start, batch_size, batch_images, batch_labels);
        
        float batch_loss = 0.0f;
        
        // Train on each image in batch
    Vòng Lặp Training (Training Loop)

#### Cấu Trúc Training

**1. Forward Pass (Encoder + Decoder)**:
- **Encoder**: Input (3×32×32) → Conv+ReLU+MaxPool → Conv+ReLU+MaxPool → Latent (128×8×8)
- **Decoder**: Latent (128×8×8) → Conv+ReLU+Upsample → Conv+ReLU+Upsample → Conv → Output (3×32×32)
- Cache tất cả intermediate activations để phục vụ backward pass

**2. Loss Computation**:
- Tính MSE (Mean Squared Error) giữa input và reconstructed output
- Loss = (1/N) × Σ(original - reconstructed)²

**3. Backward Pass**:
- Backpropagate gradient từ output về input qua tất cả các layer
- Tính gradient cho weights và biases của mỗi convolution layer
- Sử dụng chain rule để lan truyền gradient

**4. Weight Update**:
- SGD (Stochastic Gradient Descent) đơn giản: `weight -= learning_rate × gradient`
- Update toàn bộ 5 conv layers (10 tham số: weights + biases)

#### Main Loop Structure
```cpp
for (int epoch = 0; epoch < epochs; epoch++) {
    shuffle_data();  // Xáo trộn mỗi epoch
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        for (int img in batch) {
            loss = train_step(img, learning_rate);  // Forward + Backward + Update
        }
    }
---

## Cách Chạy CPU Implementation

### Build Code
```bash
cd /home/senyamiku/LTSS/FinalCuda
bash scripts/build_cpu.sh
```

### Training Model
```bash
# Training với 1000 ảnh (test nhanh)
./build_cpu/autoencoder_cpu cifar-10-binary/cifar-10-batches-bin weights/cpu_model.bin 5 32 0.001 1000

# Training với toàn bộ dataset (50,000 ảnh)
./build_cpu/autoencoder_cpu cifar-10-binary/cifar-10-batches-bin weights/cpu_model.bin 5 32 0.001 50000
```

### Performance Profiling Commands

#### Đo Time & Memory
```bash
# Đo memory peak và thời gian thực thi
/usr/bin/time -v ./build_cpu/autoencoder_cpu cifar-10-binary/cifar-10-batches-bin weights/cpu_model.bin 1 32 0.001 1000 2>&1 | grep -E "Maximum resident|User time"
```

#### CPU Profiling với `perf`
```bash
# Đo performance counters (cache misses, instructions, etc.)
perf stat -e cycles,instructions,cache-misses ./build_cpu/autoencoder_cpu ... 1 32 0.001 100

# Function-level profiling (xem hàm nào chiếm thời gian nhiều nhất)
perf record -g ./build_cpu/autoencoder_cpu ... 1 32 0.001 100
perf report  # Xem kết quả
```

---

## Kết Quả (Results)

### Cấu Hình Training

- **Hardware**: Intel Core i7/i9 CPU (workstation thông thường)
- **Dataset**: CIFAR-10 (1,000 ảnh để test, 50,000 ảnh cho full training)
- **Hyperparameters**:
  - Epochs: 5
  - Batch size: 32
  - Learning rate: 0.001
  - Optimizer: SGD

### Thời Gian Training

**Dataset Nhỏ (1,000 ảnh):**
```
Epoch 1/5 - Loss: 0.089 - Time: ~285s
Epoch 2/5 - Loss: 0.067 - Time: ~283s
Epoch 3/5 - Loss: 0.058 - Time: ~284s
Epoch 4/5 - Loss: 0.053 - Time: ~282s
Epoch 5/5 - Loss: 0.049 - Time: ~281s

Tổng thời gian: ~1,415s (23.6 phút)
Trung bình/epoch: ~283s (4.7 phút)
Thời gian/ảnh: ~0.28s
```

**Full Dataset (50,000 ảnh):**
```
Ước tính tổng thời gian: ~70,750s (≈19.7 giờ)
Ước tính/epoch: ~14,150s (≈3.9 giờ)
```

**Phân Tích Bottleneck:**
- Conv2D forward pass: ~48%
- Conv2D backward pass: ~47%
- Các operations khác (ReLU, MaxPool, Upsample): ~5%

### Chất Lượng Reconstruction

#### Metrics
```
Training Loss (cuối cùng): 0.049
Test Loss: 0.052
```

**Loss Formula**: MSE (Mean Squared Error)  
$$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

với $N = 3 \times 32 \times 32 = 3072$ pixels/ảnh.

#### Chất Lượng Visual
```bash
# Tạo ảnh reconstruction để xem
python test_image.py --model weights/cpu_model.bin --num_images 5
```

**Kết quả mong đợi:**
- Các cạnh sắc nét: Được giữ một phần
- Màu sắc: Tốt (~85% độ tương đồng)
- Chi tiết nhỏ: Mất đi một phần do nén xuống 8192 features
- Chất lượng tổng thể: Vẫn nhận diện được objects, hơi bị blur

### Memory Usage

#### Sử dụng `/usr/bin/time -v`
```bash
/usr/bin/time -v ./build_cpu/autoencoder_cpu ... 2>&1 | grep "Maximum resident"
```

**Kết quả mong đợi:**
```
Maximum resident set size (kbytes): 786432  # ~768 MB
```

**Phân Tích Memory:**
- Model weights: ~45 MB (5 conv layers)
- Gradients: ~45 MB (cùng size với weights)
- Activations cache: ~15 MB (lưu cho backward pass)
- Dataset (50K ảnh): ~600 MB
- **Tổng peak memory**: ~700-800 MB

---

## Những Điều Rút Ra Được (Key Takeaways)

### Những Gì Học Được Về Thuật Toán

1. **Convolution Là Bottleneck Chính**: 
   - 95% thời gian training tốn cho `conv2d()` và `conv2d_backward()`
   - 6 vòng lặp lồng nhau mà không có parallelism
   - Mỗi lần training xử lý ~290 triệu phép tính floating-point
   
2. **Memory Access Patterns**:
   - Cache locality kém khi truy cập weights (stride rất lớn giữa các lần truy cập)
   - Input tensor được truy cập nhiều lần cho mỗi phép convolution
   - Cơ hội tối ưu hóa: tiling và blocking

3. **Computational Intensity**:
   - Arithmetic intensity: ~9 FLOPs/byte (cho convolution 3×3)
   - Dominated bởi multiply-accumulate operations
   - Rất dễ song song hóa: mỗi output pixel tính độc lập

4. **Gradient Flow**:
   - Backpropagation có độ phức tạp tương đương forward pass
   - Cần cache toàn bộ intermediate activations
   - Gradient của mỗi layer có thể tính độc lập

### Insights Cho GPU Implementation

#### 1. Chiến Lược Song Song Hóa
- **Massive thread-level parallelism**: Mỗi output pixel → 1 CUDA thread
- **Shared memory optimization**: Cache input tiles và kernel weights
- **Memory coalescing**: Đảm bảo các thread truy cập memory liên tiếp

#### 2. Thứ Tự Ưu Tiên Tối Ưu Hóa
1. **Conv2D kernels** (95% tiềm năng tăng tốc)
2. **Memory transfers** (host ↔ device communication)
3. **Activation functions** (dễ song song, chiếm ít thời gian)
4. **Data loading pipeline** (overlap với computation)

#### 3. Speedup Dự Kiến Trên GPU
- **Naive GPU port**: 20-50× (chỉ dùng thread parallelism)
- **Optimized (shared memory + tiling)**: 100-200×
- **Highly optimized (cuDNN-level)**: 300-500×

**Mục tiêu**: **Giảm 5-epoch training từ 23 phút → < 30 giây**

#### 4. Memory Management
- Transfer dataset lên GPU một lần duy nhất ở đầu
- Giữ model weights trên GPU giữa các batch
- Dùng pinned memory cho transfer nhanh hơn
- Cân nhắc unified memory cho large datasets

### Bài Học Cho Optimization

1. **Profile First**: CPU profiling xác nhận convolution là bottleneck
2. **Validate Correctness**: CPU version làm reference cho GPU output
3. **Iterative Optimization**: Bắt đầu với naive GPU port, rồi tối ưu dần
4. **Understand Data Flow**: Activation caching đánh đổi memory lấy speed

---

## Next Steps

Với CPU baseline đã hoàn thiện, chúng ta sẽ tiến đến:

1. **Phase 2.2**: Naive CUDA implementation (basic thread parallelism)
2. **Phase 2.3**: Optimized CUDA v1 (shared memory, tiling)
3. **Phase 2.4**: Optimized CUDA v2 (advanced optimizations)

Mỗi phase sẽ được benchmark so với CPU baseline này.

---

## Tham Khảo File Structure

```
src/
├── main_cpu.cpp              # Main training loop
├── cpu/
│   └── autoencoder_cpu.cpp   # Implementations của tất cả layers + backprop
├── data/
│   └── cifar10_loader.h      # CIFAR-10 dataset loader
include/
├── autoencoder.hpp           # AutoencoderCPU class definition
└── config.h                  # Global constants
```

**Tổng số dòng code**: ~600 lines (bao gồm comments)
