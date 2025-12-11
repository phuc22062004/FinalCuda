# QUICKSTART - CUDA Autoencoder

## 🚀 Build & Run (5 phút)

### 1. Build GPU Version
```bash
./build_gpu_basic.sh
```

### 2. Run Quick Test (1K images, 10 epochs, ~2 phút)
```bash
cd build_cuda
./autoencoder_cuda_basic ../cifar-10-binary/cifar-10-batches-bin test.bin 10 32 0.001 1000
```

### 3. Run Full Training (1K images, 15 epochs, ~3 phút)
```bash
./autoencoder_cuda_basic ../cifar-10-binary/cifar-10-batches-bin model.bin 15 32 0.001 1000
```

## 📊 Expected Results

**Quick Test (1K, 10 epochs):**
- Time: **~118 seconds** (2 phút)
- Initial Loss: **0.047** (epoch 1)
- Final Loss: **0.0156** (epoch 10) ✅
- Loss giảm: **47 → 15.6 (~3x improvement)**
- Batch loss: **0.015-0.017** (rất ổn định)
- GPU Memory: ~500 MB

**Loss Trajectory:**
```
Epoch 1:  0.047
Epoch 2:  0.028
Epoch 5:  0.020
Epoch 10: 0.0156 ✅
```

**Full Training (1K, 15 epochs):**
- Expected Final Loss: **~0.012-0.015** (tiếp tục giảm)
- Time: ~3 minutes

## 🔧 Parameters

```
./autoencoder_cuda_basic [cifar_dir] [model_path] [epochs] [batch_size] [lr] [max_images]
```

**Defaults:**
- epochs: 20
- batch_size: 32
- learning_rate: 0.001
- max_images: 10000

## 📖 Full Documentation

See `README.md` for complete guide.
