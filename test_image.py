import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# 1. Model Definition
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2) 
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2) 
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(256, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(self.up1(x)))
        x = self.conv5(self.up2(x))
        return x

# ==========================================
# 2. Weight Loading Logic
# ==========================================
def load_weights_from_binary(model, filepath):
    print(f"Loading weights from {filepath}...")
    if not os.path.exists(filepath):
        print("Error: Weight file not found.")
        sys.exit(1)

    weights = np.fromfile(filepath, dtype=np.float32)
    offset = 0
    
    def load_layer(layer):
        nonlocal offset
        w_count = np.prod(layer.weight.shape)
        layer.weight.data.copy_(torch.from_numpy(weights[offset:offset+w_count].reshape(layer.weight.shape)))
        offset += w_count
        
        b_count = np.prod(layer.bias.shape)
        layer.bias.data.copy_(torch.from_numpy(weights[offset:offset+b_count]))
        offset += b_count

    load_layer(model.conv1)
    load_layer(model.conv2)
    load_layer(model.conv3)
    load_layer(model.conv4)
    load_layer(model.conv5)
    print("Weights loaded successfully.")

# ==========================================
# 3. Main Logic
# ==========================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_folder_vertical.py <weights.bin> <folder_path>")
        return

    weight_path = sys.argv[1]
    folder_path = sys.argv[2]
    
    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    load_weights_from_binary(model, weight_path)
    model.eval()

    # Find Images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([
        f for f in os.listdir(folder_path) 
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        print(f"No images found in {folder_path}!")
        return

    print(f"Found {len(image_files)} images.")
    
    inputs = []
    for fname in image_files:
        path = os.path.join(folder_path, fname)
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((32, 32))
            inputs.append(transforms.ToTensor()(img))
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not inputs: return

    # Inference
    input_batch = torch.stack(inputs).to(device)
    with torch.no_grad():
        output_batch = model(input_batch)

    # Convert to Numpy
    inputs_np = input_batch.cpu().permute(0, 2, 3, 1).numpy()
    outputs_np = output_batch.cpu().permute(0, 2, 3, 1).numpy()
    outputs_np = np.clip(outputs_np, 0.0, 1.0)

    # --- NEW STITCHING LOGIC ---
    print("Stitching images (Input TOP, Output BOTTOM)...")
    columns = []
    
    # Define a separator (White line, 2 pixels wide, 64 pixels high)
    # 64 height because 32 (Input) + 32 (Output) = 64
    sep_width = 2
    separator = np.ones((64, sep_width, 3)) # White color (1.0)

    for i in range(len(inputs_np)):
        # 1. Vertically Stack: Input on TOP of Output
        # axis=0 is vertical stacking for (H, W, C) arrays
        col = np.concatenate((inputs_np[i], outputs_np[i]), axis=0)
        
        columns.append(col)
        
        # Add separator after every column except the last one
        if i < len(inputs_np) - 1:
            columns.append(separator)

    # 2. Horizontally Stack all columns
    final_strip = np.concatenate(columns, axis=1)

    # Save
    output_filename = "res.png"
    print(f"Saving to {output_filename}...")
    
    plt.figure(figsize=(len(inputs_np) * 1.5, 3)) # Adjust aspect ratio
    plt.imshow(final_strip)
    plt.axis('off')
    plt.title("Top: Input | Bottom: Reconstruction")
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    main()