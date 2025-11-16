#!/usr/bin/env python3
"""
U-Net Model - Simple Working Implementation
Complete model architecture and dataset for background removal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class DoubleConv(nn.Module):
    """Double convolution block with proper padding"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # padding=1 keeps size
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # padding=1 keeps size
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net Architecture with proper skip connections"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Store encoder outputs for skip connections
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder with skip connections
        for idx in range(len(self.decoder)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]

            # Handle size mismatch (if any)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:],
                                  mode='bilinear', align_corners=True)

            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder[idx](x)

        # Final output (no sigmoid here if using BCEWithLogitsLoss)
        return self.final_conv(x)


class BackgroundRemovalDataset(Dataset):
    """Fixed dataset class with proper error handling"""

    def __init__(self, image_dir, mask_dir, transform=None, target_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

        # Get all image files
        self.images = []
        if os.path.exists(image_dir):
            self.images = sorted([f for f in os.listdir(image_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {image_dir}. "
                               f"Please check the path and file formats.")

        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Try different mask extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(self.mask_dir,
                                     os.path.splitext(img_name)[0] + ext)
            if os.path.exists(mask_path):
                break
        else:
            raise FileNotFoundError(f"Mask not found for {img_name}")

        # Load and convert images
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale
        except Exception as e:
            raise RuntimeError(f"Error loading {img_name}: {str(e)}")

        # Resize to fixed size
        target_size = (self.target_size, self.target_size)
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0

        # Apply transforms if any
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        return image, mask


def test_model():
    """Test model with sample input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Test forward pass
    x = torch.randn(1, 3, 256, 256).to(device)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (1, 1, 256, 256), "Output shape mismatch!"
    print("✓ Model test passed!")


if __name__ == "__main__":
    test_model()