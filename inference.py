#!/usr/bin/env python3
"""
Background Removal Inference Script
Remove backgrounds from images using trained U-Net
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from pathlib import Path
import argparse


def load_model(model_path, device='cuda'):
    """Load trained model"""
    from models.u2net_model import UNet

    model = UNet(in_channels=3, out_channels=1)

    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"✓ Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, target_size=256):
    """
    Load and preprocess image for inference

    Args:
        image_path: Path to input image
        target_size: Size to resize image (must match training size)

    Returns:
        tensor: Preprocessed image tensor
        original_size: Original image size for resizing back
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # Resize
    image = image.resize((target_size, target_size), Image.BILINEAR)

    # Convert to numpy and normalize
    image_np = np.array(image).astype(np.float32) / 255.0

    # Convert to tensor: HWC -> CHW
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, original_size


def postprocess_mask(mask_tensor, original_size, threshold=0.5):
    """
    Postprocess model output to create final mask

    Args:
        mask_tensor: Model output tensor
        original_size: Original image size (W, H)
        threshold: Threshold for binary mask

    Returns:
        mask: Binary mask as PIL Image
    """
    # Apply sigmoid to get probabilities (if not already applied)
    mask = torch.sigmoid(mask_tensor)

    # Remove batch and channel dimensions
    mask = mask.squeeze().cpu().numpy()

    # Apply threshold
    mask = (mask > threshold).astype(np.uint8) * 255

    # Convert to PIL Image
    mask = Image.fromarray(mask)

    # Resize back to original size
    mask = mask.resize(original_size, Image.BILINEAR)

    return mask


def remove_background(image_path, mask, output_path=None, background='transparent'):
    """
    Remove background from image using mask

    Args:
        image_path: Path to original image
        mask: Binary mask (PIL Image)
        output_path: Path to save result
        background: 'transparent', 'white', 'black', or RGB tuple

    Returns:
        result: Image with background removed
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')

    # Ensure mask is same size
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.BILINEAR)

    # Convert to numpy
    image_np = np.array(image)
    mask_np = np.array(mask) / 255.0

    if background == 'transparent':
        # Create RGBA image
        result = Image.new('RGBA', image.size)
        result_np = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
        result_np[:, :, :3] = image_np
        result_np[:, :, 3] = mask_np * 255
        result = Image.fromarray(result_np)

    elif background == 'white':
        result_np = image_np * mask_np[:, :, np.newaxis] + (1 - mask_np[:, :, np.newaxis]) * 255
        result = Image.fromarray(result_np.astype(np.uint8))

    elif background == 'black':
        result_np = image_np * mask_np[:, :, np.newaxis]
        result = Image.fromarray(result_np.astype(np.uint8))

    else:
        # Custom RGB background
        bg_color = np.array(background)
        result_np = image_np * mask_np[:, :, np.newaxis] + bg_color * (1 - mask_np[:, :, np.newaxis])
        result = Image.fromarray(result_np.astype(np.uint8))

    # Save if output path provided
    if output_path:
        result.save(output_path)
        print(f"✓ Saved result to {output_path}")

    return result


def process_single_image(model, image_path, output_path, device='cuda',
                         threshold=0.5, background='transparent'):
    """Process a single image"""
    print(f"Processing: {image_path}")

    # Preprocess
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)

    # Postprocess
    mask = postprocess_mask(output, original_size, threshold)

    # Remove background
    result = remove_background(image_path, mask, output_path, background)

    return result


def process_batch(model, input_dir, output_dir, device='cuda',
                  threshold=0.5, background='transparent'):
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(input_dir)
                   if Path(f).suffix.lower() in image_extensions]

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Process each image
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir,
                                   Path(img_file).stem + '.png')

        try:
            process_single_image(model, input_path, output_path,
                                 device, threshold, background)
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")

    print(f"\\n✓ Processed {len(image_files)} images")
    print(f"✓ Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Background Removal Inference')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output path or directory')
    parser.add_argument('--model', default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask (0-1)')
    parser.add_argument('--background', default='transparent',
                        choices=['transparent', 'white', 'black'],
                        help='Background type')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    try:
        model = load_model(args.model, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    # Process images
    if args.batch:
        process_batch(model, args.input, args.output,
                      args.device, args.threshold, args.background)
    else:
        process_single_image(model, args.input, args.output,
                             args.device, args.threshold, args.background)


if __name__ == "__main__":
    # If no arguments provided, show example usage
    import sys

    if len(sys.argv) == 1:
        print("Background Removal Inference Script")
        print("=" * 60)
        print("\\nExample usage:")
        print("\\n1. Single image:")
        print("   python inference.py --input photo.jpg --output result.png")
        print("\\n2. Batch processing:")
        print("   python inference.py --input photos/ --output results/ --batch")
        print("\\n3. Custom settings:")
        print("   python inference.py --input photo.jpg --output result.png \\\\")
        print("                       --threshold 0.7 --background white")
        print("\\nOptions:")
        print("  --model PATH       Path to model checkpoint")
        print("  --threshold FLOAT  Threshold for mask (0-1)")
        print("  --background TYPE  transparent|white|black")
        print("  --device TYPE      cuda|cpu")
        print("=" * 60)
    else:
        main()