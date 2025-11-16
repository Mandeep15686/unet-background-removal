#!/usr/bin/env python3
"""
Training Script - Complete U-Net Training with Multiple Modes
Supports quick test, fast training, and full training modes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import json
import argparse


def train_model(
        model,
        train_loader,
        val_loader=None,
        epochs=10,
        learning_rate=1e-4,
        device='cuda',
        save_dir='checkpoints',
        accumulation_steps=1
):
    """Train U-Net model with proper error handling"""
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    # Use BCEWithLogitsLoss (includes sigmoid, more numerically stable)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print(f"Training on {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\\n⚠️ CUDA OOM! Reduce batch_size or image_size")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        if device == 'cuda':
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\\n" + "=" * 60)
    print("Training completed!")
    print(f"Models saved in: {save_dir}")
    return history


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for Background Removal')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    from models.u2net_model import UNet, BackgroundRemovalDataset

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Check data
    image_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')

    if not os.path.exists(image_dir):
        print(f"❌ Error: {image_dir} not found!")
        return

    # Create dataset
    try:
        dataset = BackgroundRemovalDataset(image_dir, mask_dir)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )

        print(f"\\n✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}\\n")

    except Exception as e:
        print(f"❌ Dataset error: {str(e)}")
        return

    # Create model
    model = UNet(in_channels=3, out_channels=1)

    # Resume if checkpoint provided
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Resumed from {args.resume}")

    # Train
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            accumulation_steps=args.accumulation_steps
        )
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()