import torch
import numpy as np

def iou_score(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)

def dice_score(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = 2 * (pred_binary * target).sum()
    return intersection / (pred_binary.sum() + target.sum() + 1e-8)
