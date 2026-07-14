import os
import numpy as np
from skimage import measure
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def instance_max_min_normalize(input_tensor):
    b = input_tensor.shape[0]
    min_vals = torch.min(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
    max_vals = torch.max(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor


def instance_mean_std_normalize(input_tensor):
    b = input_tensor.shape[0]
    mean_vals = torch.mean(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1)
    std_vals = torch.std(input_tensor.view(b, -1), dim=1).view(b, 1, 1, 1)
    normalized_tensor = (input_tensor - mean_vals + 1e-5) / (std_vals + 1e-5)
    return normalized_tensor
