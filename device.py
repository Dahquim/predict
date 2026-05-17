"""Device selection for CUDA (NVIDIA), ROCm (AMD), or CPU."""

import torch


def get_device(use_gpu: bool = True) -> torch.device:
    """Return cuda device when available (CUDA or ROCm builds), else CPU."""
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def use_gpu_enabled(use_gpu_flag: bool) -> bool:
    return use_gpu_flag and torch.cuda.is_available()
