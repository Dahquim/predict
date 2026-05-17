#!/usr/bin/env python3
"""Verify PyTorch and GPU (CUDA/ROCm) setup."""

import sys

import torch


def main() -> int:
    print('PyTorch version:', torch.__version__)
    print('CUDA available (CUDA or ROCm):', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('Device count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  [{i}] {torch.cuda.get_device_name(i)}')
        try:
            x = torch.randn(512, 512, device='cuda')
            y = x @ x
            print('GPU matmul smoke test: OK', y.shape, y.device)
        except Exception as e:
            print('GPU matmul smoke test: FAILED', e)
            return 1
    else:
        ver = torch.__version__.lower()
        if 'cu' in ver and 'rocm' not in ver:
            print(
                '\nHint: Installed build looks like NVIDIA CUDA (e.g. +cu130) on a machine '
                'that may need ROCm. See README.md "AMD GPU" section.'
            )
        elif 'rocm' not in ver:
            print('\nHint: For AMD GPUs: bash ./scripts/setup_rocm_venv.sh (Python 3.11–3.14 + ROCm wheels).')
        print('Training will fall back to CPU unless you fix the environment.')
        return 0

    if 'rocm' in torch.__version__.lower():
        print('ROCm build detected.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
