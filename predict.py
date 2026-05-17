#!/usr/bin/env python3
"""CLI image classification using a trained checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

from device import get_device, use_gpu_enabled
from inference import load_inference_model, predict_image


def parse_args():
    p = argparse.ArgumentParser(description='Predict class for an image')
    p.add_argument('image', help='Path to input image')
    p.add_argument('--checkpoint', default=os.getenv(
        'CHECKPOINT', 'datasets/oxford102/model/resnet152_oxford102.pth',
    ))
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        print(f'Image not found: {args.image}', file=sys.stderr)
        sys.exit(1)

    use_gpu = use_gpu_enabled(not args.cpu)
    device = get_device(use_gpu)

    model, class_names, transform, meta = load_inference_model(
        args.checkpoint, device=device, use_gpu=use_gpu,
    )
    image = Image.open(args.image).convert('RGB')
    predictions = predict_image(
        model, transform, class_names, image, device, top_k=args.top_k,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {meta.get('dataset', '?')}  Arch: {meta.get('arch', '?')}  "
          f"Classes: {meta.get('num_classes', len(class_names))}")
    print('-' * 60)
    for i, pred in enumerate(predictions, 1):
        print(f"{i:2d}. {pred['class_name']:<40} {pred['confidence']:.4f}  (id={pred['class_id']})")


if __name__ == '__main__':
    main()
