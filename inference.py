"""Shared inference loading for API and CLI."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from data import build_eval_transform, default_normalize
from model import build_model, model_info_path


def load_meta(checkpoint_path: str) -> dict[str, Any]:
    """Load metadata from checkpoint or sibling model_info.json."""
    info_path = model_info_path(checkpoint_path)
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('meta', data)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'meta' in ckpt:
            return ckpt['meta']

    return _fallback_oxford_meta()


def _fallback_oxford_meta() -> dict[str, Any]:
    """Backwards compatibility when no metadata in checkpoint."""
    names_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'datasets', 'oxford102', 'class_names.txt',
    )
    class_names = []
    if os.path.exists(names_path):
        with open(names_path, 'r', encoding='utf-8') as f:
            class_names = [line.split(':', 1)[1].strip() for line in f if ':' in line]

    norm = default_normalize()
    return {
        'arch': 'resnet152',
        'num_classes': len(class_names) or 102,
        'class_names': class_names,
        'image_size': 256,
        'crop_size': 224,
        'normalize': norm,
        'dataset': 'oxford102',
    }


def build_transform_from_meta(meta: dict[str, Any]) -> transforms.Compose:
    normalize = meta.get('normalize') or default_normalize()
    image_size = meta.get('image_size', 256)
    crop_size = meta.get('crop_size', 224)
    return build_eval_transform(image_size, crop_size, normalize)


def load_state_dict(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    model.load_state_dict(state)


def load_inference_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
) -> Tuple[nn.Module, list[str], transforms.Compose, dict[str, Any]]:
    """
    Load model, class names, eval transform, and metadata for inference.
    """
    from device import get_device

    device = device or get_device(use_gpu)
    meta = load_meta(checkpoint_path)

    arch = meta.get('arch', 'resnet152')
    num_classes = meta.get('num_classes', len(meta.get('class_names', [])) or 102)
    class_names = meta.get('class_names', [])

    model = build_model(arch, num_classes, pretrained=False)

    if os.path.exists(checkpoint_path):
        load_state_dict(model, checkpoint_path, device)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    transform = build_transform_from_meta(meta)
    return model, class_names, transform, meta


def predict_image(
    model: nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
    image,
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """Run inference on a PIL image. Returns list of {class_id, class_name, confidence}."""
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        k = min(top_k, probs.shape[0])
        top_prob, top_idx = torch.topk(probs, k)

    results = []
    for prob, idx in zip(top_prob, top_idx):
        i = idx.item()
        name = class_names[i] if i < len(class_names) else str(i)
        results.append({
            'class_id': i,
            'class_name': name,
            'confidence': float(prob.item()),
        })
    return results
