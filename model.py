"""Model factories, checkpoint I/O, and training metrics."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from torchvision.models import (
    alexnet,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    inception_v3,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    shufflenet_v2_x1_0,
    squeezenet1_0,
    vgg11,
    wide_resnet50_2,
    wide_resnet101_2,
)

from device import get_device

PYTORCH_MODELS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3,
    'alexnet': alexnet,
    'squeezenet': squeezenet1_0,
    'shufflenet': shufflenet_v2_x1_0,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'vgg11': vgg11,
    'mobilenet_v3_large': mobilenet_v3_large,
    'mobilenet_v3_small': mobilenet_v3_small,
}

TIMM_MODELS = {
    'inception_resnet_v2', 'inception_v4', 'efficientnet_b0', 'efficientnet_b1',
    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'vit_base_patch16_224',
}


def set_seed(seed: int, use_gpu: bool = True, print_out: bool = True) -> None:
    if print_out:
        print(f'Seed:\t {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_correct_per_class(batch_output, batch_y, d):
    predicted_class = torch.argmax(batch_output, dim=-1)
    for true_label, predicted_label in zip(batch_y, predicted_class):
        if true_label == predicted_label:
            d[true_label.item()] += 1


def update_correct_per_class_topk(batch_output, batch_y, d, k):
    topk_labels_pred = torch.argsort(batch_output, axis=-1, descending=True)[:, :k]
    for true_label, predicted_labels in zip(batch_y, topk_labels_pred):
        d[true_label.item()] += torch.sum(true_label == predicted_labels).item()


def count_correct_topk(scores, labels, k):
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a classification model by architecture name."""
    if arch in PYTORCH_MODELS and not pretrained:
        if arch == 'inception_v3':
            return PYTORCH_MODELS[arch](weights=None, num_classes=n_classes, aux_logits=False)
        return PYTORCH_MODELS[arch](weights=None, num_classes=n_classes)

    if arch in PYTORCH_MODELS and pretrained:
        weights = 'DEFAULT'
        if arch in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                    'wide_resnet50_2', 'wide_resnet101_2', 'shufflenet'}:
            model = PYTORCH_MODELS[arch](weights=weights)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif arch in {'alexnet', 'vgg11'}:
            model = PYTORCH_MODELS[arch](weights=weights)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_classes)
        elif arch in {'densenet121', 'densenet161', 'densenet169', 'densenet201'}:
            model = PYTORCH_MODELS[arch](weights=weights)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_classes)
        elif arch == 'mobilenet_v2':
            model = PYTORCH_MODELS[arch](weights=weights)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        elif arch == 'inception_v3':
            model = inception_v3(weights=weights, aux_logits=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif arch == 'squeezenet':
            model = PYTORCH_MODELS[arch](weights=weights)
            model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = n_classes
        elif arch in {'mobilenet_v3_large', 'mobilenet_v3_small'}:
            model = PYTORCH_MODELS[arch](weights=weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)
        else:
            raise NotImplementedError(arch)
        return model

    if arch in TIMM_MODELS:
        return timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)

    raise NotImplementedError(f"Unknown architecture: {arch}")


def get_model(args, n_classes: int) -> nn.Module:
    """Legacy helper: build model from argparse-style namespace."""
    return build_model(args.model, n_classes, pretrained=args.pretrained)


def model_info_path(checkpoint_path: str) -> str:
    base, _ = os.path.splitext(checkpoint_path)
    return f"{base}_info.json"


def weights_path(checkpoint_path: str) -> str:
    base, ext = os.path.splitext(checkpoint_path)
    return f"{base}.weights{ext or '.pth'}"


def save(
    model: nn.Module,
    optimizer,
    epoch: int,
    location: str,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Save training checkpoint with embedded metadata and sibling model_info.json."""
    dir_ = os.path.dirname(location)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_)

    meta = dict(meta or {})
    meta.setdefault('pytorch_version', torch.__version__)
    meta.setdefault('created_at', datetime.now(timezone.utc).isoformat())

    payload = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'meta': meta,
    }
    torch.save(payload, location)

    info_path = model_info_path(location)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({'epoch': epoch, 'meta': meta}, f, indent=2)

    # Inference-only bare weights
    torch.save(model.state_dict(), weights_path(location))


def load_checkpoint(
    model: nn.Module,
    filename: str,
    use_gpu: bool = True,
    load_optimizer: bool = False,
    optimizer=None,
) -> dict[str, Any]:
    """Load weights (and optionally optimizer) from checkpoint. Returns full payload."""
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    device = get_device(use_gpu)
    payload = torch.load(filename, map_location=device, weights_only=False)

    if isinstance(payload, dict) and 'model' in payload:
        model.load_state_dict(payload['model'])
    else:
        model.load_state_dict(payload)
        payload = {'model': payload, 'epoch': 0, 'meta': {}}

    if load_optimizer and optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])

    return payload


def load_model(model, filename, use_gpu):
    """Legacy: load model weights only, return epoch."""
    payload = load_checkpoint(model, filename, use_gpu=use_gpu)
    return payload.get('epoch', 0)


def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    device = get_device(use_gpu)
    d = torch.load(filename, map_location=device, weights_only=False)
    optimizer.load_state_dict(d['optimizer'])


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_schedule, epoch):
    if epoch in lr_schedule:
        optimizer = decay_lr(optimizer)
    return optimizer
