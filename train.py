#!/usr/bin/env python3
"""Train an image classifier on Oxford 102 or ImageFolder datasets."""

from __future__ import annotations

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from data import default_normalize, get_data
from device import get_device, use_gpu_enabled
from model import (
    build_model,
    count_correct_topk,
    model_info_path,
    save,
    set_seed,
    weights_path,
)


def parse_args():
    p = argparse.ArgumentParser(description='Train flower/plant classifier')
    p.add_argument('--data-root', default='datasets/oxford102')
    p.add_argument('--dataset', default=None, help='Dataset name (auto-detect if omitted)')
    p.add_argument('--model', default='resnet152', dest='model_arch')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--crop-size', type=int, default=224)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--no-pretrained', action='store_false', dest='pretrained')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-swap-splits', action='store_true',
                   help='Use official Oxford splits (Oxford102 only)')
    p.add_argument('--checkpoint-out',
                   default='datasets/oxford102/model/resnet152_oxford102.pth')
    p.add_argument('--lr-step', type=int, default=10)
    p.add_argument('--lr-gamma', type=float, default=0.1)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int, use_amp: bool) -> dict:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss_sum += loss.item() * labels.size(0)
        correct1 += count_correct_topk(outputs.float(), labels, 1).item()
        correct5 += count_correct_topk(outputs.float(), labels, min(5, n_classes)).item()
        total += labels.size(0)

    return {
        'loss': loss_sum / max(total, 1),
        'top1': correct1 / max(total, 1),
        'top5': correct5 / max(total, 1),
        'total': total,
    }


def train_one_epoch(model, loader, optimizer, device, use_amp: bool, scaler) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return running_loss / max(n, 1)


def build_meta(args, attrs: dict, best_val_top1: float) -> dict:
    split_strategy = attrs.get('split_strategy')
    if split_strategy is None:
        split_strategy = 'official'
    elif split_strategy is True:
        split_strategy = 'swap'
    return {
        'arch': args.model_arch,
        'num_classes': attrs['n_classes'],
        'class_names': attrs['class_names'],
        'image_size': args.image_size,
        'crop_size': args.crop_size,
        'normalize': attrs['normalize'],
        'dataset': attrs['dataset'],
        'split_strategy': split_strategy,
        'best_val_top1': best_val_top1,
        'pretrained_init': args.pretrained,
    }


def main():
    args = parse_args()
    use_gpu = use_gpu_enabled(not args.cpu)
    device = get_device(use_gpu)
    use_amp = use_gpu and not args.no_amp

    print(f'Device: {device}  AMP: {use_amp}')
    set_seed(args.seed, use_gpu=use_gpu)

    normalize = default_normalize()
    trainloader, valloader, testloader, attrs = get_data(
        root=args.data_root,
        image_size=args.image_size,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
        normalize=normalize,
        swap_train_test=not args.no_swap_splits,
    )

    n_classes = attrs['n_classes']
    model = build_model(args.model_arch, n_classes, pretrained=args.pretrained)
    model = model.to(device)

    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_top1 = 0.0
    best_epoch = 0
    os.makedirs(os.path.dirname(args.checkpoint_out) or '.', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, trainloader, optimizer, device, use_amp, scaler)
        val_metrics = evaluate(model, valloader, device, n_classes, use_amp)
        scheduler.step()

        print(
            f'Epoch {epoch}/{args.epochs}  '
            f'train_loss={train_loss:.4f}  val_loss={val_metrics["loss"]:.4f}  '
            f'val_top1={val_metrics["top1"]:.4f}  val_top5={val_metrics["top5"]:.4f}  '
            f'({time.time() - t0:.1f}s)'
        )

        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            best_epoch = epoch
            meta = build_meta(args, attrs, best_val_top1)
            meta['best_val_top5'] = val_metrics['top5']
            save(model, optimizer, epoch, args.checkpoint_out, meta=meta)
            print(f'  -> saved best checkpoint to {args.checkpoint_out}')

    print(f'\nBest val top-1: {best_val_top1:.4f} at epoch {best_epoch}')

    # Reload best and evaluate test split
    if os.path.exists(args.checkpoint_out):
        ckpt = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, testloader, device, n_classes, use_amp)
    print(
        f'Test top-1: {test_metrics["top1"]:.4f}  '
        f'test top-5: {test_metrics["top5"]:.4f}'
    )

    metrics = {
        'best_epoch': best_epoch,
        'val_top1': best_val_top1,
        'test_top1': test_metrics['top1'],
        'test_top5': test_metrics['top5'],
        'model': args.model_arch,
        'dataset': attrs['dataset'],
        'checkpoint': args.checkpoint_out,
        'weights_only': weights_path(args.checkpoint_out),
        'model_info': model_info_path(args.checkpoint_out),
    }
    metrics_path = os.path.join(
        os.path.dirname(args.checkpoint_out) or '.', 'metrics.json',
    )
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics written to {metrics_path}')


if __name__ == '__main__':
    main()
