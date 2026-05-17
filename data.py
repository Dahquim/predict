"""Dataset registry and loaders for image classification."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np
import scipy.io
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int, crop_size: int, normalize: dict) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])


def build_eval_transform(image_size: int, crop_size: int, normalize: dict) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])


def default_normalize() -> dict:
    return {'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}


def detect_dataset(root: str) -> str:
    """Auto-detect dataset layout from directory contents."""
    if all(os.path.exists(os.path.join(root, f)) for f in ('jpg', 'imagelabels.mat', 'setid.mat')):
        return 'oxford102'
    if os.path.isdir(os.path.join(root, 'train')):
        return 'imagefolder'
    raise ValueError(
        f"Unable to auto-detect dataset in {root}. "
        "Expected Oxford 102 (jpg/, imagelabels.mat, setid.mat) or ImageFolder (train/)."
    )


class Oxford102Dataset:
    """Oxford 102 Category Flower Dataset (official .mat splits)."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        swap_train_test: bool = True,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.swap_train_test = swap_train_test
        self.img_dir = os.path.join(root, 'jpg')
        self.labels_path = os.path.join(root, 'imagelabels.mat')
        self.splits_path = os.path.join(root, 'setid.mat')
        self.class_names_path = os.path.join(root, 'class_names.txt')

        self.img_files: list[str] = []
        self._label_list: list[int] = []

        self._load_data()
        self.labels = np.array(self._label_list, dtype=np.int64)
        self.classes = self._load_class_names()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    @property
    def targets(self) -> list[int]:
        return self.labels.tolist()

    def _resolve_split_key(self) -> str:
        """Map logical split to setid.mat key, optionally swapping train/test sizes."""
        if not self.swap_train_test:
            return {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}[self.split]

        # Swapped: train on official tstid (~6149), val on valid, test on trnid
        return {'train': 'tstid', 'val': 'valid', 'test': 'trnid'}[self.split]

    def _load_data(self) -> None:
        labels_data = scipy.io.loadmat(self.labels_path)
        labels = labels_data['labels'].flatten()

        splits_data = scipy.io.loadmat(self.splits_path)
        key = self._resolve_split_key()
        setid = splits_data[key][0].flatten()

        for idx in setid:
            img_id = int(idx) if hasattr(idx, 'item') else int(idx)
            self.img_files.append(f'image_{img_id:05d}.jpg')
            self._label_list.append(int(labels[img_id - 1]) - 1)

    def _load_class_names(self) -> list[str]:
        repo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'datasets', 'oxford102', 'class_names.txt',
        )
        for path in (repo_path, self.class_names_path):
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    return [line.split(':', 1)[1].strip() for line in lines if ':' in line]
        raise FileNotFoundError(
            f"class_names.txt not found under {self.root} or datasets/oxford102/"
        )

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageFolderSplit(ImageFolder):
    """ImageFolder for train/val/test subdirectories under root."""

    def __init__(self, root: str, split: str, **kwargs):
        self.dataset_root = root
        self.split = split
        super().__init__(os.path.join(root, split), **kwargs)


def _build_oxford102_loaders(
    root: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    normalize: dict,
    swap_train_test: bool = True,
    **_,
) -> tuple:
    transform_train = build_train_transform(image_size, crop_size, normalize)
    transform_eval = build_eval_transform(image_size, crop_size, normalize)

    trainset = Oxford102Dataset(root, 'train', transform_train, swap_train_test=swap_train_test)
    valset = Oxford102Dataset(root, 'val', transform_eval, swap_train_test=swap_train_test)
    testset = Oxford102Dataset(root, 'test', transform_eval, swap_train_test=swap_train_test)

    split_note = 'swapped' if swap_train_test else 'official'
    print(
        f"Oxford102 ({split_note} splits): train={len(trainset)} val={len(valset)} test={len(testset)}"
    )

    return _make_loaders(trainset, valset, testset, batch_size, num_workers, 'oxford102', swap_train_test)


def _build_imagefolder_loaders(
    root: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    normalize: dict,
    **_,
) -> tuple:
    transform_train = build_train_transform(image_size, crop_size, normalize)
    transform_eval = build_eval_transform(image_size, crop_size, normalize)

    trainset = ImageFolderSplit(root, 'train', transform=transform_train)
    valset = ImageFolderSplit(root, 'val', transform=transform_eval)
    testset = ImageFolderSplit(root, 'test', transform=transform_eval)

    print(f"ImageFolder: train={len(trainset)} val={len(valset)} test={len(testset)}")

    return _make_loaders(trainset, valset, testset, batch_size, num_workers, 'imagefolder', None)


def _make_loaders(trainset, valset, testset, batch_size, num_workers, dataset_type, split_strategy):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    n_classes = len(trainset.classes)
    dataset_attributes = {
        'dataset_type': dataset_type,
        'split_strategy': split_strategy,
        'n_train': len(trainset),
        'n_val': len(valset),
        'n_test': len(testset),
        'n_classes': n_classes,
        'class_names': list(trainset.classes),
        'class2num_instances': {
            'train': Counter(trainset.targets),
            'val': Counter(valset.targets),
            'test': Counter(testset.targets),
        },
        'class_to_idx': trainset.class_to_idx,
        'normalize': None,  # filled by get_data
    }
    return trainloader, valloader, testloader, dataset_attributes


DATASETS: dict[str, Callable] = {
    'oxford102': _build_oxford102_loaders,
    'imagefolder': _build_imagefolder_loaders,
}


def get_data(
    root: str,
    image_size: int = 256,
    crop_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset: Optional[str] = None,
    normalize: Optional[dict] = None,
    swap_train_test: bool = True,
    **kwargs,
) -> tuple:
    """Build train/val/test dataloaders. Auto-detects dataset unless `dataset` is set."""
    name = dataset or detect_dataset(root)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(DATASETS.keys())}")

    norm = normalize or default_normalize()
    trainloader, valloader, testloader, attrs = DATASETS[name](
        root=root,
        image_size=image_size,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=norm,
        swap_train_test=swap_train_test,
        **kwargs,
    )
    attrs['normalize'] = norm
    attrs['dataset'] = name
    attrs['image_size'] = image_size
    attrs['crop_size'] = crop_size
    return trainloader, valloader, testloader, attrs
