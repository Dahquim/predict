# predict — Flower / plant image classification

Train and serve image classifiers with a pluggable dataset registry. **Phase 1** targets the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset; **Phase 2** can add PlantNet-300K or any `train/val/test` folder layout.

## Project layout

```text
datasets/oxford102/
  jpg/                  # 8189 images
  imagelabels.mat
  setid.mat
  class_names.txt
  model/
    resnet152_oxford102.pth      # training checkpoint (weights + optimizer + meta)
    resnet152_oxford102_info.json
    resnet152_oxford102.weights.pth
    metrics.json
```

## Environment setup

### Check GPU

```bash
python check_env.py
```

You want `torch.cuda.is_available()` → **True**. On ROCm builds the version string contains **`rocm`** (PyTorch still uses the `torch.cuda` API on AMD).

### AMD GPU (e.g. RX 7900 XTX on CachyOS/Arch)

System ROCm should be installed (`rocm-smi` works). Use a venv with **Python 3.11–3.14** (your system **3.14** is fine) and ROCm wheels from PyTorch — avoid `pacman -S python-pytorch-rocm` on RDNA3 if you hit GPU hangs.

```bash
cd /path/to/predict

# Automated (bash script — do not run with python):
# Picks python3.14, then python3, then 3.13/3.12/3.11
bash ./scripts/setup_rocm_venv.sh
```

Or manually (example with system Python 3.14):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# Option A: PyTorch.org ROCm index
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Option B: AMD wheels matched to ROCm 7.2 — see
# https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html

pip install -r requirements.txt
python check_env.py
```

Optional:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### NVIDIA GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Training (Oxford 102)

Default: **swapped splits** (train ~6149, val ~1020, test ~1020). ImageNet-pretrained ResNet152, AMP on GPU.

```bash
source .venv/bin/activate
python train.py \
  --data-root datasets/oxford102 \
  --epochs 30 \
  --batch-size 32 \
  --checkpoint-out datasets/oxford102/model/resnet152_oxford102.pth
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--dataset oxford102` | Force dataset (else auto-detect) |
| `--model resnet50` | Smaller / faster model |
| `--cpu` | Train on CPU |
| `--no-amp` | Disable mixed precision |
| `--no-swap-splits` | Official paper splits |

## Inference

### CLI

```bash
python predict.py datasets/oxford102/jpg/image_00001.jpg --checkpoint datasets/oxford102/model/resnet152_oxford102.pth
```

### Flask API

```bash
export CHECKPOINT=datasets/oxford102/model/resnet152_oxford102.pth
export USE_GPU=true   # if ROCm/CUDA available
python api.py
```

`POST /predict` with JSON `{"image": "<base64>"}`.

Checkpoints are **self-describing**: `meta` in the `.pth` and sibling `*_info.json` hold `arch`, `num_classes`, `class_names`, and normalization — swap in a PlantNet model later without code changes.

## Dataset registry

| Name | Layout |
|------|--------|
| `oxford102` (auto) | `jpg/`, `imagelabels.mat`, `setid.mat` |
| `imagefolder` (auto) | `train/`, `val/`, `test/` with class subfolders |

```python
from data import get_data, detect_dataset
print(detect_dataset('datasets/oxford102'))  # oxford102
```

## Docker

The included `Dockerfile` targets CPU inference. GPU serving with ROCm in Docker is not configured in this pass.

## Phase 2 (not implemented)

- Register `plantnet300k` in `data.DATASETS`
- Long-tail sampling / class-balanced loss for 1000+ classes
- Open-set “unknown plant” rejection
