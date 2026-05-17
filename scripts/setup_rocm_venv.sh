#!/usr/bin/env bash
# Recreate .venv with ROCm PyTorch (AMD RX 7900 XTX etc.)
# Run with: bash ./scripts/setup_rocm_venv.sh  (NOT python)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON=""
for candidate in python3.14 python3 python3.13 python3.12 python3.11; do
  if command -v "$candidate" &>/dev/null; then
    # Need 3.10+ for current PyTorch
    major_minor=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$major_minor" in
      3.10|3.11|3.12|3.13|3.14) PYTHON="$candidate"; break ;;
    esac
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "No suitable Python found (need 3.10–3.14)."
  echo "On Arch/CachyOS: sudo pacman -S python"
  exit 1
fi

echo "Using $PYTHON ($("$PYTHON" --version))"

echo "Removing old .venv (backup first if needed)..."
rm -rf .venv

"$PYTHON" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -U pip wheel

echo "Installing ROCm PyTorch (pytorch.org rocm6.3 index)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

pip install -r requirements.txt

python check_env.py
echo "Done. Activate with: source .venv/bin/activate"
