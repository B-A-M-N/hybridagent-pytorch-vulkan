#!/usr/bin/env bash
# Build and install everything needed for HybridAgent-on-Vulkan.
#
#   bash scripts/install.sh
#
# Creates the project venv, installs the pinned Python stack, compiles the
# SPIR-V shaders, builds the raw-Vulkan backend (tvkvk.so, the only native
# extension the canonical path uses), and installs the Python packages.
# Kompute + torch-vulkan are LEGACY (unused by the canonical path) and are
# built only if their sources are complete; the build warns and skips them
# otherwise instead of failing. Does not touch system packages or GPU drivers.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "=== 1/6 venv ==="
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
fi

echo "=== 2/6 python stack ==="
.venv/bin/pip install -q \
  "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q \
  "transformers==5.13.0" "accelerate==1.14.0" "safetensors==0.8.0" \
  "tokenizers==0.22.2" "huggingface-hub==1.22.0" "numpy" "tqdm"

echo "=== 3/6 shaders ==="
bash backend/torch_vulkan_ext/csrc/shaders/compile.sh

echo "=== 4/6 raw-Vulkan backend (tvkvk.so) ==="
cmake -S backend/torch_vulkan_ext -B backend/torch_vulkan_ext/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build backend/torch_vulkan_ext/build -j"$(nproc)"

echo "=== 5/6 legacy Kompute/torch-vulkan (optional) ==="
if [ -f backend/torch_vulkan_ext/csrc/vk_ops.cpp ] && [ -f backend/torch_vulkan_ext/csrc/tvk_registry.cpp ]; then
  echo "legacy sources complete; building Kompute v0.9.0 + torch-vulkan"
  if [ ! -f third_party/kompute_install/lib/libkompute.a ]; then
    if [ ! -d third_party/kompute ]; then
      git clone --depth 1 --branch v0.9.0 https://github.com/KomputeProject/kompute.git \
        third_party/kompute
    fi
    cmake -S third_party/kompute -B third_party/kompute/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DKOMPUTE_OPT_BUILD_TESTS=OFF -DKOMPUTE_OPT_BUILD_DOCS=OFF \
      -DKOMPUTE_OPT_BUILD_PYTHON=OFF -DKOMPUTE_OPT_INSTALL=ON \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC" \
      -DCMAKE_INSTALL_PREFIX="$ROOT/third_party/kompute_install"
    cmake --build third_party/kompute/build -j"$(nproc)"
    cmake --install third_party/kompute/build
  fi
  VENV_TORCH="$(pwd)/.venv/lib/python3.11/site-packages/torch"
  cmake -S third_party/torch-vulkan -B third_party/torch-vulkan/build \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$VENV_TORCH/share/cmake"
  cmake --build third_party/torch-vulkan/build -j"$(nproc)"
  .venv/bin/pip install -q -e third_party/torch-vulkan --no-build-isolation
else
  echo "WARNING: legacy torch-vulkan sources incomplete (vk_ops.cpp /"
  echo "tvk_registry.cpp missing); skipping Kompute/torch-vulkan build."
  echo "The canonical path uses only torch_vulkan_ext (raw Vulkan) and is unaffected."
fi

echo "=== 6/6 python packages ==="
.venv/bin/pip install -q -e backend/torch_vulkan_ext --no-build-isolation
.venv/bin/pip install -q -e third_party/HybridAgent --no-build-isolation
.venv/bin/pip install -q -e . --no-build-isolation

echo
echo "=== verify ==="
.venv/bin/python -c "
import torch, torch_vulkan_ext
torch_vulkan_ext.init(0)
print('torch        ', torch.__version__)
print('vulkan device', torch_vulkan_ext.device_name())
print('vram total   ', round(torch_vulkan_ext.vram_total()/1e9, 2), 'GB')
"
echo "install complete"
