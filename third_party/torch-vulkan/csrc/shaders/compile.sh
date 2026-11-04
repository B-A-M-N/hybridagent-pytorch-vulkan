#!/bin/bash
# Compile all GLSL compute shaders to SPIR-V
# Requires: glslangValidator (from Vulkan SDK or `pacman -S glslang`)
#
# We target Vulkan 1.1, which produces SPIR-V 1.3. Several of the quantized
# matmul kernels (matmul_q*k*, matmul_gpuq*) use GL_KHR_shader_subgroup
# reductions, and those subgroup ops require SPIR-V >= 1.3. The default
# `glslangValidator -V` emits SPIR-V 1.0 and fails on them, so the explicit
# --target-env is required to compile the full shader set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

fail=0
for comp in *.comp; do
    spv="${comp%.comp}.spv"
    echo "Compiling $comp -> $spv"
    if ! glslangValidator --target-env vulkan1.1 -V "$comp" -o "$spv"; then
        echo "  ERROR: failed to compile $comp" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "One or more shaders failed to compile." >&2
    exit 1
fi

echo "Done. $(ls *.spv | wc -l) shaders compiled."
