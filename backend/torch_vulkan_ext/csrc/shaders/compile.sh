#!/bin/bash
# Compile every GLSL compute shader in this directory to SPIR-V.
#
# Targets Vulkan 1.1 (SPIR-V 1.3) because some torch-vulkan kernels use
# GL_KHR_shader_subgroup reductions, and because Kompute 0.9.0 requests
# SPIR-V >= 1.3. `glslangValidator -V` alone would emit SPIR-V 1.0.
set -euo pipefail
cd "$(dirname "$0")"
fail=0
for comp in *.comp; do
    spv="${comp%.comp}.spv"
    if glslangValidator --target-env vulkan1.1 -V "$comp" -o "$spv" 2>/tmp/spv_err.txt; then
        echo "ok   $comp"
    else
        echo "FAIL $comp"; cat /tmp/spv_err.txt; fail=1
    fi
done
if [ "$fail" -ne 0 ]; then echo "one or more shaders failed" >&2; exit 1; fi
echo "compiled $(ls *.spv | wc -l) shaders"
