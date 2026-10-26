// Minimal SPIR-V reflection: report how many storage-buffer descriptor bindings
// a shader declares.
//
// Why this exists: the host must declare exactly as many descriptor bindings as
// the shader reads. Declaring MORE than the shader has is harmless; declaring
// FEWER, or allocating a descriptor pool sized for a different count, produces
// undefined behaviour on RADV rather than a clean error -- it silently
// corrupted two kernels during development. This helper makes the host ask the
// shader instead of guessing.
//
// Parsing is two-pass over the instruction stream:
//   pass 1 collects OpEntryPoint interface variable ids
//   pass 2 collects OpDecorate bindings on those variables
// A single pass does not work: SPIR-V emits decorations before the entry point.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace tvk {

enum : uint32_t {
  SpvOpNop = 0,
  SpvOpDecorate = 71,
  SpvOpEntryPoint = 15,
  SpvDecorationBinding = 33,
  SpvDecorationDescriptorSet = 34,
};

inline int spirv_max_binding(const std::vector<uint32_t>& spv) {
  if (spv.size() < 5) return 0;
  const uint32_t* p = spv.data();
  const size_t n = spv.size();
  auto u32 = [&](size_t i) -> uint32_t { return (i < n) ? p[i] : 0u; };

  // ---- pass 1: entry point interface variables ----
  std::unordered_set<uint32_t> iface;
  size_t i = 5;
  while (i < n) {
    uint32_t inst = u32(i);
    uint32_t wc = inst >> 16;
    uint32_t op = inst & 0xFFFFu;
    if (wc == 0) break;
    if (op == SpvOpEntryPoint) {
      for (uint32_t k = 3; k < wc; k++) iface.insert(u32(i + k));
    }
    i += wc;
  }

  // ---- pass 2: Binding decorations on those variables ----
  int maxb = -1;
  i = 5;
  while (i < n) {
    uint32_t inst = u32(i);
    uint32_t wc = inst >> 16;
    uint32_t op = inst & 0xFFFFu;
    if (wc == 0) break;
    if (op == SpvOpDecorate && wc >= 4) {
      uint32_t target = u32(i + 1);
      uint32_t dec = u32(i + 2);
      if (dec == SpvDecorationBinding && iface.count(target)) {
        maxb = std::max(maxb, (int)u32(i + 3));
      }
    }
    i += wc;
  }
  return maxb + 1;   // binding indices are 0-based
}

}  // namespace tvk
