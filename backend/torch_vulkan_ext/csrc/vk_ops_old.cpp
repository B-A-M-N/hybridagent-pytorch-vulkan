#include "vk_backend.h"
#include "tvk_registry.h"
#include "torch_vulkan_shim.h"

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <cstring>
#include <vector>

// torch-vulkan registers PrivateUse1 as "vulkan"; importing it here also
// installs the backend name and the allocator.
#include <torch_vulkan_shim.h>

namespace tvk_ops {

namespace tvk = ::tvk;

static constexpr c10::DeviceType VK_DEV = c10::DeviceType::PrivateUse1;

inline bool on_vk(const at::Tensor& t) {
  return t.device().type() == VK_DEV;
}

// A PrivateUse1 tensor created by torch-vulkan's allocator is backed by a
// Kompute device tensor. The registry (populated by the patched torch-vulkan
// allocator) maps data_ptr() -> VkBuffer, which is what our dispatches bind.
static VkBuffer vk_buffer_of(const at::Tensor& t) {
  return tvk::Registry::instance().buffer(t.data_ptr());
}

// Allocate a fresh PrivateUse1 tensor with torch-vulkan's allocator.
static at::Tensor vk_empty(at::IntArrayRef sizes, at::ScalarType dtype) {
  auto opts = at::TensorOptions().dtype(dtype).device(VK_DEV, 0);
  return at::empty(sizes, opts);
}

// ---------------------------------------------------------------------------
// GELU (erf variant) -- matches torch.nn.GELU() default ("none").
// torch-vulkan's `gelu` kernel uses the tanh approximation, which diverges
// from HybridLink's nn.GELU() by up to ~1e-3 in absolute terms. HybridLink is
// the component under test, so we match PyTorch exactly here.
// ---------------------------------------------------------------------------
at::Tensor gelu_erf(const at::Tensor& self) {
  TORCH_CHECK(on_vk(self), "gelu_erf: tensor must be on the vulkan device");
  TORCH_CHECK(self.scalar_type() == at::kFloat,
              "gelu_erf: only float32 is supported");
  auto c = self.contiguous();
  uint32_t n = (uint32_t)c.numel();
  at::Tensor out = vk_empty(c.sizes(), at::kFloat);

  struct P {
    uint32_t n;
  } p{n};

  auto& be = tvk::Backend::instance();
  be.dispatch("gelu_erf", {vk_buffer_of(c), vk_buffer_of(out)}, {},
             &p, sizeof(p), (n + 255) / 256, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// RMSNorm -- matches Qwen3RMSNorm (fp32 statistics, weight applied in fp32).
// ---------------------------------------------------------------------------
at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight,
                   double eps) {
  TORCH_CHECK(on_vk(x) && on_vk(weight),
              "rmsnorm: tensors must be on the vulkan device");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "rmsnorm: only float32");

  auto xc = x.contiguous();
  auto wc = weight.contiguous();

  int64_t row_size = wc.numel();
  int64_t rows = xc.numel() / row_size;

  at::Tensor out = vk_empty(xc.sizes(), at::kFloat);

  struct P {
    uint32_t rows;
    uint32_t row_size;
    float eps;
  } p{(uint32_t)rows, (uint32_t)row_size, (float)eps};

  auto& be = tvk::Backend::instance();
  be.dispatch("rmsnorm", {vk_buffer_of(xc), vk_buffer_of(wc), vk_buffer_of(out)},
             {}, &p, sizeof(p), (uint32_t)rows, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// RoPE -- in-place on the head dimension, Qwen3 interleaved pairing.
//   x[2i], x[2i+1] rotated with theta_i = 1000000 ** (-2i / head_dim)
//   q_embed = q * cos + rotate_half(q) * sin
// Works on any tensor whose last dim is `head_dim`; `n_heads` heads are
// processed by `n_heads` workgroups, one per head.
// ---------------------------------------------------------------------------
at::Tensor rope(const at::Tensor& x, int64_t n_heads, int64_t head_dim,
                int64_t pos) {
  TORCH_CHECK(on_vk(x), "rope: tensor must be on the vulkan device");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "rope: only float32");
  auto c = x.contiguous().clone();   // in-place on the head dim
  TORCH_CHECK(c.size(-1) % head_dim == 0,
              "rope: last dim must be a multiple of head_dim");

  int64_t rows = c.numel() / c.size(-1);   // [tokens, heads] rows

  struct P {
    uint32_t offset;
    uint32_t head_dim;
    uint32_t pos;
  } p;

  auto& be = tvk::Backend::instance();
  // One dispatch per (row, head) so each workgroup owns one head slice.
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t h = 0; h < n_heads; h++) {
      p.offset = (uint32_t)((r * n_heads + h) * head_dim);
      p.head_dim = (uint32_t)head_dim;
      p.pos = (uint32_t)pos;
      be.dispatch("rope", {vk_buffer_of(c)}, {}, &p, sizeof(p),
                  (uint32_t)(head_dim / 2), 1, 1);
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// Fused SiLU gate: out[i] = silu(gate[i]) * up[i]
// gate_up is [.., 2*N] with gate first.
// ---------------------------------------------------------------------------
at::Tensor silu_gate(const at::Tensor& gate_up) {
  TORCH_CHECK(on_vk(gate_up), "silu_gate: tensor must be on the vulkan device");
  TORCH_CHECK(gate_up.scalar_type() == at::kFloat, "silu_gate: only float32");
  auto c = gate_up.contiguous();
  int64_t N = c.numel() / 2;
  TORCH_CHECK(c.numel() % 2 == 0, "silu_gate: last dim must be even");

  auto sizes = c.sizes().vec();
  sizes.back() = sizes.back() / 2;
  at::Tensor out = vk_empty(sizes, at::kFloat);

  struct P {
    uint32_t N;
  } p{(uint32_t)N};

  auto& be = tvk::Backend::instance();
  be.dispatch("silu_gate", {vk_buffer_of(c), vk_buffer_of(out)}, {}, &p,
            sizeof(p), (uint32_t)((N + 255) / 256), 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Broadcast add: c[i] = a[i] + alpha * b[bidx(i)]
// b may be a single element (global scale) or a row vector.
// ---------------------------------------------------------------------------
at::Tensor broadcast_add(const at::Tensor& a, const at::Tensor& b, double alpha) {
  TORCH_CHECK(on_vk(a) && on_vk(b), "broadcast_add: tensors must be on vulkan");
  auto ac = a.contiguous();
  auto bc = b.contiguous();
  uint32_t n = (uint32_t)ac.numel();
  uint32_t bn = (uint32_t)bc.numel();

  at::Tensor out = vk_empty(ac.sizes(), at::kFloat);

  struct P {
    uint32_t n;
    uint32_t b_size;
    float alpha;
  } p{n, bn, (float)alpha};

  auto& be = tvk::Backend::instance();
  be.dispatch("broadcast_add", {vk_buffer_of(ac), vk_buffer_of(bc), vk_buffer_of(out)},
             {}, &p, sizeof(p), (n + 255) / 256, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Residual add: c = a + b  (equal shapes)
// ---------------------------------------------------------------------------
at::Tensor residual_add(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(on_vk(a) && on_vk(b), "residual_add: tensors must be on vulkan");
  auto ac = a.contiguous();
  auto bc = b.contiguous();
  uint32_t n = (uint32_t)ac.numel();
  TORCH_CHECK(n == (uint32_t)bc.numel(), "residual_add: shape mismatch");

  at::Tensor out = vk_empty(ac.sizes(), at::kFloat);

  struct P {
    uint32_t n;
  } p{n};

  auto& be = tvk::Backend::instance();
  be.dispatch("residual_add", {vk_buffer_of(ac), vk_buffer_of(bc), vk_buffer_of(out)},
             {}, &p, sizeof(p), (n + 255) / 256, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Slice: out[i] = in[offset + i]
// ---------------------------------------------------------------------------
at::Tensor slice(const at::Tensor& x, int64_t offset, int64_t count) {
  TORCH_CHECK(on_vk(x), "slice: tensor must be on the vulkan device");
  auto c = x.contiguous();
  auto sizes = c.sizes().vec();
  sizes.back() = count;
  at::Tensor out = vk_empty(sizes, at::kFloat);

  struct P {
    uint32_t count;
    uint32_t offset;
  } p{(uint32_t)count, (uint32_t)offset};

  auto& be = tvk::Backend::instance();
  be.dispatch("slice", {vk_buffer_of(c), vk_buffer_of(out)}, {}, &p, sizeof(p),
            (uint32_t)((count + 255) / 256), 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Gather: out[i] = in[offset + idx[i]]
// ---------------------------------------------------------------------------
at::Tensor gather(const at::Tensor& x, const at::Tensor& idx, int64_t offset) {
  TORCH_CHECK(on_vk(x) && on_vk(idx), "gather: tensors must be on vulkan");
  auto c = x.contiguous();
  auto ic = idx.contiguous();
  auto sizes = ic.sizes().vec();
  at::Tensor out = vk_empty(sizes, at::kFloat);

  struct P {
    uint32_t count;
    uint32_t offset;
  } p{(uint32_t)ic.numel(), (uint32_t)offset};

  auto& be = tvk::Backend::instance();
  be.dispatch("gather", {vk_buffer_of(c), vk_buffer_of(ic), vk_buffer_of(out)}, {},
            &p, sizeof(p), (uint32_t)((ic.numel() + 255) / 256), 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// KV store: write K and V into the persistent cache at slot `pos`.
//   kv:     [kv_len * 2]  (K first, then V), fp32
//   k_cache/v_cache: [n_kv_heads, cap, head_dim]
// ---------------------------------------------------------------------------
void kv_store(const at::Tensor& kv, at::Tensor& k_cache, at::Tensor& v_cache,
              int64_t pos, int64_t head_dim, int64_t kv_len) {
  TORCH_CHECK(on_vk(kv) && on_vk(k_cache) && on_vk(v_cache),
              "kv_store: tensors must be on vulkan");
  auto kvc = kv.contiguous();
  auto kc = k_cache.contiguous();
  auto vc = v_cache.contiguous();
  int64_t cap = kc.size(1);

  struct P {
    uint32_t pos;
    uint32_t cap;
    uint32_t head_dim;
    uint32_t kv_len;
  } p{(uint32_t)pos, (uint32_t)cap, (uint32_t)head_dim, (uint32_t)kv_len};

  auto& be = tvk::Backend::instance();
  be.dispatch("kv_store", {vk_buffer_of(kvc), vk_buffer_of(kc), vk_buffer_of(vc)},
             {}, &p, sizeof(p), (uint32_t)((2 * kv_len + 255) / 256), 1, 1);
}

// ---------------------------------------------------------------------------
// Single-token causal attention over the persistent KV cache (GQA aware).
//   q:      [n_heads, head_dim]
//   k_cache/v_cache: [n_kv_heads, cap, head_dim]
//   mask:   [n_heads, new_len]  (0 = attend, -inf = skip)
//   out:    [n_heads, head_dim]
// ---------------------------------------------------------------------------
at::Tensor attention(const at::Tensor& q, const at::Tensor& k_cache,
                     const at::Tensor& v_cache, const at::Tensor& mask,
                     int64_t pos, double scaling, int64_t n_heads,
                     int64_t n_kv_heads, int64_t head_dim) {
  TORCH_CHECK(on_vk(q) && on_vk(k_cache) && on_vk(v_cache) && on_vk(mask),
              "attention: tensors must be on the vulkan device");
  auto qc = q.contiguous();
  auto kc = k_cache.contiguous();
  auto vc = v_cache.contiguous();
  auto mc = mask.contiguous();
  TORCH_CHECK(head_dim == 128, "attention: head_dim must be 128 (workgroup size)");

  int64_t new_len = pos + 1;
  auto sizes = qc.sizes().vec();
  at::Tensor out = vk_empty(sizes, at::kFloat);

  struct P {
    uint32_t pos;
    uint32_t new_len;
    uint32_t n_heads;
    uint32_t heads_per_kv;
    uint32_t head_dim;
    float scaling;
  } p;
  p.pos = (uint32_t)pos;
  p.new_len = (uint32_t)new_len;
  p.n_heads = (uint32_t)n_heads;
  p.heads_per_kv = (uint32_t)(n_heads / n_kv_heads);
  p.head_dim = (uint32_t)head_dim;
  p.scaling = (float)scaling;

  auto& be = tvk::Backend::instance();
  be.dispatch("attention",
            {vk_buffer_of(qc), vk_buffer_of(kc), vk_buffer_of(vc),
             vk_buffer_of(mc), vk_buffer_of(out)},
            {}, &p, sizeof(p), (uint32_t)n_heads, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Row-wise argmax over a [rows, cols] matrix.
// ---------------------------------------------------------------------------
at::Tensor argmax(const at::Tensor& x) {
  TORCH_CHECK(on_vk(x), "argmax: tensor must be on the vulkan device");
  auto c = x.contiguous();
  TORCH_CHECK(c.dim() == 2, "argmax: expects a 2D tensor");
  int64_t rows = c.size(0), cols = c.size(1);

  at::Tensor idx = vk_empty({rows}, at::kFloat);
  at::Tensor val = vk_empty({rows}, at::kFloat);
  at::Tensor out = vk_empty({1}, at::kLong);

  struct P {
    uint32_t rows;
    uint32_t cols;
  } p{(uint32_t)rows, (uint32_t)cols};

  auto& be = tvk::Backend::instance();
  be.dispatch("argmax", {vk_buffer_of(c), vk_buffer_of(idx), vk_buffer_of(val),
                       vk_buffer_of(out)},
            {}, &p, sizeof(p), 1, 1, 1);
  return out;
}

// ---------------------------------------------------------------------------
// Linear: out = x @ weight.T + bias  (weight is [out, in], as in nn.Linear)
// Implemented on top of torch-vulkan's resident `mm`, then a broadcast add.
// ---------------------------------------------------------------------------
at::Tensor linear(const at::Tensor& x, const at::Tensor& weight,
                  const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(on_vk(x) && on_vk(weight), "linear: tensors must be on vulkan");
  auto wc = weight.contiguous();
  auto xc = x.contiguous();
  // flatten leading dims into rows
  int64_t K = xc.size(-1);
  int64_t M = xc.numel() / K;
  auto x2 = xc.view({M, K});
  auto w2 = wc.view({wc.size(0), wc.size(1)});

  // torch-vulkan's mm takes the weight as [K, N]; nn.Linear stores [N, K],
  // so transpose to [K, N]. The transpose is a CPU-side metadata op that
  // torch-vulkan implements as a copy; it is done once at load time, not per
  // decode step.
  auto w_t = w2.t().contiguous();
  auto prod = at::mm(x2, w_t);   // [M, N]

  if (bias.has_value() && bias->defined()) {
    auto b = bias->contiguous();
    // broadcast over rows
    auto b_row = b.view({1, b.numel()});
    return broadcast_add(prod, b_row, 1.0).view(xc.sizes().vec());
  }
  return prod.view(xc.sizes().vec());
}

}  // namespace tvk_ops

TORCH_LIBRARY(torch_vulkan_ext, m) {
  m.def("gelu_erf(Tensor self) -> Tensor");
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
  m.def("rope(Tensor x, int n_heads, int head_dim, int pos) -> Tensor");
  m.def("silu_gate(Tensor gate_up) -> Tensor");
  m.def("broadcast_add(Tensor a, Tensor b, float alpha) -> Tensor");
  m.def("residual_add(Tensor a, Tensor b) -> Tensor");
  m.def("slice(Tensor x, int offset, int count) -> Tensor");
  m.def("gather(Tensor x, Tensor idx, int offset) -> Tensor");
  m.def("kv_store(Tensor kv, Tensor k_cache, Tensor v_cache, int pos, int head_dim, int kv_len) -> ()");
  m.def("attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor mask, int pos, float scaling, int n_heads, int n_kv_heads, int head_dim) -> Tensor");
  m.def("argmax(Tensor x) -> Tensor");
  m.def("linear(Tensor x, Tensor weight, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_vulkan_ext, PrivateUse1, m) {
  m.impl("gelu_erf", &tvk_ops::gelu_erf);
  m.impl("rmsnorm", &tvk_ops::rmsnorm);
  m.impl("rope", &tvk_ops::rope);
  m.impl("silu_gate", &tvk_ops::silu_gate);
  m.impl("broadcast_add", &tvk_ops::broadcast_add);
  m.impl("residual_add", &tvk_ops::residual_add);
  m.impl("slice", &tvk_ops::slice);
  m.impl("gather", &tvk_ops::gather);
  m.impl("kv_store", &tvk_ops::kv_store);
  m.impl("attention", &tvk_ops::attention);
  m.impl("argmax", &tvk_ops::argmax);
  m.impl("linear", &tvk_ops::linear);
}

// ---------------------------------------------------------------------------
// Python-facing setup + diagnostics
// ---------------------------------------------------------------------------
bool py_init(int64_t device_index) {
  return tvk::Backend::instance().init((uint32_t)device_index);
}

std::string py_device_name() { return tvk::Backend::instance().device_name(); }

void py_set_shader_dir(const std::string& path) {
  tvk::Backend::instance().set_shader_dir(path);
}

int64_t py_dispatch_count() { return (int64_t)tvk::Backend::instance().dispatch_count(); }
void py_reset_timing() { tvk::Backend::instance().reset_timing(); }
int64_t py_vram_total() { return (int64_t)tvk::Backend::instance().vram_total(); }
int64_t py_vram_used() { return (int64_t)tvk::Backend::instance().vram_used(); }

PYBIND11_MODULE(_C, m) {
  m.def("init", &py_init, "Create the Vulkan instance/device");
  m.def("device_name", &py_device_name);
  m.def("set_shader_dir", &py_set_shader_dir);
  m.def("dispatch_count", &py_dispatch_count);
  m.def("reset_timing", &py_reset_timing);
  m.def("vram_total", &py_vram_total);
  m.def("vram_used", &py_vram_used);
}
