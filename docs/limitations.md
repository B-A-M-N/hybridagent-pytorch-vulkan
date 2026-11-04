# Limitations (summary)

Full evidence: RELEASE_QUALIFICATION.md. In brief:
- Experimental research implementation, not a production runtime.
- CPU-only is fastest for the measured single-token latent workload
  (~403 ms/step); Vulkan hetero is functional but not performance-leading.
- Only Ryzen 9 5900X + RX 6600 XT tested; no other GPU/CPU claimed.
- `_int_mm` activation quantization ~1% approximate vs FP32 dequant.
- 20 GPU layers hangs (FP32 VRAM ceiling); Q4 path unqualified.
- Harness RAM footprint 18-20 GB; checkpoints not distributed.
- Legacy Kompute/torch-vulkan build skipped (lost sources).
