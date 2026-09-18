# 02 — Memory Transfers and Compute Efficiency on GPU

## Overview
Profiling study of two primary cost centers in GPU workloads: host-to-device
data transfer latency and Tensor Core utilization during matrix multiply. A
follow-up investigation traces why achieved throughput falls short of
theoretical peak and identifies the L2 cache boundary responsible.

## Motivation
GPU optimization discussions tend to focus on compute. In practice, data
movement and configuration gaps limit effective throughput before any
kernel-level work is relevant. Both are addressable without code changes —
making them the correct starting point before any deeper investigation.

## Methodology
Three experiments, each isolating a single variable:

1. **Memory transfer** — 200 MB host-to-device copy with pageable vs pinned
   allocation (`pin_memory=False/True`). 50 iterations, 10 warmup. Repeated
   at 1000 MB to rule out size effects.
2. **TF32 matmul** — 2048×2048 float32 square matmul with
   `torch.backends.cuda.matmul.allow_tf32` toggled. 50 iterations, 10 warmup.
3. **Matrix size scaling** — TF32-enabled square matmuls at five sizes (512 to
   8192) to identify where L2 cache capacity becomes the binding constraint.

## Results (NVIDIA L4, AWS g6.xlarge)

### Memory Transfer

| Method | Time | Bandwidth | Speedup |
|--------|------|-----------|---------|
| Pageable | 16.0 ms | 13.1 GB/s | 1x |
| Pinned | 15.6 ms | 13.5 GB/s | 1.03x |

B200 reference (bare metal): 5.39x speedup, pinned reaching 54.8 GB/s.

### TF32 Matrix Multiply

| Mode | Time | Throughput | Speedup |
|------|------|------------|---------|
| FP32 (TF32 off) | 1.183 ms | 14.5 TFLOPS | 1x |
| TF32 (TF32 on) | 0.480 ms | 35.8 TFLOPS | **2.46x** |

### TF32 Utilization vs Matrix Size

| Size | TFLOPS | % of Peak | Working Set |
|------|--------|-----------|-------------|
| 512×512 | 18.2 | 15% | 1 MB |
| 1024×1024 | 24.4 | 20% | 4 MB |
| 2048×2048 | **37.0** | **31%** | 16 MB |
| 4096×4096 | 26.3 | 22% | 64 MB |
| 8192×8192 | 26.4 | 22% | 256 MB |

L4 L2 cache: 48 MB. Performance cliff at 4096×4096 where working set exceeds cache.

## Key Findings
- Pinned memory delivered no meaningful speedup on this instance — AWS
  virtualization caps PCIe bandwidth before the staging copy overhead matters.
  The 5.4x B200 result confirms the optimization is real on bare metal.
- A single flag (`allow_tf32=True`) delivers 2.46x on a compute-bound matmul
  with no accuracy impact. This is the highest-leverage configuration change
  available before any kernel work.
- TF32 utilization peaks at 31% of theoretical peak at 2048×2048, then drops
  to 22% at 4096×4096 — not because of compute limits, but because the working
  set spills from L2 cache to VRAM. Tensor Cores stall waiting for data while
  nvidia-smi reports 100% utilization.
- The cache-capacity boundary is the same bottleneck pattern from experiment 01:
  fix launch overhead, expose memory bandwidth; enable Tensor Cores, expose
  cache capacity. Performance optimization is a chain.

## Environment
- GPU: NVIDIA L4 (Ada Lovelace, SM 8.9, 23 GB VRAM)
- Instance: AWS g6.xlarge (us-east-1c)
- PyTorch: 2.5.1+cu121 | CUDA Driver: 13.2
- Date: 2026-09-18

## Files
- `results/` — raw benchmark output
- `experiments/observations.md` — extended analysis and open questions
- `experiments/matrix_size_scaling/` — investigation script and results
