# 01 — Kernel Launch Overhead in GPU Matrix Multiplication

## Overview
Profiling study of GPU kernel launch overhead in batched GEMM (General Matrix
Multiply) operations. The experiment quantifies how much runtime is consumed by
CPU-GPU dispatch versus actual compute, and evaluates two batching strategies
to amortize that cost.

## Motivation
In deep learning workloads, small matrix operations are issued repeatedly in
tight loops. A common assumption is that GPU utilization is the bottleneck.
This experiment challenges that assumption by isolating launch overhead as the
primary cost driver for small, repeated operations.

## Methodology
Three implementations of 40 GEMMs (M=32, N=256, K=256) were benchmarked:

1. **Individual launches** — one `cublasSgemm` call per matrix (40 kernel launches)
2. **Batched** — single `cublasSgemmBatched` call (1 launch, pointer array)
3. **Strided** — single `cublasGemmStridedBatchedEx` call (1 launch, contiguous memory)

Each implementation performs identical floating point operations. All timing
uses CUDA events for GPU-side measurement, with warmup iterations excluded.

## Results (NVIDIA L4, AWS g6.xlarge)

| Implementation | Time     | TFLOPS | Kernel Launches |
|----------------|----------|--------|-----------------|
| Individual     | 0.366 ms | 0.46   | 40              |
| Batched        | 0.015 ms | 11.03  | 1               |
| Strided        | 0.014 ms | 12.05  | 1               |

**26x speedup** from launch consolidation alone. Arithmetic intensity unchanged.

## Key Findings
- Kernel launch overhead dominates runtime when operations are small and frequent
- A single batched launch recovers ~24x of wasted dispatch time
- Strided layout adds a further ~7% gain through memory coalescing — contiguous
  memory allows the GPU to read all 40 matrices sequentially rather than
  following scattered pointers
- The optimization pattern is architecture-agnostic: equivalent results were
  reproduced on Blackwell (B200) hardware, confirming the bottleneck is
  structural rather than hardware-specific

## Environment
- GPU: NVIDIA L4 (Ada Lovelace, SM89, 23GB VRAM)
- Instance: AWS g6.xlarge
- CUDA: 13.2 / cuBLAS
- Date: 2026-09-09

## Files
- `results/` — raw benchmark output
- `experiments/observations.md` — extended analysis and open questions

---
*Benchmark methodology based on patterns from* AI Systems Performance Engineering *(Fregly, O'Reilly).*
