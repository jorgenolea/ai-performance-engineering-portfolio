# Extended Analysis — Kernel Launch Overhead

## Hardware context
Benchmarks executed on NVIDIA L4 (Ada Lovelace, SM89) via AWS g6.xlarge.
Reference results from Blackwell B200 hardware included for cross-architecture
comparison.

## Experiment 1: Baseline results (batch=40)

| Implementation  | Time     | TFLOPS | Launches | Speedup |
|-----------------|----------|--------|----------|---------|
| Individual      | 0.366 ms | 0.46   | 40       | 1x      |
| Batched         | 0.015 ms | 11.03  | 1        | ~24x    |
| Strided         | 0.014 ms | 12.05  | 1        | ~26x    |
| B200 (reference)| 0.364 ms | —      | 40       | ~29x    |

## Experiment 2: Batch count scaling (batch=400)

**Hypothesis:** baseline scales linearly (~3.6ms), batched scales with math only (~0.150ms)

**Results:**

| Implementation | Batch=40 | Batch=400 | Scaling factor |
|----------------|----------|-----------|----------------|
| Individual     | 0.366 ms | 3.258 ms  | 8.9x           |
| Batched        | 0.015 ms | 0.524 ms  | 35x            |
| Strided        | 0.014 ms | 0.523 ms  | 37x            |

**Finding:** Baseline scaled nearly linearly (8.9x for 10x more launches) —
consistent with launch overhead being a fixed per-call cost.

Batched versions scaled super-linearly (35x for 10x more matrices) — unexpected.
Root cause: at batch=400, total matrix data (~800MB) exceeds GPU L2 cache capacity.
The GPU becomes memory bandwidth bound — compute cores sit idle waiting for data
from VRAM. This is visible in the TFLOPS drop: 11.03 → 3.20 TFLOPS for batched.

**Key insight:** batching eliminates launch overhead but exposes the next bottleneck —
memory bandwidth. Performance optimization is a chain: fix one bottleneck and the
next one becomes visible.

## Experiment 3: PyTorch abstraction (torch.bmm vs manual loop)

**Question:** Does PyTorch automatically protect framework users from launch overhead,
or does a Python loop over torch.mm suffer the same problem?

| Implementation          | Time     | Speedup |
|-------------------------|----------|---------|
| torch.mm loop (x40)     | 0.783 ms | 1x      |
| torch.bmm               | 0.056 ms | 13.91x  |
| CUDA individual (ref)   | 0.366 ms | —       |
| CUDA batched (ref)      | 0.015 ms | —       |

**Finding:** PyTorch does abstract batching — torch.bmm internally calls the same
cublasSgemmBatched routine. Framework users are partially protected.

However, PyTorch adds measurable overhead versus raw CUDA:
- Loop: 0.783ms (PyTorch) vs 0.366ms (CUDA) — 2.1x slower due to Python + dispatcher overhead
- Batched: 0.056ms (PyTorch) vs 0.015ms (CUDA) — 3.7x slower due to framework layers
- Speedup: 13.9x (PyTorch) vs 24x (CUDA) — framework overhead narrows the gap

**Key insight:** PyTorch trades performance for usability. The abstraction is real
but not free. For maximum throughput, raw CUDA still wins. For most production
workloads, the PyTorch overhead is acceptable — but knowing it exists matters
when diagnosing performance regressions.

## Cross-architecture observation
The L4 baseline (0.366ms) and B200 baseline (0.364ms) are nearly identical
despite a significant GPU generation gap. This confirms that launch overhead
is a CPU-side bottleneck — it scales with the number of API calls, not GPU
compute throughput.

## Open questions for further investigation
- At what matrix size (M, N, K) does compute time dominate and launch overhead
  becomes negligible? (batch count fixed, matrix dimensions scaled)
- Does torch.compile() close the gap between PyTorch and raw CUDA for the batched case?
