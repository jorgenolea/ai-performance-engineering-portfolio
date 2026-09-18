# Extended Analysis — Memory Transfers and Compute Efficiency

## Hardware context
Benchmarks executed on NVIDIA L4 (Ada Lovelace, SM 8.9) via AWS g6.xlarge.
Reference results from Blackwell B200 hardware included for cross-architecture comparison.

---

## Experiment 1: Pageable vs Pinned Memory Transfer

**Hypothesis:** pinned memory eliminates the hidden staging copy before DMA,
delivering a measurable speedup in host-to-device transfer throughput.

| Method | Time | Bandwidth | Speedup |
|--------|------|-----------|---------|
| Pageable | 16.006 ms | 13.1 GB/s | 1x |
| Pinned | 15.587 ms | 13.5 GB/s | 1.03x |
| B200 pageable (ref) | 19.64 ms | 10.2 GB/s | — |
| B200 pinned (ref) | 3.65 ms | 54.8 GB/s | 5.39x |

**Finding:** Hypothesis not confirmed on this hardware configuration.
Both paths cap at ~13 GB/s — well below the PCIe 4.0 x16 theoretical
maximum of ~32 GB/s. Transfer size had no effect (tested at 200 MB and 1000 MB).

The AWS virtualization layer introduces a PCIe bandwidth ceiling that affects
both methods equally. The staging copy overhead that pinned memory eliminates
is smaller than this ceiling, so neither path can pull ahead.

The B200 reference result (5.39x) demonstrates the optimization is real on
bare metal hardware with direct PCIe access and HBM3e bandwidth.

**Key insight:** optimization impact is environment-dependent. Characterize
actual PCIe throughput on target hardware before assuming pinned memory
will improve transfer latency in a given deployment context.

---

## Experiment 2: FP32 vs TF32 cuBLAS

**Hypothesis:** enabling TF32 routes 2048×2048 matmuls through Tensor Cores,
delivering a speedup proportional to the Tensor Core vs FP32 CUDA core ratio.

| Mode | Time | TFLOPS | % of Peak | Speedup |
|------|------|--------|-----------|---------|
| FP32 (TF32 off) | 1.183 ms | 14.5 | 46% FP32 | 1x |
| TF32 (TF32 on) | 0.480 ms | 35.8 | 30% TF32 | 2.46x |
| B200 TF32 (ref) | 0.091 ms | ~185 | — | 3.95x |

**Finding:** Hypothesis confirmed. 2.46x speedup from a single flag change
with no accuracy impact for standard training workloads. The operation is
compute-bound at this matrix size (arithmetic intensity ~340 FLOPS/byte),
placing it in the correct regime for Tensor Core acceleration.

Achieved TF32 efficiency: 30% of theoretical peak. This opened a follow-up
question — see Experiment 3.

---

## Experiment 3: TF32 Utilization vs Matrix Size (original investigation)

**Question:** why does a compute-bound operation achieve only 30% of TF32
theoretical peak? Is matrix size the limiting factor, and where is the
inflection point?

| Matrix Size | Time | TFLOPS | % of Peak | Working Set |
|-------------|------|--------|-----------|-------------|
| 512×512 | 0.015 ms | 18.2 | 15% | 1 MB |
| 1024×1024 | 0.088 ms | 24.4 | 20% | 4 MB |
| 2048×2048 | 0.464 ms | 37.0 | **31%** | 16 MB |
| 4096×4096 | 5.221 ms | 26.3 | 22% | 64 MB |
| 8192×8192 | 41.678 ms | 26.4 | 22% | 256 MB |

L4 L2 cache: 48 MB.

**Finding:** Utilization peaks at 2048×2048 (31%) and drops sharply at
4096×4096. At that size, each matrix is 64 MB — the working set exceeds the
48 MB L2 cache and spills to VRAM. The operation transitions from
compute-bound to memory-bandwidth-bound. Tensor Cores can process data faster
than VRAM (~300 GB/s) can supply it. Cores stall waiting, while nvidia-smi
continues to show 100% utilization — the same misleading signal as Experiment 1
in the kernel launch series.

Small matrices (512×512) underperform for a different reason: insufficient
parallelism to occupy all 58 SMs simultaneously.

**Key insight:** peak utilization is not a function of raw matrix size — it is
a function of fit within the L2 cache. The same pattern from the kernel launch
series repeats: fixing one bottleneck (launch overhead) exposed the next one
(cache capacity). Here, enabling TF32 made Tensor Cores fast enough that VRAM
bandwidth became the new ceiling.

In transformer training, effective matrix dimensions are determined by
`batch_size × sequence_length` against `hidden_dim`. Production training
frameworks tune micro-batch sizes and gradient accumulation steps to keep
the working set in the cache-friendly regime for each target GPU.

---

## Open questions for further investigation
- Does torch.compile close the gap between 30% and theoretical peak by fusing
  operations that reduce VRAM round trips?
- At what sequence length does the attention matrix in a real transformer model
  cross the L2 cache boundary on the L4?
- How does arithmetic intensity compare between attention (QK^T matmul) and
  feedforward (linear projection) layers at typical batch/sequence sizes?
