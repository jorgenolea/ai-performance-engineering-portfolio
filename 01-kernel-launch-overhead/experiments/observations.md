# Extended Analysis — Kernel Launch Overhead

## Hardware context
Benchmarks executed on NVIDIA L4 (Ada Lovelace, SM89) via AWS g6.xlarge.
Reference results from Blackwell B200 hardware included for cross-architecture
comparison.

## Results summary

| Implementation  | Time     | TFLOPS | Launches | Speedup |
|-----------------|----------|--------|----------|---------|
| Individual      | 0.366 ms | 0.46   | 40       | 1x      |
| Batched         | 0.015 ms | 11.03  | 1        | ~24x    |
| Strided         | 0.014 ms | 12.05  | 1        | ~26x    |
| B200 (reference)| 0.364 ms | —      | 40       | ~29x    |

## Cross-architecture observation
The L4 baseline (0.366 ms) and B200 baseline (0.364 ms) are nearly identical
despite a significant GPU generation gap. This confirms that launch overhead
is a CPU-side bottleneck — it scales with the number of API calls made, not
with GPU compute throughput.

The optimized speedup is slightly lower on L4 (~26x) than B200 (~29x), which
is consistent with B200's higher memory bandwidth enabling faster execution of
the batched kernel once dispatch overhead is removed.

## Memory coalescing effect
Strided outperforms batched by ~7% despite both using a single kernel launch.
The difference is memory layout:
- Batched: GPU dereferences 40 separate device pointers to locate each matrix
- Strided: GPU computes each matrix address via a fixed stride offset from a
  single base pointer — no pointer chasing, predictable prefetching

This is a concrete example of memory coalescing improving throughput without
changing launch count or arithmetic intensity.

## Implications for production workloads
Any workload issuing many small GPU operations in a loop — attention heads,
expert routing in MoE models, multi-layer inference — is a candidate for this
class of optimization. The fix is not algorithmic; it is structural: batch
the dispatch, not just the data.

## Open questions for further investigation
- How does speedup scale as batch count increases from 40 to 400 to 4000?
- At what matrix size does compute time dominate and launch overhead become negligible?
- Can equivalent batching gains be demonstrated through PyTorch's `torch.bmm`
  versus a manual loop — and does the framework abstract this automatically?
