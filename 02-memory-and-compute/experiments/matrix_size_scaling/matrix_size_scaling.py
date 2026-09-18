"""
TF32 utilization vs matrix size — roofline investigation.
Identifies the L2 cache boundary where compute-bound flips to memory-bound.

Hardware: NVIDIA L4 (SM 8.9), PyTorch 2.5.1 + CUDA 12.1
"""

import torch
import time

SIZES = [512, 1024, 2048, 4096, 8192]
WARMUP = 5
ITERATIONS = 20
TF32_PEAK_TFLOPS = 120.0  # L4 Tensor Core TF32 peak


def bench_size(s: int) -> tuple[float, float]:
    torch.backends.cuda.matmul.allow_tf32 = True
    A = torch.randn(s, s, device="cuda", dtype=torch.float32)
    B = torch.randn(s, s, device="cuda", dtype=torch.float32)

    for _ in range(WARMUP):
        torch.mm(A, B)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        torch.mm(A, B)
    torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) / ITERATIONS * 1000
    tflops = (2 * s ** 3) / (ms / 1000) / 1e12
    return ms, tflops


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA GPU found.")
        return

    props = torch.cuda.get_device_properties(0)
    l2_mb = props.l2_cache_size / 1e6

    print(f"GPU: {props.name}  |  L2 cache: {l2_mb:.0f} MB")
    print(f"{'Size':>10}  {'Time':>8}  {'TFLOPS':>8}  {'% Peak':>8}  {'Working Set':>12}")
    print("-" * 60)

    for s in SIZES:
        ms, tflops = bench_size(s)
        working_set_mb = (s * s * 4 * 2) / 1e6
        pct = tflops / TF32_PEAK_TFLOPS * 100
        spill = " <- L2 spill" if working_set_mb > l2_mb else ""
        print(f"{s:>5}x{s:<5}  {ms:>7.3f}ms  {tflops:>7.1f}T  {pct:>7.0f}%  {working_set_mb:>8.0f} MB{spill}")


if __name__ == "__main__":
    main()
