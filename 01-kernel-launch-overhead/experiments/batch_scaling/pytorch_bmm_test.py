"""
Experiment: PyTorch manual loop vs torch.bmm
Question: Does PyTorch abstract batch launch overhead automatically,
or does a manual loop suffer the same overhead as raw CUDA individual launches?
"""

import torch
import time

device = torch.device("cuda")
batch = 40
M, N, K = 32, 256, 256

A = torch.randn(batch, M, K, device=device)
B = torch.randn(batch, K, N, device=device)

# Warmup — let PyTorch/CUDA compile kernels and allocate memory
# before timing starts, so setup cost doesn't pollute results
for _ in range(10):
    _ = torch.bmm(A, B)
torch.cuda.synchronize()

# Manual loop — one torch.mm call per matrix (40 separate GPU dispatches)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    for i in range(batch):
        torch.mm(A[i], B[i])
torch.cuda.synchronize()  # wait for GPU to finish before stopping timer
t1 = time.perf_counter()
loop_ms = (t1 - t0) / 100 * 1000

# Batched — single torch.bmm call (1 GPU dispatch for all 40 matrices)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    torch.bmm(A, B)
torch.cuda.synchronize()
t1 = time.perf_counter()
bmm_ms = (t1 - t0) / 100 * 1000

print(f"Manual loop (torch.mm x{batch}): {loop_ms:.3f} ms")
print(f"Batched     (torch.bmm):         {bmm_ms:.3f} ms")
print(f"Speedup: {loop_ms/bmm_ms:.2f}x")
