# AI Performance Engineering

Training and serving large AI models efficiently is an infrastructure problem as much as a modeling one. This repo documents hands-on GPU performance engineering experiments — profiling where compute time actually goes, understanding why hardware limits are rarely achieved in practice, and identifying what moves the needle across the full stack from CUDA kernels to distributed inference.

Experiments span GPU generations from L4 and T4 through A100 and Grace-Blackwell, covering kernel efficiency, memory hierarchy, compute precision, distributed communication, and inference scaling.

Each folder covers a specific bottleneck category with benchmark scripts, raw results, and analysis.
