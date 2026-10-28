# 04 — Apple-silicon (M5, 16 GB) hardware and optimisation study

*Research and measurements for the video-to-animated-low-poly-scene model.*
*Date: 2026-10-28. Machine: Apple M5 (10-core GPU, `applegpu_g17g`), 16 GB unified memory, macOS 27.0 (26A428), model identifier Mac17,2. Software: MLX 0.32.4, PyTorch 2.15.0, coremltools 9.0, Python 3.12.*

All numbers in this file come from the scripts in `research/bench/`. Raw JSON is in `research/bench/results/`. Web claims have URLs; anything I could not check is marked **unverified**.

> **Caveats.**
> - The machine was shared with other agents during the runs (load average about 4–5). Absolute timings are therefore a few percent pessimistic, and where I ran something twice the results agreed within about 10%.
> - The shared web-search budget ran out partway through. After that I fetched official pages (Apple, MLX, PyTorch docs) directly by URL, but some literature items could not be re-checked; they are flagged.
> - `research/bench/02_dense_raster_bench.py` was written by a different agent. It is not part of this study.

---

## 0. Decisions in one screen

| Question | Answer | Key evidence (measured here) |
|---|---|---|
| Training framework | **MLX** (with PyTorch kept for offline teachers and data preparation) | ViT-S training step is 1.35–1.45x faster in MLX. SDPA backward is 3.2x faster in MLX with O(L) memory, while torch materialises the L×L matrix and runs out of memory at 16k tokens. Custom Metal kernels plug into MLX autodiff directly. |
| Precision | bf16 compute with fp32 master weights, and the whole step under `mx.compile` | ViT-S b8: fp32 eager 148 ms, amp-bf16 + compile 50–56 ms (2.6–2.9x). |
| Achievable dense throughput | About 14 TFLOPS fp16/bf16 GEMM; about 8.5 TFLOPS "fp32" in MLX (silently TF32); 3.4 TFLOPS true fp32 | §2.1–2.2 |
| Realistic sustained training throughput | 4.4–4.7 TFLOPS for ViT-S (about 30% of peak), 6.2–6.5 TFLOPS for ViT-B/L. No throttling over 150 s. | ViT-S @224 trains at **168 img/s**, about 0.6 M img/h or about 19k steps/h at batch 32. |
| Renderer | **Custom Metal kernel** (`mx.fast.metal_kernel` + `mx.custom_function` VJP), not plain ops | Soft triangle rasteriser, 128², fwd+bwd: 1.8 ms (1k tris) / 12.8 ms (10k) / 127 ms (100k). That is 40–78x faster than the best plain-MLX version and 62–126x faster than torch, using 10–100x less memory. |
| Largest trainable model | About 0.5 B parameters with full AdamW (18 B/param) plus checkpointing. Far larger with frozen or LoRA bases. | ViT-L/16 (304 M), batch 8, checkpointing: 5.2 GB, 16.7 img/s |
| Inference for frozen teachers | Core ML on the ANE (batch 1) or Core ML on the GPU (batch ≥ 8), run offline with results cached | ViT-S: ANE 1.97 ms/img at b1; Core ML GPU 973 img/s at b8; MLX 605 img/s at b8 |

---

## 1. What is new in the M5 for ML (web research)

### 1.1 Hardware

- **A Neural Accelerator in every GPU core.**
  - Apple's M5 press release (15 Oct 2025): a 10-core GPU "with a Neural Accelerator in each core", "over 4x the peak GPU compute performance for AI compared to M4, and over 6x compared to M1".
  - Also from the release: a 16-core Neural Engine, a 10-core CPU (4P + 6E), third-generation 3 nm, and **153 GB/s** unified-memory bandwidth (nearly 30% over M4's 120 GB/s).
  - The release says "Metal 4 Tensor APIs" let developers program the Neural Accelerators directly, and that Core ML, MPS and Metal 4 apps get the speed-up automatically.
  - Source: https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- **Peak FP16/BF16 TFLOPS.**
  - Apple does not publish a TFLOPS number (neither source above gives one).
  - Measured here: about **13.5–14.1 TFLOPS** dense fp16/bf16 GEMM (both MLX and torch MPS), **8.3–9.2 TFLOPS** for MLX "fp32" (TF32), and **2.8–3.5 TFLOPS** true fp32 (§2.1).
  - fp16 versus true-fp32 ALU throughput on the same chip is about 4.1x, which is consistent with Apple's ">4x" claim.
- **Memory bandwidth.**
  - Specification: 153 GB/s (source above).
  - Measured effective bandwidth: about **131 GB/s**. This comes from a bf16 4096×4096 matrix-vector product, which reads 32 MB in 255 µs (§2.10).
- **Memory limits on this 16 GB machine** (from `mx.device_info()`):
  - `max_recommended_working_set_size` = **12.71 GB**
  - `max_buffer_length` = 9.53 GB
  - `memory_size` = 17.18 GB

### 1.2 Software: Metal 4 tensor APIs

- **MTLTensor** is a new multi-dimensional resource usable in compute, render and ML contexts.
- **MTL4MachineLearningCommandEncoder** runs whole Core ML (ML program) networks on the GPU timeline, synchronised with other Metal work.
- **Shader ML** lets shaders call **Metal Performance Primitives** (`tensor_ops::matmul2d`, convolution). The `matmul2d` descriptor takes a "reduced precision" flag.
- Source: WWDC25 "Combine Metal 4 machine learning and graphics", https://developer.apple.com/videos/play/wwdc2025/262/
- For us this means a custom Metal kernel can issue matmuls on the Neural Accelerators from inside a fused shader, for example a small MLP evaluated inside the rasteriser. MLX already relies on this internally.

### 1.3 How MLX uses the M5 Neural Accelerators ("NAX")

- **The mechanism.** Apple ML Research says MLX uses "Tensor Operations (TensorOps) and Metal Performance Primitives framework introduced with Metal 4" to reach the Neural Accelerators, and requires macOS 26.2 or later (https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- **Apple's M5 vs M4 results with MLX-LM.**
  - Time-to-first-token (compute-bound): **3.33–4.06x** faster.
  - Generation (bandwidth-bound): 1.19–1.27x faster.
  - FLUX-dev-4bit image generation: more than 3.8x faster.
- **The MLX version.** MLX **v0.30.0** (dated 19 Nov on the release page, year not shown; late 2025) added "Support for Neural Accelerators on M5 (macOS >= 26.2)", PR #2772 (https://github.com/ml-explore/mlx/releases/tag/v0.30.0).
- **Later releases (0.32.x)** extended NAX coverage (https://github.com/ml-explore/mlx/releases):
  - NAX attention for head dims 72, 80 and 256.
  - Thin-M/N NAX GEMMs.
  - `qmm_t_nax` tuning.
  - **SDPA VJP**, in v0.32.4 (dated 14 Oct, the version installed here). This is what makes fused attention backward fast (see §2.3).
  - v0.32.0 notes that NAX builds need `MACOSX_DEPLOYMENT_TARGET=26.2`.
- **Which ops use NAX, from local inspection** (`strings libmlx.dylib`, MLX 0.32.4):
  - Dense GEMM: `steel_gemm_fused_nax_*`, `steel_gemm_splitk_nax_*`, `steel_gemm_thin_nax_*`.
  - Gather/MoE matmul: `steel_gather_mm_rhs_nax_*`.
  - Quantised matmul: `qmm_t_nax`, `qmm_n_nax`, `gather_qmm_*_nax`.
  - NAX attention and `gated_delta_fused_nax`/`gated_delta_vjp_fused_nax`.
  - An environment variable **`MLX_ENABLE_TF32`**.
  - An error string reading "... head dim 512 on NAX GPUs (float32 also requires TF32)".
- **Dtypes.** fp16 and bf16 run natively. **fp32 matmul runs as TF32 by default on M5**, which I verified numerically (§2.2). Quantised 4- and 8-bit (and nvfp4/mxfp8) weights have NAX paths.
- **Not accelerated.** Element-wise ops, reductions, scatter and gather run on the normal shader ALUs. That is why `mx.compile` fusion matters as much as it does.

---

## 2. Measurements on this machine

How to run (one at a time, each under about 3 minutes):

```bash
cd research/bench
PY=.venv/bin/python
$PY bench_matmul.py mlx; $PY bench_matmul.py torch     # §2.1
$PY bench_fp32_precision.py                             # §2.2
$PY bench_sdpa.py                                       # §2.3
$PY bench_vit.py mlx; $PY bench_vit.py torch; $PY bench_vit.py torch_compile   # §2.4
$PY bench_memory.py mlx; $PY bench_memory.py torch      # §2.5
$PY bench_sustained.py 150                              # §2.6
$PY bench_raster.py gauss_local; $PY bench_raster.py gauss_dense; $PY bench_raster.py tri_soft  # §2.7
$PY bench_metal_kernel.py; $PY bench_torch_shader.py    # §2.8
$PY bench_coreml.py                                     # §2.9
$PY bench_extras.py                                     # §2.10, §2.11
$PY probe_ops.py                                        # §2.12
```

Shared code lives in `common.py` (timing helpers, peak footprint through `proc_pid_rusage`, which counts GPU buffers), `vit_models.py` (identical ViT in functional MLX and torch) and `raster_ops.py` (plain-op splatting and soft rasterisation).

### 2.1 Matmul throughput (TFLOPS; chained products, median of 7)

| shape (M×K×N) | MLX fp32 (=TF32) | MLX fp16 | MLX bf16 | torch fp32 | torch fp16 | torch bf16 |
|---|---|---|---|---|---|---|
| 1024³ | 8.55 | 12.73 | 12.20 | 3.32 | 13.06 | 12.70 |
| 2048³ | 9.15 | 13.53 | 13.18 | 3.32 | 14.02 | 13.90 |
| 4096³ | 8.26 | 13.53 | 12.61 | 2.82 | 14.11 | 13.95 |
| 1576×384×1536 (ViT-S fc1, b8) | 6.21 | 8.19 | 8.20 | 3.13 | 11.31 | 11.43 |
| 1576×384×1152 (ViT-S qkv, b8) | 5.98 | 7.81 | 8.19 | 3.07 | 10.78 | 11.05 |
| 1576×1536×384 (ViT-S fc2, b8) | 5.90 | 11.60 | 11.01 | 3.17 | 9.91 | 10.39 |

Per-call times at 4096³: MLX fp16 10.2 ms, torch fp16 9.7 ms, torch fp32 48.8 ms.

**Takeaways**
- For large fp16/bf16 GEMMs the two frameworks are level, at about 14 TFLOPS. Both reach the Neural Accelerators (torch through MPS/MPSGraph).
- For the skinny ViT-S shapes, MPS is about 1.4x faster than MLX on the two K = 384 GEMMs (qkv, fc1), while MLX is faster on fc2 (K = 1536). MLX still wins whole training steps (§2.4) because of fusion and lower overhead.
- MLX "fp32" is about 2.6x torch fp32, but only because it is TF32 (next section).

### 2.2 Is fp32 really fp32? (`bench_fp32_precision.py`, 2048³, error is max relative error vs a float64 reference)

| path | max rel. error | TFLOPS |
|---|---|---|
| MLX GPU fp32, default | **7.8e-4** | 8.53 |
| MLX GPU fp32, `MLX_ENABLE_TF32=0` | 1.8e-6 | 3.36 |
| MLX CPU fp32 | 2.1e-6 | – |
| torch MPS fp32 (`set_float32_matmul_precision` highest/high/medium: no effect) | 2.1e-6 | 3.1–3.5 |
| fp16 (MLX and torch) | 4.3e-4 | – |
| bf16 (MLX and torch) | 3.6e-3 | – |

**Takeaways**
- On M5, MLX runs fp32 matmuls as TF32 by default: about a 10-bit mantissa, worse than fp16 inputs here. This is fine for networks.
- It is not fine for gradient checks or numerically delicate geometry such as camera solves or least squares. In the rasteriser test (§2.8) TF32 caused about 1% gradient differences.
- Use `MLX_ENABLE_TF32=0` for those paths and for unit tests.

### 2.3 Scaled dot-product attention (B=1, 6 heads, head dim 64; `bench_sdpa.py`)

Effective TFLOPS follows the FlashAttention convention: forward = 4·B·H·L²·D, forward+backward = 3.5× forward. The memory column is peak allocator memory. At 1k and 4k, MLX's figure covers 20 or 4 concurrent calls (they are batched to amortise dispatch), so it is an upper bound; at 16k there is a single call. Torch frees memory between calls, so its figure is per call.

**Forward**

| L | dtype | MLX fused ms (TFLOPS) | torch fused ms (TFLOPS) | MLX naive ms | torch naive ms |
|---|---|---|---|---|---|
| 1k | fp16 | 0.12 (13.6) | 0.14 (11.8) | 1.51 | 1.61 |
| 1k | bf16 | 0.12 (14.0) | 0.14 (11.8) | – | – |
| 1k | fp32 | 0.18 (9.0) | 0.60 (2.7) | – | – |
| 4k | fp16 | 1.90 (13.6) | 2.06 (12.5) | 24.3 | 37.7 |
| 4k | bf16 | 1.90 (13.6) | 2.79 (9.3) | – | – |
| 4k | fp32 | 4.08 (6.3) | 9.04 (2.9) | – | – |
| 16k | fp16 | 32.9 (12.5), 0.06 GB | 37.0 (11.2), 1.07 GB | – | – |
| 16k | bf16 | 33.4 (12.4) | 39.3 (10.5) | – | – |

**Forward + backward**

| L | dtype | MLX fused ms (TFLOPS) | MLX mem | torch fused ms (TFLOPS) | torch mem |
|---|---|---|---|---|---|
| 1k | fp16 | 1.39 (4.1) | – | 5.16 (1.1) | 1.12 GB |
| 1k | bf16 | 1.47 (3.8) | – | 5.40 (1.0) | 1.12 GB |
| 1k | fp32 | 2.68 (2.1) | – | 4.54 (1.2) | 1.12 GB |
| 4k | fp16 | **23.5 (3.8)** | 0.31 GB (4 calls) | **75.9 (1.2)** | **2.22 GB** |
| 4k | bf16 | 24.1 (3.7) | 0.31 GB | 78.5 (1.2) | 2.22 GB |
| 4k | fp32 | 45.1 (2.0) | 0.55 GB | 69.8 (1.3) | 2.22 GB |
| 16k | fp16 | **347 (4.2)** | **0.23 GB** | **OOM** with a 5.9 GB cap | > 5.9 GB |
| 16k | bf16 | 357 (4.0) | 0.23 GB | OOM | > 5.9 GB |

**Takeaways**
- The forward pass is near GEMM peak in both frameworks.
- **Backward is where they diverge.**
  - MLX 0.32.4 has a fused, memory-linear SDPA VJP.
  - Torch MPS's SDPA backward behaves like naive attention: the same speed as the materialised version and O(L²) memory (2.2 GB at 4k; it ran out of memory at 16k).
- For any global attention over many tokens (a whole clip, or slots × frames) during training, MLX is 3.2x faster at 4k and is the only option at 16k.

### 2.4 ViT-S/16 training step (12 layers, dim 384, 6 heads, 224 px, batch 8, AdamW, 1000-way cross-entropy; `bench_vit.py`)

| lib | precision | compile | ms/step | img/s | achieved TFLOPS | first step (s) | lib mem (GB) |
|---|---|---|---|---|---|---|---|
| MLX | fp32 (TF32 matmul) | no | 148.3 | 54 | 1.49 | 2.95 | 1.47 |
| MLX | fp32 | `mx.compile` | 90.4 | 89 | 2.44 | 6.0 | 1.90 |
| MLX | amp-bf16 (fp32 master) | no | 77.4 | 103 | 2.85 | 4.2 | 0.95 |
| MLX | **amp-bf16** | **`mx.compile`** | **56.2** (50.3 on a re-run) | **142–159** | 3.9–4.4 | 1.6 | 1.25 |
| MLX | pure bf16 (params + Adam in bf16) | `mx.compile` | 53.7 | 149 | 4.11 | 0.9 | 0.98 |
| torch MPS | fp32 | no | 124.2 | 64 | 1.78 | 1.6 | 0.96 |
| torch MPS | autocast fp16 | no | 79.4 | 101 | 2.78 | 1.7 | 0.80 |
| torch MPS | autocast bf16 | no | 81.2 | 99 | 2.72 | 1.5 | 0.80 |
| torch MPS | pure bf16 | no | 72.8 | 110 | 3.03 | 0.8 | 0.62 |
| torch MPS | fp32 | `torch.compile` (inductor/MPS) | 107.9 | 74 | 2.04 | 6.1 | 0.97 |
| torch MPS | autocast bf16 | `torch.compile` | 170.1 | 47 | 1.30 | 8.8 | 0.76 |

FLOPs are counted analytically as 6 × MACs, about 27.6 GFLOP per image. Fused AdamW was used in torch.

**Takeaways**
- The best MLX configuration (amp-bf16 + compile, 50–56 ms) beats the best torch configuration (pure bf16 72.8 ms, autocast 79–81 ms) by **1.3–1.45x**.
- `mx.compile` gives 1.2–1.6x on a training step. `torch.compile` on MPS works but gives only 1.15x in fp32 and is 2x *slower* with autocast-bf16.
- MLX has no autocast. The pattern used here keeps fp32 master parameters and casts them inside the loss function (`tree_map(astype(bf16))`), so gradients and AdamW stay fp32. Pure bf16 is marginally faster but loses master-weight precision.

### 2.5 Memory: how large a model and batch fits (`bench_memory.py`; amp-bf16; MLX compiled, torch eager; peak allocator memory)

| model (params) | batch | grad ckpt | MLX ms/step | MLX img/s | MLX TFLOPS | MLX GB | torch ms/step | torch img/s | torch GB |
|---|---|---|---|---|---|---|---|---|---|
| ViT-S (22.1 M) | 8 | no | 50.3 | 159 | 4.39 | 1.25 | – | – | – |
| ViT-S | 32 | no | 189.3 | 169 | 4.66 | 2.30 | 255.7 | 125 | 2.42 |
| ViT-S | 32 | yes | 233.7 | 137 | 3.78 | 1.22 | 333.8 | 96 | 1.27 |
| ViT-S | 128 | yes | 911.7 | 140 | 3.87 | 2.24 | 1301.9 | 98 | 2.47 |
| ViT-B (86.6 M) | 8 | no | 130.6 | 61 | 6.45 | 2.56 | 186.0 | 43 | 2.25 |
| ViT-B | 32 | yes | 569.4 | 56 | 5.92 | 2.38 | 808.4 | 40 | 2.54 |
| ViT-L (304 M) | 8 | yes | 479.1 | 16.7 | 6.17 | **5.19** (footprint 5.54) | – | – | – |

**Derived rules of thumb**

- **Fixed cost of AdamW training with fp32 master weights:** about 16 B/param (weights, gradients, two moments), plus 2 B/param for the bf16 copy. ViT-L: 304 M × 18 B ≈ 5.5 GB, which matches the 5.2–5.5 GB measured.
- **Activations at 224 px for ViT-S:**
  - About 44 MB per image without checkpointing.
  - About 11 MB per image with `mx.checkpoint` per block.
  - Checkpointing costs about 23% in MLX and 31% in torch.
- **Usable budget.** About 10–11 GB is realistic: `max_recommended_working_set_size` is 12.7 GB, and the OS and apps need the rest.
- **Full AdamW training:** at most about **0.5 B parameters** with checkpointing. ViT-H (632 M) would not fit without 8-bit optimiser states or LoRA.
- **ViT-S batch limits:** about 200 images per batch at 224 px without checkpointing, or about 900 with it. That is 12 or 55 clips of 16 frames.
- **Frozen bf16 base plus LoRA:** about 2 B/param, so a 1–2 B-parameter frozen teacher plus adapters fits.
- **4-bit frozen base (QLoRA):** about 0.55 B/param, so 3–4 B parameters is possible. MLX-LM does exactly this for LLMs (§3).
- **Default limits allow swapping.**
  - MLX's default memory limit and cache limit are both **16.3 GB** (95% of RAM), and the wired limit is 0.
  - Torch's `PYTORCH_MPS_HIGH_WATERMARK_RATIO` defaults to **1.7 × recommendedMaxWorkingSetSize ≈ 21.6 GB** on this machine (https://docs.pytorch.org/docs/2.14/mps_environment_variables.html).
  - Both can therefore page to swap, which turns a 1-second step into minutes; swap was already 1.7 GB used on this machine during the runs.
  - **Set the caps explicitly:** `mx.set_memory_limit(10<<30)` and `mx.set_cache_limit(2<<30)` in MLX; `torch.mps.set_per_process_memory_fraction(0.8)` in torch. That fraction is relative to the 12.7 GB working set, as the 0.5 → 5.92 GiB OOM message confirmed.
- **Wiring memory.** `mx.set_wired_limit` (macOS ≥ 15) keeps buffers resident. Raising the system ceiling needs `sudo sysctl iogpu.wired_limit_mb=…`, which is a system setting: I did not change it, and it is not needed at our model sizes (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).

### 2.6 Sustained throughput (`bench_sustained.py`, ViT-S, MLX amp-bf16 compiled, batch 32, 150 s)

| t (s) | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 |
|---|---|---|---|---|---|---|---|---|---|
| img/s | 168.0 | 168.1 | 168.3 | 168.0 | 168.1 | 168.1 | 167.4 | 166.4 | 166.8 |

- About 1% drift over 2.5 minutes, so no meaningful thermal throttling at this length. Throughput is steady at **4.6 TFLOPS**.
- I could not test hour-long runs within the 3-minute rule. Expect some throttling on a fanless chassis.
- Forward-only ViT-S inference (bf16, b64, compiled): **636 img/s**.

### 2.7 Differentiable splatting and soft rasterisation in plain ops (128×128 RGB; `bench_raster.py`, `raster_ops.py`)

There are three methods, each fully differentiable with respect to per-primitive parameters, and MSE against a target image:

- **gauss_local:** each 2D Gaussian splats into a 9×9 footprint through scatter-add, O(N·81).
- **gauss_dense:** every pixel against every Gaussian, O(HW·N).
- **tri_soft:** SoftRas-style triangles, O(HW·N). Coverage is the product of three edge sigmoids, aggregation is a depth softmax with a background term, and colour is flat (low-poly).

Library variants:
- **mlx:** eager, chunked over N with `mx.checkpoint`.
- **mlx-compile:** the whole forward and the whole `value_and_grad` under `mx.compile`.
- **mlx-stream:** chunked with an `mx.eval` per chunk and a manual per-chunk VJP (`raster_ops.mlx_stream_value_and_grad`).
- **torch:** eager, chunked, with `torch.utils.checkpoint`.

Images and gradients agree across libraries: checksums match to 1e-4, and the differences come from TF32.

| method | N | lib | fwd ms | fwd+bwd ms | fwd+bwd peak GB |
|---|---|---|---|---|---|
| gauss_local | 1k | mlx / mlx-compile / torch | 0.74 / 0.52 / 1.87 | 1.73 / 1.39 / 1.25 | 0.02 / 0.02 / – |
| gauss_local | 10k | mlx / mlx-compile / torch | 3.07 / 3.48 / 3.63 | 5.97 / 5.00 / 8.09 | 0.17 / 0.15 / – |
| gauss_local | 100k | mlx / mlx-compile / torch | 34.8 / **16.9** / 43.5 | 80.5 / **60.6** / 79.9 | 1.22 / 1.41 / (footprint 1.32) |
| gauss_dense | 1k | mlx / mlx-compile / mlx-stream / torch | 16.6 / 2.87 / 2.18 / 23.6 | 51.6 / **32.3** / 54.1 / 65.7 | 1.25 / 1.12 / 1.25 / – |
| gauss_dense | 10k | mlx / mlx-compile (one graph) | 177 / 15.5 | 715 / 311 | **6.06 / 6.49** |
| gauss_dense | 10k | mlx-stream / torch | 17.4 / 261 | **593** / 893 | 1.28 / (footprint 1.39) |
| gauss_dense | 100k | mlx-stream / torch | **171** / 2298 | **5056** / 8672 | 1.29 / (footprint 1.46) |
| tri_soft | 1k | mlx / mlx-compile / mlx-stream / torch | 27.2 / 3.62 / 3.02 / 47.0 | 95.4 / **72.1** / 109.4 / 112.2 | 1.84 / 1.97 / 1.90 / – |
| tri_soft | 10k | mlx-stream / torch | **26.2** / 454 | **992** / 1590 | 1.95 / (footprint 2.45) |
| tri_soft | 100k | mlx-stream / torch | **253** / 4508 | **9582** / 16071 | 1.97 / (footprint 2.53) |

**Takeaways**

1. **Compiled forward passes are 8–18x faster** in MLX than in eager torch (tri_soft at 100k: 253 ms vs 4.5 s). Element-wise fusion is what counts here.
2. **Backward passes are not fused well by any plain-op route.**
   - MLX fwd+bwd is 11–38x its compiled forward pass and needs about 116 B per pixel-primitive pair, even per chunk. Compiling the per-chunk VJP helps only about 1.4x (88 → 62 ms per 1024-triangle chunk).
   - MLX is still 1.5–2x faster than torch end to end.
3. **Lazy-graph memory pitfall in MLX.** Chunking with `mx.checkpoint` *inside one* `mx.eval` does **not** bound memory: about 6 GB at 10k, and an estimated more than 60 GB at 100k. This held with `MLX_BFS_MAX_WIDTH=1` as well. Evaluating per chunk with a manual VJP (`mlx-stream`) bounds it to about 1.3–2 GB at any N. Torch's eager execution frees memory per op, so its checkpointing works as expected.
4. **O(HW·N) dense soft rasterisation is not viable at N ≥ 10k in plain ops.** The O(N·K²) local footprint is cheap (60 ms at 100k in MLX). Hence the custom kernel in §2.8.

### 2.8 Custom Metal kernels

**MLX `mx.fast.metal_kernel` works** (`bench_metal_kernel.py`).

*Part A, a tiny kernel.* A fused SiLU over 16 M floats:

| | ms | Notes |
|---|---|---|
| custom kernel | 1.24–1.43 | about 94–108 GB/s; max error vs MLX ops 4.8e-7 |
| MLX ops | 2.85–3.12 | |
| MLX compiled | 1.35–1.72 | |

*Part B, a real kernel.* A SoftRas-style triangle rasteriser as two Metal kernels inside `mx.custom_function` with a hand-written `.vjp`:
- **Forward:** one thread per pixel, looping over triangles.
- **Backward:** one thread per (triangle, pixel block), looping over the block's pixels and accumulating 19 per-triangle gradients in registers. The 16 partials are summed in MLX, so no atomics are needed.
- MLX autodiff handles the per-triangle preprocessing (edge vectors, inverse lengths, depth weights) in both directions.
- It computes the same maths as `tri_soft` above.

| N triangles | culling | fwd ms | fwd+bwd ms | fwd+bwd mem | vs best plain MLX fwd+bwd | vs torch fwd+bwd |
|---|---|---|---|---|---|---|
| 1k | no | 0.75 | **1.80** | 4 MB | 40x (72.1 ms) | 62x (112 ms) |
| 1k | edge-cull | 1.10 | 2.68 | 4 MB | | |
| 10k | no | 5.20 | **12.8** | 20 MB | 78x (992 ms) | 124x (1590 ms) |
| 10k | edge-cull | 6.33 | 20.4 | 20 MB | | |
| 100k | no | 51.6 | **127** | 169 MB | 75x (9582 ms) | 126x (16071 ms) |
| 100k | edge-cull | 63.7 | 203 | 169 MB | | |

**Correctness.** Against the plain-op MLX autodiff reference, the loss and gradient errors are 1.7e-4 and 0.5–1% with default TF32. With `MLX_ENABLE_TF32=0` they are **1.4e-7 and 1.3e-6 (N=1k)** and **0.0 and 2.5e-6 (N=10k)**. The kernel is exact; the 1% came from TF32 in the reference's `w @ col` matmul.

**Notes.**
- Naive per-pixel culling with `break` is *slower*, because of SIMD divergence.
- The real next step is tile binning: sort triangles into 16×16 tiles, as 3DGS and nvdiffrast do. With low-poly scenes of about 1k triangles per view, the unculled kernel is already about 1.8 ms per view at 128², roughly 29 ms for a 16-frame clip.
- Two API gotchas:
  - For a single-input `custom_function`, the VJP receives the primal array itself, not a tuple.
  - Kernels are JIT-compiled once per name and template; create them at module scope.
- API reference: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html. It documents `atomic_outputs`, `init_value`, the `custom_function`+`vjp` pattern and a `grid_sample` example that is 8x faster forward and 40x faster backward than plain ops on an M1 Max.

**PyTorch also works** (`bench_torch_shader.py`). `torch.mps.compile_shader(src)` compiles Metal source and exposes kernels as Python callables (https://docs.pytorch.org/docs/2.14/generated/torch.mps.compile_shader.html). Wrapped in `torch.autograd.Function`, a fused SiLU forward and backward matched `F.silu` exactly (forward error 0, backward error 2.4e-7) at the same speed: 1.06 vs 1.03 ms for 16 M floats. So custom kernels are possible in torch too, but MLX's `metal_kernel` has more conveniences: shape and stride injection, templates, `init_value`, atomics, and a `custom_function` VJP that composes with `mx.compile`.

### 2.9 Core ML and Neural Engine inference (`bench_coreml.py`; ViT-S/16 converted to an FP16 ML Program, macOS 15 target)

| batch | compute units | ms/call | img/s | rel. error vs fp32 torch |
|---|---|---|---|---|
| 1 | CPU_ONLY | 5.05 | 198 | 1.0e-2 |
| 1 | CPU_AND_GPU | 3.19 | 314 | 1.3e-3 |
| 1 | **CPU_AND_NE (ANE)** | **1.97** | **506** | 8.3e-3 |
| 1 | ALL | 2.01 | 497 | 8.3e-3 |
| 1 | MLX GPU bf16, compiled | 2.69 | 372 | – |
| 8 | CPU_ONLY | 28.2 | 284 | 1.2e-2 |
| 8 | **CPU_AND_GPU** | **8.22** | **973** | 1.3e-3 |
| 8 | CPU_AND_NE | 23.6 | 340 | 1.1e-2 |
| 8 | ALL | 23.6 | 339 | 1.1e-2 |
| 8 | MLX GPU bf16, compiled | 13.2 | 605 | – |

Conversion takes 2–3 s. coremltools 9.0 warns that torch 2.15 is untested and fails on traced `aten::size`/`int` casts. A static-shape model definition works around this (`static_vit` in the script). `MLComputePlan` op-placement inspection failed (a path error), so ANE placement is inferred from timing.

**Takeaways**
- The ANE is the fastest option at batch 1 (2 ms per frame for ViT-S) and leaves the GPU free.
- It does not scale with batch: 340 img/s at b8. Its fp16 error is about 1%.
- At batch 8, Core ML on the GPU is 1.6x faster than MLX inference. MPSGraph seems better tuned for these skinny GEMMs, consistent with §2.1.
- **Use Core ML (ANE or GPU) for frozen-teacher feature extraction and caching, and for the final deployed encoder.** It is not a training path.

### 2.10 Quantised matmul in MLX (affine, group 64; `bench_extras.py`; `x @ W.T`, times in µs)

| K→N | M | bf16 | q4 | q4 rel. err | q8 | q8 rel. err |
|---|---|---|---|---|---|---|
| 384→1536 | 1 | 11.9 | 14.1 | 9.9e-2 | 18.2 | 9.6e-3 |
| 384→1536 | 1576 | 215.7 | 139.3 | 1.0e-1 | 142.4 | 9.3e-3 |
| 1536→384 | 1576 | 128.9 | 136.1 | 1.1e-1 | 143.9 | 8.2e-3 |
| 4096→4096 | 1 | 255.4 | **61.3** | 9.7e-2 | 129.0 | 7.8e-3 |
| 4096→4096 | 16 | 288.9 | 316.6 | 1.0e-1 | 412.7 | 7.1e-3 |
| 4096→4096 | 1576 | 5730 | 3951 | 8.7e-2 | 4750 | 8.5e-3 |

**Takeaways**
- 4-bit weights pay off in the bandwidth-bound regime: 4.2x at M=1 for a 4096² weight. That regime is LLM decoding, which is not our case.
- At ViT-S sizes they are neutral, and they add about 10% output error (q4) or 0.8% (q8) on random weights.
- Use 8-bit for *frozen* teachers when memory is tight. Do not quantise the trained encoder.

### 2.11 Cost versus tokens (progressive resolution, token-merging proxy; ViT-S, MLX amp-bf16 compiled, batch 32)

| image px | tokens | ms/step | img/s | TFLOPS | GB |
|---|---|---|---|---|---|
| 112 | 50 | 47.7 | **670** | 4.42 | 1.22 |
| 160 | 101 | 93.4 | 343 | 4.66 | 1.60 |
| 224 | 197 | 188.8 | 170 | 4.68 | 2.30 |
| 320 | 401 | 424.7 | 75 | 4.57 | 4.17 |

Cost is linear in token count at these lengths, since attention is a small share of the work. Halving the tokens, whether by resolution, ToMe, or skipping static patches, halves the step time, and efficiency stays at about 4.5 TFLOPS.

### 2.12 Operator coverage relevant to the project (`probe_ops.py`; torch with `PYTORCH_ENABLE_MPS_FALLBACK=0`)

| op | torch MPS | MLX GPU |
|---|---|---|
| `grid_sample` 2D/3D (+bwd), bicubic antialias resize, `unfold`, `cdist`, `histc`, `bincount`, `nonzero`, `linalg.det`, `lstsq` | ok | **no built-in op**. `nonzero` is impossible because the output shape is dynamic. The others need a custom kernel or a few lines of code. |
| `linalg.svd / eigh / inv / qr / cholesky` | ok (with bwd) | **GPU: not supported.** CPU stream only, and **no VJP** ("Not implemented for SVD/Inverse/Eigh/QRF/Cholesky") |
| `linalg.solve` | ok | CPU only. Backward fails. |
| conv3d, max_pool3d, scatter add/max (+bwd), sort 1M, topk, cumsum, FFT, complex64 matmul, erfinv, RMSNorm, SDPA with boolean mask (+bwd), `unique` (MLX needs a static `size=`), `searchsorted` | ok | ok |
| float64 | fails (MPS has no float64) | fails on GPU (CPU only) |

**Consequence for the design.** Parameterise rotations and poses without differentiable SVD or inverse: use the 6D rotation representation or quaternions, axis-angle exponential maps written element-wise, and closed-form 3×3 or 4×4 inverses for rigid transforms. Write `grid_sample` as a Metal kernel; the MLX documentation's example is exactly this. Torch's operator coverage is now broad, which is its main remaining advantage.

---

## 3. MLX versus PyTorch MPS for training

| Aspect | MLX 0.32.4 | PyTorch 2.15 MPS |
|---|---|---|
| Dense GEMM fp16/bf16 | about 13.5 TFLOPS (NAX) | about 14 TFLOPS (MPS); better on skinny K=384 shapes |
| fp32 | TF32 by default (8.5 TFLOPS, 7.8e-4 error); true fp32 3.4 | true fp32 only (3.4); `set_float32_matmul_precision` has no effect |
| Fused attention forward | yes (`mx.fast.scaled_dot_product_attention`), 12.5–14 TFLOPS | yes, 9–12.5 TFLOPS |
| Fused attention **backward** | **yes, since v0.32.4**: 4 TFLOPS, O(L) memory | **no**: O(L²) memory, out of memory at 16k tokens, 3.2x slower at 4k |
| Graph compilation | `mx.compile` of the whole step: 1.2–1.6x; reliable; recompiles on shape change unless `shapeless=True` (https://ml-explore.github.io/mlx/build/html/usage/compile.html) | `torch.compile` (inductor → Metal) runs: 1.15x in fp32 and 2x slower with autocast-bf16; 6–9 s compile |
| Mixed precision | no autocast; cast params in the loss (fp32 master) or train in pure bf16 | `torch.autocast("mps", fp16 or bf16)` works; `torch.amp.GradScaler` exists (not tested on MPS here) |
| Gradient checkpointing | `mx.checkpoint(fn)`: −47% peak memory (ViT-S b32), +23% time. Only bounds memory if the graph is evaluated in pieces (§2.7). | `torch.utils.checkpoint`: −48% peak memory, +31% time |
| Custom kernels | `mx.fast.metal_kernel` + `mx.custom_function`/`.vjp`; composes with compile | `torch.mps.compile_shader` + `autograd.Function`; works |
| Quantised training (QLoRA) | first-class in MLX-LM: QLoRA is used automatically when the base is quantised; `--fine-tune-type lora/dora/full`, `--grad-checkpoint` (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md); `nn.quantize` (affine / nvfp4 / mxfp8) (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html) | bitsandbytes 0.50 supports 4-bit and LLM.int8() on MPS, but **8-bit optimisers are not supported on MPS** (https://github.com/bitsandbytes-foundation/bitsandbytes/releases) |
| Operator coverage | gaps: linalg (CPU only, no VJP), `grid_sample`, `nonzero`, dynamic shapes | broad (§2.12); `PYTORCH_ENABLE_MPS_FALLBACK=1` falls back to CPU for missing ops (https://docs.pytorch.org/docs/2.14/mps_environment_variables.html) |
| Memory model | arrays live in unified memory and each op picks CPU or GPU; no copies; CPU and GPU streams can run concurrently (https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html). Lazy evaluation needs `mx.eval` boundaries to bound peaks. | eager; caching allocator; high watermark 1.7x working set by default (can swap) |
| Ecosystem / pretrained weights | must port weights (DINOv2/v3, Depth-Anything) or run teachers elsewhere | everything (DINO, DA-V2/V3, TransNetV2, PyTorch3D on CPU) |

**Verdict: MLX for the trainable model; PyTorch for offline preprocessing and teachers** (or export them to Core ML for the ANE).

The measured reasons:
- 1.35–1.45x faster transformer training steps (§2.4, §2.5).
- 3.2x faster and memory-linear attention backward, which matters for slots × frames attention (§2.3).
- 8–18x faster fused forward for renderer-like element-wise maths (§2.7).
- Custom-kernel autodiff that composes cleanly with compile (§2.8).

The costs:
- Port DINO weights, which is straightforward for a ViT.
- Avoid differentiable linear algebra.
- Write two or three kernels: the rasteriser, `grid_sample`, and the scatter or z-buffer.
- Be disciplined about `mx.eval` and memory limits.

---

## 4. Core ML and the Apple Neural Engine for inference

- **What is realistic.** ViT-S runs at 2 ms per frame on the ANE at batch 1 (measured, §2.9). Apple's Core ML Depth-Anything-V2-Small (24.8 M) is reported at 24.6 ms on an M3 Max ANE (https://huggingface.co/apple/coreml-depth-anything-v2-small; via research agent, not re-checked). A 1-minute clip at 4 fps (240 frames) through a ViT-S encoder is therefore about 0.5 s on the ANE, plus depth teachers of a few seconds. Inference of the whole pipeline on a 1-minute video is comfortably interactive.
- **ANE design rules** (Apple, "Deploying Transformers on the Apple Neural Engine", https://machinelearning.apple.com/research/neural-engine-transformers; reference code https://github.com/apple/ml-ane-transformers):
  - Use a (B, C, 1, S) channels-first layout with `Conv2d` in place of `Linear`.
  - Split attention per head.
  - Avoid reshapes and transposes.
  - The last axis must be contiguous and 64-byte aligned.
  - With these, DistilBERT ran "up to 10 times faster" with 14x less peak memory on iPhone 13.
  - The ANE is fp16 (8-bit weights are supported for storage), and batch scaling is poor (our b8 result).
- **Not for training.** Core ML has no training path for our model. Use it for:
  1. offline teacher features (depth, DINO, masks) that are cached to disk;
  2. the deployed encoder, so the GPU is left for rendering and decoding.
- **Metal 4 can run a Core ML ML program on the GPU timeline** (`MTL4MachineLearningCommandEncoder`) synchronised with render passes (WWDC25 262). That is relevant if the final viewer renders the low-poly scene in the same command stream.

---

## 5. Techniques for cheap video transformers

Literature numbers come from the cited papers, collected by a research sub-agent; I have not re-checked each one, and the agent's own "unverified" flags are kept. The "here" column links each technique to our measurements.

| Technique | Key sources (URLs) | Reported effect | Applicability here |
|---|---|---|---|
| **Token merging (ToMe) and successors** | ToMe https://arxiv.org/abs/2210.09461; vid-TLDR (CVPR'24) https://arxiv.org/abs/2403.13347; "Video, How Do Your Tokens Merge?" https://arxiv.org/abs/2506.03885; STTM (ICCV'25) https://github.com/HYUNJS/STTM; PiToMe https://arxiv.org/abs/2405.16148; ToFu https://arxiv.org/pdf/2312.01026; DiffRate https://arxiv.org/pdf/2305.17997; ATC https://vap.aau.dk/atc/ | ToMe: 2x image and 2.2x video throughput for 0.2–0.3% accuracy, training-free; up to 2x faster MAE video fine-tuning. vid-TLDR: −39.5% FLOPs with *better* retrieval. STTM: 2x at −0.5%. DiffRate: 1.5x throughput at −0.16% on ViT-H. | **Build in.** Cost is linear in tokens here (§2.11), so 2x fewer tokens gives 2x speed. Use a fixed *r* per layer so shapes stay static for `mx.compile`. Merge only in the encoder; slot queries cross-attend to merged tokens, with size-weighted attention as in ToMe. |
| Token pruning | DynamicViT https://arxiv.org/pdf/2106.02034.pdf; EViT/A-ViT comparison https://openaccess.thecvf.com/content/WACV2024/supplemental/Liu_Revisiting_Token_Pruning_WACV_2024_supplemental.pdf | DynamicViT: −66% tokens, −31–37% FLOPs, under 0.5% accuracy loss | Input-dependent counts break static shapes. Use only fixed-budget top-k variants. |
| **Shot detection and keyframes** | TransNetV2 https://github.com/soCzech/TransNetV2 (F1 77.9 ClipShots, 96.2 BBC); PySceneDetect `AdaptiveDetector` https://www.scenedetect.com/docs/api.html; DINO-diversity keyframe selection, e.g. DIG https://arxiv.org/pdf/2512.04000v1 | essentially free offline | **Build in.** Cuts define where camera slots reset and location slots switch. Sample 2–4 fps within shots, more where there is motion. |
| **Run-length / skip-static-patch tokens** | RLT, "Don't Look Twice" (NeurIPS'24) https://arxiv.org/html/2411.05222; EVEREST (ICML'24) https://arxiv.org/abs/2211.10636v6; motion-guided compression https://arxiv.org/abs/2402.18577v1 | RLT: −30% fine-tuning wall-clock at equal accuracy; +35% throughput training-free; up to 80% fewer tokens on long videos | **Build in.** Static backgrounds (sets) are exactly the redundant patches, and this is a natural fit for "location slots". Pad to a fixed token budget per clip (NaViT-style packing) to keep shapes static. |
| Codec-domain inputs (H.264/HEVC motion vectors) | CoViAR https://arxiv.org/abs/1712.00636 (4.6x faster than Res3D); MM-ViT https://arxiv.org/pdf/2108.09322.pdf; CoPE-VideoLM https://arxiv.org/abs/2602.13191 (up to 93% fewer tokens); PyAV `flags2=+export_mvs` → `frame.side_data` https://raw.githubusercontent.com/PyAV-Org/PyAV/main/tests/test_decode.py; `motion-vector-extractor` https://pypi.org/project/motion-vector-extractor/ | MVs are free but noisy flow; I-frames hint at cuts | **Medium.** Use MVs for patch-skip masks and camera-motion initialisation, not as a replacement for RGB, which the photometric loss needs. Edited, re-encoded clips have encoder-specific MVs. Whether HEVC/VideoToolbox can export MVs is unverified. |
| **Distillation / tiny pretrained students** | DINOv2 ViT-S/14 (21 M) https://github.com/facebookresearch/dinov2; DINOv3 ViT-S/16 (21 M) https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m; TinyViT (caches teacher logits on disk) https://arxiv.org/pdf/2207.10666; FastViT (Apple) https://arxiv.org/abs/2303.14189v1; Depth-Anything-V2-Small (24.8 M) https://github.com/DepthAnything/Depth-Anything-V2; Video-Depth-Anything-Small (28.4 M) https://github.com/DepthAnything/Video-Depth-Anything; Depth-Anything-3 Small/Base (80/120 M, Apache-2.0, depth plus pose) https://github.com/ByteDance-Seed/Depth-Anything-3 | – | **Build in.** From-scratch ViT-S at ImageNet scale (about 1e19 FLOP) would take about 27 days at our 4.5 TFLOPS. Start from DINO ViT-S weights; use DA-V2/V3-Small as cached teachers. |
| Linear attention / SSMs | Vision Mamba https://www.arxiv.org/abs/2401.09417; VideoMamba https://arxiv.org/abs/2403.06977 (6x faster than TimeSformer, 40x less memory at 64 frames); GLA/FLA https://github.com/sustcsonglin/flash-linear-attention (Triton, no Metal); MLX ports: mlx-lm `mamba.py`/`mamba2.py`/`gated_delta.py` https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models; mamba.py (MLX) https://github.com/alxndrTL/mamba.py | large gains only at long L | **Low priority.** With 2–4 fps × 60 s ≈ 120–240 frames, about 50–100 merged tokens per frame and factorised attention, sequences stay ≤ 4k, where MLX fused SDPA is 13.6 TFLOPS forward. MLX 0.32 has native gated-delta-net kernels (NAX) if a recurrent slot updater is needed. |
| **LoRA / DoRA** | LoRA; QLoRA https://arxiv.org/abs/2305.14314; DoRA https://arxiv.org/html/2402.09353v4; MLX-LM LoRA/QLoRA/DoRA https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md | QLoRA: 65 B on one 48 GB GPU at 16-bit quality | **Build in** for the pretrained encoder: freeze DINO-S and add LoRA rank 8–16 on qkv/proj. This saves about 16 B/param of optimiser state and the encoder's backward weight-gradient work. |
| 4-bit / 8-bit weights | MLX `quantize` (measured §2.10); RepQ-ViT W4A4 DeiT-S 69.0% vs 79.9% full precision https://arxiv.org/pdf/2212.08254 | q4 is a 4.2x win only when bandwidth-bound | 8-bit frozen teachers only. Do not quantise targets or the trained encoder. |

**Factorised attention (my recommendation).** Run spatial attention within each frame (about 100 tokens) and temporal attention through slots, rather than full space-time attention. Full attention over 16 frames × 197 tokens is L = 3152, which costs about 23 ms fwd+bwd per head-group at 4k in MLX; per-frame attention costs about 0.1 ms. Cost is dominated by the MLP/QKV GEMMs either way (§2.11).

---

## 6. Efficient differentiable rendering on Metal

**Status of existing libraries** (from the research agent; URLs given):

- **Not usable on Mac GPUs.**
  - PyTorch3D: GPU support is CUDA-only, and on Mac it runs on the CPU (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
  - nvdiffrast: NVIDIA only (https://nvlabs.github.io/nvdiffrast/).
  - Pulsar: same situation as PyTorch3D.
- **Metal Gaussian splatting.**
  - OpenSplat, Metal by default on macOS: https://github.com/pierotofy/OpenSplat
  - msplat, pure Metal: https://github.com/rayanht/msplat. Reports Garden at 30k iterations, 27.14 PSNR in 700 s on an M4 Max. It does not expose gradients to Python.
  - metalsplat, PyTorch MPS through `torch.mps.compile_shader`, 3DGS/2DGS with differentiable depth and normals, v0.2: https://github.com/tchauffi/metalsplat
  - gsplat-mps: https://github.com/iffyloop/gsplat-mps
  - Brush, wgpu/Burn: https://github.com/ArthurBrussee/brush
- **MLX.** mlx3D has a soft triangle rasteriser and Metal 3DGS kernels with `custom_function`, but is early (https://github.com/amirhossein-razlighi/mlx3D).
- **Mitsuba 3.9** is reported to add `metal_*` and `metal_ad_*` variants (https://mitsuba.readthedocs.io/en/v3.9.1/release_notes.html). Interop with MLX tensors is **unverified**.
- **Low-poly-friendly primitives:** convex polyhedra (https://arxiv.org/html/2407.15686v1), triangle splatting (https://papers.cool/arxiv/2506.18575), MeshSplatting (https://papers.cool/arxiv/2512.06818), superquadrics (https://arxiv.org/pdf/2603.16133).

**Measured conclusion.** Write our own MLX Metal kernels.
- Our roughly 100-line soft-triangle kernel already gives 1.8 ms fwd+bwd per 128² view at 1k triangles, and 127 ms at 100k. That is 40–126x faster than plain ops and uses less than 0.2 GB of memory.
- Plain-op SoftRas is unusable beyond about 1k triangles: 72–112 ms per view at 1k, and memory-bound at 116 B per pixel-triangle pair in backward.
- Planned kernel roadmap:
  1. Soft triangles with per-face colour (done, `bench_metal_kernel.py`).
  2. Tile binning: per-tile triangle lists from a bounding-box test, with a prefix sum on the GPU.
  3. Extra output channels for masks (per-entity id softmax), depth and DINO-feature splatting. These are the same loop with more accumulators.
  4. A hard z-buffer forward (atomic min on packed depth|id) with SoftRas-style gradients for sharp final renders.
  5. A `grid_sample` kernel for sampling feature maps. The MLX documentation's example can be reused.
- Keep the renderer's maths in fp32 inside the kernel, and keep accumulation order-independent (sums) so tiles and chunks compose.

---

## 7. Training-recipe tricks for small compute

| Trick | Evidence | How to use it here |
|---|---|---|
| **Progressive resolution / multi-resolution** | FFCV progressive resizing https://arxiv.org/pdf/2306.12517; Composer's recipe (`initial_scale=0.5`, `finetune_fraction=0.2`) https://docs.mosaicml.com/projects/composer/en/latest/method_cards/progressive_resizing.html; FlexiViT patch-size randomisation https://arxiv.org/pdf/2212.08013; ViT in under 24 h on one GPU with a size curriculum https://arxiv.org/pdf/2211.05187. **Measured here:** 112 px trains 4x faster than 224 px (670 vs 170 img/s). | Train at 112 px with 64² renders, then 160 px/96², then 224 px/128² for the last 20%. Use `mx.compile(..., shapeless=False)` and accept one recompile per stage (about 1–6 s). |
| **Curriculum over clip complexity** | general practice; Budgeted Training for ViT https://iclr.cc/virtual/2023/poster/11289 | Order the data as: single shot, static camera, one entity → camera motion → multiple entities → cuts and close-ups. Shorter clips (4–8 frames) come first. |
| **Cache frozen-teacher features to disk** | TinyViT stores sparsified teacher logits https://arxiv.org/pdf/2207.10666. Caveat: Beyer et al. find that *fixed* teacher targets plus augmentation generalise worse than "consistent and patient" distillation (https://arxiv.org/abs/2106.05237). | Our targets belong to the exact frames we render, so caching is correct. Precompute DA-V2/V3 depth, DINO-S features (patch tokens at 1/14 resolution, fp16), masks and shot boundaries once with Core ML on the ANE or GPU. Apply only geometric augmentations that can be applied identically to the cached maps. Rough storage: DINO-S 16×16×384 fp16 ≈ 196 KB per frame, so 240 frames ≈ 47 MB per clip; 2k clips ≈ 94 GB. Use 8×8 pooled or PCA-64 features (about 8 KB per frame) to make it tiny. |
| **Mixed precision** | §2.4: amp-bf16 is 1.6x over fp32 compiled | bf16 compute with fp32 master weights; fp32 in renderer kernels and pose maths |
| **Gradient checkpointing** | §2.5: −50% activations for +23% time | Turn it on only when batch or clip length is memory-bound |
| **LoRA on a pretrained encoder** | §5 | Removes the encoder's optimiser state and most of its backward pass |
| **Static shapes everywhere** | `mx.compile` recompiles on shape change | Use fixed token budgets per frame (merge or skip to *k*), a fixed slot count with existence masks, and padded triangle buffers |
| **Explicit `mx.eval` boundaries and memory caps** | §2.5, §2.7 | Evaluate after each micro-batch and renderer chunk; `mx.set_memory_limit(10<<30)` |

---

## 8. Compute budget we realistically have

Measured building blocks (MLX, amp-bf16, compiled; sustained, with no throttling seen over 150 s):

| block | throughput |
|---|---|
| ViT-S encoder training (fwd+bwd+AdamW) @224 / @160 / @112 | 168 / 343 / 670 frames/s |
| ViT-B encoder training @224, b8 | 61 frames/s |
| ViT-S frozen forward (inference, bf16) | 636 frames/s (MLX b64); about 970 frames/s with Core ML GPU at b8 |
| Soft-triangle render fwd+bwd, 128², custom kernel | 1.8 ms/view (1k tris), 12.8 ms/view (10k tris) |
| Sustained matmul-heavy compute | about 4.5 TFLOPS (ViT-S), 6.2–6.5 TFLOPS (ViT-B/L): about 4e17 FLOP/day |

**Headline numbers**
- **ViT-S/16 @224 training: about 19k steps/h at batch 32** (189 ms/step), or **about 64–72k steps/h at batch 8** (50–56 ms/step). That is about 0.6 M images per hour and about 14 M images per 24 h day, roughly 11 ImageNet-1k epochs per day.
- DeiT-S-from-scratch (300 epochs, about 1.1e19 FLOP) would take about 27 days, so pretrained initialisation is mandatory.

**Estimated full-model step.** One training sample is one clip: 16 frames at 160 px with the encoder trainable through LoRA, a 6-layer slot transformer, 16 rendered 128² views at 1k triangles, and cached teacher targets.

| component | per clip |
|---|---|
| encoder | about 47 ms |
| slot transformer and decoders | about 20 ms (estimate) |
| renderer (16 × 1.8 ms) | about 29 ms |
| losses and data | about 10 ms |
| **total** | **about 0.1 s** |

That gives about 30–35k clip-steps per hour at batch 1, or **about 8k steps per hour at batch 4 clips**. An overnight 10-hour run is about 80k steps, and a full week of nights about 0.5 M steps.

This is a respectable research budget if the model stays at about 30–60 M trainable parameters, about 100 tokens per frame and about 1–5k triangles per view. It is not enough to pre-train anything large from scratch.

---

## 9. Optimisations to build in from day one (ranked)

1. **MLX, whole-step `mx.compile`, bf16 compute with fp32 master weights.** 2.6–2.9x over fp32 eager, and 1.35–1.45x over the best torch MPS setup.
2. **Custom Metal rasteriser** (`mx.fast.metal_kernel` + `custom_function` VJP) from the start: 40–126x over plain ops, at less than 0.2 GB. Add tile binning next, and mask, depth and feature channels in the same loop.
3. **Pretrained, mostly frozen encoder** (DINOv2/v3 ViT-S) with LoRA, plus **offline teacher caching** (DA-V2/V3-Small depth, DINO features, masks, shot cuts) computed with Core ML on the ANE or GPU. Teachers never run inside the training loop.
4. **Token budget control with static shapes:** low resolution (112–160 px), static-patch skipping (RLT-style) and fixed-*r* ToMe, all padded to a fixed *k* tokens per frame; factorised spatial versus slot-temporal attention; 2–4 fps keyframes within detected shots. Cost is linear in tokens (§2.11).
5. **Memory discipline:**
   - `mx.set_memory_limit(~10 GB)` and `mx.set_cache_limit(~2 GB)`.
   - An explicit `mx.eval` per micro-batch and per render chunk; MLX's lazy graph otherwise holds every chunk.
   - Gradient checkpointing switchable per block.
   - In torch, always set `set_per_process_memory_fraction`.
6. **Numerics hygiene:**
   - Remember MLX fp32 matmul is TF32 on M5. Run gradient checks and pose solvers with `MLX_ENABLE_TF32=0`.
   - Avoid differentiable SVD, inverse and eigh (no GPU support or VJP in MLX). Use 6D rotations or quaternions and closed-form rigid inverses.
7. **Progressive curriculum:** resolution 112 → 160 → 224, render 64² → 128², clip length 4 → 16 → 64 frames, and scene complexity from a single static shot to multi-shot with entities.

---

## References

**Apple / MLX / PyTorch (fetched directly)**
- Apple M5 press release: https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU": https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- WWDC25 session 262, "Combine Metal 4 machine learning and graphics": https://developer.apple.com/videos/play/wwdc2025/262/
- Apple, "Deploying Transformers on the Apple Neural Engine": https://machinelearning.apple.com/research/neural-engine-transformers ; code https://github.com/apple/ml-ane-transformers
- MLX v0.30.0 release (M5 Neural Accelerator support): https://github.com/ml-explore/mlx/releases/tag/v0.30.0 ; all releases https://github.com/ml-explore/mlx/releases
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX compile: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX unified memory: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX `set_wired_limit`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html
- MLX `nn.quantize`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html
- MLX-LM LoRA/QLoRA: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- PyTorch MPS notes: https://docs.pytorch.org/docs/2.14/notes/mps.html ; MPS environment variables: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html ; `torch.mps.compile_shader`: https://docs.pytorch.org/docs/2.14/generated/torch.mps.compile_shader.html
- bitsandbytes releases (MPS status): https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- Apple Core ML Depth-Anything-V2-Small: https://huggingface.co/apple/coreml-depth-anything-v2-small

**Literature and repositories (via the research agent)** are listed inline in §5–§7: ToMe, vid-TLDR, STTM, PiToMe, ToFu, DiffRate, ATC, DynamicViT, TransNetV2, PySceneDetect, RLT, EVEREST, CoViAR, MM-ViT, CoPE-VideoLM, PyAV, mv-extractor, DINOv2/v3, TinyViT, FastViT, Depth-Anything V2/V3, Video-Depth-Anything, Vision Mamba, VideoMamba, FLA, mamba.py, QLoRA, DoRA, RepQ-ViT, PyTorch3D, nvdiffrast, OpenSplat, msplat, metalsplat, gsplat-mps, Brush, mlx3D, Mitsuba 3.9, FFCV, Composer, FlexiViT, NaViT (https://arxiv.org/pdf/2307.06304.pdf), Beyer et al.

**Local evidence (this machine)**
- `strings libmlx.dylib` for NAX kernel names and `MLX_ENABLE_TF32`.
- `mx.device_info()` for working-set limits.
- Raw JSON for every table in `research/bench/results/`.
