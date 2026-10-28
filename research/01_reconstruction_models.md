# 01: Feed-forward 3D and 4D reconstruction from images and video

*Research survey for the "video to animated 3D scene" project. State of the field as of 2026-10-28.*

## How this was researched, and how far to trust it

- Facts come from GitHub READMEs and LICENSE files (read through `gh api`), Hugging Face model metadata (`huggingface.co/api/models/...`, which gives the licence tag, gating and safetensors parameter count), and arXiv abstract and HTML pages.
- Three helper agents covered three slices in parallel: dynamic/4D, human-scene and 4D Gaussians, and Apple Silicon support. I checked their most load-bearing claims myself, including licences, gating, repo existence, the Dr.Jit Metal release and the existence of the 2026 papers.
- **Search budget.** The shared web-search budget for this session ran out partway through. After that, work used only direct fetches of known URLs. Some 2026 papers are therefore known from one abstract only. Coverage of late-2026 work is probably incomplete. A follow-up session with a fresh search budget could fill this in.
- **Markers:**
  - **[U]** means unverified: an author's own claim, a third-party benchmark, or something read in code but not run.
  - **[E]** means my own estimate or inference.
- **Nothing was downloaded or run.** All speeds are as reported by the source, on the hardware stated, and are not measured on an M5.

---

## 1. Key takeaways

1. **Four families exist, and none of them does our task.**
   - Pairwise pointmap regression: DUSt3R and its descendants.
   - One-shot multi-view transformers: VGGT, π3, MapAnything, DA3, VGGT-Ω.
   - Streaming / recurrent-state models: CUT3R, TTT3R, StreamVGGT, STream3R.
   - Dynamic / 4D models: MonST3R, Any4D, D4RT, V-DPM, St4RTrack, SpatialTrackerV2. Human-specific variants (Human3R, JOSH3R, UniSH) and feed-forward 4D Gaussian models (L4GM, 4DGT, BTimer, MoVieS) build on these.

   All of them output dense, per-pixel, per-frame geometry: depth, pointmaps, rays, Gaussians or tracks. None outputs persistent objects, rigs, low-poly sets or an edit structure.
2. **Every video method assumes one continuous shot.**
   - The assumption takes several forms: frame-1 anchoring (VGGT reference frame, Any4D, St4RTrack), recurrent state (CUT3R, TTT3R, Human3R), temporal tubelets (D4RT) and flow-based alignment (MonST3R, MegaSaM).
   - None of the reviewed papers detects or handles cuts.
   - The unordered-set models (π3, MapAnything, DA3, VGGT) accept shuffled frames. They always force all inputs into one coordinate frame, even when the frames come from different places. MapAnything's benchmark deliberately samples only connected view graphs ([paper](https://arxiv.org/html/2509.13414)).
3. **The 2026 trend matches our slot design.**
   - VGGT-Ω (May 2026) adds *registers* that hold a compact scene summary. Its "register attention" limits exchange between frames to those registers, and it adds self-supervised training on unlabeled video ([project](https://vggt-omega.github.io/), [arXiv](https://arxiv.org/abs/2605.15195)). This closely resembles our "location slot".
   - Query-based decoders (D4RT, UniQuery4R) decode any (pixel, source time, target time, camera) query on its own. That suits per-entity motion decoding on low memory ([D4RT](https://arxiv.org/html/2512.08924v2), [UniQuery4R](https://arxiv.org/abs/2608.17283)).
4. **Strong, permissive, ungated teachers that should fit in 16 GB.**
   - MapAnything-apache, Apache-2.0: multi-view metric geometry, 1.23B params, official MPS support.
   - DA3-Small and DA3-Base, Apache-2.0: any-view depth, rays and pose.
   - DA3Metric-Large and DA3Mono-Large, Apache-2.0: single-image metric and relative depth.
   - MoGe-2, MIT: single-image metric pointmap with intrinsics; works on MPS.
   - Video Depth Anything Small, Apache-2.0.
   - Depth Pro: Apple sample-code licence.
   - SAM 2.1, Apache-2.0.
   - SEA-RAFT (BSD-3) and AllTracker (MIT code).
   - MHR, Apache-2.0: the body rig, with no login needed.
   - L4GM, Apache-2.0: object-level 4D Gaussians.

   Sources are in §7.
5. **Most of the famous strong models are non-commercial (NC), gated, or both.**
   - NC: DUSt3R, MASt3R, MASt3R-SLAM, CUT3R, TTT3R, MonST3R, π3/Pi3X weights, VGGT-1B, Human3R, CoTracker3, SpatialTrackerV2, 4DGT.
   - Gated (manual approval), which our no-gated-downloads constraint excludes: VGGT-1B-Commercial, VGGT-Ω, SAM 3, SAM 3D Body, SAM 3D Objects.
   - VGGT-Ω's FAIR NC licence also forbids using its *outputs* commercially ([LICENSE](https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE)). Its pseudo-labels would therefore taint any commercial use.
6. **SMPL/SMPL-X and BEDLAM are NC and need registration.**
   - Sources: [SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html), [BEDLAM licence](https://bedlam.is.tue.mpg.de/license.html).
   - Almost every human-scene model inherits this. Meta's **MHR** is Apache-2.0 and its assets come from a GitHub release ([repo](https://github.com/facebookresearch/MHR)). NAVER's **Anny** is Apache-2.0 code with CC0 assets ([repo](https://github.com/naver/anny)). Both avoid the problem.
7. **Apple Silicon is now viable for inference with several teachers.**
   - MapAnything (official MPS PR [#131](https://github.com/facebookresearch/map-anything/pull/131)).
   - VGGT (MPS fix [#112](https://github.com/facebookresearch/vggt/pull/112)).
   - DA3 (MPS fix [#26](https://github.com/ByteDance-Seed/Depth-Anything-3/pull/26)).
   - DA2 (official Core ML from [Apple](https://huggingface.co/apple/coreml-depth-anything-v2-small)).
   - Depth Pro, MoGe-2, DUSt3R and MASt3R.
   - Differentiable rendering on Metal now exists:
     - [Dr.Jit 1.4 Metal backend](https://github.com/mitsuba-renderer/drjit/releases/tag/v1.4.0), BSD-3, released 2026-08.
     - [metal-gauss](https://github.com/nandometzger/metal-gauss) and [MetalSplat](https://github.com/tchauffi/metalsplat), both MIT, for Gaussians.
     - [mtldiffrast](https://github.com/pedronaugusto/mtldiffrast), MIT, an nvdiffrast-style API; young.
   - Still CUDA-only: MASt3R-SLAM, MegaSaM (lietorch / DROID kernels), MoGe-3 (Triton), and gsplat/nvdiffrast-based code.
8. **Training these models took 8 to 128 datacentre GPUs.** DA3 used 128 H100s for 200k steps ([paper](https://arxiv.org/html/2511.10647)). D4RT used 64 TPUs for about 2 days ([paper](https://arxiv.org/html/2512.08924v2)). 4DGT used 64 H100s ([paper](https://arxiv.org/html/2506.08015)). π3 reports that training from scratch "converges poorly" and initialises from VGGT ([paper](https://arxiv.org/html/2507.13347)). The cheapest recipe found is Human3R: visual prompt tuning on a *frozen* CUT3R, about 1 day on one 48 GB GPU ([paper](https://arxiv.org/html/2510.06219v2)). **Our design should freeze or borrow a backbone and train a small "compiler" on cached features, not train a ViT from scratch on the laptop.** [E]

---

## 2. How these models work: a short technical primer

- **Pointmap regression (DUSt3R, [arXiv](https://arxiv.org/abs/2312.14132)).**
  - A Siamese ViT-L encoder and two cross-attending ViT-B decoders map an image pair to two pixel-aligned 3D pointmaps, both in the first camera's frame, plus confidence maps ([README](https://github.com/naver/dust3r)).
  - Cameras are not predicted directly. Intrinsics and pose are recovered from the pointmaps, and many pairs are merged with a *global alignment* optimisation over a pair graph.
  - **MASt3R** adds a dense local-feature head for matching and metric pointmaps ([arXiv](https://arxiv.org/abs/2406.09756)).
  - **MASt3R-SfM** uses the encoder as an image retriever, so it can handle "any collection of images, ordered or not" ([arXiv](https://arxiv.org/abs/2409.19152)).
- **One-shot multi-view transformers (VGGT, [paper](https://arxiv.org/html/2503.11651)).**
  - DINOv2 patch tokens plus one camera token and four register tokens per frame pass through 24 pairs of *frame-wise* and *global* self-attention layers, about 1.2B params.
  - Heads output camera (Huber loss), depth and pointmaps (uncertainty-weighted, with a gradient term) and track features (a CoTracker2-style head).
  - The first frame gets special tokens and defines the world frame.
  - **π3** removes the reference frame: it predicts affine-invariant poses and scale-invariant local pointmaps with a fully permutation-equivariant design ([paper](https://arxiv.org/html/2507.13347)).
  - **MapAnything** *factors* geometry into per-view ray maps, depth along ray, pose and one global metric scale. It optionally ingests intrinsics, poses or depth ([paper](https://arxiv.org/html/2509.13414)).
  - **DA3** shows that a plain DINO ViT with cross-view attention in its later layers, predicting only depth and ray maps, is enough ([paper](https://arxiv.org/html/2511.10647)).
  - **VGGT-Ω** keeps only camera and depth as targets. It adds registers and "register attention", and trains on 15× more labelled data plus unlabeled video ([arXiv](https://arxiv.org/abs/2605.15195)).
- **Streaming / recurrent models (CUT3R, [paper](https://arxiv.org/html/2501.12387)).**
  - A persistent state of 768 tokens of width 768 is read and updated by each new frame. Each frame outputs pointmaps in its own frame and in the world frame, plus pose.
  - The state can be queried with a ray map of a *virtual* view to hallucinate unseen regions.
  - The authors say it drifts on long sequences because there is no global alignment.
  - **TTT3R** makes the state update a confidence-gated test-time-training step ([arXiv](https://arxiv.org/abs/2509.26645)). StreamVGGT and STream3R use causal attention with a KV cache. Point3R keeps an explicit 3D pointer memory.
- **Dynamic / 4D models.**
  - **MonST3R** fine-tunes DUSt3R on dynamic data so each timestep gets its own pointmap ([arXiv](https://arxiv.org/abs/2410.03825)).
  - **DPM / V-DPM** predict pointmaps of both frames at both times, which gives scene flow ([DPM](https://arxiv.org/html/2503.16318v1), [V-DPM](https://arxiv.org/abs/2601.09499)).
  - **St4RTrack** predicts frame-1 content at time j, which gives world-frame tracks ([project](https://st4rtrack.github.io/)).
  - **D4RT** encodes a clip with a ViT-g, and a small cross-attention decoder answers per-point queries "where is pixel (u,v) of time s at time t, in camera c" ([paper](https://arxiv.org/html/2512.08924v2)).
  - **Trace Anything** predicts B-spline control points per pixel, so each pixel gets a continuous trajectory ([arXiv](https://arxiv.org/abs/2510.13802)).
- **Feed-forward Gaussians.** Pixel-aligned Gaussian heads sit on a geometry backbone: DA3's GS-DPT head, AnySplat, YoNoSplat and WorldMirror. 4D variants add time:
  - **4DGT**: Gaussians with a temporal centre, a lifespan, velocity and angular velocity ([paper](https://arxiv.org/html/2506.08015)).
  - **BTimer**: Gaussians for one queried "bullet" time ([project](https://research.nvidia.com/labs/toronto-ai/bullet-timer/)).
  - **L4GM**: per-frame object Gaussians ([project](https://research.nvidia.com/labs/toronto-ai/l4gm/)).
- **Typical losses:**
  - Confidence-weighted (aleatoric) regression on points and depth, after an optimal scale alignment. π3 uses depth-weighted L1 after a closed-form s* ([paper](https://arxiv.org/html/2507.13347)).
  - Normal and gradient-matching terms, usually on synthetic data only.
  - Camera rotation and translation losses.
  - BCE for confidence and visibility.
  - Robust losses with top-k outlier rejection: MapAnything drops the top 5% of per-pixel losses ([paper](https://arxiv.org/html/2509.13414)).
  - Teacher pseudo-labels: DA3 trains a synthetic-only monocular teacher and aligns it to noisy real depth with RANSAC scale-shift ([paper](https://arxiv.org/html/2511.10647)).

---

## 3. Comparison tables

Column notes:
- **FF**: feed-forward. **TTO**: test-time optimisation. **NC**: non-commercial.
- **Params** come from HF safetensors where available, otherwise from the paper.
- **Speed** is as reported by the source, with its hardware.
- The **Licence** column gives code / weights.

### 3a. Pairwise pointmap, SfM and SLAM family

| Name | Year / venue | Inputs | Outputs | Params | Speed (reported) | Licence (code / weights) | URL |
|---|---|---|---|---|---|---|---|
| DUSt3R | 2024, CVPR | Image pairs, then global alignment (GA) over N images | Pointmaps and confidence; poses and intrinsics recovered via GA | About 0.6B (ViT-L encoder, ViT-B decoder; same architecture as MonST3R's 0.6B card) | 1.25 FPS ([π3 table](https://arxiv.org/html/2507.13347)). M2 Mac: 3 images = 6 s inference + 15 s GA ([issue #31](https://github.com/naver/dust3r/issues/31)) | CC BY-NC-SA 4.0 / same | [github](https://github.com/naver/dust3r) |
| MASt3R | 2024, ECCV | Pairs | Metric pointmaps, dense matching descriptors | About 0.7B [U] (ViT-L/ViT-B) | MPS full model 1204 ms per pair on an M4 Max, MLX 767 ms [U] ([mlx-mast3r](https://github.com/Aedelon/mlx-mast3r)) | CC BY-NC-SA 4.0 / same | [github](https://github.com/naver/mast3r) |
| MASt3R-SfM | 2024 arXiv (3DV 2025 [U]) | Unordered collections (retrieval-built graph) | Global poses and points | MASt3R | n/r | CC BY-NC-SA 4.0 | [arXiv](https://arxiv.org/abs/2409.19152) |
| MASt3R-SLAM | 2025, CVPR | Monocular video, no fixed camera model | Poses, dense map, loop closure | MASt3R | 15 FPS ([arXiv](https://arxiv.org/abs/2412.12392)) | CC BY-NC-SA 4.0 ([LICENSE](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)) | [github](https://github.com/rmurai0610/MASt3R-SLAM) |
| MUSt3R | 2025, CVPR | Image collections or video (online, multi-layer memory) | Pointmaps, poses | n/r | n/r | MUSt3R NC licence; mapfree training data "very restrictive" | [github](https://github.com/naver/must3r) |
| SLAM3R | 2025, CVPR | Video | Dense points without explicit camera estimation | n/r | Real-time [U] | CC BY-NC-SA 4.0 | [github](https://github.com/PKU-VCL-3DV/SLAM3R) |

### 3b. One-shot multi-view transformers

| Name | Year / venue | Inputs | Outputs | Params | Speed (reported) | Licence (code / weights) | URL |
|---|---|---|---|---|---|---|---|
| Fast3R | 2025, CVPR | 1000+ unordered images in one pass | Pointmaps, poses | n/r | n/r | FAIR NC / FAIR NC | [github](https://github.com/facebookresearch/fast3r) |
| VGGT | 2025, CVPR best paper | 1 to hundreds of images; frame 1 is the reference | Cameras (extrinsics and intrinsics), depth, pointmaps, track features | 1.26B ([HF](https://huggingface.co/facebook/VGGT-1B)) | H100: 1 frame 0.04 s / 1.9 GB; 50 frames 1.04 s / 11.4 GB; 100 frames 3.12 s / 21 GB ([paper](https://arxiv.org/html/2503.11651)). A May 2026 memory fix allows 2–3× more frames ([README](https://github.com/facebookresearch/vggt)) | VGGT licence, commercial use OK except military / VGGT-1B is **CC BY-NC 4.0**, ungated. **VGGT-1B-Commercial** is commercial but **gated (manual)** | [github](https://github.com/facebookresearch/vggt) |
| VGGT-Ω (Omega) | 2026, CVPR 2026 best-paper finalist | Images or video, static or dynamic | Camera, depth, registers (compact scene tokens), optional text-aligned embedding | About 1B (checkpoint names "1b") | A100, 624×416 input, peak memory including weights: 1 frame 6.0 GB, 50 frames 9.7 GB, 100 frames 13.4 GB ([README](https://github.com/facebookresearch/vggt-omega)) | **FAIR Noncommercial Research License** (outputs too) / **gated (manual)** | [project](https://vggt-omega.github.io/) |
| π3 | 2025, arXiv | Unordered set; permutation-equivariant, no reference view | Local pointmaps, c2w poses, confidence | 959M ([HF](https://huggingface.co/yyfz233/Pi3)) | 57.4 FPS on an A800 (KITTI) vs VGGT 43.2 ([paper](https://arxiv.org/html/2507.13347)) | BSD-3 / **CC BY-NC 4.0** ([README](https://github.com/yyfz/Pi3)) | [github](https://github.com/yyfz/Pi3) |
| Pi3X | Dec 2025 | Same, plus optional poses, intrinsics and depth | Same, plus approximate metric scale and better confidence | 1.36B ([HF](https://huggingface.co/yyfz233/Pi3X)) | n/r | BSD-3 / CC BY-NC 4.0 | [github](https://github.com/yyfz/Pi3) |
| MapAnything (v1.1, Jan 2026) | 2025–26, arXiv / 3DV [U] | 1 to 2000 images plus optional intrinsics, poses, depth | Metric pointmaps, ray dirs, depth, poses, intrinsics, metric scale, confidence, masks | 1.23B ([HF](https://huggingface.co/facebook/map-anything-apache)). DINOv2 ViT-G features plus a 16-layer alternating transformer | Up to 2000 views on 140 GB in memory-efficient mode ([README](https://github.com/facebookresearch/map-anything)) | Apache-2.0 / **map-anything-apache: Apache-2.0, ungated**; map-anything: CC BY-NC 4.0 | [github](https://github.com/facebookresearch/map-anything) |
| Depth Anything 3 (DA3) | Nov 2025, arXiv | 1 to N images, optional poses | Depth plus ray maps (gives camera), pose, optional 3DGS (Giant), sky mask (Mono/Metric) | Small 22M backbone (34M checkpoint), Base 86M (135M), Large 0.30–0.35B, Giant 1.13–1.15B ([README](https://github.com/ByteDance-Seed/Depth-Anything-3), [HF](https://huggingface.co/depth-anything/DA3-SMALL)) | A100 at 504×336: Small 160 FPS, Base 127, Large 78, Giant 38. Max images on 80 GB: Small about 4000, Giant about 900 ([paper](https://arxiv.org/html/2511.10647)) | Apache-2.0 / **Small and Base Apache-2.0; Metric-Large and Mono-Large Apache-2.0**; Large and Giant CC BY-NC 4.0 | [github](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| WorldMirror / WorldMirror 2.0 | 2025 / Apr 2026 | Images plus any priors (intrinsics, pose, depth) | Pointmaps, depth, normals, cameras, 3DGS | About 1.2B (v2.0) | n/r | Tencent community licence; does not apply in EU, UK or South Korea | [github](https://github.com/Tencent-Hunyuan/HY-World-2.0) |

### 3c. Streaming, long-sequence and efficiency variants

| Name | Year / venue | Inputs | Outputs | Params | Speed (reported) | Licence | URL |
|---|---|---|---|---|---|---|---|
| Spann3R | 2024–25 | Ordered frames, spatial memory | Pointmaps in a global frame | n/r | n/r | CC BY-NC-SA 4.0 | [github](https://github.com/HengyiWang/spann3r) |
| CUT3R | 2025, CVPR oral | Stream or unordered photos (claimed); recurrent state | Per-frame self and world pointmaps, pose; virtual-view queries | ViT-L encoder, 2× ViT-B decoders; total n/r | About 16.6–17 FPS on an A100 at 512×144 ([paper](https://arxiv.org/html/2501.12387)) | CC BY-NC-SA 4.0 (code and weights) | [github](https://github.com/CUT3R/CUT3R) |
| TTT3R | 2025, ICLR 2026 | Ordered stream; `--reset_interval` | As CUT3R, with better long-range pose | CUT3R | 20 FPS, 6 GB ([arXiv](https://arxiv.org/abs/2509.26645)) | MIT code + CUT3R CC BY-NC-SA weights ([LICENSE](https://github.com/Inception3D/TTT3R)) | [github](https://github.com/Inception3D/TTT3R) |
| StreamVGGT | 2025, arXiv | Ordered stream, causal attention, KV cache | VGGT outputs online | About VGGT size | n/r | CC BY-NC-SA 4.0 / CC BY-NC 4.0 ([HF](https://huggingface.co/lch01/StreamVGGT)) | [github](https://github.com/wzzheng/StreamVGGT) |
| STream3R | 2025, arXiv | Ordered stream, decoder-only causal transformer | Pointmaps, poses, static and dynamic | n/r | H200 table in README | S-Lab NC licence | [github](https://github.com/NIRVANALAN/STream3R) |
| Point3R | 2025, arXiv | Stream, explicit 3D pointer memory | Pointmaps, depth, pose | n/r | n/r | CC BY-NC-SA 4.0 | [github](https://github.com/YkiWu/Point3R) |
| WinT3R | 2025, arXiv | Stream, sliding window plus camera-token pool | Online pointmaps and poses | n/r | "fastest … to date" [U] | Non-commercial | [github](https://github.com/LiZizun/WinT3R) |
| VGGT-Long | 2025, arXiv | km-scale video, chunk plus align plus loop closure | Poses, points | VGGT | 0.175 s per alignment iteration | VGGT licence | [github](https://github.com/DengKaiCQ/VGGT-Long) |
| VGGT-SLAM 2.0 | Jan 2026 | Video | Real-time dense SLAM; optional SAM 3 object detection | VGGT | Real-time [U] | BSD-2 code; VGGT weights | [github](https://github.com/MIT-SPARK/VGGT-SLAM), [arXiv](https://arxiv.org/abs/2601.19887) |
| DA3-Streaming | Dec 2025 | Ultra-long video, sliding window | DA3 outputs | DA3 | Under 12 GB GPU memory ([README](https://github.com/ByteDance-Seed/Depth-Anything-3)) | Follows DA3 weights | [readme](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) |
| FastVGGT | 2025 | 1000 frames, training-free token merging | VGGT outputs | VGGT | Speed-up claims in README [U] | VGGT licence | [github](https://github.com/mystorm16/FastVGGT) |
| QuantVGGT | 2025, ICLR 2026 | Same as VGGT | VGGT outputs, 4-bit | VGGT | 3.7× less memory, 2.5× faster, accuracy over 98% retained ([arXiv](https://arxiv.org/abs/2509.21302)) | Apache-2.0 code ([repo](https://github.com/wlfeng0509/QuantVGGT)); VGGT weights | [github](https://github.com/wlfeng0509/QuantVGGT) |

### 3d. Monocular depth and geometry teachers

| Name | Year | Inputs | Outputs | Params | Speed | Licence (code / weights) | URL |
|---|---|---|---|---|---|---|---|
| Depth Anything V2 | 2024, NeurIPS | Single image | Relative depth (disparity) | S 24.8M, B 97.5M, L 335M | Core ML Small on an M3 Max ANE: 24.6 ms at 518×396 ([HF](https://huggingface.co/apple/coreml-depth-anything-v2-small)) | Apache-2.0 / **Small Apache-2.0**; B, L, G CC BY-NC 4.0 ([README](https://github.com/DepthAnything/Depth-Anything-V2)) | [github](https://github.com/DepthAnything/Depth-Anything-V2) |
| DA3Mono-Large / DA3Metric-Large | Nov 2025 | Single image | Relative depth / metric depth (metric = focal × out / 300), sky mask | 0.35B (334M, [HF](https://huggingface.co/depth-anything/DA3METRIC-LARGE)) | n/r | **Apache-2.0** | [github](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| Video Depth Anything | 2025, CVPR | Arbitrarily long video | Temporally consistent depth | Small checkpoint 0.12 GB | n/r | Apache-2.0 / **Small Apache-2.0**; B and L CC BY-NC 4.0 ([README](https://github.com/DepthAnything/Video-Depth-Anything)) | [github](https://github.com/DepthAnything/Video-Depth-Anything) |
| MoGe-2 | 2025 | Single image | **Metric** pointmap, depth, **intrinsics**, normals, mask | ViT-S 35M, ViT-B 104M, ViT-L 326–331M | M1 Max MPS, ViT-S at 384 px: 212 ms [U] ([report](https://dev.to/kiarina/estimating-surface-orientation-and-3d-from-one-image-with-moge-2-on-apple-silicon-2n0h)) | **MIT / MIT**, ungated ([README](https://github.com/microsoft/MoGe)) | [github](https://github.com/microsoft/MoGe) |
| MoGe-3 | Aug 2026 | Single image | Fine-detail metric pointmap via sparse volumetric refinement | ViT-L 370M, ViT-G 1.25B | n/r | MIT; **"macOS is not supported"** (Triton) | [github](https://github.com/microsoft/MoGe) |
| Depth Pro | 2024, Apple | Single image | Metric depth plus focal length | n/r | MPS fp16 3.76 s, 7 GB at 3024×2268 on an M4 Pro [U] ([MLX port](https://huggingface.co/ntoan199/DepthPro-mlx)) | Apple sample-code licence (code and weights) | [github](https://github.com/apple/ml-depth-pro) |

### 3e. Dynamic / 4D reconstruction and tracking

| Name | Year / venue | FF or TTO | Inputs | Outputs | Params | Speed | Licence (code / weights) | URL |
|---|---|---|---|---|---|---|---|---|
| MonST3R | 2025, ICLR | FF pairs plus global optimisation (real-time mode is FF) | Video | Per-timestep pointmaps, poses, intrinsics, dynamic masks | 0.6B ([HF](https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt)) | About 33 GB for 65 frames | CC BY-NC-SA / same | [github](https://github.com/Junyi42/monst3r) |
| Easi3R | 2025, ICCV | Training-free (DUSt3R attention) | Video | Dynamic masks, poses, 4D pointmaps | DUSt3R | n/r | CC BY-NC-SA | [github](https://github.com/Inception3D/Easi3R) |
| Align3R | 2025, CVPR | FF plus global alignment | Video plus mono-depth prior | Consistent video depth, poses | DUSt3R | n/r | CC BY-NC-SA | [github](https://github.com/jiah-cloud/Align3R) |
| POMATO | 2025, ICCV | FF pairs plus 12-frame temporal module | Video | Depth, dynamic masks, 3D tracks, poses | n/r | n/r | CC BY-NC-SA | [arXiv](https://arxiv.org/abs/2504.05692) |
| D²USt3R | 2025, NeurIPS | FF pairs | Pairs | Static-dynamic aligned pointmaps | n/r | n/r | No LICENSE file | [github](https://github.com/cvlab-kaist/DDUSt3R) |
| DPM / V-DPM | 2025 / CVPR 2026 | FF | Pairs / video (V-DPM on VGGT) | Pointmaps at both times, giving scene flow, motion segmentation, pose | n/r | n/r | V-DPM: MIT code plus VGGT terms | [V-DPM](https://github.com/eldar/vdpm) |
| St4RTrack | 2025, ICCV | FF pairs (frame 1, frame j), optional test-time adaptation | Video | World-frame 3D tracks plus geometry | n/r | n/r | NC research only | [project](https://st4rtrack.github.io/) |
| Stereo4D / DynaDUSt3R | 2025, CVPR oral | Dataset pipeline; FF pair model | VR180 YouTube clips (more than 100k) | Metric poses, 3D tracks; pair to points plus motion | n/a | n/a | CC0 code, CC BY 4.0 annotations; videos are YouTube IDs only | [github](https://github.com/Stereo4D/stereo4d-code) |
| Geo4D | 2025, ICCV | Video diffusion (DynamiCrafter) plus fusion optimisation | 16-frame windows | Disparity, pointmaps, ray maps | n/r | 1.89 s/frame | Unverified | [arXiv](https://arxiv.org/html/2504.07961) |
| Sora3R | 2025, arXiv | FF video diffusion | Video | 4D pointmaps | n/r | n/r | No code found | [arXiv](https://arxiv.org/abs/2503.21082) |
| PAGE-4D | ICLR 2026 | FF (VGGT fine-tuned with a dynamics mask) | Video | Poses, depth, pointmaps | VGGT | n/r | Not stated | [project](https://page4d.github.io/) |
| VGGT4D | 2025, arXiv | Training-free (VGGT attention Gram masks) | Over 500 frames | Depth, poses, dynamic masks | VGGT | n/r | MIT plus VGGT weights | [github](https://github.com/3DAgentWorld/VGGT4D) |
| Any4D | CVPR 2026 | FF (from MapAnything) | N frames (frame 1 is the reference) plus optional depth, IMU, radar | Metric scale, rays, depth, poses, scene flow from frame 1 | n/r (7.7 GB checkpoint) | 0.5 s per 50 frames on an H100 | **Apache-2.0** / weights untagged ([HF](https://huggingface.co/airlabshare/any4d-checkpoint)) | [github](https://github.com/Any-4D/Any4D) |
| D4RT | Dec 2025, arXiv; DeepMind blog Jan 2026 | FF query decoder | Ordered clip (48 frames at 256² in training), Sim(3)-chained chunks | Any (u, v, t_src, t_tgt, camera) query to a 3D point, giving depth, tracks, poses, intrinsics | Encoder about 1B (ViT-g), decoder 144M | Pose over 200 FPS on an A100 | **No code or weights released** | [paper](https://arxiv.org/html/2512.08924v2), [blog](https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/) |
| Trace Anything | ICLR 2026 | FF | Up to 40 frames at 512 px (temporal order assumed in code) | Per-pixel B-spline trajectories | n/r | Needs at least 48 GB VRAM | Apache-2.0 / **CC BY-NC 4.0** | [github](https://github.com/ByteDance-Seed/TraceAnything) |
| SpatialTrackerV2 | 2025, ICCV | FF with differentiable bundle adjustment (BA) | Video (12–48 frames in training) | World 3D tracks, poses, focal, depth, dynamic scores | Offline checkpoint 69M ([HF](https://huggingface.co/Yuxihenry/SpatialTrackerV2-Offline)) | About 50× faster than MegaSaM on video depth | CC BY-NC 4.0 | [arXiv](https://arxiv.org/html/2507.12462) |
| TAPIP3D | 2025, NeurIPS | FF 3D tracker | Video **plus depth and poses** | 3D tracks | n/r | n/r | **Apache-2.0 / MIT** ([HF](https://huggingface.co/zbww/tapip3d)) | [github](https://github.com/zbw001/TAPIP3D) |
| CoTracker3 | 2024–25 | FF 2D tracker | Video | 2D tracks plus visibility | n/r | n/r | **CC BY-NC 4.0** ([LICENSE](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md)) | [github](https://github.com/facebookresearch/co-tracker) |
| MegaSaM | 2025, CVPR | **TTO** (DROID-style BA plus depth priors) | Continuous video | Poses, focal, consistent depth | n/a | Tracking 0.7–1.0 s/frame | Apache-2.0 | [github](https://github.com/mega-sam/mega-sam) |
| Uni4D | 2025, CVPR | **TTO** over frozen foundation models; needs an OpenAI API key | Video | Static plus dynamic 4D, poses, dense 3D tracks | n/a | n/r | MIT | [github](https://github.com/Davidyao99/uni4d) |
| MoRe | CVPR 2026 | FF (VGGT plus MagiAttention), full and streaming | Video | Poses, depth, motion masks | n/r | n/r | Not stated | [github](https://github.com/HellexF/MoRe) |
| UniQuery4R | Aug 2026, arXiv | FF query decoder (D4RT-like) | Clip plus (source view, target view, uv) queries | Correspondence, target-time 3D, scene flow, depth, cameras | n/r | n/r | Code status unknown | [arXiv](https://arxiv.org/abs/2608.17283) |
| Causal Spatiotemporal State Evolution | Oct 2026, arXiv | FF causal (Mamba state) | Monocular video | Depth, pose, optical flow | n/r | n/r | No code linked | [arXiv](https://arxiv.org/abs/2610.27594) |
| Mesh4D | Jan 2026, arXiv | FF latent diffusion | Object video | Mesh plus deformation field | n/r | n/r | Unverified | [arXiv](https://arxiv.org/abs/2601.05251) |

### 3f. Human + scene and human-object

| Name | Year / venue | FF or TTO | Inputs | Outputs | Params | Speed | Licence | URL |
|---|---|---|---|---|---|---|---|---|
| Human3R | ICLR 2026 | FF online (visual prompt tuning on frozen CUT3R plus Multi-HMR tokens) | Monocular video | Multi-person world-frame **SMPL-X**, pointmaps, cameras; tracklet IDs via OT with a dustbin | ViT-S, B, L variants | 15 FPS (ViT-S) on a 4090, 8 GB; trained in about 1 day on one 48 GB GPU ([paper](https://arxiv.org/html/2510.06219v2)) | CC BY-NC-SA plus NAVER NC; SMPL-X NC | [github](https://github.com/fanegg/Human3R) |
| JOSH / JOSH3R | 2025–26, arXiv | JOSH TTO; JOSH3R FF (MASt3R plus a head) | Video | Cameras, SMPL motion, scene point cloud, contact | n/r | JOSH3R 15.4 FPS, JOSH 0.8 FPS (4090) | **No LICENSE file**; SMPL NC | [github](https://github.com/genforce/JOSH) |
| UniSH | CVPR 2026 | FF | Monocular video | Scene, cameras, metric SMPL | n/r | n/r | README badge says Apache-2.0 but **no LICENSE file in repo**; needs SMPL | [github](https://github.com/murphylmf/UniSH) |
| UniCon3R | 2026, arXiv | FF online, contact-aware | Video | Human plus scene | n/r | n/r | Apache-2.0 file; code "soon" | [github](https://github.com/surtantheta/UniCon3R) |
| WildHSR | Sep 2026, arXiv | FF | Video | Metric human plus scene; **persistent identities** via dustbin Sinkhorn | n/r | 10.1 FPS | No code | [arXiv](https://arxiv.org/abs/2609.29106) |
| CARI4D | CVPR 2026 | Hybrid (foundation models, render-and-compare network, joint optimisation) | Video, one human plus one object | Metric 4D human (SMPL-H) plus object mesh | n/a | About 45 min per 300 frames on an A100 | NVIDIA NC | [arXiv](https://arxiv.org/html/2512.11988v2) |
| SAM 3D Body | 2025–26 | FF single image, promptable | Image plus box, mask or keypoint prompts | **MHR** body plus hands | 840M (DINOv3-H+) or 631M (ViT-H) | n/r | SAM License (commercial OK); **HF gated (manual)** | [github](https://github.com/facebookresearch/sam-3d-body) |
| SAM 3D Objects | Nov 2025 | FF flow models | Image plus mask | Object mesh or 3DGS plus layout (R, t, s) | 1.2B geometry plus 600M texture | Needs at least 32 GB NVIDIA | SAM License; **gated** | [github](https://github.com/facebookresearch/sam-3d-objects) |
| PromptHMR, GVHMR, TRAM, CameraHMR, Multi-HMR | 2024–25 | FF HMR (+ SLAM for world frame) | Image / video | SMPL(-X) | Multi-HMR S/B/L | Multi-HMR 29–126 ms on a V100 | Mostly NC (TRAM code MIT, but SMPL NC) | [Multi-HMR](https://github.com/naver/multi-hmr), [TRAM](https://github.com/yufu-wang/tram) |
| MHR (body model) | Nov 2025 | Parametric rig | — | 45 shape, 204 pose, 72 expression params; 7 levels of detail; FBX and TorchScript | — | `pymomentum-cpu` available | **Apache-2.0**, assets via GitHub release, no login | [github](https://github.com/facebookresearch/MHR) |
| Anny (body model) | Nov 2025 | Parametric rig, infants to elders | — | 104- or 163-bone rig | — | — | Apache-2.0 code, CC0 assets (its optional SMPL-X topology is NC) | [github](https://github.com/naver/anny) |

### 3g. Feed-forward (4D) Gaussian and large reconstruction models

| Name | Year / venue | Inputs | Outputs | Params | Speed | Licence | URL |
|---|---|---|---|---|---|---|---|
| L4GM | 2024, NeurIPS | Single-view **object** video | Per-frame 3D Gaussians plus 3× interpolation | n/r (from LGM) | About 1 s | **Apache-2.0 code and weights** ([HF](https://huggingface.co/jiawei011/L4GM)) | [github](https://github.com/nv-tlabs/L4GM-official) |
| 4DGT | 2025, NeurIPS | **Posed** monocular video, 64-frame window | 4D Gaussians with lifespan and velocity | n/r | 25 ms/frame on an A100 | CC BY-NC / CC BY-NC-SA | [github](https://github.com/facebookresearch/4DGT) |
| BTimer | 2025, NeurIPS | Context frames plus Plücker rays plus timestamp | 3DGS at the queried time | n/r | 150 ms | No code found | [project](https://research.nvidia.com/labs/toronto-ai/bullet-timer/) |
| 4D-LRM | 2025, NeurIPS | Posed views at any time | Per-pixel 4D Gaussians | About 300M | 24 frames in under 1.5 s on an A100 | Unofficial code only | [arXiv](https://arxiv.org/abs/2506.18890) |
| MoVieS | CVPR 2026 | Monocular video | Pixel-aligned dynamic Gaussians, tracks, flow | n/r | About 1 s per scene | Unverified | [arXiv](https://arxiv.org/abs/2507.10065) |
| C4G | May 2026 | Unposed video | **Compact** set of Gaussian query tokens plus a global motion field | n/r | n/r | No code | [arXiv](https://arxiv.org/abs/2605.31595) |
| NoPo4D | May 2026 | Unposed multi-view videos | Dynamic Gaussians | n/r | n/r | Unverified | [arXiv](https://arxiv.org/abs/2605.22190) |
| AnySplat / YoNoSplat / NoPoSplat | 2025–26 | Unposed, uncalibrated images | Static 3DGS plus poses | n/r | n/r | MIT code; weights licences not stated | [AnySplat](https://github.com/InternRobotics/AnySplat), [YoNoSplat](https://github.com/cvg/YoNoSplat) |
| Apple SHARP | Dec 2025 | Single image | Metric 3DGS (static) | n/r | Under 1 s; **prediction runs on MPS** | Apple code licence / research-only model licence | [github](https://github.com/apple/ml-sharp) |
| Lyra / Lyra 2.0 (NVIDIA) | 2025–26 | Image or video plus video diffusion | 3D and 4D Gaussians (generative) | Large | About 43 GB even with offloading | Apache-2.0 code; NVIDIA model licences | [github](https://github.com/nv-tlabs/lyra) |

### 3h. Supporting teachers: masks, flow and tracks

| Name | Outputs | Licence / gating | Mac | URL |
|---|---|---|---|---|
| SAM 2.1 | Promptable image and video masks, within-video object tracking | **Apache-2.0, ungated** ([HF](https://huggingface.co/facebook/sam2.1-hiera-large)) | Community MPS PR, tested on an M3 Pro ([#778](https://github.com/facebookresearch/sam2/pull/778)) | [github](https://github.com/facebookresearch/sam2) |
| SAM 3 / 3.1 | Text-promptable detection, segmentation and tracking | SAM License, **gated (manual)** | PR tested on an **M5 16 GB**: 720p tracking about 1.3 FPS, about 7 GB peak ([#627](https://github.com/facebookresearch/sam3/pull/627)) | [github](https://github.com/facebookresearch/sam3) |
| SEA-RAFT | Optical flow | BSD-3-Clause | [U] | [github](https://github.com/princeton-vl/SEA-RAFT) |
| AllTracker | Dense long-range 2D tracking | MIT code; weights licence [U] | [U] | [github](https://github.com/aharley/alltracker) |
| DINOv2 | Generic ViT features (backbone of most models above) | Apache-2.0 | Works on MPS | [github](https://github.com/facebookresearch/dinov2) |

---

## 4. Unordered frames, multiple shots, and frames from different places

| Model family | Order assumption | Unordered input? | Frames from *different places* | Cuts | Evidence |
|---|---|---|---|---|---|
| DUSt3R / MASt3R GA | Pair graph; none | Yes (any pairs) | Each pair gets a pointmap regardless. Low confidence and few matches are the only signal. GA still forces one frame [E] | No notion | [DUSt3R](https://github.com/naver/dust3r) |
| MASt3R-SfM | None (retrieval graph) | **Yes**, "any collection … ordered or not" | Retrieval scores could separate components; not reported for this purpose [E] | No notion | [arXiv](https://arxiv.org/abs/2409.19152) |
| VGGT | Frame 1 = world frame (special tokens) | Yes, but output depends on which frame is first | Forced into one frame. Std of DTU accuracy over first-frame choices is 0.033 vs π3's 0.003 | No | [VGGT](https://arxiv.org/html/2503.11651), [π3](https://arxiv.org/html/2507.13347) |
| π3 / Pi3X | **Permutation-equivariant** | **Yes**, the most robust to order | Forced into one affine frame [E] | No | [π3](https://arxiv.org/html/2507.13347) |
| MapAnything | Any; reference-free factored output | Yes | Not handled. Benchmark sampling "prevents disjoint sets of images" | No | [paper](https://arxiv.org/html/2509.13414) |
| DA3 | Any; selectable `ref_view_strategy` | Yes | Not discussed [E: forced into one frame] | No | [README](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| VGGT-Ω | Video or images | Yes [U] | Not discussed | No | [project](https://vggt-omega.github.io/) |
| CUT3R | Recurrent state | Claims photo collections; evaluated only on 2–5 views | A frame from a new place gets written into the same state, so it would corrupt it [E] | No | [paper](https://arxiv.org/html/2501.12387) |
| TTT3R, StreamVGGT, STream3R, Point3R, WinT3R, SLAM3R, MASt3R-SLAM | Ordered stream | No | State or map corruption. Only a manual reset helps (TTT3R `--reset_interval`) | No | [TTT3R](https://github.com/Inception3D/TTT3R) |
| MonST3R, Align3R, Geo4D, MegaSaM | Ordered windows plus flow | No | Fails. MegaSaM explicitly assumes "a continuous video" | No | [MegaSaM](https://arxiv.org/html/2412.04463) |
| Any4D, St4RTrack | Frame-1 anchored | No | Any4D lists "no visual overlap" as a failure case | No | [Any4D](https://arxiv.org/html/2512.10935v1) |
| D4RT | Spatio-temporal tubelets | No | Not discussed | No | [paper](https://arxiv.org/html/2512.08924v2) |
| Human3R | Recurrent state, reset every 100 frames | No | Re-associates a person only when their head reappears in the same stream | No | [paper](https://arxiv.org/html/2510.06219v2) |

**Implications for edited video** [E]:
- **Shot boundaries must be explicit.** They are cheap to detect, and every method above silently misbehaves across them.
- **Within a shot, all the video models apply.**
- **Across shots in the same location**, the unordered multi-view models are the right tool, especially π3 (order-robust) and MapAnything-apache (metric, permissive). Feed them keyframes from several shots *after* deciding that the shots share a location. They are the only family designed for wide-baseline, unordered views.
- **"Same location?" is an open sub-problem.** Candidate signals:
  - Retrieval similarity: MASt3R-style or DINOv2 global features.
  - Teacher confidence and cross-view consistency after a joint pass: run MapAnything on a pair of shots and check whether the confidence collapses and whether the static geometry is mutually consistent.
  - A small classifier trained on our synthetic data.
  
  I found no published method that does this for feed-forward reconstruction.
- **Shot/reverse-shot coverage** (two cameras facing opposite ways in the same room) has almost no shared pixels. Geometric overlap-based methods will fail on it. It needs semantic and layout reasoning or entity anchors (the same people seen from the front and the back).

---

## 5. Small, distilled and quantized versions

| Base | Small / efficient option | Notes | Source |
|---|---|---|---|
| DA3 | **DA3-Small** (22M backbone, 34M checkpoint), **DA3-Base** (86M / 135M), both Apache | Small runs at 160 FPS on an A100; same any-view interface as Giant | [README](https://github.com/ByteDance-Seed/Depth-Anything-3), [paper](https://arxiv.org/html/2511.10647) |
| DA2 / Video DA | DA2-Small 24.8M; Video-DA-Small; Core ML Small from Apple | All Apache | [DA2](https://github.com/DepthAnything/Depth-Anything-V2), [VDA](https://github.com/DepthAnything/Video-Depth-Anything) |
| MoGe-2 | ViT-S 35M, ViT-B 104M | MIT | [README](https://github.com/microsoft/MoGe) |
| VGGT | VGGT-500M and VGGT-200M are promised but **not released** (checklist unticked) | — | [README](https://github.com/facebookresearch/vggt) |
| VGGT | StreamVGGT ships **distillation code from VGGT** | NC | [README](https://github.com/wzzheng/StreamVGGT) |
| VGGT | QuantVGGT 4-bit: 3.7× less memory, 2.5× faster, over 98% accuracy | Apache code | [arXiv](https://arxiv.org/abs/2509.21302) |
| VGGT | FastVGGT, training-free token merging | VGGT licence | [github](https://github.com/mystorm16/FastVGGT) |
| MASt3R | **DUNE**: ViT-S (110 MB) and ViT-B (420 MB) encoders distilled from DINOv2, MASt3R and Multi-HMR; DUNE-MASt3R decoders | Licence not stated in repo (NAVER NC likely [U]) | [github](https://github.com/naver/dune), [arXiv](https://arxiv.org/abs/2503.14405) |
| CUT3R | 224 px linear-head checkpoint | NC | [README](https://github.com/CUT3R/CUT3R) |
| Human3R / Multi-HMR | ViT-S variants (15 FPS) | NC | [paper](https://arxiv.org/html/2510.06219v2) |
| SAM 3D Objects | 4-step shortcut distillation (down from 25 steps) | Gated | [arXiv](https://arxiv.org/html/2511.16624v2) |
| D4RT | ViT-B (90M) encoder ablation | Not released | [paper](https://arxiv.org/html/2512.08924v2) |

No published model yet combines a small size with multi-view, metric, dynamic and permissive. DA3-Small/Base and MoGe-2-S/B are the closest. Distilling MapAnything-apache into a small student, on our own data, is a plausible step for us [E].

---

## 6. Apple Silicon (MPS, MLX, Core ML) status

| Model | Status | Numbers / blocker | Source |
|---|---|---|---|
| DUSt3R | Works on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1` | M2: 8 images, 56 pairs = 70 s plus 90 s GA | [issue #31](https://github.com/naver/dust3r/issues/31) |
| MASt3R | MPS works; MLX port | M4 Max 512×672: MPS 1.2 s, MLX fp16 0.77 s; port has **no licence** [U] | [mlx-mast3r](https://github.com/Aedelon/mlx-mast3r) |
| MASt3R-SfM | Partly blocked | ASMK `build_ivf` segfaults on a Mac | [issue #128](https://github.com/naver/mast3r/issues/128) |
| MASt3R-SLAM, MegaSaM | **Blocked** | CUDA kernels, lietorch, DROID | [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) |
| CroCo-family (CUT3R, TTT3R, MonST3R, Spann3R, Human3R) | Probably runs [U] | `curope` CUDA RoPE is optional, with a PyTorch fallback | [croco pos_embed](https://github.com/naver/croco/blob/master/models/pos_embed.py) |
| VGGT | Official MPS fix (float64 to float32) | MLX port on an M4 16 GB: 1.95 s/frame for two 126×518 views, capped at 4 views on 16 GB; port has no licence [U] | [PR #112](https://github.com/facebookresearch/vggt/pull/112), [vggt-mlx](https://github.com/koernergb/vggt-mlx) |
| π3 / Pi3X | Community PR, verified on an M1 Pro 16 GB | fp16 plus fallback | [PR #153](https://github.com/yyfz/Pi3/pull/153) |
| **MapAnything** | **Official MPS (merged 2026-03-23)**; MLX port in mlx-spatial (MIT) | Device-aware AMP | [PR #131](https://github.com/facebookresearch/map-anything/pull/131), [mlx-spatial](https://github.com/appautomaton/mlx-spatial) |
| **DA3** | MPS fix merged; MLX and Swift ports; Apple Core AI recipe for DA3-small | M2 Pro 16 GB at 504×378, 1 new image plus 5 keyframes: Small 166 ms, Base 371 ms, Large 1.08 s; Large swapped until the MLX cache was capped [U] | [PR #26](https://github.com/ByteDance-Seed/Depth-Anything-3/pull/26), [da3-engine](https://github.com/tg1482/da3-engine), [Apple coreai](https://github.com/apple/coreai-models/tree/main/models/depth-anything) |
| DA2 | Official MPS plus Apple Core ML | 24.6 ms on M3 Max ANE | [HF](https://huggingface.co/apple/coreml-depth-anything-v2-small) |
| MoGe-2 | Works on MPS unpatched [U]; MoGe-3 unsupported on macOS | 212 ms ViT-S on M1 Max | [report](https://dev.to/kiarina/estimating-surface-orientation-and-3d-from-one-image-with-moge-2-on-apple-silicon-2n0h), [README](https://github.com/microsoft/MoGe) |
| Depth Pro | Official MPS; MLX port | 3.8 s, 7 GB on M4 Pro (fp16) | [MLX](https://huggingface.co/ntoan199/DepthPro-mlx) |
| CoTracker3 | demo defaults to MPS | NC licence | [github](https://github.com/facebookresearch/co-tracker) |
| SAM 2 / SAM 3 | Community PRs; SAM 3 tested on an **M5 16 GB** | 720p at 1.3 FPS, 7 GB | [#778](https://github.com/facebookresearch/sam2/pull/778), [#627](https://github.com/facebookresearch/sam3/pull/627) |
| SAM 3D Body | MHR TorchScript fails on MPS; PR runs MHR on CPU and the rest on MPS | — | [PR #125](https://github.com/facebookresearch/sam-3d-body/pull/125) |
| SHARP | Prediction on MPS is official; rendering needs CUDA | — | [github](https://github.com/apple/ml-sharp) |
| **Differentiable renderers** | **Dr.Jit 1.4 / Mitsuba 3.9: Metal backend with autodiff and HW ray tracing** (BSD-3) | Mature | [release](https://github.com/mitsuba-renderer/drjit/releases/tag/v1.4.0) |
| | metal-gauss (MIT): fused forward and backward 3DGS on PyTorch MPS | Author benchmark on an **M5**: 69 FPS forward, 600k splats, 768² [U] | [github](https://github.com/nandometzger/metal-gauss) |
| | MetalSplat (MIT): 3DGS and 2DGS with depth and normals | Pinhole cameras only | [github](https://github.com/tchauffi/metalsplat) |
| | mtldiffrast (MIT): nvdiffrast-compatible mesh rasterizer with autograd | Very new; check gradients | [github](https://github.com/pedronaugusto/mtldiffrast) |
| | msplat (Apache), Brush (Apache, WebGPU), mlx3D (MIT, includes a soft mesh rasterizer) | Not PyTorch autograd (msplat, Brush) | [msplat](https://github.com/rayanht/msplat), [Brush](https://github.com/ArthurBrussee/brush), [mlx3D](https://github.com/amirhossein-razlighi/mlx3D) |
| | Avoid: OpenSplat and gsplat-mps (AGPL); PyTorch3D (CPU-only on Mac); nvdiffrast and gsplat (CUDA) | — | per helper-agent check [U] |
| M5 hardware | GPU Neural Accelerators reached through Metal 4 TensorOps; MLX needs macOS 26.2 or later | Prefill 3.3–4.1× faster than M4; generation limited by bandwidth (153 GB/s) | [Apple ML research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |

---

## 7. Licence and gating screen: teachers we can actually use on 16 GB

**Tier A: permissive and ungated.** These are the default teachers. Memory figures are my estimates [E] for a 16 GB M5, assuming roughly 10–11 GB usable by the GPU and fp16/bf16. None of this has been tested.

| Teacher | Role for us | Licence | Fit on 16 GB [E] |
|---|---|---|---|
| **MapAnything-apache** (1.23B) | Multi-view **metric** cameras, intrinsics and depth, within a shot and across shots in the same location | Apache-2.0 code and weights ([HF](https://huggingface.co/facebook/map-anything-apache)) | Weights about 2.5 GB in fp16. Expect roughly 8–16 views at 518 px using `memory_efficient_inference` and minibatch 1 [E] |
| **DA3-Base / DA3-Small** | Fast any-view depth, rays and pose; per-frame features; candidate **student backbone initialisation** | Apache-2.0 | Easy. About 0.2–0.4 s/frame reported on an M2 Pro [U] |
| **DA3Metric-Large, DA3Mono-Large** | Per-frame metric or relative depth, plus sky mask | Apache-2.0 | Easy |
| **MoGe-2 (S/B/L)** | Per-frame **intrinsics / FoV** (close-ups and zooms), metric pointmap, normals | MIT | Easy |
| **Video Depth Anything Small** | Temporally consistent per-shot depth | Apache-2.0 | Easy [U] |
| **Depth Pro** | Sharp metric depth plus focal (slow, offline) | Apple sample-code licence | 7 GB peak in fp16 |
| **SAM 2.1** | Entity masks and within-shot mask tracks | Apache-2.0 | Fits [U] |
| **SEA-RAFT / AllTracker** | Flow and dense 2D tracks; replaces NC CoTracker3 | BSD-3 / MIT code | Fits [U] |
| **TAPIP3D** | 3D tracks given depth and cameras | Apache-2.0 / MIT | [U] |
| **L4GM** | Per-object 4D Gaussians from crops (object-centric) | Apache-2.0 | Uses a CUDA rasterizer, so likely needs porting [U] |
| **Any4D** | Metric scene flow (frame-1 anchored) | Apache-2.0 code; **weights have no licence tag** (ask the authors) | 7.7 GB checkpoint |
| **MHR, Anny** | Body rigs for human entities | Apache-2.0 (+ CC0) | CPU |

Licence caveat: MapAnything's own dataset table marks MPSD as CC BY-NC-SA, yet it is in the Apache-model training set ([paper](https://arxiv.org/html/2509.13414)). This needs a legal check before any commercial use.

**Tier B: non-commercial but ungated.** Acceptable for research, ablations and evaluation only:
- VGGT-1B, π3 / Pi3X.
- CUT3R / TTT3R, MonST3R, MASt3R / DUSt3R.
- Human3R, SpatialTrackerV2, CoTracker3, 4DGT, Trace Anything weights.

**Tier C: gated.** Excluded by our "no gated downloads" constraint, even though some allow commercial use:
- VGGT-1B-Commercial, VGGT-Ω.
- SAM 3 / 3.1, SAM 3D Body, SAM 3D Objects.
- SMPL / SMPL-X and BEDLAM (registration).

---

## 8. What to borrow

### 8.1 As frozen teachers and pseudo-label sources
- **Per-shot geometry and cameras.**
  - Use MapAnything-apache on keyframes (for example 2–4 fps) for metric poses, intrinsics and depth.
  - Densify with DA3-Base or Video-DA-Small depth, aligned per frame by scale and shift.
  - Use MoGe-2 intrinsics as a per-frame focal prior to catch zooms, which every multi-view model assumes away.
- **Cross-shot location registration.** After a "same location" decision, run MapAnything or DA3 jointly on representative keyframes from the candidate shots. Their pose output links shot cameras into one location frame. Static-background masks from SAM 2.1 help, because the dynamic people otherwise dominate.
- **Entities.**
  - SAM 2.1 masks and tracklets.
  - SEA-RAFT / AllTracker 2D motion, lifted to 3D with teacher depth and cameras (TAPIP3D does exactly this).
  - Humans: fit **MHR** to keypoints and depth. A permissive 2D keypoint detector is still needed; it is not covered here.
- **Cache everything once.** Teachers run once per training video. Their outputs and features are stored to disk, and our model trains on the cache. This is the only way to fit a 1B teacher and a trainable model into 16 GB [E].

### 8.2 Architectural ideas
- **Alternating frame/global attention** (VGGT, π3, MapAnything, DA3) as the cross-frame mixer.
- **π3's permutation equivariance** (no reference frame), for unordered shot keyframes ([paper](https://arxiv.org/html/2507.13347)).
- **MapAnything's factored output**: ray map, depth along ray, pose, and one global metric scale. Optional geometric inputs can be fed in ([paper](https://arxiv.org/html/2509.13414)). This maps directly to "camera slot + location slot".
- **DA3's minimal targets**: depth plus rays, with no separate point head ([paper](https://arxiv.org/html/2511.10647)).
- **VGGT-Ω registers with register-only cross-frame attention.** This is independent evidence that a compact per-scene latent can carry the multi-view information ([arXiv](https://arxiv.org/abs/2605.15195)).
- **CUT3R's queryable state.** A ray map of an unseen virtual view reads geometry out of the state ([paper](https://arxiv.org/html/2501.12387)). This is the right interface for "view from any angle".
- **D4RT / UniQuery4R query decoding.** A light cross-attention decoder answers (pixel, t_src, t_tgt, camera) queries on their own, so training can subsample queries. That keeps memory small ([D4RT](https://arxiv.org/html/2512.08924v2)).
- **Trace Anything's B-spline trajectories** and **4DGT's lifespan Gaussians** as compact motion and appearance parameterisations for entities that enter and leave ([TA](https://arxiv.org/abs/2510.13802), [4DGT](https://arxiv.org/html/2506.08015)).
- **TTT3R's confidence-gated state update.** Gate slot updates by evidence, and close the gate at a cut ([arXiv](https://arxiv.org/abs/2509.26645)).
- **Persistent-identity association** for entity slots across cuts:
  - Human3R: tracklet memory plus optimal transport with a dustbin ([paper](https://arxiv.org/html/2510.06219v2)).
  - WildHSR: dustbin Sinkhorn on backbone query/key features ([arXiv](https://arxiv.org/abs/2609.29106)).
- **Adapter training on a frozen backbone** (Human3R's visual prompt tuning: about 1 day on one GPU). This is the training-cost model to copy.

### 8.3 Losses and training recipes
- Confidence-weighted regression with a learned confidence (BCE or aleatoric), applied after closed-form scale alignment (π3, VGGT).
- MapAnything's robust loss with top-5% rejection, which tolerates noisy pseudo-labels.
- DA3's recipe: synthetic-only teacher, then RANSAC scale-shift alignment to real data, then switch supervision mid-training. This is almost exactly our planned "warm-start on procedural data, then self-supervise on real video".
- VGGT-Ω's self-supervised protocol on unlabeled video. Its details are in the paper, not yet read beyond the abstract.
- Normal and gradient-matching losses on synthetic data only (MapAnything).

### 8.4 Tools
- **Renderers.** Dr.Jit / Mitsuba Metal (mature, BSD-3); metal-gauss / MetalSplat (MIT) for Gaussian surrogates; mtldiffrast (MIT) for meshes.
- **Data generator.** Kubric (Apache-2.0, [github](https://github.com/google-research/kubric)) is the generator behind several 4D training sets.
- **Real dynamic clips.** Stereo4D supplies more than 100k clips with metric 3D tracks under a CC BY 4.0 annotation licence. The videos themselves must be fetched from YouTube ([github](https://github.com/Stereo4D/stereo4d-code)).

---

## 9. Gaps: what is missing for edited, multi-shot video with persistent objects

1. **No edit awareness.** Cuts break all four forms of continuity assumption listed in §1. No method outputs an edit decision list or reasons about film grammar: shot/reverse-shot, cutaways, inserts, match cuts.
2. **No location grouping.** Unordered multi-view models assume one connected scene. Nothing decides that shots 1, 3 and 7 share a set while shot 5 is elsewhere, or handles shots of the same room with no shared pixels.
3. **No persistent entities.** Outputs are per-pixel and per-frame. There are no object identities, no canonical shapes and no cross-shot re-identification. Human3R, WildHSR and Multi-HMR 2 track only within continuous footage.
4. **No generic articulation.** Human models give rigs: SMPL family (NC) and MHR (permissive). Animals, game mobs, vehicles and props get no skeleton. Mesh4D (deformation field) is the closest, and it is object-centric.
5. **No compact or editable geometry.** Everything is dense pointmaps, depth or Gaussians. Nobody outputs low-poly sets, layout primitives or watertight entity meshes.
6. **Camera-model fragility.**
   - Zoom and focal change break MegaSaM and Geo4D ([MegaSaM](https://arxiv.org/html/2412.04463)).
   - VGGT has no fisheye support and degrades under extreme rotation ([paper](https://arxiv.org/html/2503.11651)).
   - Close-ups where dynamic people dominate the frame are a documented failure mode.
7. **Scale consistency across shots.** Only the metric models (MapAnything, MoGe-2, DA3Metric, Any4D, Pi3X approximately) give absolute scale. Per-shot scale drift must be reconciled.
8. **Amodal completion.** "View from any angle" needs the backs of people, objects and sets. Deterministic regressors blur when extrapolating (CUT3R says so). The generative options are gated (SAM 3D Objects) or heavy (Lyra, about 43 GB).
9. **Domain gap.** Training data is photoreal or indoor/driving. Stylised and game footage such as Minecraft, and heavy motion blur in dance and handheld footage, are absent.
10. **Licences and compute.** The best models are NC or gated. Training needs datacentre GPUs. Many training sets (BEDLAM, mapfree, some in MapAnything's non-Apache mix) are NC.

---

## 10. Ideas: what a better approach could look like

### 10.1 Challenges to the current sketch
- **"A single network trained from scratch" will not converge on a laptop** [E]. π3 says geometry transformers trained from scratch "converge poorly" even at scale. Every successful model starts from DINOv2, DUSt3R or VGGT weights ([π3](https://arxiv.org/html/2507.13347)). **Recommendation:** initialise image encoding from an Apache backbone (DA3-Base/Small or DINOv2-S/B), freeze it, and train only the slot model.
- **"Mainly self-supervised by re-rendering" is a weak primary signal** [E]. Photometric losses are ambiguous with a moving camera, moving entities, textureless walls and motion blur. The hybrid systems that do well (MegaSaM, JOSH, CARI4D) lean on strong priors. **Recommendation:** make *agreement with cached teacher pseudo-labels* the primary loss, weighted by teacher confidence:
  - Cameras, depth and intrinsics from MapAnything, DA3 and MoGe-2.
  - Masks from SAM 2.1.
  - Tracks from AllTracker / SEA-RAFT.

  Rendering losses (colour, mask, depth) then act as the consistency term that ties everything into one persistent scene.
- **"One slot per frame" scales badly.** At 30 fps, one minute is 1,800 slots. **Recommendation:** use one slot per *shot*, holding a smooth camera spline over position, rotation *and focal length* plus a location pointer. Per-frame values are decoded by querying the spline at time t. Shot boundaries come from a detector and are not learned latents.
- **Location assignment is a discrete, open-set problem.** Model it explicitly as a soft assignment of shot slots to location slots via Sinkhorn with a "new location" dustbin. Supervise it with synthetic ground truth and with the edit-augmentation trick in §10.3.
- **Decoding straight to low-poly meshes gives poor gradients** [E]. Mesh rasterizer gradients concentrate at silhouettes. **Recommendation:** decode to a primitive layer for training and render it through a Gaussian surrogate (metal-gauss / MetalSplat), then export clean low-poly meshes at the end. Primitives: planes, boxes or a heightfield for sets; a template mesh plus skinning for entities. mtldiffrast or Dr.Jit can fine-tune silhouettes.
- **Body rig.** Use **MHR**, which is Apache-licensed with no login and has an FBX rig, or Anny for children. Do not use SMPL/SMPL-X, because of the NC licence and gating.

### 10.2 Proposed structure: a "scene compiler" over cached teacher outputs
1. **Stage 0: edit parsing (classical plus tiny learned).** Shot boundary detection, per-shot keyframes, and per-frame zoom/focal from MoGe-2.
2. **Stage 1: frozen perception, cached once per video.**
   - DA3-Base tokens for all keyframes.
   - MapAnything-apache per shot: cameras, metric depth, confidence.
   - SAM 2.1 tracklets and AllTracker / SEA-RAFT tracks.
   - Depth densified from Video-DA-Small.
3. **Stage 2: the trainable compiler, about 30–100M params [E].** A transformer over cached tokens with four slot types:
   - **Shot slots:** a camera spline (R, t, f) and a location pointer.
   - **Location slots:** set geometry as primitives plus a surface Gaussian appearance latent, queryable from any ray (CUT3R-style).
   - **Entity slots:** a class (human / quadruped / rigid / deformable), an identity embedding, a canonical shape latent (MHR betas or a template code), and a motion decoder queried per time step (D4RT-style), giving rig pose or a rigid SE(3).
   - **Association heads:** tracklet-to-entity and shot-to-location assignment, both via dustbin Sinkhorn.

   Cross-attention from slots to tokens follows VGGT-Ω's register-attention pattern, so memory grows with the number of slots, not with frames² [E].
4. **Stage 3: short per-video refinement (optional, minutes on the M5).** Render-and-compare optimisation of the decoded parameters against the frames, masks, depth and tracks. This is the "feed-forward initialisation + test-time refinement" pattern that wins quality in MegaSaM, JOSH and CARI4D. For clips under one minute it is affordable.
5. **Output.** glTF/USD containing low-poly sets per location, rigged entity meshes with animation clips, one camera per shot, and the cut list. Playback replays shots in edit order.

### 10.3 Data ideas that address the gaps directly
- **Edit augmentation of real single-shot videos** [E]. This is free supervision for cross-shot association.
  - Take unlabeled single-shot clips and cut them into segments.
  - Simulate close-ups with crop-and-zoom, which changes focal length.
  - Interleave segments from 2–3 different source clips.
  - Labels then come for free: same source clip means same location, and within-clip teacher tracks carry entity identity across the synthetic cut.

  No public dataset provides cross-shot location or identity labels. This manufactures them from any video.
- **The procedural synthetic generator should emulate film grammar:**
  - establishing shots, shot/reverse-shot (180° rule), inserts and close-ups, zooms, handheld shake;
  - characters that leave and re-enter, location revisits;
  - stylised and voxel worlds (Minecraft-like), MHR/Anny humans with procedural motion, and simple rigged quadrupeds and props.
  
  Exact ground truth for identities, locations, rigs and per-shot cameras is the unique thing synthetic data offers here. Kubric (Apache) or Dr.Jit/Mitsuba Metal can render it on the Mac.
- **Pseudo-label distillation, DA3-style.** First train the compiler on synthetic data. Then run it plus the teachers on real edited video, keep high-confidence agreements as pseudo-labels, and continue training.

### 10.4 Risks and open questions
- MapAnything-apache's quality on dynamic content and close-ups is untested; its paper says it does not model motion ([paper](https://arxiv.org/html/2509.13414)). Mask out entities before running it on sets.
- Same-location detection between shot/reverse-shot pairs with no shared pixels may need semantic cues. Shared entities are a strong anchor: the same person in both shots constrains relative placement.
- MPS numerics: there is no float64. Fast attention kernels are missing, and some ops need CPU fallback. MLX TF32 conv precision on the M5 is a known issue ([mlx #4595](https://github.com/ml-explore/mlx/issues/4595)). Budget engineering time for porting.
- Licence hygiene: if the project may ever be commercial, keep Tier B (NC) teachers out of the training-label pipeline. VGGT-Ω-style licences cover outputs too.

### 10.5 Suggested first experiments (not run; for the experiments track)
1. Run MapAnything-apache, DA3-Base and MoGe-2 on MPS for 8 and 16 frames at 518 px. Record time and peak memory on the M5.
2. **Different-place probe.** Feed MapAnything and π3 keyframes from two different places, and from a shot/reverse-shot pair. Check whether confidence and cross-view consistency separate them.
3. **Cut probe.** Run TTT3R/CUT3R (NC, research only) and DA3-Streaming across a synthetic cut, and measure how the state or trajectory corrupts.
4. Check gradients of metal-gauss and mtldiffrast on a toy inverse-rendering task on the M5.

---

## Appendix: main sources

**Static multi-view and streaming**
- [DUSt3R](https://github.com/naver/dust3r) ([arXiv](https://arxiv.org/abs/2312.14132)), [MASt3R](https://github.com/naver/mast3r) ([arXiv](https://arxiv.org/abs/2406.09756)), [MASt3R-SfM](https://arxiv.org/abs/2409.19152), [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) ([arXiv](https://arxiv.org/abs/2412.12392)), [MUSt3R](https://github.com/naver/must3r)
- [VGGT](https://github.com/facebookresearch/vggt) ([paper](https://arxiv.org/html/2503.11651)), [VGGT-Ω](https://vggt-omega.github.io/) ([repo](https://github.com/facebookresearch/vggt-omega), [arXiv](https://arxiv.org/abs/2605.15195)), [π3](https://github.com/yyfz/Pi3) ([paper](https://arxiv.org/html/2507.13347))
- [MapAnything](https://github.com/facebookresearch/map-anything) ([paper](https://arxiv.org/html/2509.13414)), [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) ([paper](https://arxiv.org/html/2511.10647)), [DA2](https://github.com/DepthAnything/Depth-Anything-V2), [Video DA](https://github.com/DepthAnything/Video-Depth-Anything), [MoGe](https://github.com/microsoft/MoGe), [Depth Pro](https://github.com/apple/ml-depth-pro)
- [CUT3R](https://github.com/CUT3R/CUT3R) ([paper](https://arxiv.org/html/2501.12387)), [TTT3R](https://github.com/Inception3D/TTT3R), [StreamVGGT](https://github.com/wzzheng/StreamVGGT), [STream3R](https://github.com/NIRVANALAN/STream3R), [Point3R](https://github.com/YkiWu/Point3R), [WinT3R](https://github.com/LiZizun/WinT3R), [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R), [Spann3R](https://github.com/HengyiWang/spann3r), [Fast3R](https://github.com/facebookresearch/fast3r), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), [FastVGGT](https://github.com/mystorm16/FastVGGT), [QuantVGGT](https://arxiv.org/abs/2509.21302), [DUNE](https://github.com/naver/dune), [WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)

**Dynamic / 4D**
- [MonST3R](https://github.com/Junyi42/monst3r), [St4RTrack](https://st4rtrack.github.io/), [Stereo4D](https://github.com/Stereo4D/stereo4d-code), [MegaSaM](https://arxiv.org/html/2412.04463), [Uni4D](https://github.com/Davidyao99/uni4d), [D4RT](https://arxiv.org/html/2512.08924v2), [Any4D](https://github.com/Any-4D/Any4D)
- [DPM](https://arxiv.org/html/2503.16318v1), [V-DPM](https://github.com/eldar/vdpm), [Easi3R](https://github.com/Inception3D/Easi3R), [POMATO](https://arxiv.org/abs/2504.05692), [Geo4D](https://arxiv.org/html/2504.07961), [Align3R](https://github.com/jiah-cloud/Align3R), [D²USt3R](https://github.com/cvlab-kaist/DDUSt3R), [PAGE-4D](https://page4d.github.io/), [VGGT4D](https://github.com/3DAgentWorld/VGGT4D)
- [SpatialTrackerV2](https://arxiv.org/html/2507.12462), [TAPIP3D](https://github.com/zbw001/TAPIP3D), [Trace Anything](https://github.com/ByteDance-Seed/TraceAnything), [CoTracker3](https://github.com/facebookresearch/co-tracker), [UniQuery4R](https://arxiv.org/abs/2608.17283), [Causal SSE](https://arxiv.org/abs/2610.27594), [MoRe](https://github.com/HellexF/MoRe), [Mesh4D](https://arxiv.org/abs/2601.05251)

**Humans, objects and Gaussians**
- [Human3R](https://github.com/fanegg/Human3R) ([paper](https://arxiv.org/html/2510.06219v2)), [JOSH](https://github.com/genforce/JOSH), [UniSH](https://github.com/murphylmf/UniSH), [WildHSR](https://arxiv.org/abs/2609.29106), [CARI4D](https://arxiv.org/html/2512.11988v2)
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), [MHR](https://github.com/facebookresearch/MHR), [Anny](https://github.com/naver/anny), [SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html)
- [L4GM](https://github.com/nv-tlabs/L4GM-official), [4DGT](https://github.com/facebookresearch/4DGT), [BTimer](https://research.nvidia.com/labs/toronto-ai/bullet-timer/), [4D-LRM](https://arxiv.org/abs/2506.18890), [MoVieS](https://arxiv.org/abs/2507.10065), [C4G](https://arxiv.org/abs/2605.31595), [AnySplat](https://github.com/InternRobotics/AnySplat), [YoNoSplat](https://github.com/cvg/YoNoSplat), [SHARP](https://github.com/apple/ml-sharp), [Lyra](https://github.com/nv-tlabs/lyra)

**Apple Silicon and rendering**
- [MapAnything MPS PR](https://github.com/facebookresearch/map-anything/pull/131), [VGGT MPS PR](https://github.com/facebookresearch/vggt/pull/112), [DA3 MPS PR](https://github.com/ByteDance-Seed/Depth-Anything-3/pull/26), [Pi3 MPS PR](https://github.com/yyfz/Pi3/pull/153), [SAM 3 MPS PR](https://github.com/facebookresearch/sam3/pull/627)
- [Dr.Jit 1.4](https://github.com/mitsuba-renderer/drjit/releases/tag/v1.4.0), [metal-gauss](https://github.com/nandometzger/metal-gauss), [MetalSplat](https://github.com/tchauffi/metalsplat), [mtldiffrast](https://github.com/pedronaugusto/mtldiffrast), [mlx-spatial](https://github.com/appautomaton/mlx-spatial), [Apple MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
