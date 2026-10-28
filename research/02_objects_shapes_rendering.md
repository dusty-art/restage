# 02: Object-centric scene representations, primitive and low-poly 3D, and differentiable rendering on an Apple M5

**Date:** 2026-10-28
**Scope:** (a) object-centric learning, (b) shape representations for low-poly output, (c) differentiable rendering without CUDA, (d) conditioning the re-render-and-compare loss. These feed a critique of the design and a recommendation.
**Hardware:** Apple M5 laptop (10-core GPU, Metal 4, 16 GB), macOS 27.0. MLX 0.32.4 and PyTorch 2.15 (MPS) were already installed in `.venv`.

**Method and caveats**
- **Search budget ran out.** The shared web-search budget for this session was used up partway through. After that, claims were checked by fetching primary pages directly: arXiv abstract pages, GitHub READMEs, LICENSE files through the GitHub API, and Hugging Face model cards. A few 2026 items rest only on their arXiv abstracts and are marked as such. Some areas, especially the newest low-poly and scene-level work, are less complete than they would otherwise be.
- **Licences.** Licence fields come from the GitHub API (`gh api repos/...`) and LICENSE files, read on 2026-10-28.
- **Local measurements.** Section 3.3 has my own micro-benchmarks on this M5. The script is `research/bench/02_dense_raster_bench.py`. Other agents were running GPU jobs on the same machine at the same time, so treat the timings as roughly ±20%.

---

## 0. TL;DR

1. **Pure emergent slots are not good enough on real video yet.** The best unsupervised video object-centric models in 2026 report YT-VIS mBO of about 0.33 to 0.63 ([Semantic Slots](https://arxiv.org/abs/2608.21636), [DSSA](https://arxiv.org/abs/2606.12601), [UniSlot](https://arxiv.org/abs/2606.23436)). None of them handles shot cuts.
   - Use slot attention as the binding and persistence mechanism.
   - Seed and supervise the slots with a permissively licensed segmenter/tracker: [SAM 2](https://github.com/facebookresearch/sam2), Apache-2.0, not gated.
   - Add explicit shot detection ([TransNetV2](https://github.com/soCzech/TransNetV2), MIT) and a video-level identity bank.
2. **Shape representation: use fixed-topology parametric parts in the training loop.**
   - Textured superquadric meshes, as in [DBW](https://arxiv.org/abs/2307.05473), on a kinematic tree.
   - A licence-clean human template: [Anny](https://github.com/naver/anny) or [MHR](https://github.com/facebookresearch/MHR), both Apache-2.0 with no sign-up. SMPL-family models are gated.
   - Autoregressive artist-mesh generators (MeshGPT → MeshAnything → BPT → DeepMesh → FastMesh / MeshWeaver / MeshFIM) are slow, non-differentiable, and often non-commercial. They are at most an optional export-time remesher.
   - Structured 3D latents (TRELLIS / TRELLIS.2) need at least 24 GB of NVIDIA memory and gated dependencies. They do not fit the M5.
3. **Differentiable rendering on Apple Silicon is now viable without CUDA.**
   - [mtldiffrast](https://github.com/pedronaugusto/mtldiffrast) is an nvdiffrast-API Metal port (MIT, very new, no benchmarks).
   - Mature Metal 3DGS rasterisers: [metal-gauss](https://github.com/QuintonQu/metal-gauss) (MIT, benchmarked on an M5) and [metalsplat](https://github.com/tchauffi/metalsplat) (MIT, 3DGS and 2DGS).
   - [mlx3D](https://github.com/amirhossein-razlighi/mlx3D) (MIT) does soft triangle rasterisation, NeRF and 3DGS in MLX.
   - [Mitsuba 3.9](https://mitsuba.readthedocs.io/en/v3.9.1/release_notes.html) now ships `metal_ad_*` variants.
   - PyTorch3D has no MPS rasteriser. SoftRas and nvdiffrast are CUDA-only.
4. **Measured on this M5:**
   - MLX and PyTorch-MPS both reach about 15 TFLOP/s on bf16 matmul.
   - A dense SoftRas-style rasteriser written as plain tensor ops costs about 16 ns and about 200 B per pixel–triangle pair (forward plus backward, MLX compiled). That is fine at feature-map resolution (32² to 64²) but too expensive at 256².
   - A fused, tiled Metal kernel is the one piece of custom engineering worth paying for.
5. **Loss conditioning.**
   - Do most of the re-render-and-compare work at token resolution, in feature space (DINOv2 Apache-2.0 features, SAM 2 masks, scale-invariant depth), with sigma annealing and multi-hypothesis poses.
   - Bring in photometric RGB late.
   - Give each entity an explicit shadow/effects channel, so shadows are not explained by phantom objects.

---

## 1. Object-centric learning

### 1.1 The 2D/video slot lineage

| Method | Year / venue | Input → reconstruction target | Real video? | How identity persists over time | Code / licence |
|---|---|---|---|---|---|
| Slot Attention | NeurIPS 2020 [arXiv 2006.15055](https://arxiv.org/abs/2006.15055) | Image → pixels; iterative competitive attention, softmax over slots | No (CLEVR-like) | n/a | [google-research](https://github.com/google-research/google-research/tree/master/slot_attention), Apache-2.0 |
| SAVi | ICLR 2022 [arXiv 2111.12594](https://arxiv.org/abs/2111.12594) | Video → optical flow / RGB; slots conditioned on first-frame cues (boxes or points) | Synthetic (MOVi) | Recurrent predictor carries slots frame to frame; conditioning fixes identity at t=0 | [slot-attention-video](https://github.com/google-research/slot-attention-video), Apache-2.0 |
| SAVi++ | NeurIPS 2022 [arXiv 2206.07764](https://arxiv.org/abs/2206.07764) | Video → depth (LiDAR) targets | Yes: Waymo driving, with first-frame conditioning | Same as SAVi; depth targets help on real scenes | same repo, Apache-2.0 |
| DINOSAUR | ICLR 2023 [arXiv 2209.14860](https://arxiv.org/abs/2209.14860) | Image → **frozen DINO features** (not pixels) | Real images (COCO) | n/a | [OCL framework](https://github.com/amazon-science/object-centric-learning-framework), Apache-2.0 |
| VideoSAUR | NeurIPS 2023 [arXiv 2306.04829](https://arxiv.org/abs/2306.04829) | Video → DINO features plus a temporal feature-similarity loss | Yes (YT-VIS, DAVIS) | Previous-frame slots initialise the next frame | [videosaur](https://github.com/martius-lab/videosaur), MIT |
| SOLV | NeurIPS 2023 [arXiv 2310.06907](https://arxiv.org/abs/2310.06907) | Masked spatio-temporal feature reconstruction, slot merging | Yes (YT-VIS) | Temporal binding plus merging | [SOLV](https://github.com/gorkaydemir/SOLV), **no licence file** |
| SlotContrast | CVPR 2025 [arXiv 2412.14295](https://arxiv.org/abs/2412.14295) | Feature reconstruction plus **slot-slot contrastive loss across consecutive frames** | Yes | Contrastive loss explicitly ties slot k at t to slot k at t+1. The authors say it often re-identifies objects after long full occlusion, but not always ([paper](https://arxiv.org/pdf/2412.14295)) | [slotcontrast](https://github.com/martius-lab/slotcontrast), MIT |
| SlotMatch | 2025 [arXiv 2508.03411](https://arxiv.org/html/2508.03411) | Distils SlotContrast into a smaller student | Yes | Inherited | — |
| SRL ("vicious to virtuous cycles") | 2026 [arXiv 2602.03390](https://arxiv.org/html/2602.03390) | Synergy between encoder attention and decoder maps | MOVi-C | — | — |
| Slot curriculum | 2026 [arXiv 2603.22758](https://arxiv.org/abs/2603.22758) | Reconstruction-guided curriculum against over-fragmentation | Yes | — | — |
| DSSA | ACM MM 2026 [arXiv 2606.12601](https://arxiv.org/abs/2606.12601) | **Splits each slot into a per-frame appearance state and a persistent identity state**; competition-modulated aggregation stops weak slots stealing tokens | Yes. Reports +9.0 ARI, +9.7 FG-ARI, +6.4 mBO on YT-VIS over the best baseline ([search-indexed text](https://arxiv.org/html/2606.12601)) | A recurrent identity filter, decoupled from appearance | [project page](https://uark-aicv.github.io/dssa/) |
| TSA | 2026 [arXiv 2606.13714](https://arxiv.org/abs/2606.13714) | **Per-slot, per-frame activation score** learned without visibility labels. Inactive slots are frozen and masked out of the decoder | MOVi-C/E, YT-VIS, OVIS | Explicit "object hidden" state; largest gains on occlusions longer than 20 frames | no code found |
| UniSlot | 2026 [arXiv 2606.23436](https://arxiv.org/abs/2606.23436) | Trained on YT-VIS 2021, zero-shot on DAVIS-17 | Yes | — | — |
| SSync | 2026 [arXiv 2606.15527](https://arxiv.org/abs/2606.15527) | Fuses slot-attention maps with decoder maps for sharper boundaries | Yes | — | — |
| Semantic Slots | BMVC 2026 [arXiv 2608.21636](https://arxiv.org/abs/2608.21636) | **Transformer decoder with image context**; slots become position-invariant semantic queries, so slots from one frame can decompose later frames | Yes: 86.6 ARI and 62.8 mBO on YT-VIS, +31 mBO over VideoSAUR | Identity comes from *semantic* matching rather than spatial recurrence. This is the property that survives cuts | not stated |
| Loci-Looped | 2023 [arXiv 2310.10372](https://arxiv.org/abs/2310.10372) | Slot model that fuses its own latent predictions with observations | Synthetic | "Imagines" hidden objects and predicts reappearance (object permanence) | — |
| Better Slots, Better Worlds | RLC 2026 workshop [arXiv 2608.12078](https://arxiv.org/abs/2608.12078) | Study of object-centric world models | Robotics | Planning success rises with FG-ARI/mBO and then plateaus | — |

### 1.2 How well this works on real video

| Model | Benchmark | Reported | Source |
|---|---|---|---|
| VideoSAUR | YT-VIS / DAVIS | mBO in roughly the 0.30–0.41 range depending on dataset version and backbone (index excerpt: 41.3 YT-VIS-19, 34.0 DAVIS) | [NeurIPS 2023 paper](https://papers.nips.cc/paper_files/paper/2023/file/c1fdec0d7ea1affa15bd09dd0fd3af05-Paper-Conference.pdf) |
| SOLV (with merging) | YT-VIS | mBO 45.3 (as cited by VideoSAUR's related-work discussion) | same |
| SlotContrast vs VideoSAUR | YT-VIS 2021 | +6.8 FG-ARI, +4 mBO | [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.pdf) |
| SlotMatch / SlotContrast | YT-VIS | video mBO 32.8 / 32.4 | [arXiv 2508.03411](https://arxiv.org/html/2508.03411) |
| UniSlot (zero-shot) | DAVIS-17 | FG-ARI 0.148, mBO 0.379, J&F 0.242 | [arXiv 2606.23436](https://arxiv.org/html/2606.23436) |
| Semantic Slots | YT-VIS | ARI 86.6, mBO 62.8 | [arXiv 2608.21636](https://arxiv.org/abs/2608.21636) |

**Reading these numbers.**
- mBO of 0.3–0.6 means slots are coarse blobs that often merge parts or split objects.
- Prompted foundation trackers are in a different regime. The SAM 2 paper reports J&F around 90 on DAVIS-17 in semi-supervised VOS ([arXiv 2408.00714](https://arxiv.org/abs/2408.00714); figure from memory, not re-verified here, and a different task setting).
- Practical consequence: entity discovery should come from SAM 2 masks, with slots doing binding, 3D lifting and persistence. Slots should not be asked to discover entities from scratch.
- **Licences.**
  - SAM 2 is Apache-2.0 and its Hugging Face weights are not gated ([model card](https://huggingface.co/facebook/sam2.1-hiera-small)).
  - SAM 3 is gated ("You need to agree to share your Hugging Face information", per the [model card](https://huggingface.co/facebook/sam3)) under the custom [SAM License](https://github.com/facebookresearch/sam3). That violates the no-gated-downloads constraint.

### 1.3 Object-centric 3D and scene representation transformers

| Method | Venue | What it does | Data | Relevance | Code / licence |
|---|---|---|---|---|---|
| SRT | CVPR 2022 [arXiv 2111.13152](https://arxiv.org/abs/2111.13152) | Set-latent scene representation from posed images; light-field decoder (one forward pass per ray, no volume rendering) | Synthetic MSN, Street View | Fast rendering by attention, but not object-centric and produces no explicit geometry | unofficial [stelzner/srt](https://github.com/stelzner/srt), MIT |
| OSRT | NeurIPS 2022 [arXiv 2206.06922](https://arxiv.org/abs/2206.06922) | Slot attention over the SRT set-latent; "Slot Mixer" decoder | Synthetic MSN-Hard | Shows 3D-consistent slots emerging from multi-view rendering losses | unofficial [stelzner/osrt](https://github.com/stelzner/osrt), MIT |
| uORF | ICLR 2022 [arXiv 2107.07905](https://arxiv.org/abs/2107.07905) | Slots → per-object NeRFs from a single image | Synthetic rooms | Background/foreground slot split worth copying | [KovenYu/uORF](https://github.com/KovenYu/uORF), MIT |
| ObSuRF | 2021 [arXiv 2104.01148](https://arxiv.org/abs/2104.01148) | Slots → object NeRFs, trained using RGB-D | Synthetic | Early use of depth to make the 3D slot loss tractable | [stelzner/obsurf](https://github.com/stelzner/obsurf), MIT |
| SlotLifter | ECCV 2024 [arXiv 2408.06697](https://arxiv.org/abs/2408.06697) | Slot-guided feature lifting into object-centric radiance fields | 4 synthetic and 4 real datasets | Most real-data-capable of the NeRF-slot line | — |
| Scene-agnostic OCL for 3DGS | CVPR 2026 3DMV workshop [arXiv 2604.09045](https://arxiv.org/abs/2604.09045) | Pre-trained slot model (GOCL) plus a scene-agnostic object codebook supervising identity features on 3D Gaussians | n/a in abstract | A **dataset-level identity codebook** is the right idea for cross-shot re-ID | — |
| 4D Primitive-Mâché | arXiv Dec 2025 [2512.16564](https://arxiv.org/abs/2512.16564) | Monocular video → rigid 3D primitives with per-primitive SE(3) motion from dense correspondences; **extrapolates motion for invisible objects**; replayable persistent 4D reconstruction | Casual real video | Closest existing system to the "persistent entity" goal, though optimisation-based, not amortised | [project](https://makezur.github.io/4DPM/) |
| D-Prism | CVPR 2026 [arXiv 2604.17082](https://arxiv.org/abs/2604.17082) | Differentiable primitives for dynamic scenes; 3DGS bound to primitive surfaces; deformation network; adaptive primitive count | Multi-part / jointed objects | Validates the "Gaussian skin on primitives" renderer idea | [zju3dv/D-Prism](https://github.com/zju3dv/D-Prism), MIT |

**Takeaway.** The 3D slot models (SRT, OSRT, uORF, ObSuRF) were demonstrated on synthetic multi-view data with known cameras. None yields low-poly geometry, and none handles cuts. The line moving toward real video in 2025–26 is primitive- or Gaussian-based optimisation (4DPM, D-Prism) supported by foundation-model correspondences. Your amortised network would be new in combining these.

### 1.4 Persistence through occlusion and cuts

| Mechanism | Example | Handles occlusion? | Handles a cut? |
|---|---|---|---|
| Recurrent slot carry-over | SAVi, VideoSAUR | Partly; slots drift while hidden | **No.** A cut changes all appearance and position at once |
| Temporal contrastive identity | SlotContrast | Often | No (assumes consecutive-frame similarity) |
| Identity/appearance split | DSSA | Better under fast motion and partial occlusion | Partly. The identity state could survive a cut if matching is semantic |
| Activation gating (freeze hidden slots) | TSA | Yes, gains above 20-frame occlusions | Partly. The "hidden" state is the right primitive for off-screen entities |
| Latent imagination | Loci-Looped, 4DPM motion extrapolation | Yes | No (motion extrapolation across a cut is wrong) |
| Semantic, position-invariant slot queries | Semantic Slots | Yes | **Most promising**: slots from frame A can decompose frame B |
| Memory-bank trackers | SAM 2 ([Apache-2.0](https://github.com/facebookresearch/sam2)); SAMURAI adds Kalman motion ([Apache-2.0](https://github.com/yangchris11/samurai)) | Yes | Motion priors break at cuts; appearance memory sometimes re-acquires |
| 3D-aware association | PHALP ([repo](https://github.com/brjathu/PHALP)), 4DHumans ([MIT](https://github.com/shubham-goel/4D-Humans)) | Yes, via 3D appearance, pose and location | Appearance part transfers; location part does not |
| Shot-boundary detection | [TransNetV2](https://github.com/soCzech/TransNetV2) (MIT), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (BSD-3) | — | Detects cuts; identity must then come from re-ID |

**Conclusion.** No object-centric paper found treats edited video with cuts. Persistence across cuts has to be built from four parts:
1. a cut detector, or a learned per-frame "shot-change" bit;
2. a video-level entity and location memory;
3. assignment by appearance and identity embeddings (DSSA-style identity state, codebook à la [2604.09045](https://arxiv.org/abs/2604.09045));
4. verification by re-rendering (Idea 2 below).

---

## 2. Shape representations for a low-poly, predictable, differentiably renderable output

Criteria:
- **P:** a network can regress it with fixed output size and smooth gradients.
- **R:** cheap to render differentiably on Metal.
- **L:** directly exportable as low-poly.
- **E:** editable.
- **A:** animatable.

### 2.1 Primitives and part-based abstractions

| Method | Venue | Primitive | Input | Notes | P / R / L / E / A | Code / licence |
|---|---|---|---|---|---|---|
| Superquadrics Revisited | CVPR 2019 [arXiv 1904.09970](https://arxiv.org/abs/1904.09970) | Superquadrics | 3D / voxels | Network regresses M superquadrics; parsimony loss | ✓/✓/✓/✓/~ | [repo](https://github.com/paschalidoud/superquadric_parsing), licence unclear (NOASSERTION) |
| CvxNet | CVPR 2020 [arXiv 1909.05736](https://arxiv.org/abs/1909.05736) | Convexes (half-space intersections) | Image / 3D | Exportable as low-poly polytopes; needs an indicator-function renderer | ✓/~/✓/✓/~ | Google (TF) |
| Neural Parts | CVPR 2021 [arXiv 2103.10429](https://arxiv.org/abs/2103.10429) | Sphere deformed by an invertible net | Image | Expressive parts, still genus-0, mesh-exportable | ✓/✓/~/~/~ | [MIT](https://github.com/paschalidoud/neural_parts) |
| **Differentiable Blocks World (DBW)** | NeurIPS 2023 [arXiv 2307.05473](https://arxiv.org/abs/2307.05473) | **Textured superquadric meshes**, with transparency for pruning | Calibrated multi-view images | Per-scene optimisation of about 4 h on one GPU; env pins PyTorch3D 0.7.1 and CUDA 11.3 ([repo](https://github.com/monniert/differentiable-blocksworld)) | ✓/✓/✓/✓/~ | MIT |
| SuperDec | ICCV 2025 [arXiv 2504.00992](https://arxiv.org/abs/2504.00992) | Superquadrics, feed-forward per object | Point clouds plus instance masks | Fast amortised fitting, trained on ShapeNet | ✓/✓/✓/✓/~ | [MIT](https://github.com/elisabettafedele/superdec) |
| PrimitiveAnything | SIGGRAPH 2025 [arXiv 2505.04622](https://arxiv.org/abs/2505.04622) | Cuboids, elliptical cylinders, ellipsoids; **autoregressive** "human-crafted" assemblies | Point clouds (Michelangelo encoder) | Learns how humans abstract shapes | ✓/✓/✓/✓/~ | [**GPL-3.0**](https://github.com/PrimitiveAnything/PrimitiveAnything) |
| 3D Convex Splatting | CVPR 2025 [arXiv 2411.14974](https://arxiv.org/abs/2411.14974) | Smooth convexes as radiance primitives | Multi-view | Splatting renderer for hard-edged primitives | ~/✓/~/~/✗ | [Apache-2.0](https://github.com/convexsplatting/convex-splatting) |
| Triangle Splatting / + | 2025 [2505.19175](https://arxiv.org/abs/2505.19175), [2509.25122](https://arxiv.org/abs/2509.25122) | Triangles; "+" enforces **opaque triangles with shared vertices**, so the output is directly game-engine-ready | Multi-view | State of the art for mesh-based NVS on Mip-NeRF360 / T&T | ~/✓/✓/~/✗ | [Apache-2.0](https://github.com/trianglesplatting/triangle-splatting) |
| MeshSplatting | Dec 2025 [arXiv 2512.06818](https://arxiv.org/abs/2512.06818) | Opaque connected meshes; restricted Delaunay connectivity | Multi-view | Reports about 200 FPS rendering on a MacBook M4 | ~/✓/✓/~/✗ | — |
| D-Prism | CVPR 2026 [arXiv 2604.17082](https://arxiv.org/abs/2604.17082) | Primitives plus bound 3DGS, deformation net | Multi-view dynamic | Articulated structure plus Gaussian appearance | ✓/✓/✓/✓/✓ | MIT |
| 4D Primitive-Mâché | [2512.16564](https://arxiv.org/abs/2512.16564) | Rigid primitives with per-primitive SE(3) | **Monocular video** | Persistent, replayable | ✓/✓/~/✓/✓ | — |

### 2.2 Low-poly and artist-mesh generation (autoregressive and similar)

| Model | Year | Key idea | Face budget / efficiency | Code / licence |
|---|---|---|---|---|
| PolyGen | 2020 [2002.10880](https://arxiv.org/abs/2002.10880) | Vertex then face transformers | small | — |
| MeshGPT | CVPR 2024 [2311.15475](https://arxiv.org/abs/2311.15475) | Learned face tokens (graph VQ) plus GPT | ≈800 faces | unofficial [lucidrains](https://github.com/lucidrains/meshgpt-pytorch), MIT |
| MeshXL | 2024 [2405.20853](https://arxiv.org/abs/2405.20853) | Coordinate-sequence LLM | ≈800 | — |
| MeshAnything / V2 | ICLR 2025 / ICCV 2025 [2406.10163](https://arxiv.org/abs/2406.10163), [2408.02555](https://arxiv.org/abs/2408.02555) | Shape-conditioned artist meshes; V2 adds adjacent mesh tokenization | 800 → 1.6k ([survey excerpt](https://arxiv.org/pdf/2409.18114)) | [**S-Lab License 1.0, non-commercial**](https://github.com/buaacyw/MeshAnythingV2) |
| EdgeRunner | 2024 [2409.18114](https://arxiv.org/abs/2409.18114) | Autoregressive auto-encoder, compact tokens | ≈4k | — |
| BPT | CVPR 2025 [2411.07025](https://arxiv.org/abs/2411.07025) | Blocked and patchified tokenization, about 75% shorter sequences | >8k faces; point-cloud or image conditioning | [Tencent-Hunyuan/bpt](https://github.com/Tencent-Hunyuan/bpt), custom licence |
| Meshtron | 2024 [2412.09548](https://arxiv.org/abs/2412.09548) | Hourglass transformer, truncated training | very high face counts (NVIDIA) | — |
| TreeMeshGPT | CVPR 2025 [2503.11629](https://arxiv.org/abs/2503.11629) | Tree sequencing, 2 tokens per face (about 22% of naive) | — | — |
| DeepMesh | ICCV 2025 [2503.15265](https://arxiv.org/abs/2503.15265) | New tokenizer plus **DPO from human preference** | high | [Apache-2.0](https://github.com/zhaorw02/DeepMesh) |
| Mesh Silksong | 2025 [2507.02477](https://arxiv.org/abs/2507.02477) | Each vertex visited once | — | — |
| FastMesh | 3DV 2026 [2508.19188](https://arxiv.org/abs/2508.19188) | Autoregressive vertices only, plus a **bidirectional one-shot face predictor** (adjacency) | ≈23% of the most compact tokenizer, more than 8× faster | — |
| FlashMesh | 2025 [2511.15618](https://arxiv.org/abs/2511.15618) | Structured speculative decoding | faster autoregression | — |
| FACE | 2026 [2603.01515](https://arxiv.org/abs/2603.01515) | One token per face | — | — |
| MeshWeaver | CVPR 2026 [2606.04688](https://arxiv.org/abs/2606.04688) | Sparse-voxel-guided "weaving", whole-vertex tokens | up to 16k faces, 18% compression | — |
| MeshFIM | 2026 [2605.08744](https://arxiv.org/abs/2605.08744) | **Fill-in-the-middle local editing of low-poly meshes** | — | paper CC-BY-4.0; no code found |

**Verdict.**
- These models are point-cloud- or image-conditioned generators of 0.5–1B+ parameters, trained on large 3D asset collections.
- Inference is token-by-token, and the discrete topology output is not differentiable with respect to an image loss.
- They cannot sit inside the self-supervised loop on 16 GB.
- Use them only as an optional **export-time remesher** of the trained primitive scene, and only those with permissive licences (DeepMesh, Apache-2.0; others unknown or non-commercial).
- FastMesh's split (autoregressive vertices plus a one-shot adjacency predictor) and MeshFIM's local editing are the design ideas most worth borrowing for a future learned low-poly decoder.

### 2.3 Structured 3D latents

| Model | Representation | Hardware reality | Licence |
|---|---|---|---|
| 3DShape2VecSet | Set of latent vectors → cross-attention occupancy decoder ([2301.11445](https://arxiv.org/abs/2301.11445)) | Small, trainable on M5; decoding needs marching cubes | [MIT](https://github.com/1zb/3DShape2VecSet) |
| TRELLIS | Sparse structured latent → GS / RF / mesh ([2412.01506](https://arxiv.org/abs/2412.01506)) | CUDA kernels | [MIT](https://github.com/microsoft/TRELLIS) |
| TRELLIS.2 | **O-Voxel** sparse "omni-voxel" (geometry plus PBR, arbitrary topology), 16× compression VAE, 4B flow model ([2512.14692](https://arxiv.org/abs/2512.14692)) | Needs ≥24 GB NVIDIA; tested on A100/H100 ([README](https://github.com/microsoft/TRELLIS.2)). The Mac port takes about 3.5 min and about 18 GB peak on an M4 Pro 24 GB, inference only, and needs **gated DINOv3 and non-commercial RMBG-2.0** ([trellis-mac](https://github.com/shivampkumar/trellis-mac)) | MIT (weights too), but dependencies are gated |
| TripoSG | Rectified-flow SDF ([2502.06608](https://arxiv.org/abs/2502.06608)) | Large | [MIT](https://github.com/VAST-AI-Research/TripoSG) |
| Hunyuan3D-2 / 2.1 | VecSet-style DiT | Large | Tencent community licence ([repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)); check territory restrictions |
| SAM 3D Objects | Single image → geometry, texture and **layout** ([2511.16624](https://arxiv.org/abs/2511.16624)) | Large | SAM License ([repo](https://github.com/facebookresearch/sam-3d-objects)); SAM-family Hugging Face repos are gated |
| PartCrafter / OmniPart | Part-aware compositional latents ([PartCrafter](https://github.com/wgsxm/PartCrafter), [OmniPart](https://github.com/HKU-MMLab/OmniPart)) | Large | MIT / MIT |

**Verdict.** The *idea* of a structured latent is right: a set of local latents tied to space, as in VecSet or sparse voxels. Running these models on 16 GB, and the licences and gated dependencies around them, do not fit. A small VecSet-style latent per entity, decoded to **primitive parameters plus per-primitive residuals** rather than to an occupancy field, gives the same structure at a fraction of the cost (Section 6).

### 2.4 Articulated templates for people and animals

| Asset | Type | Access / licence | Use |
|---|---|---|---|
| SMPL / SMPL-X / SMAL / MANO / FLAME | Parametric body, animal, hand and face models | **Registration required and restrictive licences** (e.g. [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)) | **Violates "no sign-ups"; avoid** |
| **Anny** (NAVER LABS, Nov 2025) | Scan-free, all-ages (infant to elder) parametric body in PyTorch; regressors to and from SMPL-X; matches or slightly beats SMPL-X on 3DPW/EHF HMR ([arXiv 2511.03589](https://arxiv.org/abs/2511.03589), [blog](https://europe.naverlabs.com/blog/anny-a-free-to-use-3d-human-parametric-model-for-all-ages/)) | **Apache-2.0, no sign-up** ([repo](https://github.com/naver/anny)) | **Recommended human template** |
| **MHR**, Momentum Human Rig (Meta, Nov 2025) | Decoupled skeleton and shape (ATLAS lineage), pose correctives; used by SAM 3D Body ([arXiv 2511.15586](https://arxiv.org/abs/2511.15586)) | **Apache-2.0** ([repo](https://github.com/facebookresearch/MHR)) | Alternative template, game-rig-like |
| SAM 3D Body | HMR model on MHR ([arXiv 2602.15989](https://arxiv.org/abs/2602.15989)) | SAM License, **gated** ([HF](https://huggingface.co/facebook/sam-3d-body-dinov3)) | Not usable under the constraints |
| 4DHumans / HMR 2.0 | HMR plus tracking | MIT code, but depends on SMPL | Not usable as-is |
| MagicPony / 3D-Fauna / Ponymation | Category-level articulated animals learned from web images and video with **DINO feature rendering** ([2211.12497](https://arxiv.org/abs/2211.12497), [2401.02400](https://arxiv.org/abs/2401.02400), [2312.13604](https://arxiv.org/abs/2312.13604)) | Research code | The recipe (feature loss plus multi-hypothesis viewpoints) transfers |
| CASA / CAMM | Category-agnostic skeleton discovery from video ([2211.03568](https://arxiv.org/abs/2211.03568), [2304.06937](https://arxiv.org/abs/2304.06937)) | Research | Skeleton-from-motion idea |
| MoCapAnything | Monocular video plus any rigged asset → animation ([2512.10881](https://arxiv.org/abs/2512.10881)) | — | Retargeting stage |
| UniRig / Puppeteer / MagicArticulate | Auto-rigging (skeleton plus skinning) of meshes ([UniRig](https://github.com/VAST-AI-Research/UniRig) MIT; [Puppeteer](https://github.com/Seed3D/Puppeteer) Apache-2.0; [MagicArticulate](https://github.com/Seed3D/MagicArticulate) Apache-2.0) | Code permissive; check checkpoints and training-data terms | Optional export-time rigging of non-human entities |

---

## 3. Differentiable rendering on an Apple M5 (no CUDA)

### 3.1 Hardware facts

- **M5:** 10-core GPU with a Neural Accelerator in each core, "over 4×" the M4's peak GPU AI compute, 153 GB/s unified memory bandwidth ([Apple newsroom](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)).
- **MLX** uses the accelerators through Metal 4 TensorOps on macOS 26.2 or later ([Apple ML research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).
- **Measured here (4096³ matmul):**

  | | bf16 | fp16 | fp32 |
  |---|---|---|---|
  | MLX | 15.3 TFLOP/s | 15.2 TFLOP/s | 8.4 TFLOP/s |
  | PyTorch-MPS 2.15 | 14.4 TFLOP/s | 14.5 TFLOP/s | 3.2 TFLOP/s |

  So both frameworks now use the matrix units in half precision. MLX is about 2.6× faster in fp32.

### 3.2 Renderer survey

| Renderer | Primitive | Runs on M5? | Differentiable | Licence | Performance evidence | Maturity / notes |
|---|---|---|---|---|---|---|
| PyTorch3D | Mesh (soft raster), points | **CPU only.** "Several components have underlying implementation in CUDA… a subset… have CPU implementations" ([INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)). The only MPS work is an Oct 2026 RFC for point-cloud ops ([#2049](https://github.com/facebookresearch/pytorch3d/issues/2049)); MPS bus error in `sample_points_from_meshes` ([#1983](https://github.com/facebookresearch/pytorch3d/issues/1983)) | Yes | BSD-3 | CPU raster too slow for training | Do not build on it for the Mac |
| SoftRas | Mesh | CUDA only | Yes | [MIT](https://github.com/ShichenLiu/SoftRas) | — | The *algorithm* is easy to re-implement (3.3) |
| nvdiffrast | Mesh (hard raster plus antialias) | CUDA only | Yes | [NVIDIA Source Code License (1-Way Commercial)](https://github.com/NVlabs/nvdiffrast) | — | Reference design |
| **mtldiffrast** | Mesh: rasterize / interpolate / texture (mip) / antialias, all with autograd | **Yes, Metal, zero-copy MPS tensors** | Yes | MIT | None published | 11 commits, 5 stars; **API-compatible with nvdiffrast** ([repo](https://github.com/pedronaugusto/mtldiffrast)). Used by TRELLIS.2 and Pixal3D Mac ports for baking ([Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac)) |
| **Mitsuba 3.9 `metal_ad_*`** | Physically based (rays, meshes, BSDFs) | **Yes, since 3.9.0 (2026-06-26)**, with hardware ray tracing and Dr.Jit JIT ([release notes](https://mitsuba.readthedocs.io/en/v3.9.1/release_notes.html)) | Yes (AD variants) | BSD-style ([repo](https://github.com/mitsuba-renderer/mitsuba3)) | None published | Ideal for gradient checking, shadow-aware refinement and synthetic data. Heavy for a per-step training renderer |
| SlangPy | Any (write your own shader) | Metal listed as a backend ([docs](https://slangpy.shader-slang.org/en/latest/)) | Slang autodiff; **Metal autodiff and Torch-MPS interop unverified** | Apache-2.0 WITH LLVM-exception | — | [Splatelier](https://github.com/shuta-ochiai/Splatelier) compiles differentiable Slang kernels to Metal with an MLX-Swift trainer, so the path exists |
| Taichi | Any | Metal backend | Yes | [Apache-2.0](https://github.com/taichi-dev/taichi) | — | Viable for custom kernels; ecosystem momentum lower |
| NVIDIA Warp | Any | CPU only on Mac (no Metal) | Yes | [Apache-2.0](https://github.com/NVIDIA/warp) | — | Not useful for GPU here |
| **metal-gauss** | 3DGS, fused fwd+bwd Metal kernels on PyTorch-MPS, runtime-compiled | Yes | Yes | [MIT](https://github.com/QuintonQu/metal-gauss) | **On M5:** 69 fps forward at 600k splats and 768²; 481 fps at 100k and 384²; 2,000 training iterations in 57–68 s; 31.9 dB mean on NeRF-synthetic in 15 min | Best evidence of M5 splat throughput |
| metalsplat | 3DGS and **2DGS** (depth, normals, distortion) | Yes (MSL via PyTorch runtime compiler) | Yes | [MIT](https://github.com/tchauffi/metalsplat) | 15k iterations in 42 min (about 6 it/s) at 1297×840 with 330k Gaussians on an "M-series" GPU | Pure-PyTorch reference implementation used as a gradient oracle |
| mlx3D | **Soft triangle raster**, NeRF / hash-grid, 3DGS (Metal kernels via `mx.custom_function`), marching cubes, mesh losses | Yes | Yes | [MIT](https://github.com/amirhossein-razlighi/mlx3D) | About 30 fps forward at 720p with 100k Gaussians | The most complete MLX 3D library found; 150 commits |
| gsplat-mlx | 3DGS / 2DGS | Yes | Yes | README says Apache-2.0, but **no LICENSE file detected** ([repo](https://github.com/RobotFlow-Labs/gsplat-mlx)) | 128², 1k Gaussians: 788 ms forward+backward (Python loop rasteriser) | Slow until its Metal raster lands |
| OpenSplat | 3DGS | Yes (MPS default on macOS) | Yes | **AGPL-3.0** ([repo](https://github.com/WebODM/OpenSplat)) | — | C++ CLI, no Python API; AGPL is a problem |
| Brush | 3DGS (Rust/Burn/WebGPU) | Yes | Yes | [Apache-2.0](https://github.com/ArthurBrussee/brush) | "faster than gsplat" (self-reported) | No Python API |
| gsplat | 3DGS | CUDA only | Yes | [Apache-2.0](https://github.com/nerfstudio-project/gsplat) | — | Reference |
| Gaussian Mesh Renderer | **Mesh → one analytic Gaussian per triangle**, rendered by a 3DGS rasteriser; smoother gradients | Portable (swap in a Metal GS rasteriser) | Yes | [MIT](https://github.com/huntorochi/Gaussian-Mesh-Renderer) | — | ICASSP 2026 ([2602.14493](https://arxiv.org/abs/2602.14493)). Exactly the "mesh output, splat gradients" bridge |
| Meshtryoshka | SDF → nested mesh shells, **non-differentiable off-the-shelf rasteriser**; gradients flow only through vertex opacities | Portable to Metal's hardware rasteriser | Yes (through the SDF) | — | Scales to unbounded real scenes ([2606.28622](https://arxiv.org/abs/2606.28622)) | Uses the GPU's hardware raster path |
| Stochastic-gradient rasteriser | Any rasteriser plus parameter perturbation and ID/UV buffers | Portable | Stochastic gradients | — | Millions of parameters in-engine ([2404.09758](https://arxiv.org/abs/2404.09758)) | Noisier than nvdiffrast |

### 3.3 Local measurement: dense, untiled differentiable rendering in plain tensor ops

The script ([research/bench/02_dense_raster_bench.py](bench/02_dense_raster_bench.py)) implements two renderers. Each computes **every pixel against every primitive**, with no culling and no custom kernels, and times loss plus a full backward pass:
1. a SoftRas-style triangle rasteriser: signed edge distance → sigmoid coverage, clamped barycentric depth, softmax depth aggregation, soft silhouette;
2. dense 2D-Gaussian alpha compositing with exclusive cumulative transmittance.

| Resolution × primitives | Pairs | MLX soft-raster (eager / `mx.compile`) | MLX peak memory | PyTorch-MPS soft-raster | MLX dense 2D-GS (compiled) |
|---|---|---|---|---|---|
| 64² × 256 | 1.0 M | 29 / **15.5 ms** | 0.43 GB | 27 ms | 6.1 ms, 0.17 GB |
| 128² × 512 | 8.4 M | 175 / **131 ms** | 1.9 GB | 218 ms | 50 ms, 1.3 GB |
| 128² × 1024 | 16.8 M | 404 / **301 ms** | 3.2–3.5 GB | 440 ms | 99 ms, 1.75 GB |
| 128² × 2048 | 33.6 M | 701 / **556 ms** | 6.0–7.0 GB | 883 ms | 202 ms, 3.4 GB |
| 256² × 512 | 33.6 M | 699 / **538 ms** | 6.0–7.0 GB | 905 ms | 177 ms, 3.4 GB |

An earlier attempt at 256² × 2048 (134 M pairs) exhausted the 16 GB and swapped, so it was stopped.

**What this means.**
- Dense soft rasterisation costs about **16 ns and about 200 B per pixel–primitive pair** (MLX compiled). Dense Gaussian compositing costs about 6 ns and about 100 B per pair.
- MLX-compiled is 1.6–1.7× faster than PyTorch-MPS eager on the same code.
- **At token resolution, dense is fine.** For example, 8 frames × 48² pixels × 512 visible primitives is 9.4 M pairs, about 150 ms and about 2 GB. That is enough to train the feature-, mask- and depth-space losses (Section 4) without writing any kernel.
- **At image resolution it is not.** RGB at 128–256 px needs a tiled kernel with culling: each pixel touches about 10–50 primitives instead of 2,000, a reduction of roughly 50–200×.
- Fused kernels are known to deliver this on Metal: metal-gauss's M5 numbers, and MLX's own docs, which show a fused Metal VJP 40× faster than the composed-ops version (`grid_sample`, 676 ms → 16.7 ms on an M1 Max) ([MLX custom kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).

### 3.4 Kernel-writing building blocks on the Mac

- **MLX.** `mx.fast.metal_kernel` takes the kernel body. `mx.custom_function` plus `.vjp` defines the backward pass. `atomic_outputs=True` plus `init_value=0` handles gradient scatter. `simd_sum` pre-reduction makes atomics cheap ([docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).
- **PyTorch.** `torch.mps.compile_shader` "compiles compute shader from source and allows one to invoke kernels defined there" ([docs](https://docs.pytorch.org/docs/2.14/mps.html)). This is what metal-gauss and metalsplat use. You wrap it in `torch.autograd.Function` yourself.
- **Hardware raster path.** Metal render pipelines, including Mitsuba 3.9's Metal ray tracing, can be paired with Meshtryoshka-style or stochastic-gradient tricks when exact triangle rasterisation at high resolution is needed.

### 3.5 Renderer recommendation

| Stage | Renderer | Why |
|---|---|---|
| Warm start and most self-supervised training | **A custom MLX tiled soft rasteriser over tessellated primitives**, outputting RGB, feature, slot-ID, depth and normal in one pass. Prototype it first as the dense tensor-op version above at ≤64² | DBW-style superquadric meshes; differentiable with respect to primitive parameters in closed form; same code path for every entity type |
| Fallback or complement | **Gaussian skin on primitives**, i.e. triangle → analytic Gaussian ([2602.14493](https://arxiv.org/abs/2602.14493)) or [D-Prism](https://arxiv.org/abs/2604.17082)-style bound Gaussians, rendered with metal-gauss (PyTorch) or mlx3D kernels (MLX) | Tile-based Metal kernels already exist and are benchmarked on the M5; smoother gradients |
| Late fine-tuning and export check | **mtldiffrast** hard raster plus antialias (nvdiffrast semantics) | What you train equals what you ship (opaque low-poly) |
| Reference / gradient oracle / shadows / synthetic data | **Mitsuba 3.9 `metal_ad_*`** | Physically based, BSD-licensed, now on Metal |

---

## 4. Making re-render-and-compare well-conditioned

### 4.1 Signals

| Signal | Source (licence) | Strength | Weakness | When |
|---|---|---|---|---|
| **Per-slot masks / soft ID map** | SAM 2 (Apache-2.0) masks, Hungarian-matched to rendered soft slot IDs | Strong shape and pose signal; separates entities; cheap at low res | No gradient when rendered and target masks don't overlap ([2503.05953](https://arxiv.org/abs/2503.05953)); front/back flip ambiguity ([2103.15627](https://arxiv.org/abs/2103.15627)); SAM masks sometimes include or exclude shadows | From step 0. Add a distance-transform term so gradients exist without overlap |
| **Feature-metric (DINO)** | [DINOv2](https://github.com/facebookresearch/dinov2) Apache-2.0, not gated; [registers variant](https://arxiv.org/abs/2309.16588). **DINOv3 is gated** under a custom licence ([trellis-mac README](https://github.com/shivampkumar/trellis-mac), [licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) | Wide basins, robust to lighting, part-level correspondences. **MagicPony's ablation shows viewpoint prediction collapses without the feature-rendering loss** ([supp.](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wu_MagicPony_Learning_Articulated_CVPR_2023_supplemental.pdf)); also used by Ponymation and LASSIE ([2312.13604](https://arxiv.org/abs/2312.13604), [2207.03434](https://arxiv.org/abs/2207.03434)); featuremetric refinement works in SfM ([PixSfM 2108.08291](https://arxiv.org/abs/2108.08291)) | Low resolution (patch 14); view-dependent; positional artefacts (use registers) | From step 0, at token resolution |
| **Depth** | Monocular depth models: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (code Apache-2.0; **Large weights CC-BY-NC** ([HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)); check the Small checkpoint), [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache-2.0 code), [MoGe](https://github.com/microsoft/MoGe) (MIT code), [Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro) (Apple licence) | Fixes depth ordering and occlusion, and resolves the scale of the location set | Scale/shift ambiguity per frame; flicker; smeared edges | Scale-and-shift-invariant or pairwise-ordinal loss, per shot |
| **Correspondence / point tracks** | [TAPNet/BootsTAP/TAPNext](https://github.com/google-deepmind/tapnet) Apache-2.0 (CoTracker is CC-BY-NC-4.0: [repo](https://github.com/facebookresearch/co-tracker)) | Directly constrains motion; 4DPM is built on it ([2512.16564](https://arxiv.org/abs/2512.16564)) | Tracks die at cuts and occlusions | Motion and articulation |
| **Photometric RGB** | the frames | Fine detail, texture | Lighting, shadows, speculars, motion blur, compression; many local minima | Late; low weight; after poses settle |
| **Histogram / locally orderless loss** | [CVPR 2025, 2503.21931](https://arxiv.org/abs/2503.21931) | Gives gradient support where image gradients are sparse (moving content); more reliable than image pyramids per the authors | Extra cost | Pose and translation recovery |

### 4.2 Known failure modes and mitigations

| Failure mode | Evidence | Mitigation |
|---|---|---|
| Plateaus and vanishing gradients (no overlap) | [2503.05953](https://arxiv.org/abs/2503.05953); "extensive plateaus" in image-space loss for differentiable-render autoencoders ([2506.08191](https://arxiv.org/abs/2506.08191)) | Soft rasterisation with sigma annealing (SoftRas avoided the NMR local minimum, [1904.01786](https://arxiv.org/abs/1904.01786)); distance-transform mask loss; amortised initialisation from the network; latent-space losses on synthetic data |
| Pose flips / symmetric minima | [2103.15627](https://arxiv.org/abs/2103.15627); MagicPony multi-hypothesis viewpoints ([2211.12497](https://arxiv.org/abs/2211.12497)) | 4–8 orientation hypotheses per entity, winner-take-all with a learned score |
| **Shadows and reflections claimed as objects**, or baked into the wrong layer | Omnimatte family: "matching objects and their effects is challenging" ([2105.06993](https://arxiv.org/abs/2105.06993)); [Generative Omnimatte](https://arxiv.org/abs/2411.16683); [OmnimatteZero](https://arxiv.org/abs/2503.18033) notes attention maps carry effect cues | Entity-owned shadow decals (Idea 4); a per-slot existence cost; slots must be supported by SAM masks *and* 3D consistency |
| Over-fragmentation and slot swapping | [slot curriculum 2603.22758](https://arxiv.org/abs/2603.22758); DSSA ([2606.12601](https://arxiv.org/abs/2606.12601)) | SAM-mask seeding; identity/appearance split; competition-modulated updates |
| Hidden-object drift | TSA ([2606.13714](https://arxiv.org/abs/2606.13714)) | Activation gating; freeze state while invisible |
| Texture "paints" geometry (a flat card with a picture explains everything) | General analysis-by-synthesis ambiguity, the reason DBW-style regularisers exist | Feature and depth losses; multi-view (multi-frame) consistency; low-res texture budget; parsimony |
| Camera vs object motion ambiguity; depth-scale ambiguity | Monocular | Static-location assumption per shot; correspondences plus depth for cameras; entity scale priors (human height from Anny) |
| Cuts break temporal smoothness priors | — | Per-shot camera reset; never apply velocity priors across a detected cut |
| Soft-train / hard-ship gap (semi-transparent soft geometry looks different once opaque) | Triangle Splatting+ enforces opacity for this reason ([2509.25122](https://arxiv.org/abs/2509.25122)) | Anneal sigma and opacity to hard; final phase on mtldiffrast |

### 4.3 Suggested curriculum

1. **Synthetic warm start** with exact ground truth. Losses on parameters (pose, primitive params, camera) plus all render losses.
2. **Real video, token resolution (32²–64²).** DINOv2 feature loss, matched-mask cross-entropy plus distance transform, ordinal/SSI depth, track reprojection. Sigma high to low. Multi-hypothesis poses.
3. **128² RGB** added at low weight. Shadow decals and per-shot lighting enabled.
4. **Hard-raster fine-tuning** (mtldiffrast) on exported geometry, plus per-video test-time optimisation (Idea 5).

---

## 5. Critique of the current design sketch

| Sketch element | Concern | Suggested change |
|---|---|---|
| Emergent slots for entities | Real-video OCL is still at mBO 0.3–0.6 (1.2) | SAM 2-seeded, SAVi-style **conditional** slots. Slot attention binds, it does not discover |
| One slot per frame (camera plus location) | Re-render loss alone conditions cameras poorly; cuts make this a discrete assignment problem | Per-**shot** latent (location ID, intrinsics, light) plus a per-frame pose head; cameras additionally constrained by tracks and depth (another report covers geometry models) |
| One slot per location (set geometry) | Fine. Needs a background/far-field term and a ground model | Location = ground (plane or heightfield) + ≤32 superquadric/cuboid blocks + sky/far sphere texture |
| One slot per entity (identity, shape latent, pose) | Needs an "is hidden" state, an "is off-shot" state, and a shadow/effects term | Add TSA-style activation, DSSA-style identity/appearance split, effects decal |
| Decode to low-poly meshes | A variable-topology mesh decoder is not differentiable and is costly | **Fixed-topology tessellated primitives and templates** inside the loop; optional remesh at export |
| Body rig | SMPL is gated | Anny or MHR (Apache-2.0); primitive trees for everything else |
| "Single network" | Amortisation gap on out-of-distribution video | Network plus a short test-time optimisation with the same loss, distilled back (Idea 5) |

---

## 6. Recommended representation ("SlotPrim")

**Per video**
- Shot list with cut times.
- Per shot: location ID, intrinsics, per-frame camera pose, lighting (ambient plus one directional light).
- A bank of L location slots and E entity slots shared by all shots.

**Location slot ℓ**
- Ground: a plane plus a coarse heightfield (16×16).
- Up to 32 textured superquadrics (ε₁, ε₂, scale, SE(3); cuboids at small ε).
- Far-field sky sphere with a low-res equirectangular texture.
- Static within a location.

**Entity slot e**
- Identity embedding: persistent and used for re-identification.
- Class gate over three bodies:
  - **human:** Anny or MHR decimated to about 1–2k triangles, with transferred skinning;
  - **articulated:** a tree of up to 12 superquadric parts, joints at parent–child contact;
  - **rigid:** up to 6 parts, one root.
- Shape latent → primitive parameters plus small per-vertex residuals on a fixed tessellation, under a Laplacian penalty.
- Per-primitive texture: 16²–32² in the superquadric (η, ω) UV parametrisation, as in DBW, plus a 32-dimensional feature code (for the DINO loss; dropped at export).
- Per frame: root SE(3), joint rotations (6D representation), activation a∈[0,1] (visible / hidden / off-shot), shadow-decal opacity.

**Export**
- Tessellated primitives become a low-poly glTF: about 100–200 triangles per primitive, so roughly 2–6k triangles per location and 0.3–2k per entity.
- Skins and animations for entities, animated cameras.
- A JSON edit decision list (shots → camera) plus a Blender/USD importer for cuts.
- Optional: an Apache-licensed remesher (DeepMesh) or auto-rigger (UniRig/Puppeteer) as post-processing.

**Why this representation.**
- Every output is a **fixed-size regression target**, so it batches, is warm-startable on synthetic data with exact ground truth, and has closed-form vertex Jacobians.
- It renders through one triangle path, or through a Gaussian skin of the same triangles.
- It is low-poly *by construction*, and editable: moving a block is changing six numbers.

**16 GB budget (estimate)**

| Item | Estimate |
|---|---|
| Trainable network, about 60M params (bf16 weights, fp32 Adam) | ≈1.0 GB |
| Frozen teachers (DINOv2-S/B, SAM 2-S, depth) run **offline**; features cached to disk (e.g. 32×32×384 fp16 is about 0.8 MB/frame, so about 0.4 GB per 1-min video at 8 fps) | 0 GB at train time |
| Activations: 32 frames × 1,024 tokens, factorised attention, checkpointed | ≈1.5–3 GB |
| Renderer: dense at 48² for 8 frames × 512 visible primitives (measured scale) | ≈2 GB, ≈150 ms |
| **Compute** | 6·N·D ≈ 11.5 TFLOP per step for 32k tokens. At about 40% of the measured 15 TFLOP/s, that is about 2 s/step, so about 1 day per 50k steps |

---

## 7. Ideas (original proposals)

**Idea 1: Token-resolution "feature rasterisation" as the main training loss.**
- Render per-primitive feature codes, soft slot IDs and depth at the *DINO patch grid* (32²–37²), and compare against cached DINOv2 tokens, SAM 2 masks and ordinal depth.
- Each target pixel is a 14×14 patch with semantic content, so the basin is wide.
- At this resolution the naive dense renderer costs about 15 ms for 64²×256 on the M5. You can train for weeks before writing a Metal kernel.
- RGB at full resolution becomes a late, low-weight refinement.
- *Test:* on synthetic scenes, measure the pose convergence basin (percentage converged versus initial rotation error) for RGB-only versus feature-only versus feature+mask losses.

**Idea 2: Shot-aware entity bank with re-identification by re-rendering ("ShotBank").**
- Each shot's encoder proposes candidate slots. They are matched to the video-level bank by Sinkhorn over identity embeddings, with a "new entity" dustbin (SuperGlue-style) and the same for locations.
- The novelty is that **assignment is verified by synthesis**. A candidate match is accepted if the bank entity's canonical shape and texture, re-posed, explains the new shot's masks and features at lower loss than spawning a new entity.
- The re-render loss then trains the re-ID head. Wrong merges raise reconstruction error across shots; wrong splits pay a parsimony cost.
- The synthetic generator produces edited videos with random cuts, close-ups, reverse angles and location revisits, so the matcher gets exact supervision during warm-up.

**Idea 3: Rigs that grow from motion ("split-on-residual primitive trees").**
- Every non-human entity starts as one rigid superquadric.
- When point tracks inside one primitive disagree with a single SE(3) (residual above a threshold over a window), that primitive is split into two children joined at their contact point. This is the articulated analogue of 3DGS densification, and of D-Prism's adaptive count control, but driven by *motion* rather than appearance.
- The result is a skeleton and a low-poly part mesh for animals, robots, Minecraft mobs or doors without any template.
- Humans skip this path through the class gate (Anny/MHR).

**Idea 4: Entity-owned effects to stop phantom objects.**
- Each entity carries a cheap **shadow decal**: its primitives are projected onto the location's ground along the shot's light direction, softly rasterised from the light's view, blurred, and applied multiplicatively.
- Each entity also carries a small residual "effects" layer.
- Spawning a new slot costs an MDL-style penalty, and needs SAM-mask support across at least k frames plus 3D consistency.
- Shadows then lower the loss *more cheaply* through the owner's decal than through a new object. This turns the Omnimatte insight (associate effects with their object) into an inductive bias of the renderer.

**Idea 5: Amortise, then test-time optimise, then distil.**
- The feed-forward network gives a full scene in one pass.
- Then 200–500 iterations of test-time optimisation per video on the M5 refine it, using the same differentiable renderer and loss.
- The refined scenes become pseudo-labels for the next round of network training (expert iteration). This is a way to keep improving on real video without any labelled data, and the M5 can run it overnight.

**Idea 6: Train soft, ship hard, with an "export gap" loss.**
- Every N steps, export the current low-poly scene, render it with a hard rasteriser (mtldiffrast; or Metal hardware raster with Meshtryoshka-style opacity gradients), and penalise the difference between soft and hard renders.
- Combine with sigma and opacity annealing so the final soft renderer is nearly binary.
- This guarantees the glTF looks like the training renders.

**Idea 7: Native voxel locations for blocky domains.**
- A location slot may alternatively decode a coarse sparse voxel grid (≤64³ with soft occupancy, a tiny TRELLIS-like structured latent), greedy-meshed to low-poly at export.
- This makes Minecraft gameplay a first-class case instead of a pathological one for superquadrics.
- The gate between "blocks" and "voxels" is learned per location.

---

## 8. Licence summary (as used here)

| Component | Licence | Constraint check |
|---|---|---|
| MLX | [MIT](https://github.com/ml-explore/mlx) | ✓ |
| PyTorch3D | BSD-3 | ✓ licence; ✗ no MPS raster |
| mtldiffrast / metal-gauss / metalsplat / mlx3D | MIT | ✓ |
| gsplat-mlx | README Apache-2.0, no LICENSE file | ⚠ |
| Mitsuba 3 | BSD-style | ✓ |
| SlangPy | Apache-2.0 WITH LLVM-exception | ✓ |
| OpenSplat | AGPL-3.0 | ⚠ copyleft |
| nvdiffrast | NVIDIA Source Code License (1-Way Commercial) | ⚠, and CUDA-only |
| SAM 2 | Apache-2.0, not gated | ✓ |
| SAM 3 / SAM 3D Objects / SAM 3D Body | SAM License, gated | ✗ |
| DINOv2 | Apache-2.0 | ✓ |
| DINOv3 | Custom, gated | ✗ |
| Anny / MHR | Apache-2.0 | ✓ |
| SMPL family | Registration required | ✗ |
| DBW / SuperDec / Neural Parts / D-Prism | MIT | ✓ |
| PrimitiveAnything | GPL-3.0 | ⚠ |
| MeshAnything V1/V2 | S-Lab 1.0 (non-commercial) | ✗ commercial |
| DeepMesh | Apache-2.0 | ✓ |
| TRELLIS / TRELLIS.2 | MIT, but gated and non-commercial dependencies; NVIDIA-only | ⚠ |
| TransNetV2 | MIT | ✓ |
| PySceneDetect | BSD-3 | ✓ |
| TAPNet | Apache-2.0 | ✓ |
| Depth Anything V2 Large | CC-BY-NC | ⚠ |
| UniRig | MIT | ✓ |
| Puppeteer / MagicArticulate | Apache-2.0 | ✓ (check checkpoints) |

## 9. What to prototype first

1. Build a superquadric tessellator and the dense MLX soft rasteriser (RGB, feature, ID, depth). Already shown to fit at ≤64².
2. Run the synthetic basin-of-convergence study for Idea 1.
3. Integrate SAM 2 and TransNetV2 preprocessing, then a ShotBank matcher on synthetic cut videos.
4. Write the tiled MLX Metal soft-raster kernel with a custom VJP. Target: ≥10× the dense numbers in Section 3.3.
5. Test the mtldiffrast hard-raster export check. Benchmark it, since no numbers exist yet.
