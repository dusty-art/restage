# 05 — Unconventional ideas: approaches outside mainstream 3D reconstruction

*Contrarian-inventor research note. Date: 2026-10-28. Target: video (<1 min, may be edited) → animated low-poly 3D scene with persistent entities, a camera per shot, and the edit. Constraints: Apple M5, 16 GB, $0 cloud, training and inference on the laptop.*

---

## 0. How to read this

- **Part A** surveys the eight areas I was asked about: what exists, how mature it is, its licence and compute, and whether it fits in 16 GB.
- **Part B** reports a small experiment I ran on this laptop. It uses real codecs, a synthetic edited video and no downloads, and tests the codec ideas with numbers.
- **Part C** gives seven original ideas, ranked. Each has its key insight, why it should work, a cheap laptop test, the main risk and a novelty check.
- **Part D** challenges the current design sketch.
- **Part E** says what I would build first.
- **Part F** lists caveats and unverified claims.

**Sourcing rules.** Every claim taken from the web has its URL next to it. Claims from my own background knowledge, not checked in this session, are marked **[unverified]**.

**Search budget.** The web-search budget ran out (a shared 200-search cap) partway through the novelty checks for Ideas 5 to 7, so those novelty claims are weaker. They are flagged.

---

## TL;DR

1. **The edit is the rig ("Cut-Graph Stereo").** A monocular video gets wide-baseline views of a set only through its cuts. Continuity editing (180° rule, shot/reverse-shot, eyeline match, match-on-action) tells you how those views relate. Register shots through the people, not the pixels, using a toric-space camera model (Lino & Christie 2015) plus grammar priors.
   - I found no work that inverts film grammar to register cameras or sets across cuts.
   - HumanMM aligns human motion across shots, but does not register the set ([arXiv 2503.07597](https://arxiv.org/abs/2503.07597)).
2. **The codec is the judge.** Give a real encoder the rendered frame as its reference picture and count the bits it needs for the true frame. This is model-aided coding (Eisert et al., ~2000) turned into an MDL objective for the decisions gradients cannot make: how many entities, which shots share a location, how many polygons.
   - My toy test: the score falls monotonically as the camera error shrinks from 8° to 0°.
   - A render of the wrong location saves nothing (712 KB vs 698 KB for a blank reference).
   - The correct scene beats ordinary temporal coding (161 KB vs 257 KB).
3. **Probe re-encoding.** Re-encode the video yourself with a probe configuration, then read the motion vectors. On the M5 this runs at thousands of fps and gives:
   - cuts, exactly, even when the GOP is fixed;
   - per-shot camera rotation (0.3–0.5° drift over a 25° pan when frames are anchored);
   - self-calibrated focal length;
   - a supervision signal for the network's rendered motion field.
   The MVs inside the uploaded bitstream are unusable for this (27° drift), because FFmpeg does not export the reference index.
4. **Closed-vocabulary decompilation, voxel worlds first.** Minecraft-like video → block world plus camera path. I found no prior work in that direction. Unlimited, exactly-labelled training data can be generated on the laptop (Luanti is LGPL).
5. **Free metrology.** Free-fall acceleration is exactly *g*, and the frame rate is known, so depth Z = g·f/a_px. In ideal tracking the noise floor is about 1% for a 0.33–0.5 s flight. On-screen telemetry (Minecraft F3 screen, dashcam speed) gives exact scale or pose.
6. **Repetition is multi-view.** N identical instances in one frame are N views of one prototype. *(Novelty not verified.)*
7. **The soundtrack is a sensor.** Room-tone continuity across a cut suggests the same location; reverb suggests room size; an off-screen voice implies an off-screen entity. *(Novelty not verified.)*

**Build first:** a laptop-generated "director + editor" synthetic world (CineSynth), with the codec-probe front end and the cross-cut registration solver (Idea 1). Measure against SfM on reverse-shot pairs before training any big network. Details are in Part E.

---

## Part A — Survey of the eight areas

### A1. Generative world models and video diffusion as 3D priors

| System | What it is | Maturity | Licence / access | Compute | Fits 16 GB? |
|---|---|---|---|---|---|
| **Genie 3** (DeepMind, Aug 2025) | Real-time interactive world model, 720p at 24 fps. Consistency is emergent and frame-by-frame, with **no explicit 3D**. Visual memory about 1 min; sessions about a few minutes ([DeepMind blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)). A GDC 2026 report says consistency is about 1 minute ([TweakTown](https://www.tweaktown.com/news/110471/genie-3-ai-generated-worlds-fall-apart-after-few-minutes-google-admits/index.html)). | Research preview | Closed; limited testers ([DeepMind blog](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)). The site navigation now mentions a "Genie 4" [unverified] | Datacentre | **No** |
| **HunyuanWorld-Mirror / WorldMirror** (Tencent, Oct 2025; ICML 2026) | Feed-forward. Images or video in; cameras, depth, normals, point maps and 3DGS out, in one pass. "Any-prior prompting" injects known intrinsics, pose or depth ([arXiv 2510.10726](https://arxiv.org/abs/2510.10726), [ICML](https://icml.cc/virtual/2026/poster/65052)). WorldMirror 2.0 (~1.2B) accepts casual videos at 50K–500K px ([HF HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0), [arXiv 2604.14268](https://arxiv.org/abs/2604.14268)). Dynamic scenes not mentioned. | Released weights and code | Tencent community licence ([HF](https://huggingface.co/tencent/HY-World-2.0), [README](https://huggingface.co/tencent/HunyuanWorld-Mirror/blob/main/README.md)). Check territorial restrictions [unverified] | CUDA 12.8 recommended; no VRAM figures; no MPS mention ([HF](https://huggingface.co/tencent/HY-World-2.0)) | **Probably, for inference** (1.2B is about 2.4 GB in bf16). MPS port unproven |
| **World Labs Marble** (Nov 2025; 1.1 in Apr 2026) | Text, images, **video**, panoramas or coarse 3D layout in. Exports Gaussian splats (~2M or 500k) and meshes ([Radiance Fields](https://radiancefields.com/world-labs-formally-launches-marble-a-generative-world-model), [SiliconANGLE](https://siliconangle.com/2025/11/12/world-labs-launches-marble-commercial-world-model-generating-entire-virtual-environments/), [Marble 1.1](https://radiancefields.com/world-labs-releases-marble-1.1-and-marble-1.1-plus)) | Commercial product | Freemium; commercial rights on paid tiers; open weights not stated ([Radiance Fields](https://radiancefields.com/world-labs-formally-launches-marble-a-generative-world-model)) | Cloud | **No** (breaks the $0 / local constraint) |
| **NVIDIA Lyra** (Sep 2025; ICLR 2026) | **Self-distillation.** A 3DGS decoder sits on a frozen camera-controlled video diffusion model (GEN3C). It is trained only on the model's own generated videos; no real multi-view data. Supports 4D from monocular video ([project](https://research.nvidia.com/labs/toronto-ai/lyra), [arXiv](https://arxiv.org/html/2509.19296v1), [ICLR](https://iclr.cc/virtual/2026/poster/10006990)) | Code released | Code Apache-2.0 ([GitHub](https://github.com/nv-tlabs/lyra)); weights under NVIDIA model licence ([HF card](https://huggingface.co/nvidia/Lyra/blob/refs%2Fpr%2F2/README.md)) | Not stated for v1 | **No** |
| **Lyra 2.0** (Apr 2026) | Image + camera path → long walkthrough video → 3DGS and mesh. Built on Wan2.1-14B; 14B params ([HF](https://huggingface.co/nvidia/Lyra-2.0), [arXiv 2604.13036](https://arxiv.org/pdf/2604.13036)) | Released | Weights: *NVIDIA Internal Scientific Research and Development Model License*, with **no production use** ([HF](https://huggingface.co/nvidia/Lyra-2.0)) | H100/GB200, Linux. Trained on 32 H100 nodes ([HF](https://huggingface.co/nvidia/Lyra-2.0)) | **No** |
| **CAT4D** (CVPR 2025) | Multi-view video diffusion turns a monocular video into multi-view videos, which are fitted with deformable 3DGS. Trained on a mix of static multi-view, fixed-view dynamic and synthetic 4D data ([arXiv 2411.18613](https://arxiv.org/abs/2411.18613), [project](https://cat-4d.github.io)) | Paper | I found no released weights [unverified] | Large | **No** |
| **SV4D 2.0** (Stability, 2025) | **Object-centric.** A 12-frame 576² video becomes 4 views × 12 frames (or 8 views × 5) ([HF](https://huggingface.co/stabilityai/sv4d2.0), [arXiv 2503.16396](https://arxiv.org/abs/2503.16396v3)) | Released | Stability Community Licence: free under US$1M annual revenue ([HF](https://huggingface.co/stabilityai/sv4d2.0), [blog](https://stability.ai/news/stable-video-4d-20-new-upgrades-for-high-fidelity-novel-views-and-4d-generation-from-a-single-video)) | VRAM not stated | **Maybe, slowly**, as an offline per-entity teacher [unverified] |
| **Wonderland** (Snap, CVPR 2025) | Feed-forward 3DGS reconstruction **directly from video-diffusion latents**, said to be about 256× compressed in space-time ([arXiv 2412.12091](https://arxiv.org/abs/2412.12091v1), [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/34401), [secondary](https://bohrium.dp.tech/paper/arxiv/2412.12091)) | Paper | — | Large backbone | **No** (the backbone) |

**"Video diffusion knows 3D" probing.**
- *How Much 3D Do Video Foundation Models Encode?* (Dec 2025) probed frozen WAN2.1-14B, Open-Sora 2.0, CogVideoX, Aether and V-JEPA features with shallow read-outs ([arXiv 2512.19949](https://arxiv.org/html/2512.19949)):
  - WAN2.1-14B came close to Fast3R on CO3Dv2 (point error 0.284 vs 0.262) and **beat it on DL3DV** (1.051 vs 1.379).
  - DINOv2 was far weaker (CO3Dv2 point error 0.559).
  - The best features are **mid-layer, at early-but-not-first diffusion timesteps**.
  - Replacing DINO with WAN features in VGGT ("VidFM-VGGT") beat the original VGGT, and surpassed full-data VGGT with **<10% of the 3D training data**.
- *Geometry Forcing* (ICLR 2026) argues raw-video-trained models have weak geometry under its probe, and aligns them to VGGT features ([arXiv 2507.07982](https://arxiv.org/abs/2507.07982v1)).
- A June 2026 probe reports that diffusion models encode physical plausibility (~81% on IntPhys/InfLevel) ([arXiv 2606.05328](https://arxiv.org/pdf/2606.05328)).

**Running a video model on 16 GB.**
- Wan2.2-TI2V-5B in MLX q8 was validated on a **64 GB** Mac: 15 min for 41 frames at 720p, "32 GB+ recommended" ([HF](https://huggingface.co/Anes1032/Wan2.2-TI2V-5B-mlx-q8)).
- A Q4 bundle was tested on a 24 GB M2, with a ≥12 GB memory gate ([HF](https://huggingface.co/jc-builds/Wan2.2-TI2V-5B-iOS)).
- So generating video on this laptop is marginal at best.

**What this implies for a small model.**
1. **Do not put a generative video model in the inference path.** The useful ones are 5–14B, CUDA-first, and often non-commercial (Lyra 2.0).
2. **Copy Lyra's trick, not its teacher.** Lyra's real lesson is that a 3D decoder can be trained purely on a generator's own outputs. Our cheapest 3D-native generator is a procedural *director + editor* we write ourselves. It is exact, unlimited, free, and already knows the cut structure (Part E).
3. **The 3D knowledge sits in mid-layer features.** A small video DiT (e.g. Wan2.1-1.3B, which appears in the probe's scaling ablation; its numbers I did not read) used as a *frozen feature extractor* is the one diffusion-derived component that might fit. Test it against DINOv2-S before committing.
4. **Prior prompting is a pattern worth copying.** WorldMirror takes optional camera, intrinsics and depth priors as prompts. Our cheap sensors (Part B) produce exactly those priors: rotation and focal per shot, and cut positions.
5. **Generative priors hallucinate unseen geometry, and in an edited video the "unseen" side is often shown by another shot.** Exploiting the edit (Idea 1) beats hallucinating.

### A2. Video codecs as free computation (compressed-domain vision)

**What exists.**
- **CoViAR (2018)** showed that motion vectors and residuals carry action cues cheaply, as cited in [HY-Himmel](https://arxiv.org/html/2605.08158).
- 2025–26 video-LLMs now tokenise codec primitives:
  - **EMA**: a GOP encoder that fuses I-frame and motion ([arXiv 2503.13016](https://arxiv.org/pdf/2503.13016)).
  - **CoPE-VideoLM**: motion vectors and residuals through a light delta-encoder; up to 93% fewer tokens and 86% lower time-to-first-token ([arXiv 2602.13191](https://arxiv.org/abs/2602.13191)).
  - **HY-Himmel**: reads MVs directly from the native H.264 bitstream ([arXiv 2605.08158](https://arxiv.org/html/2605.08158)).
- **Codec MVs as a flow prior:**
  - **MVFlow** (MM 2023) ([review](https://liner.com/review/mvflow-deep-optical-flow-estimation-compressed-videos-with-motion-vector)).
  - **AV1/HEVC MV fidelity** (Oct 2025) ([arXiv 2510.17427](https://arxiv.org/html/2510.17427v1)):
    - mean end-point error 1.58–2.02 px; median 0.36–0.43 px;
    - coverage of moving pixels ~73–81%, but poor for motion of 40 px or more;
    - AV1 MVs as a RAFT warm start: 5 iterations reach 1.68 px EPE, against 2.07 px for vanilla RAFT at 20 iterations (about a 4× speed-up).
- **AV1 global motion.** libaom estimates per-reference global motion models (translation up to a full 8-parameter homography), feature-based or by DIS flow, and signals them in the bitstream when worth it ([libaom header](https://aomedia.googlesource.com/aom/+/refs/heads/research/av1/encoder/global_motion.h)). That is a free camera-motion estimate.
- **Cut detection** in the compressed domain is old and mature. It uses I-frame DC images and macroblock-type statistics (intra MBs in P-frames, P_Skip counts) and runs about 11× real time in one report ([core.ac.uk](https://core.ac.uk/download/55694135.pdf), [TRECVID 2003](https://www-nlpir.nist.gov/projects/tvpubs/tvpapers03/kddi.final2.paper.pdf), [Strathclyde](https://strathprints.strath.ac.uk/29270/1/final5.pdf)).

**Tools.**
- FFmpeg exports MVs as frame side-data with `-flags2 +export_mvs` ([patch](https://ffmpeg.org/pipermail/ffmpeg-devel/2014-August/161322.html), [extract_mvs example](https://ffmpeg.org/doxygen/4.4/extract__mvs_8c_source.html)). H.264 works. HEVC returned no side-data in a 2015 report ([libav-user](https://ffmpeg.org/pipermail/libav-user/2015-June/008242.html)).
- The `AVMotionVector` struct carries only a ±1 "direction", not the reference index. The header itself has an "XXX" note about this (read locally in `/opt/homebrew/include/libavutil/motion_vector.h`). **This matters a lot (Part B).**
- `mv-extractor` is x86-64 Linux only (Docker elsewhere) and patches FFmpeg internals ([PyPI](https://pypi.org/project/motion-vector-extractor/)). On the Mac I wrote a 70-line C tool against Homebrew FFmpeg 8.0.1 instead.

**Licence and compute.** FFmpeg is LGPL/GPL; x264 and x265 are GPL [unverified for this build; the configure line shows `--enable-gpl`]. Fine for research; matters if you ship a binary. Compute is trivial (Part B: thousands of fps). **Fits 16 GB: yes, trivially.**

### A3. Inverse procedural modelling and program synthesis of scenes

| Work | Gist | Maturity / licence | Fit |
|---|---|---|---|
| **IG-LLM** (TMLR 2024) | An LLM decodes a frozen visual embedding into a compositional scene program, with a continuous numeric head for precise values ([arXiv 2404.15228](https://arxiv.org/abs/2404.15228), [project](https://ig-llm.is.tue.mpg.de/)) | Code and data released | Small LLM fine-tune; feasible on 16 GB with LoRA [unverified] |
| **BlenderGym** (CVPR 2025) | Benchmark of VLM systems editing Blender scenes in code. Even state-of-the-art VLMs underperform humans. Inference compute helps most when split between generation and verification ([arXiv 2504.01786](https://arxiv.org/abs/2504.01786v1), [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_BlenderGym_Benchmarking_Foundational_Model_Systems_for_Graphics_Editing_CVPR_2025_paper.html)) | Benchmark | — |
| **VIGA** (Jan 2026) | Training-free write → run → render → compare → revise agent in Blender Python. +35% on BlenderGym and +125% on BlenderBench over one-shot baselines ([arXiv 2601.11109](https://arxiv.org/abs/2601.11109)). Code reportedly MIT ([CGWorld](https://cgworld.jp/flashnews/01-202602-VIGA.html), [arXiv v1](https://arxiv.org/abs/2601.11109v1)) | Code | Needs a strong VLM; local 16 GB VLMs are weak at this [unverified] |
| **SEIG** (Jun 2026) | "Thinking in Blender". Staged code refinement of geometry, then materials, composition and lighting, from **one image**, with no SAM or 3D foundation models. Beats VIGA on 5 of 6 metrics ([arXiv 2606.02580](https://arxiv.org/abs/2606.02580), [HTML](https://arxiv.org/html/2606.02580)) | Paper | Same VLM dependence |
| **SceneScript** (Meta, ECCV 2024) | Video → **structured language commands** (`make_wall`, `make_door`, boxes) via an encoder-decoder; 100k-scene synthetic dataset ([arXiv 2403.13064](https://arxiv.org/abs/2403.13064), [Meta](https://ai.meta.com/research/publications/scenescript-reconstructing-scenes-with-an-autoregressive-structured-language-model/)) | Research | The *architecture pattern* fits a laptop |
| **SpatialLM** (2025) | Point cloud → walls, doors, windows and oriented boxes via Llama-1B / Qwen-0.5B ([arXiv 2506.07491](https://arxiv.org/abs/2506.07491v1), [GitHub](https://github.com/manycore-research/SpatialLM)) | Released | Encoder weights CC-BY-NC ([HF](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)). Size fits |
| **DI-PCG** (Dec 2024) | Small diffusion transformer predicts procedural-generator parameters from images ([arXiv 2412.15200](https://web3.arxiv.org/abs/2412.15200)) | Paper | Small; fits |
| **NNProc** (EG 2025) | Learns forward and inverse maps of any procedural model; combined with optimisation ([EG](https://diglib.eg.org/handle/10.1111/cgf70024)) | Paper | Fits |
| Infinigen in reverse | **I found no inverse-Infinigen work.** The closest are DI-PCG, NNProc and inverse procedural materials ([arXiv 1912.01067](https://arxiv.org/pdf/1912.01067)) | Gap | — |

**Assessment.** LLM/VLM agent loops (VIGA, SEIG) are impressive but wrong for this project. They need a frontier VLM (cloud $ or too big for 16 GB), take many seconds per Blender render-compare iteration, and work per image. The transferable idea is the **output representation**: a small scene DSL decoded by a small trained network (SceneScript/IG-LLM style), whose continuous parameters are then polished by gradient. Programs also give the description-length accounting that Idea 2 needs.

### A4. Analysis-by-synthesis with game engines; known-vocabulary codebooks

- **Minecraft.** I found **no** paper or tool that reconstructs a block world from gameplay video.
  - World2Minecraft (ICLR 2026) goes the other way: real indoor scenes → voxel builds via `/setblock` ([papernotes](https://en.papernotes.org/ICLR2026/3d_vision/world2minecraft_occupancy-driven_simulated_scenes_construction/), [HF mirror](https://huggingface-proxy.pages.dev/papers/2604.27578.md)).
  - GANcraft renders block worlds photorealistically ([arXiv 2104.07659](https://ar5iv.labs.arxiv.org/html/2104.07659)).
  - "Minecraft to 3D" converts voxels to meshes ([site](https://minecraftto3d.pages.dev)).
  - A forum thread discusses reading grid-relative pose from screenshots informally ([Cemetech](https://www.cemetech.net/forum/viewtopic.php?p=262774)).
- **Free engine for exact synthetic data.** Luanti (formerly Minetest) is an LGPL-2.1 voxel engine with Lua modding and macOS builds ([GitHub](https://github.com/luanti-org/luanti)). A self-written voxel ray-caster in NumPy or MLX is also only a few hundred lines.
- **Rooms.** SceneScript and SpatialLM (A3) give the "room codebook": walls, doors, windows and boxes.
- **Roads.** Not researched this session (search budget). Standard lane widths and marking geometry are well known as metric priors [unverified].
- **Fit on 16 GB.** Yes. Closed-vocabulary fitting is a small discrete search plus a small classifier.

### A5. Physics and intuitive-physics priors

- **Gravity frames.** GVHMR estimates human motion in a gravity-view frame and limits error build-up along gravity ([arXiv 2409.06662](https://arxiv.org/abs/2409.06662v1), [project](https://zju3dv.github.io/gvhmr)). It needs SMPL-X, whose licence requires registration and bars redistribution ([PyPI](https://pypi.org/project/gvhmr/)).
- **Contact and support for humans in scenes:**
  - JOSH optimises scene, cameras and human motion with contact constraints (as described in [UniCon3R](https://arxiv.org/abs/2604.19923)).
  - SHARE (Oct 2025) gets metric scale for static-camera video via MoGe-2 ([arXiv 2510.15342](https://www.arxiv.org/pdf/2510.15342)).
  - PhySIC (SIGGRAPH Asia 2025) handles single-image contact and interpenetration ([arXiv 2510.11649](https://arxiv.org/html/2510.11649v1)).
  - UniSH (Jan 2026) is feed-forward metric human + scene ([alphaXiv](https://alphaxiv.org/abs/2601.01222), [project](https://murphylmf.github.io/UniSH)).
  - UniCon3R (Apr 2026) is feed-forward and contact-aware ([arXiv 2604.19923](https://arxiv.org/abs/2604.19923)).
- **Static support.** REST3D (May 2026) builds a gravity-support scene tree from a single image, then runs physics-constrained optimisation. It reportedly cuts floating and penetration failures to 0%, against up to 48% for prior methods (secondary figures) ([arXiv 2605.30338](https://arxiv.org/abs/2605.30338), [EmergentMind](https://www.emergentmind.com/papers/2605.30338)). The classic is Zheng et al., CVPR 2013 ([CVF](https://openaccess.thecvf.com/content_cvpr_2013/papers/Zheng_Beyond_Point_Clouds_2013_CVPR_paper.pdf)).
- **Maturity.** Contact and support are mature for humans; scene-wide physics is emerging.
- **Fit.** The priors are cheap loss terms and fit easily. The cited human models are under 1B params, but SMPL licensing is restrictive.
- **Not found in my searches:** using *ballistic flight* as a metric-scale anchor for a whole scene (Idea 5). The search budget ran out before a dedicated check.

### A6. Cinematography priors across cuts

- **The grammar is well codified:**
  - 180° rule / axis of action ([O'Reilly, Grammar of the Edit](https://www.oreilly.com/library/view/grammar-of-the/9780240521213/Chapter04e.html), [wolfcrow](https://wolfcrow.com/how-to-use-180-degree-rule-with-one-character-two-characters-and-three-and-more-characters/));
  - eyeline match ([Wikipedia](https://en.wikipedia.org/wiki/Eyeline_match));
  - a cognitive theory of continuity editing covering match-action, shot/reverse-shot, the 180° rule and POV ([core.ac.uk](https://core.ac.uk/download/49557361.pdf)).
- **Forward computational cinematography exists:**
  - **Toric space** (Lino & Christie, SIGGRAPH 2015): every camera that frames two subjects at given screen positions lies on a **2D manifold**, so 7 DOF becomes 4 ([HAL](https://hal.archives-ouvertes.fr/hal-01142876), [site](https://sites.google.com/site/christophelino/research/toric-space)). The ToricCam library is GPLv3 ([SourceForge](https://sourceforge.net/projects/toric-cam/)).
  - *Continuity Editing for 3D Animation* (Galvane et al., AAAI 2015) generates edits from 3D ([core.ac.uk](https://core.ac.uk/download/49557361.pdf)).
  - There is a camera-trajectory generation survey ([arXiv 2506.00974](https://arxiv.org/pdf/2506.00974)) and Cine-AI ([arXiv 2208.05701](https://arxiv.org/pdf/2208.05701)).
- **Understanding camera motion.** CameraBench (NeurIPS 2025) built a taxonomy with cinematographers over ~3,000 videos. It found that SfM fails semantic primitives such as "follow" and VLMs fail geometric ones ([arXiv 2504.15376](https://arxiv.org/pdf/2504.15376), [NeurIPS](https://neurips.cc/virtual/2025/poster/121600)).
- **Multi-shot reconstruction:**
  - **HumanMM** (CVPR 2025): shot-transition detection plus alignment of *human* pose and orientation across shots ([arXiv 2503.07597](https://arxiv.org/abs/2503.07597)).
  - **Sitcoms3D** (ECCV 2022): exploits *repetition of sets across episodes* to build NeRF sets and place humans ([arXiv 2207.14279](https://arxiv.org/abs/2207.14279), [code](https://github.com/ethanweber/sitcoms3D)).
  - Mixed-camera feed-forward reconstruction (MEOW, 2026) assumes one scene with no cut structure ([arXiv 2609.35658](https://arxiv.org/html/2609.35658v1)).
- **Answer to "can film grammar constrain 3D across cuts?"** In my searches, **no one has inverted film grammar to register cameras or sets across cuts.** Both the forward models (toric space) and the human-only alignment (HumanMM) exist. That gap is Idea 1.

### A7. Test-time training and optimisation

| Work | Mechanism | Cost | Fit |
|---|---|---|---|
| **TTT3R** (ICLR 2026) | CUT3R's recurrent state treated as fast weights with confidence-based learning rates. **No weight updates.** 2× length generalisation ([arXiv 2509.26645](https://arxiv.org/abs/2509.26645)) | ~20 FPS at ~6 GB VRAM ([papernotes](https://en.papernotes.org/ICLR2026/3d_vision/ttt3r_3d_reconstruction_as_test-time_training/)) | **Yes** |
| **Test3R** (2025) | Visual prompt tuning of DUSt3R at test time, using triplet self-consistency ([arXiv 2506.13750](https://arxiv.org/html/2506.13750v1)) | ~30 s per scene (secondary) | **Yes** (prompts only) |
| **G3R** (ECCV 2024) | **Learned optimiser.** A network takes the 3D representation plus its rendering gradient and outputs an update. At least 10× faster than per-scene optimisation; about 2 min per driving scene ([arXiv 2409.19405](https://arxiv.org/pdf/2409.19405), [Waabi](https://waabi.ai/g3r/)) | Few iterations | Pattern fits |
| **VGGT** | Feed-forward. The commercial checkpoint has a custom AUP licence; the original is CC-BY-NC ([HF](https://huggingface.co/facebook/VGGT-1B-Commercial), [mirror note](https://github.com/HardingPan/VGGT)). A community MPS port claims 8 GB+ RAM ([vggt-mps](https://gh.r-mj.com/jmanhype/vggt-mps)) | — | Possibly |

**Takeaway.** The field has converged on *amortised + a few refinement steps* (G3R; TTT3R as state-level adaptation). On 16 GB, full fine-tuning of a 1B model per video is borderline. A third-party estimate puts full-parameter training at about 1B params on a 16 GB M5 ([ApX](https://apxml.com/hardware/apple-m5-16gb)). Prompt or adapter tuning, or a learned optimiser, are the realistic options.

**Hardware facts.**
- The M5 has a Neural Accelerator in every GPU core: more than 4× M4 peak GPU AI compute ([Apple ML](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), [MacTrast](https://www.mactrast.com/2025/10/apple-unveils-its-m5-chip-featuring-next-generation-gpu-optimized-for-ai-tasks/)).
- MLX needs macOS 26.2+ to use the accelerators ([Apple ML](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)).
- Differentiable rendering on Metal is available:
  - Dr.Jit (BSD-3) JIT-compiles to Metal on macOS ([GitHub](https://github.com/mitsuba-renderer/drjit));
  - OpenSplat runs 3DGS on MPS (AGPLv3, ~2 GB per million Gaussians) ([GitHub](https://github.com/pierotofy/OpenSplat)).

### A8. Neural compression: a video as a 3D scene plus residuals

- **Model-based coding is the historical ancestor.** In the 1990s–2000s, analysis-by-synthesis face models sent only animation parameters. MPEG-4 standardised 68 Facial Animation Parameters ([AAU slides](https://ftp.itec.aau.at/itec/lehre/mm-coding/4.3%20model%20based%20video%20coding%2008_v1.pdf), [MPEG-4 FA](https://www.bib.irb.hr/107165)).
  - **Model-aided coding** (Eisert et al., ~2000) added a *model-rendered reference frame* to a standard motion-compensated coder. It reported ~35% bit savings over H.263, and up to 45% in later work ([record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/71349085), [Fraunhofer](https://publica.fraunhofer.de/handle/publica/205206)).
  - These systems were limited to head-and-shoulders scenes. **Idea 2 generalises this mechanism to whole scenes and uses it as a judge rather than a codec.**
- **2D Gaussian video codecs:**
  - GaussianVideo: deformable 2D Gaussians; 5.5× faster training and 12.5× faster decoding than NeRV ([arXiv 2503.04333](https://arxiv.org/html/2503.04333v1)).
  - GSVC: rate-distortion comparable to AV1/VVC; 1,500 fps decoding at 1080p ([arXiv 2501.12060](https://web3.arxiv.org/abs/2501.12060v1)).
- **Compression of 3D/4D Gaussian scenes:**
  - D-FCGS: feed-forward dynamic GS compression ([arXiv 2507.05859](https://arxiv.org/html/2507.05859v4));
  - GSCodec Studio ([arXiv 2506.01822](https://export.arxiv.org/abs/2506.01822));
  - SAGA ([Radiance Fields](https://radiancefields.com/papers/gaussian-splatting-based-volumetric-video-compression-with-sparse-4d-anchors));
  - Disney Research on making free-view content friendly to standard codecs ([PDF](https://studios.disneyresearch.com/app/uploads/2026/07/SmoothMotionVectors-Optimizing-Your-Content-for-Video-Codecs-in-Free-View-Video-Compression-Paper.pdf)).
  - All of these compress a *given* 3D scene. None infers the 3D scene of a monocular general video by rate-distortion.
- **MDL in vision:**
  - MDLSeg (2025) segments video into scenes by description length, with an exact dynamic-programming solution ([arXiv 2503.01201](https://arxiv.org/pdf/2503.01201));
  - older MDL work on 3D faceted scene models ([SPIE 1998](https://spiedigitallibrary.org/conference-proceedings-of-spie/3371/0000/Minimum-description-length-understanding-of-infrared-scenes/10.1117/12.323855.full)).
- **Answer.** "Scene as codec" in the strong sense is open: a monocular, edited, general video → a 3D scene chosen because it minimises bits(scene) + bits(video | scene). Part B shows a cheap way to compute bits(video | scene) with a stock encoder.

---

## Part B — Laptop experiment: the codec as sensor and judge

**Files.**
- `research/05_experiments/mvdump.c` — the MV extractor (FFmpeg 8.0.1 `export_mvs`).
- `research/05_experiments/codec_probe.py` — sensor and judge tests.
- `research/05_experiments/anchor_probe.py` — the anchor trick.
- Results in `research/05_experiments/out/results.json` and `research/05_experiments/out/anchor_results.json`.
- Run time: about 30 s in total on the M5. Nothing was downloaded.

**Setup.** A synthetic *edited* video:
- 255 frames, 640×360, 30 fps, 4 shots of 90, 60, 60 and 45 frames, cuts at frames 90, 150 and 210.
- Shots 1 and 3 share location A; shots 2 and 4 share location B (a "return to location").
- The camera is a pure-rotation camera (pan and tilt) with focal length 500 px; each location is a textured panorama.
- An independently moving 90×90 "entity" appears in shots 1 and 3.
- Gaussian sensor noise σ = 2.
- Ground truth is known exactly.

**Limitation up front.** Pure rotation, no parallax, no lighting change. Real video will be harder, so treat these as feasibility results, not accuracy claims.

### B1. Cuts

| Encode | I-frames | Cuts found |
|---|---|---|
| x264 default (scene-cut on) | 0, 90, 150, 210 | all 3, exactly |
| x265 default | 0, 90, 150, 210 | all 3, exactly |
| SVT-AV1 preset 8 | 0, 161 | **misses all** |
| x264 fixed GOP (keyint 60, scenecut 0, "platform-like") | 0, 60, 120, 180, 240 | I-frames useless. **Inter-coverage collapse found 92, 152, 212** (all three, 2 frames late from B-frame reordering) |
| **Our probe re-encode** (ultrafast, P-only, 1 ref) | — | **Inter-coverage collapse: exactly 90, 150, 210** |

### B2. Camera rotation and focal from MVs

Method: per-frame RANSAC homography on the MVs, rotation R = K⁻¹HK, and focal length by self-calibration (the f that makes K⁻¹HK most rotation-like).

| Encode | Median per-frame rotation error (GT motion 0.28°/frame) | End-of-shot drift on the 25.2° pan | Focal estimate (true 500) |
|---|---|---|---|
| x264 default (B-frames, 3 refs) | 0.225° (≈80% error) | **27.3° (fails)** | 510 |
| x264 fixed GOP | 0.220° | 27.1° (fails) | 515 |
| **Probe**, ultrafast, chained | **0.040°** | 1.75° | 540 |
| Probe, veryfast/umh, chained | 0.024° | 0.52° | — |
| **Probe + anchor interleaving** (anchor every 10–30 frames, merange 64) | 0.04–0.075° per link | **0.30–0.38°** | — |
| Single anchor for the whole shot, merange 16 / 64 | — | 5.07° / 0.70° | — |

**Why the default bitstream fails.** FFmpeg's `AVMotionVector` says only "past or future". With B-frames and multiple references, the MV often points 2–3 frames back, so a frame-to-frame model breaks.

**Fix 1: the probe re-encode.** Re-encoding ourselves with P-only and one reference costs about 0.1 s for 255 frames.

**Fix 2: the anchor trick.**
- Interleave copies of an anchor frame (A, F1, A, F2, …) and encode with one reference. Every frame's MVs then point **directly at the anchor**: a long-baseline, drift-free motion field from a stock encoder.
- In this test the main win came from choosing a better encoder preset (1.75° → 0.52°). Anchors added a further ~40% (→ 0.30°).

### B3. Moving entities

Homography outliers covered **87–90% of the entity's blocks**. Precision was only 10–29%, because other outliers are noise blocks. MVs therefore give a high-recall "maybe dynamic" mask that needs cleaning (temporal persistence, residual energy).

### B4. Speed (640×360, M5 CPU)

| Step | Speed |
|---|---|
| MV extraction (decode + export, single thread) | **2,600–3,600 fps** |
| Probe encode | ~2,500 fps |
| Dense optical flow, Farneback | 73 fps |
| Dense optical flow, DIS-medium | 117 fps |

MV extraction is about 25–50× faster than CPU optical flow. At 1080p, expect roughly 9× slower (pixel count), which is still far faster than real time [extrapolation].

### B5. "Scene as codec": bits as a hypothesis score

All tests are on shot 1 (90 frames).

**(a) Subtractive residual** (encode video − render; 10-bit luma, same quantisation for every row):

| Hypothesis | Full-res bytes | ¼-res blurred bytes |
|---|---|---|
| No model (raw video) | 73,001 | 10,050 |
| Correct scene and camera | **23,388** | **6,509** |
| Yaw off by 0.1° | 120,034 (worse than no model) | 8,178 |
| Yaw off by 0.25° | 111,882 | 9,621 |
| Yaw off by 0.5° | 112,593 | 11,087 |
| Yaw off by 1° | 112,672 | 12,314 |
| Yaw off by 2° | 113,471 | 12,598 |
| Yaw off by 4° | 116,225 | 12,352 |
| Yaw off by 8° | 118,466 | 12,264 |
| Focal 490 / 510 px | 137,232 / 132,894 | 12,160 / 12,099 |
| Wrong location | 103,970 | 11,664 |

Result: the score is **sharply peaked and non-monotone**. It works as an exact-match verifier but is useless as a search objective.

**(b) Motion-compensated "model-aided" judge.** The render R_t is given to the encoder as the *only* reference for F_t, and only the bytes of the F_t packets are counted:

| Hypothesis | Bytes for the real frames |
|---|---|
| Blank reference (≈ intra coding) | 697,868 |
| Ordinary video coding (reference = previous frame) | 257,486 |
| **Correct scene** | **161,151** (37% below ordinary coding) |
| Yaw off by 0.25° | 296,903 |
| Yaw off by 0.5° | 316,972 |
| Yaw off by 1° | 316,713 |
| Yaw off by 2° | 350,424 |
| Yaw off by 4° | 371,826 |
| Yaw off by 8° | 397,480 |
| Focal 450 / 550 px | 409,998 / 453,813 |
| **Wrong location** | **711,723** (no better than blank) |

Result: **monotone ranking over 0–8°** (one flat step between 0.5° and 1°). A wrong-location hypothesis is rejected cleanly even against an 8°-misaligned right location (712 KB vs 397 KB). The cost is about 0.5 s per hypothesis for a 3-second 360p shot, mostly Python rendering and I/O.

**Conclusions.**
1. A codec probe is a near-free front end for cuts, per-shot rotation, focal length and dynamic-region candidates. Only if you re-encode it yourself: never trust the uploaded bitstream's MVs.
2. A real encoder with the render as reference is a cheap, black-box, monotone, location-discriminating judge of a scene hypothesis. That makes "scene as codec" operational today.

---

## Part C — Original ideas, ranked

Ranking criteria: impact on *this* product (edited video → persistent low-poly 3D), originality (how far from what I found), feasibility on 16 GB, and cheapness of a first test.

| Rank | Idea | Impact | Originality (confidence) | 16 GB | First-test cost |
|---|---|---|---|---|---|
| 1 | Cut-Graph Stereo: the edit is the rig | ★★★★★ | High (searched; nothing found) | ✓ | 3–4 days, CPU minutes |
| 2 | The codec is the judge (model-aided MDL) | ★★★★ | Medium-high (ancestor: 1990s face coding) | ✓ | Done in toy; 2 days for real clips |
| 3 | Probe re-encoding + anchor trick + MV-supervised rendering | ★★★★ | Medium (compressed-domain vision exists; the probe, anchor and supervision uses not found) | ✓ | Done in toy; 2 days to wire into training |
| 4 | Closed-vocabulary decompilation (voxel worlds first) | ★★★ | Medium-high for video → blocks (nothing found) | ✓ | 1 week |
| 5 | Free metrology: ballistics + telemetry overlays | ★★★ | Medium (**not verified**: search budget ran out) | ✓ | 1 day |
| 6 | Repetition is multi-view (instancing) | ★★½ | Uncertain (**not verified**) | ✓ | 2–3 days |
| 7 | The soundtrack as a sensor | ★★ | Medium (**not verified**) | ✓ | 1 day |

### Idea 1 — Cut-Graph Stereo: the edit is the rig

**Key insight.**
- The design sketch treats cuts, close-ups and odd angles as obstacles. In a continuity-edited scene, the cuts are the **only wide-baseline views of the set** a monocular video will ever contain, and film grammar specifies how they relate.
- A shot/reverse-shot pair is a two-camera rig filmed at different moments of the same story time, in the same room.
- Its views often share **zero background pixels**: you see opposite walls. That is why appearance-based SfM and feed-forward multi-view reconstruction (which assumes overlap) fail.
- What the views share is **the people** (an over-the-shoulder shot shows the other actor's shoulder and head) plus **the grammar**.

**Mechanism.**
1. *Per shot.* The codec probe gives cuts, rotation and focal (Part B). A body model gives each person's 3D joints in that shot's camera frame, with metric scale from a height prior or Idea 5. Head pose gives a gaze ray.
2. *Edit graph.* Nodes are shots. Edges are "same set, continuous time" hypotheses scored by the codec judge (Idea 2) and audio continuity (Idea 7). A wide master or establishing shot is the root, because directors shoot coverage outward from it.
3. *Cross-cut registration through entities.*
   - At a **match-on-action** cut, entity e's pose at t⁻ (shot A) and t⁺ (shot B) must agree up to one frame of motion. That is a closed-form Procrustes/Sim(3) on ≥3 non-collinear joints, giving the full relative camera pose.
   - In a static dialogue, both actors are often visible in OTS shots, which gives several joints per entity.
4. *When each shot shows only one person (clean singles), use the grammar:*
   - **Toric parameterisation.** Write each camera in toric coordinates relative to the two subjects' 3D heads. Their screen positions then pin the camera to a 2D manifold ([Lino & Christie](https://hal.archives-ouvertes.fr/hal-01142876)).
   - **180° rule.** All shots in a scene put the camera on the same side of the axis of action: one sign constraint per shot ([O'Reilly](https://www.oreilly.com/library/view/grammar-of-the/9780240521213/Chapter04e.html)).
   - **Eyeline match.** A's gaze ray in shot 1 must pass near B's head in the joint frame. Screen direction ("looks frame-right") fixes which side B is on ([Wikipedia](https://en.wikipedia.org/wiki/Eyeline_match)).
   - **Shot scale.** Close-up, medium or wide, plus a lens prior, gives camera distance. Camera height near eye level and near-zero roll are strong defaults.
5. *Set fusion.* Each shot contributes its visible low-poly planes. Registration puts them in one frame, so the reverse shot supplies the "back of the room" for free and nothing has to be hallucinated.
6. *Train it.* The network gets a **cross-cut consistency loss**: rendering shot B's camera from the scene fused through shot A must match shot B's frames.

**Why it should work.**
- Each cue is individually weak, but they are independent and the constraint count is high. Two heads give 6 numbers; gaze gives 2 rays; the toric model leaves 2 free DOF; the 180° rule gives one sign; match-on-action supplies joints.
- Rough error estimate: head-position errors of 5–15 cm at 2–3 m give relative-camera errors of a few degrees. That is plenty for low-poly sets [own estimate].
- The closest prior work confirms the parts but not the whole:
  - HumanMM aligns human motion across shots, but not the set or cameras ([arXiv 2503.07597](https://arxiv.org/abs/2503.07597));
  - Sitcoms3D needed many episodes of footage for SfM ([arXiv 2207.14279](https://arxiv.org/abs/2207.14279));
  - toric space solves only the forward direction.

**Cheap laptop test (CineSynth).**
- Write a ~500-line renderer: box rooms with distinct wall textures, 2–3 capsule people with heads and gaze, rendered at 320×180 in NumPy or MLX.
- Write a *director* that samples coverage like a cinematographer: master, OTS-A, OTS-B, singles, with the 180° rule and occasional deliberate line crosses as negatives. Write an *editor* that assembles shot/reverse-shot sequences, including match-on-action cuts and returns to the master.
- Compare relative camera pose error across cuts on 200 sequences for:
  - (i) SIFT + essential matrix (baseline);
  - (ii) entity Procrustes only;
  - (iii) (ii) + toric + 180° + eyeline priors.
- **Gate:** (iii) under 5° and under 10% translation error on reverse-shot pairs where (i) fails. The whole test is a few days of coding and minutes of compute.

**Main risk.**
- Real edits break the rules: jump cuts, deliberate line crosses, montage, B-roll, time ellipses, speed ramps. Single-person or no-person content (driving, Minecraft, landscapes) gets little help.
- Monocular depth error of the body model propagates into camera poses.
- *Mitigation:* treat every rule as a soft prior with a per-cut "rule broken" latent, and let the codec judge veto merges.

### Idea 2 — The codec is the judge (model-aided MDL for discrete scene structure)

**Key insight.**
- The hardest choices in this project are **discrete**, and differentiable re-rendering cannot make them:
  - how many entity slots;
  - is shot 5 a return to location 1;
  - is this the same chair re-entering or a new one;
  - how many polygons this wall deserves;
  - did this entity persist across the cut.
- Rate-distortion supplies one currency for all of them: **total description length L = L(scene program) + L(video | renders of the scene).**
- Computing L(video | scene) needs no learned entropy model. Use a real encoder, with the render as the reference picture, exactly as model-aided coding did for faces ([record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/71349085)), and count the bits.
- The motion search absorbs small misregistrations, so the score degrades smoothly.
- Any structure is kept only if it pays for itself in bits. That is Occam's razor, and it **makes "low-poly" an outcome rather than a hand-tuned regulariser**.

**Uses.**
1. *Location identity across cuts.* For each existing location slot j, render shot k from j's set and compute bits. Accept the merge if the bits beat the "new location" cost.
2. *Entity birth and death.* Add a slot only if Δbits(video) exceeds the slot's own cost (parameter count × bits per parameter).
3. *Level of detail.* Choose each entity's polygon budget at the knee of its rate curve.
4. *Training signal.*
   - The amortised network proposes K discrete candidates; the codec picks the best; the network is distilled toward the winner (best-of-K self-training).
   - Or bits feed a REINFORCE reward on the discrete heads.

**Why it should work.**
- Part B: the motion-compensated score ranks camera hypotheses monotonically over 0–8°.
- A wrong location is no better than a blank reference (712 KB vs 698 KB), while the right location at 8° error is far cheaper (397 KB). That is clean location discrimination.
- The correct model beats ordinary temporal coding by 37%, so "the scene explains the video" is measurable.
- Encoders are decades of tuned CPU code and leave the GPU free for the network.

**Cheap test.**
- Run it on 10–20 real clips (user-supplied) with a crude per-shot reconstruction.
- Check that bits correlate with held-out-frame PSNR and with human preference.
- On CineSynth, measure location re-ID accuracy across returns against DINO-feature cosine.
- About 2 days.

**Main risk.**
- *Codec bias.* Block-translational prediction punishes rotation, zoom and lighting change unrelated to scene error. Use HEVC or AV1 with affine/warped motion, or judge at reduced resolution.
- *Not differentiable.* Use it for discrete search, ranking and rewards only; keep pixel, feature and motion losses for gradients.
- *Gaming.* A render that copies textures "explains" everything, so texture bits must be charged in L(scene).
- *Zero-overlap reverse shots* give no saving. That is informative ("unexplained, so ask Idea 1"), but not decisive.

### Idea 3 — Probe re-encoding: the encoder as a configurable motion sensor

**Key insight.**
- Compressed-domain vision so far *reads* whatever bitstream it is given ([CoPE-VideoLM](https://arxiv.org/abs/2602.13191), [HY-Himmel](https://arxiv.org/html/2605.08158)) or warm-starts flow ([AV1 MV fidelity](https://arxiv.org/html/2510.17427v1)).
- Contrarian move: **treat the encoder as an instrument you configure for 3D**, because re-encoding is nearly free (~2,500 fps at 360p, Part B). Run several probes:
  - **P-only, 1-ref, forward** → clean t→t−1 motion field; cuts from intra-coverage collapse.
  - **Anchor-interleaved** (A, F₁, A, F₂, …) → MVs point straight at a keyframe, giving long-baseline, drift-free per-shot rotation (0.30° drift on a 25° pan in Part B). Chain anchors every 10–30 frames, with a wider merange or a downscaled copy for large motion.
  - **Reversed-time probe** → backward field for occlusion reasoning (forward-backward consistency).
  - **AV1 global-motion probe** → libaom's per-reference homography parameters as a camera prior ([libaom](https://aomedia.googlesource.com/aom/+/refs/heads/research/av1/encoder/global_motion.h)).
- **Supervise the 3D network with it.** Render the motion field induced by the predicted scene (camera + mesh + rig), block-average it on the codec's partition grid, and compare it with the probe MVs, weighted by inverse residual energy (blocks coded cheaply are trustworthy). That gives dense correspondence supervision at thousands of fps with no flow network in memory.

**Why it should work.** The evidence in Part B:
- exact cuts even with a fixed GOP;
- 0.04° per-frame rotation error;
- focal estimate within 8% (ultrafast probe);
- entity recall of 87%;
- about 25–50× faster than CPU optical flow.

The AV1 study shows codec MVs are within about 2 px EPE of true flow on Spring ([arXiv 2510.17427](https://arxiv.org/html/2510.17427v1)).

**Cheap test.** On real clips, train a tiny per-shot camera + depth model three ways (photometric only, + probe-MV loss, + DIS-flow loss) and compare accuracy and wall-clock. About 2 days.

**Main risk.**
- MVs are rate-optimal, not truth-optimal: textureless regions, occlusions, and motion over 40 px (coverage collapses, [arXiv 2510.17427](https://arxiv.org/html/2510.17427v1)). Fast dance or Minecraft flicks are exactly that case.
- Parallax breaks the single-homography model, so per-plane or essential-matrix fits are needed.
- GPL licensing of x264 if you ship [unverified].

### Idea 4 — Closed-vocabulary decompilation (voxel worlds first)

**Key insight.**
- For some domains, "reconstruction" is really **decompilation**: the world was built from a known, small vocabulary.
- In Minecraft-like video:
  - geometry is unit cubes on a grid;
  - textures come from a known atlas;
  - FOV, eye height and physics are game constants [unverified specific values];
  - the HUD crosshair marks the principal point;
  - when the F3 debug screen is on, the frame literally prints position and facing.
- Camera rotation falls out of the Manhattan grid's vanishing points; translation comes in exact block units; each visible voxel becomes a classification over block IDs by projecting known textures.
- The output (world file + camera path + entity tracks) is a perfect, editable, low-poly environment that replays the video.
- I found **no** prior video → block-world work. World2Minecraft goes real → blocks ([papernotes](https://en.papernotes.org/ICLR2026/3d_vision/world2minecraft_occupancy-driven_simulated_scenes_construction/)); GANcraft goes blocks → images ([arXiv](https://ar5iv.labs.arxiv.org/html/2104.07659)).
- **Generalisation: a domain router.** Pick a codebook (voxel; Manhattan room in the SceneScript style, [arXiv](https://arxiv.org/abs/2403.13064); road; sports field of standard dimensions) when it fits better in bits (Idea 2). Otherwise fall back to generic primitives.

**Why it should work.**
- Closed worlds shrink the search space by orders of magnitude, and known textures make analysis-by-synthesis exact.
- Training data is free and exactly labelled: a self-written ray-caster or Luanti (LGPL-2.1, [GitHub](https://github.com/luanti-org/luanti)) renders unlimited (video, world, camera) triples on the laptop.
- It gives the project its first **measurable end-to-end win** (block IoU, camera error) within weeks.

**Cheap test.**
- Write a NumPy/MLX voxel ray-caster with an original or CC texture set.
- Render 2,000 clips at 256×144 overnight, train a small grid-aware network, and report block IoU.
- Then hand-label 5 real gameplay clips (user-supplied) to measure the domain gap (shaders, HUD, hand/item overlay, mobs).

**Main risk.**
- The domain gap on real gameplay: shaders, resource packs, HUD, particles.
- It covers one domain, so there is a risk of over-fitting the project to it.
- Mojang EULA and branding: never redistribute game assets; train on your own textures [unverified legal detail].

### Idea 5 — Free metrology: gravity, contact and on-screen telemetry

**Key insight.**
- Monocular scale and up-direction are ambiguous, except that **physics leaks metric units into every video**.
- During free flight (a jump's ballistic phase, a thrown ball, a dropped object) the centre of mass accelerates at exactly g = 9.81 m/s². The frame rate comes from the container timestamps.
- For roughly fronto-parallel motion, image-space acceleration a_px = g·f/Z, so **Z = g·f / a_px**: metric depth from a parabola. More generally, fit a ballistic segment in 3D with scale as the unknown.
- Gravity also gives the up-vector, and support/contact ("every non-flying entity rests on something") pins the floor plane and object depths along camera rays. REST3D/PhySIC do this for static scenes ([arXiv 2605.30338](https://arxiv.org/abs/2605.30338), [arXiv 2510.11649](https://arxiv.org/html/2510.11649v1)).
- **Telemetry overlays** are exact metrology: dashcam speed, GoPro/Garmin overlays, sports-broadcast graphics, Minecraft F3 coordinates. OCR them.

**Why it should work.**
- Noise-floor estimate (my calculation): for f = 1000 px, Z = 3 m, and 0.5 px tracking noise at 30 fps, the quadratic fit of a 10–15-frame flight gives about **0.4–1.2% depth error**.
- Real error will be dominated by camera motion, motion blur and centroid choice. For humans, use the pelvis/CoM of a body model, not the head.
- Persistent entities (Idea 1) carry the scale across cuts.

**Cheap test.** In CineSynth, add jumps and throws at random scene scales. Compare scale error against a human-height prior (typically ±10%+ [unverified]) as a function of airtime, fps and noise. One day.

**Main risk.**
- *Rarely present.* Talking heads have no ballistic phases.
- *Speed ramps and slow motion* (common in short-form edits) silently break the timebase. Detect them with a ramp latent, or reject parabolas whose fitted g is implausible.
- *Rolling shutter* and *variable frame rate* phone video.
- **Novelty not verified.** Ballistic analysis is standard in sports ball tracking, but its use as a scene-wide metric anchor in a general video-to-3D model was not checked (search budget exhausted).

### Idea 6 — Repetition is multi-view (instancing as stereo and compression)

**Key insight.**
- Scenes are full of repeats: blocks, chairs, windows, cars, trees, tiles, identically dressed dancers.
- **N instances of one prototype in one frame are N views of the same shape** from different relative poses, so a single frame becomes a multi-view capture of the prototype.
- The MDL criterion (Idea 2) prefers the instanced explanation (1 mesh + N poses), and low-poly outputs love instancing.
- Sitcoms3D exploits repetition *across episodes* ([arXiv 2207.14279](https://arxiv.org/abs/2207.14279)). This idea exploits repetition *within a frame or shot*.

**Why it should work.** Shape is shared while pose varies, so self-occluded sides of one instance are visible on another. The optimisation is ordinary multi-view fitting with per-instance poses.

**Cheap test.** Synthetic rooms with 4–8 identical chairs. Compare single-frame chair reconstruction (Chamfer distance) with and without a shared prototype. 2–3 days with a small differentiable rasteriser (Dr.Jit on Metal, [GitHub](https://github.com/mitsuba-renderer/drjit)).

**Main risk.**
- Near-duplicates that are not identical.
- Instance discovery is itself hard.
- Symmetric objects create pose ambiguity.
- **Novelty unverified.** Repeated-structure detection for facades is classic [unverified], and single-image multi-instance shape learning may exist.

### Idea 7 — The soundtrack is a free sensor for the edit and the set

**Key insight.** Most short videos carry audio, and editors cut picture far more freely than ambience.
- **Room-tone continuity across a cut** means the same place and continuous time. A change of ambience spectrum means a new location. This works exactly where vision fails: zero-overlap reverse shots.
- **Reverberation** relates to room volume and absorption (Sabine relation [unverified textbook formula]). That gives a coarse prior on the low-poly room box, and indoor vs outdoor from the presence of a tail.
- **An off-screen voice** implies an off-screen entity that must exist and lie on the listener's eyeline (links to Idea 1).
- **Beat-synchronous cutting** flags montage, so continuity priors should be relaxed.

**Cheap test.**
- Build edits from long takes with ambient noise (any recordings the user has). Detect same-location across cuts with MFCC/ambience embeddings.
- Simulate rooms with a NumPy image-source model to test reverb → room-size estimation.
- One day, CPU only.

**Main risk.** Music-only soundtracks (TikTok dance, Minecraft montages) and post-production ambience replacement erase the cue. Reverb → volume is ill-posed. Treat it as a soft prior that is often absent. **Novelty unverified** (audio-visual scene segmentation literature not checked).

### Glue: a three-speed inference loop (integration, not a standalone invention)

1. **Slow, discrete.** The amortised network proposes K scene programs per shot (SceneScript-style DSL), the edit graph and the slot assignments. The codec judge (Idea 2) and the grammar solver (Idea 1) choose among them.
2. **Medium.** A learned optimiser (G3R-style, [arXiv 2409.19405](https://arxiv.org/pdf/2409.19405)) takes probe-MV residuals and rendering gradients and updates the continuous parameters in about 5–20 steps.
3. **Fast.** TTT3R-style state adaptation ([arXiv 2509.26645](https://arxiv.org/abs/2509.26645)) for streaming. No weight updates, so it fits 16 GB.

---

## Part D — Challenges to the current design sketch

| Sketch element | Challenge | Proposed change |
|---|---|---|
| *One network with persistent latent slots (per frame, location, entity)* | The edit's structure (shots, location identity, entity identity) is **discrete and combinatorial**; one network will blur it, and slot counts become hyper-parameters. | Make the **edit graph explicit**, built by cheap sensors (codec probe, audio) and solved by search (Ideas 1–2). The network predicts per-shot partial scenes plus proposals. Slots are created and merged by MDL, not fixed. |
| *Decoded to low-poly meshes, rigs, camera paths* | Agree on the output, but meshes as raw tensors are hard to count, edit or compare. | Emit a **scene program** (DSL: room boxes, extrusions, primitives, instanced prototypes, body templates; cameras in toric coordinates for dialogue). Programs are compressible (MDL), editable, and naturally low-poly ([SceneScript](https://arxiv.org/abs/2403.13064), [IG-LLM](https://arxiv.org/abs/2404.15228)). |
| *Self-supervised by differentiable re-rendering vs frames* | Pixel loss on low-poly flat-shaded renders mismatches the model's capacity: texture and lighting dominate the loss, and basins are narrow (Part B5a: 0.1° error is already worse than no model at full resolution). | Combine: (1) **motion-field loss vs probe MVs** (Idea 3); (2) coarse or posterised photometric loss (B5a: wider basin at ¼ resolution); (3) masks and silhouettes; (4) physics terms (Idea 5); (5) **codec judge** for discrete choices (Idea 2); (6) **cross-cut consistency** (Idea 1). |
| *Warm-start on synthetic scenes* | Generic synthetic scenes miss what makes this product hard: the edit. | The synthetic generator should be a **director + editor** (toric/180° camera sampling, shot scales, cut patterns, returns, match-on-action), plus a voxel domain with exact labels. Lyra's lesson, generator → 3D decoder ([project](https://research.nvidia.com/labs/toronto-ai/lyra)), done with a procedural generator we own. |
| Priors from big generative models | Genie/Marble are closed; Lyra 2/CAT4D need H100s; Lyra 2's weights bar production ([HF](https://huggingface.co/nvidia/Lyra-2.0)). | At most a **frozen mid-layer feature encoder** from a small video DiT ([arXiv 2512.19949](https://arxiv.org/html/2512.19949)), A/B-tested against DINOv2-S. Prefer the edit's own other shots to hallucination. |
| "Playback reproduces the video" | Requiring geometry and materials to reproduce every pixel is the most expensive possible target. | **Contrarian principle: the video is the texture.** From the original cameras, project the frames onto the low-poly geometry (view-dependent projective texturing, the classic Façade-style trick [unverified citation]). Playback is then pixel-faithful by construction, and geometry only needs to be good enough for plausible parallax off-axis. This moves effort from appearance to structure. |
| Compute plan | GPU everywhere. | The **CPU encoders** do motion and cut sensing (thousands of fps). The **GPU (MLX)** runs only the network and a small low-poly rasteriser (Dr.Jit Metal or a custom MLX rasteriser; a few thousand triangles is cheap). |

---

## Part E — What I would build first (2 weeks, laptop only)

**Goal:** de-risk the part nobody else has solved, the edit, before training any large network.

| Days | Build | Gate (measurable) |
|---|---|---|
| 1–2 | **Probe front end** on real clips (user-supplied, ~10): cuts, per-shot rotation and focal, dynamic-block mask. Code skeleton exists in `research/05_experiments/`. | Cut F1 ≥ 0.95 against hand labels. Rotation overlays visually stable. |
| 3–6 | **CineSynth** generator (NumPy/MLX): box rooms + capsule people + gaze; voxel worlds; *director* (toric, 180°, shot scales) and *editor* (shot/reverse-shot, match-on-action, returns, montage negatives). Exact GT. | 200 edited sequences with GT cameras, sets, entities, cuts. |
| 7–10 | **Cross-cut solver** (Idea 1): entity Procrustes + toric + 180° + eyeline. Baseline: SIFT + essential matrix. | Under 5° rotation and under 10% translation error on reverse-shot pairs where SfM fails. |
| 11–12 | **Codec judge** (Idea 2) for location re-ID and entity count on CineSynth, plus 5 real clips. | Location re-ID across returns ≥ 90%. Entity count exact on ≥ 80%. |
| 13–14 | Decide the network: feature backbone (DINOv2-S vs a small video-DiT, if downloads are approved), DSL heads, and which losses survive. | Written go/no-go per idea. |

Only after these gates would I train the amortised network. It would train on CineSynth with the probe-MV loss and the cross-cut consistency loss, with the codec judge as the discrete selector.

---

## Part F — Caveats and unverified items

- **Web-search budget.** Novelty checks for Ideas 5–7 (ballistic scale, repetition multi-view, audio cues) and the road-layout part of A4 could not be completed. Treat "nobody has done this" for them as *unknown*.
- **Part B is synthetic:** pure rotation, planar texture, no parallax, no lighting change. Real numbers will be worse. The judge's monotonicity was shown only for yaw and focal on one shot.
- **Secondary sources.** Some figures come from secondary summaries and are flagged inline: REST3D's 48% → 0%, Test3R's ~30 s, Wonderland's 256×, ApX's 16 GB training limits.
- **Licences to re-check before any commercial use:**
  - Tencent community licence (WorldMirror);
  - Stability Community Licence (US$1M threshold);
  - NVIDIA R&D licence (Lyra 2.0, no production);
  - VGGT commercial AUP;
  - SpatialLM encoder CC-BY-NC;
  - SMPL-X;
  - x264/x265 GPL;
  - OpenSplat AGPLv3.
- **Items marked [unverified]** come from my background knowledge: Minecraft constants, Sabine formula, Façade, GPL status of the encoders, local VLM strength, CAT4D weights status, "Genie 4" navigation label.
- **No models or datasets were downloaded.** The experiment used only Homebrew FFmpeg 8.0.1 and the existing project `.venv` (NumPy, OpenCV).
