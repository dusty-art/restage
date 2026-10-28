# 03: Edited-video understanding and legally obtainable training data

Research date: 2026-10-28. Scope: (a) edited-video structure, (b) synthetic data with exact ground truth, (c) real video under permissive terms with no sign-up, (d) bodies and motion without sign-ups, (e) evaluation. Sections 6 to 8 give the critique, original ideas and a data plan.

## How to read this report

Each licence or access claim carries a tag:

- **[V]**: verified by fetching the primary page (GitHub, Hugging Face card, arXiv, or the official site) on 2026-10-28.
- **[S]**: taken from a search-engine summary or a secondary page. Check before relying on it.
- **[K]**: from prior knowledge. I could not re-check it because the shared web-search budget for this session ran out partway through. Treat it as a lead, not a fact.

None of this is legal advice. Where terms are ambiguous (YouTube-sourced datasets, Minecraft footage, CC BY-SA data), the report says so and leaves the decision to you.

---

## 0. Key takeaways

1. **No public dataset gives exact ground truth for edited, multi-shot video** in the sense we need: cuts, plus cross-shot camera registration, plus location identity, plus persistent entity identity, plus 3D. The closest are HumanMM's ms-Motion (multi-shot human motion built from AIST and Human3.6M) [V-ish, S] and Multi-THuMBS (July 2026), which adds artificial shot boundaries to multi-view sets [V]. **We will have to generate this data ourselves.** That is good news: a procedural "director" on top of a synthetic scene generator gives exact labels for free.
2. **Stock-footage sites cannot be used for bulk training.** Pexels Terms section 8 forbids scraping and "bulk, large-scale or systematic copying ... including ... for machine learning purposes" without permission. Pixabay's Terms say almost the same [V]. Their permissive per-file licences do not override that.
3. **The cleanest real edited video is public domain or CC.** That means the Internet Archive's Prelinger collection (8,500+ public-domain films [V via Wikipedia]), Wikimedia Commons (360,000+ videos, all free-licensed, open API [V]) and the CC BY Blender open movies [V]. Blender Studio production files, which would give exact ground truth for real edited films, sit behind a paid membership ($17/month) [V], so they break both the $0 and the no-sign-up constraints.
4. **Minecraft is a "real-domain exact ground truth" opportunity.** Every OpenAI VPT contractor segment ships the video, a per-tick action log with `xpos/ypos/zpos/yaw/pitch`, an `options.json` and a world save from the segment start. There is no sign-up and the code is MIT [V]. That gives exact static geometry plus an exact camera path for real gameplay. The data itself carries no explicit licence, and Minecraft is Microsoft IP [V]. The EULA allows videos "within reason" and forbids commercial use of Mojang's assets without agreement; I found no explicit AI clause [S]. For commercial-safe exact ground truth, use Luanti (LGPL-2.1 engine [V]) or our own voxel renderer with CC0 textures.
5. **Bodies: two permissive, no-sign-up parametric models now exist.** Meta's **MHR** (Apache-2.0, 45 shape / 204 pose / 72 expression parameters, 7 LODs, SMPL converters) [V] and NAVER's **Anny** (Apache-2.0 code, CC0 MakeHuman-derived data, infants to elders, 104-bone rig, v0.6 dated 2026-08-06) [V]. SMPL, SMPL-X, STAR, SKEL and AMASS are all non-commercial and/or require registration. The SMPL licence explicitly bans training networks "for commercial use of any kind" [V].
6. **Motion data that is free and open: CMU mocap and 100STYLE.** CMU is free, including in commercial products, but may not be resold [K]. 100STYLE is CC BY 4.0, 4M+ frames, on Zenodo [V]. LAFAN1 is CC BY-NC-ND [V]. Bandai Namco is CC BY-NC [V]. Every popular text-to-motion model I checked inherits non-commercial terms from AMASS or HumanML3D, has its own NC licence (NVIDIA GEM/GENMO [V]), or is too large for 16 GB (HY-Motion 1.0 needs 26 GB VRAM minimum [V]). **Plan to train our own small motion prior on CMU + 100STYLE, retargeted to Anny or MHR.**
7. **Vision-language models are weak at the geometric side of editing.** On CameraBench, VLMs struggle with geometric camera-motion primitives, and SfM struggles with semantic ones. A fine-tuned Qwen2.5-VL reaches about MegaSaM level [V]. On ShotBench, GPT-4o averages below 60% [S], while ShotVL-7B (Apache-2.0, a Qwen2.5-VL-7B fine-tune) reaches 70.1 [V]. On VEU-Bench, some video LLMs score *below random* on editing tasks [S]. Use VLMs for semantic labels (shot size, entity names, location descriptions), never for boundaries or geometry.
8. **Shot boundary detection is close to solved for hard cuts but not for gradual transitions or false positives.** 2026 work (OmniShotCut [V], PERSIST, BMVC 2026, Apache-2.0 with weights [V]) builds on synthetic transition generation. We can train our own in-network cut head on exact synthetic splices and warm-start it from TransNetV2 (MIT) [V].
9. **Rendering on the Mac is feasible if we keep two tiers.** Tier 1 is a fast rasteriser (wgpu-py, BSD-2 [V], or three.js WebGPU [S]) that renders the same low-poly style the model outputs, on the fly during training. Tier 2 is Blender 5.2 through the `bpy` arm64 wheel (GPL, but outputs are ours [V]), using EEVEE or Workbench, plus Cycles with MetalRT, for a smaller photoreal subset. Differentiable rasterisation on Metal exists but is immature (mtldiffrast: MIT, 11 commits [V]).
10. **Proposed benchmark ("CutScene") as a key contribution.** It would score edit structure, cross-cut camera registration, location clustering, entity persistence, geometry and re-render fidelity. It also uses *other shots of the same location as real novel views* ("shot-holdout NVS"), which fixes the main weakness of monocular 4D evaluation.

---

## 1. (a) Edited-video structure

### 1.1 Shot boundary detection (SBD)

| Method | Year | What it is | Licence / weights | Notes | URL |
|---|---|---|---|---|---|
| PySceneDetect | ongoing (v0.7.2, 11 Oct 2026) | Heuristic content, adaptive and threshold detectors | BSD-3-Clause [V] | Good fast baseline. Weak on fast motion and flashes. | https://github.com/Breakthrough/PySceneDetect |
| TransNetV2 | 2020 | Dilated 3D-CNN on frame stacks | MIT [V]. Weights in repo; PyTorch inference port also MIT [V] | Reported F1: ClipShots 77.9, BBC Planet Earth 96.2, RAI 93.9 [V, port docs]. Trains on ClipShots plus optional IACC.3 [V]. | https://github.com/soCzech/TransNetV2 , https://pypi.org/project/transnetv2-pytorch/ , paper https://arxiv.org/abs/2008.04838 [K] |
| AutoShot | CVPR-W 2023 | NAS-searched 3D-ConvNet+Transformer. SHOT dataset: 853 short videos, 11,606 shot annotations | MIT repo [V]. Dataset on Google Drive / Baidu [V] | +4.2% F1 over TransNetV2 on SHOT [V] | https://github.com/wentaozhu/AutoShot , https://arxiv.org/abs/2304.06116 |
| OmniShotCut | arXiv 2604.24762 (Apr/May 2026) | Shot-query Transformer that predicts shot ranges plus intra- and inter-shot relations. OmniShotCutBench. **Fully synthetic transition synthesis** with exact boundaries [V] | Code/weights licence not stated [V] | Claims Shot-Range F1 0.881 vs 0.814 for TransNetV2. Says classic methods reach 0.75–0.82 but fail at transition localisation and "sudden jumps" [S] | https://arxiv.org/abs/2604.24762 |
| PERSIST | arXiv 2608.29287, BMVC 2026 | Persistent latent-state discrimination: a boundary must cause a *persistent* change, not a transient flash or overlay | Apache-2.0; 3 checkpoints on GitHub Releases [V] | Removes 33–80% of flash, text-overlay and archival false positives. About halves TransNetV2's pseudo-event false positives [V] | https://arxiv.org/abs/2608.29287 , https://github.com/linty5/PERSIST |
| TransVLM, FilmShots | 2025–26 | Transition-segment formulation; dilated Conv3D+attention | Unknown | Only search snippets seen [S] | (from search on SBD 2025; not verified) |

**Takeaway for us.** The useful idea in recent SBD work is that **SBD is trained on synthetic splices**. PERSIST notes that the strongest public detector trains on corpora whose transitions are 85% synthetic [V]. We generate exact cuts anyway, so the cut head of our network can be trained on our edits plus synthetic transitions (dissolve, fade, wipe) added to real footage. PERSIST's "persistence" framing matches the slot design closely: a cut is exactly a frame where the *frame slot's* location/camera latent jumps and stays changed.

### 1.2 Shot type and camera-movement classification

| Dataset / model | Content | Size | Licence / access | URL |
|---|---|---|---|---|
| MovieShots (Rao et al., ECCV 2020) | Shot scale (5 classes) and movement (static, motion, push, pull) | ~46K shots from 7,858 movies [S] | No licence found. Access via the MovieNet project or authors [S] | https://movienet.github.io/projects/eccv20shot.html , https://arxiv.org/abs/2008.03548 |
| AVE: Anatomy of Video Editing (ECCV 2022) | 8 shot attributes: size, angle, type (OTS, single, two-shot, insert, …), motion, subject, location, … | 5,591 scenes, 196,176 shots, 1.5M+ tags [S] | Labels JSON on GitHub. Videos must be fetched from YouTube (yt-dlp), and some are gone [S]. Annotation licence not found | https://github.com/dawitmureja/AVE , https://arxiv.org/abs/2207.09812 |
| MovieCuts (Pardo et al.) | Cut-type recognition: 10 cut types (cutting on action, cut-away, match cut, J/L cuts, …) | 173,967 clips; best model 47.7 mAP [S] | Code and data said to be on GitHub; licence unverified [S] | https://arxiv.org/abs/2109.05569 |
| **CameraBench** (Lin et al., NeurIPS 2025 spotlight) | Camera-motion taxonomy built with cinematographers: object-, ground- and camera-centric frames; translation, rotation, zoom, orbit, steadiness, tracking | ~3,000 videos, 150K+ binary labels and captions. Public test split = 1,071 rows (5.6 GB) [V] | HF card: **CC BY 4.0** (an older README revision said MIT) [V]. "We do not publicly redistribute any videos"; train split by e-mail request [V] | https://linzhiqiu.github.io/papers/camerabench , https://huggingface.co/datasets/syCen/CameraBench , https://arxiv.org/abs/2504.15376 |
| **ShotBench / ShotQA / ShotVL** (NeurIPS 2025) | 8 dimensions: shot size, framing, angle, lens, lighting type/condition, composition, camera movement | 3.5k QA from 200+ acclaimed films; ShotQA ~70k QA [S] | Repo has no LICENSE file [V]. **ShotVL-7B is Apache-2.0** (Qwen2.5-VL-7B fine-tune), avg 70.1 (shot size 81.2, movement 62.9) [V] | https://github.com/Vchitect/ShotBench , https://huggingface.co/Vchitect/ShotVL-7B , https://arxiv.org/abs/2506.21356 |
| VEU-Bench (CVPR 2025) | 19 editing-understanding tasks from shot size to cut types and transitions | n/s | "Code and data will be available" [S] | https://arxiv.org/abs/2504.17828 |
| BEDLAM2.0 camera taxonomy | Static, pan, tracking, dolly, orbit, zoom and combinations, with exact ground truth | n/s | PS:License 1.0; **registration** [S] | https://bedlam2.is.tuebingen.mpg.de/ , https://arxiv.org/abs/2511.14394 |

**Takeaway.** For our *procedural director* we can reuse the *taxonomies* freely: CameraBench motion primitives, AVE shot types and sizes, and MovieCuts cut types. Taxonomies are ideas, not copyrighted data. We should also report results in those vocabularies so our camera paths can be scored against human labels.

### 1.3 Grouping shots by location (scene and location clustering)

- **Narrative "scene" and physical "location" are different things.** Scene segmentation (MovieNet/LGSS, BaSSL [K]) groups consecutive shots into story units. We need *physical set identity*, which can recur non-contiguously (A-B-A-B in cross-cutting) and can split within one scene (a walk from hallway to room).
- **Recent scene segmentation:** Scene-VLM (arXiv 2512.21778, Dec 2025) fine-tunes a VLM over frames, dialogue and metadata of neighbouring shots and reports gains on MovieNet [S]. An ICLR 2026 poster adds genre and shot-duration priors, arguing that adjacent-shot visual similarity alone is not enough [S]. URLs: https://arxiv.org/abs/2512.21778 , https://iclr.cc/virtual/2026/poster/10008536
- **Location as 3D:** "The One Where They Reconstructed 3D Humans and Environments in TV Shows" (Pavlakos, Weber, Tancik, Kanazawa, ECCV 2022) exploits *recurring sets across a season* to run SfM across shots. It builds a NeRF per set and places SMPL humans inside, then uses that 3D for re-identification and cinematography analysis [S]. https://arxiv.org/abs/2207.14279 , project: ethanweber.me/sitcoms3D. **This is the closest prior work to our location slot.**
- **Classical:** shot keyframes into bag-of-visual-words or colour histograms, then spectral clustering [S] (https://www.academia.edu/50638458/Video_Scene_Detection_Using_Compact_Bag_of_Visual_Word_Models).
- **Recommended pragmatic teacher:** DINOv2 (Apache-2.0 [K]) global descriptors for candidate pairs, then local matching and geometric verification. Verified pairs become edges of a location graph; connected components become locations. Train the in-network location slot on synthetic exact location IDs and distil this graph on real video.

### 1.4 Re-identifying people and objects across cuts

| Work | Contribution | URL |
|---|---|---|
| Human Mesh Recovery from Multiple Shots (Pavlakos, Malik, Kanazawa; arXiv Dec 2020, CVPR 2022 [K for venue]) | Treats frames either side of a cut as *multi-view* evidence. Multi-shot optimisation produces pseudo-GT meshes from edited video [V] | https://arxiv.org/abs/2012.09843 |
| HumanMM (CVPR 2025) | Global human motion across shot transitions: shot-transition detector, alignment module, foot-slide-aware integrator. New ms-Motion dataset built from AIST and Human3.6M [S] | https://arxiv.org/abs/2503.07597 |
| **Multi-THuMBS** (UNIST/Birmingham/CJ, arXiv 2607.01626, 2 Jul 2026) | Multi-person, across cuts. PySceneDetect, then 4DHumans per shot, then **VGGT on just the two boundary frames** to build a shared space, then a 3-stage mesh-to-pointcloud alignment, DROID-SLAM poses mapped through the boundary transform, Hungarian re-ID on 3D root distance + UV-texture appearance + pose (1.0 m gate). Benchmarks: EgoHumans, EgoBody and Harmony4D with *artificial* shot boundaries, plus AVA, *Friends* and *Big Bang Theory* clips. Metrics: MPJPE, MPVPE, W-MPJPE, WA-MPJPE, Accel, Jitter, ATE, ID switches, PCK*, foot sliding [V] | https://arxiv.org/html/2607.01626 |
| VideoClusterNet (ECCV 2024, Flawless AI) | Self-supervised adaptation of a face model to movie face tracks, plus parameter-free clustering. MovieFaceCluster dataset [S] | https://arxiv.org/abs/2407.12214 |
| Video person clustering (2021) | Face + body + voice, plus editing co-occurrence cues [S] | https://arxiv.org/abs/2105.09939 |
| MOSEv2 (Aug 2025) | VOS benchmark that includes multi-shot sequences: 5,024 videos, 10,074 objects. SAM2 drops from 76.4 (MOSEv1) to 50.9; reappearance accuracy only 4.1–34.9% [S] | https://arxiv.org/abs/2508.05630 |
| SAM 3 / 3.1 | Detector+tracker, 848M params. **Checkpoints gated** (HF access request) under the "SAM License" [V], so it violates our no-gated rule. SAM 2 checkpoints are Apache-2.0 and ungated [K] | https://github.com/facebookresearch/sam3 |

**Takeaway.** The 2025–26 recipe (HumanMM, Multi-THuMBS) is *modular*: detect the cut, solve each shot, then register shots through boundary geometry and person re-ID. Our slot model is the end-to-end version of that recipe. That pipeline is also the right **teacher** for pseudo-labels on real video.

### 1.5 Film-grammar and multi-shot datasets

| Dataset | Content | Size | Licence | Sign-up | URL |
|---|---|---|---|---|---|
| MovieNet | 1,100 movies, scene/shot/cast/place annotations | large | "Other" per HyperAI. Derivative MovieNet-PS is non-commercial research only [S] | likely form [K] | https://movienet.github.io/projects/eccv20movienet.html |
| CondensedMovies (CMD) | Key scenes from YouTube movie-clip channels | ~3k movies [K] | No formal licence; "freely available" [S]. Videos are YouTube content | no | https://github.com/m-bain/CondensedMovies |
| E.T. (Exceptional Trajectories), ECCV 2024 | Camera + character trajectories + captions from CMD | 115K samples, 11M frames, 120 h [S] | not found | ? | https://www.lix.polytechnique.fr/vista/projects/2024_et_courant , https://arxiv.org/abs/2407.01516 |
| DataDoP (GenDoP, ICCV 2025) | Free-moving camera trajectories + depth + directorial captions | 29K shots [V] | not stated [V] | ? | https://arxiv.org/abs/2504.07083 |
| HoloCine (Oct 2025) | Multi-shot film/TV samples with hierarchical Gemini-2.5-Flash captions | ~400k [S] | release unclear | ? | https://arxiv.org/abs/2510.20822 |
| CineTrans / Cine250K | Frame-level shot labels + dense captions | ~250K [S] | unclear | ? | https://arxiv.org/abs/2508.11484 |
| MultiShotMaster pipeline (CVPR 2026) | Automated multi-shot extraction + cross-shot grounding | n/s | dataset not found [S] | n/a | https://arxiv.org/abs/2512.03041 |
| Panda-70M | 70.7M captioned clips (167k h); shot-boundary annotations added Oct 2024 [V] | ~36 TB | Must follow HD-VILA-100M licence (research) [V] | no (Google Drive CSVs) | https://github.com/snap-research/Panda-70M |
| FineVideo (HF) | 43,751 CC-BY YouTube videos, 3,425 h; timecoded scene annotations including **editing details and transitions** [V] | 673 GB | per-video CC-BY [V] | **gated** (accept terms) [V] | https://huggingface.co/datasets/HuggingFaceFV/finevideo |

Almost all of these rest on copyrighted film or TV, or on YouTube. For a model we might ship, use them only as **statistics sources** (shot-length distributions, shot-type frequencies, camera-motion priors) and for evaluation, not as bulk pixel training data.

### 1.6 What video-language models can and cannot do here

- **Can:** name entities and locations, caption the action, and give rough shot size or framing. ShotVL-7B scores 81.2 on shot size and 90.1 on framing [V]. They can also describe edits loosely. Gemini 2.5 Flash is good enough to caption 400k multi-shot samples (HoloCine) [S].
- **Cannot (reliably):** localise cuts frame-accurately; classify geometric camera motion. CameraBench: "VLMs struggle to capture geometric primitives that require precise estimation of trajectories" [V]. ShotVL-7B is only 62.9 on movement and 45.7 on composition [V]. VEU-Bench reasoning and judging tasks put some video LLMs below random [S].
- **Fine-tuning helps a lot:** Qwen2.5-VL fine-tuned on about 1,400 CameraBench videos reaches MegaSaM-level overall AP [V].
- **On-device options with clean licences:** Qwen3-VL-8B-Instruct is Apache-2.0 and ungated, about 9B params [V]. ShotVL-7B is Apache-2.0 [V]. A 4-bit 7–9B VLM fits in 16 GB for offline labelling [K]. Note that Qwen2.5-VL-3B (the base of ShotVL-3B) is reportedly under the non-commercial Qwen Research licence [K], so prefer the 7B.

### 1.7 Continuity errors

- Research is thin. The classic paper is Pickup & Zisserman, "Automatic retrieval of visual continuity errors in movies" (CIVR 2009). It uses the edit structure to propose shot pairs and ranks unexplained visual differences [S]. https://dl.acm.org/doi/10.1145/1646396.1646406 , https://ora.ox.ac.uk/objects/uuid:2ee5e0ec-88d7-4558-ad39-1aa5ddf8840f
- Industry patents compare object attributes across clip transitions, for example a shirt that changes colour between takes: US 11070706, US 11423654 [S]. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11070706 , https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11423654
- I found no 2025–26 continuity-error benchmark. Multi-shot *generation* papers measure "cross-shot consistency" of identity and scene (HoloCine, MultiShotMaster), which is the generative mirror of our problem.
- **Why it matters to us.** Our model *assumes* continuity: persistent sets and entities. Real edits break it deliberately (cheated eyelines, moved props, "cheating" an actor closer for a close-up) and accidentally. This means: (1) the model needs a per-shot *residual* (entity offset or state change allowed at cuts, penalised by a prior), not hard persistence; (2) synthetic continuity-error injection is an easy test and training signal (Section 7).

---

## 2. (b) Synthetic data with exact ground truth

### 2.1 Existing synthetic datasets

| Source | Content / GT | Size | Licence | Sign-up | URL |
|---|---|---|---|---|---|
| Kubric MOVi-A…F | Rigid objects (GSO/ShapeNet-style), HDRI backgrounds; depth, fwd/bwd flow, segmentation, object coords, camera, 3D boxes, normals. 24 frames @ 256² (MOVi-F 512²) [V] | many 10k-clip sets [K] | Code Apache-2.0 [V]. **MOVi data licence not stated** [V]. GSO assets CC BY 4.0 [K], Poly Haven CC0 [V] | no (public GCS `gs://kubric-public/tfds`) [V] | https://github.com/google-research/kubric , https://arxiv.org/abs/2203.03570 |
| PointOdyssey | Long (avg 2,000 frames) dynamic scenes with mocap humans/animals; RGB, depth, instance masks, normals, intrinsics/extrinsics, 2D/3D tracks [V] | 159 videos, ~200k frames; **185 GB** (HF) [V] | HF card: **MIT** [V] (site footer: "All rights reserved" [V]) | no [V] | https://pointodyssey.com/ , https://huggingface.co/datasets/aharley/pointodyssey |
| Dynamic Replica | Stereo videos of humans/animals in Replica rooms; depth, masks, flow, long tracks, cameras [V] | 524 videos, 145,200 stereo frames; train 1.8 TB [V] | Code CC BY-NC [V]; data "Non-Commercial" [S]; uses RenderPeople scans | **accept licence to get links.json** [V] | https://github.com/facebookresearch/dynamic_stereo |
| TartanAir V2 | RGB, depth, semantics, pose, LiDAR, fisheye/pano, flow, IMU, events, occupancy [V] | large (n/s) | **CC BY 4.0** (data), MIT toolkit [V] | not stated | https://tartanair.org/ |
| TartanGround (2025) | Ground-robot version | n/s | paper page lists CC BY-NC-SA [S] | n/s | https://arxiv.org/abs/2505.10696 , https://tartanair.org/tartanground |
| Spring (CVPR 2023) | 47 sequences / 6,000 stereo pairs @1080p from Blender's *Spring* film; disparity, optical flow, scene flow (4× super-res) [V] | ~6k frames | Movie assets CC BY 4.0 [V]; data record lists CC BY 4.0 [S] | DaRUS repository; registration only for benchmark submissions [V] | https://spring-benchmark.org/ , https://doi.org/10.18419/darus-3376 |
| BEDLAM / BEDLAM2.0 | Humans (SMPL-X) with clothing, hair, moving cameras (static/pan/track/dolly/orbit/zoom) [S] | large | PS:License 1.0 (non-commercial) [S] | **yes** [S] | https://bedlam2.is.tuebingen.mpg.de/ |
| Infinigen / Infinigen-Indoors / Infinigen-Sim | Procedural nature and interiors; depth, normals, instance seg, flow, OBJ/USD export; Sim exports URDF/MJCF/USD [S/V] | generator | **BSD-3** [V] | no | https://github.com/princeton-vl/infinigen , https://arxiv.org/abs/2406.11824 |
| Syn4D (NAVER LABS Europe, 2026) | UE5 multiview dynamic scenes; camera, depth, dense tracks, human pose; 1,674 Objaverse animated assets + **585 BEDLAM2 humans** [S] | subset 3.8 GB; full n/s | HF card says "Licensed for AI training use", with no SPDX id [V]. BEDLAM2-derived humans make this questionable | no for subset [V] | https://huggingface.co/datasets/Syn4D/Syn4D_RGBD , https://arxiv.org/abs/2605.05207 |
| OmniWorld-Game (ICLR 2026) | 96K game clips, 18M frames; depth, camera, flow, fg masks, text [S] | very large | older README: **CC BY-NC-SA 4.0** [S] | possibly gated (check) | https://huggingface.co/datasets/InternRobotics/OmniWorld , https://arxiv.org/abs/2509.12201 |
| MPI Sintel | Shots of the open film *Sintel* with flow/depth/camera | ~1k frames [K] | research use [K] | no [K] | http://sintel.is.tue.mpg.de/ [K] |
| CARLA | Driving sim (UE5.8 branch) | generator | MIT code, CC-BY assets [V] | no | https://github.com/carla-simulator/carla . **No macOS** (Ubuntu/Windows only) [V] |
| MetaDrive | Driving sim (Panda3D); RGB/depth/semantic cameras; can replay nuScenes/Waymo scenarios [V] | generator | **Apache-2.0** [V] | no | https://github.com/metadriverse/metadrive . Runs on macOS [V] |

**Assessment.** No existing dataset combines **multi-shot edits, multiple locations, persistent entities and exact GT**. PointOdyssey (MIT, no sign-up) and TartanAir V2 (CC BY 4.0) are the best downloads for per-shot geometry and tracking pre-training. Kubric is the best *generator* to copy ideas from, but it pins old Blender and uses a Linux Docker image [V]. Port its ideas to `bpy` 5.2 on arm64 rather than run its x86 container under emulation.

### 2.2 Asset sources for our own generator

| Source | Content | Licence | Sign-up | Notes | URL |
|---|---|---|---|---|---|
| Poly Haven | HDRIs, PBR textures, models | **CC0** [V] | no | ToS forbids "web scraping or data mining without express permission". Use the public API under its API terms [V] | https://polyhaven.com/license |
| Google Scanned Objects | ~1k scanned household objects | CC BY 4.0 [K] | no | used by Kubric | https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/ [K] |
| Objaverse 1.0 / XL | 800K+ / 10M objects, some animated | ODC-By with per-object CC licences [K] | no (HF) [K] | filter to CC0/CC-BY objects; keep attribution manifest | https://objaverse.allenai.org/ |
| Quaternius | Low-poly rigged and animated humans and animals (Ultimate Animated Animal Pack, Universal Animation Library 2, …) [V] | claims CC0 [K]; check its License page | no | ideal for the low-poly target style | https://quaternius.com/ |
| Kenney | Low-poly kits, voxel textures | CC0 [K] | no | Minecraft-like voxel textures without Mojang IP | https://kenney.nl/ [K] |
| ambientCG | PBR materials | CC0 [K] | no |  | https://ambientcg.com/ [K] |
| Infinigen | Procedural terrain, plants, creatures, rooms | BSD-3 [V] | no | heavy, but generates whole sets | https://github.com/princeton-vl/infinigen |
| Blender demo files | Some open-movie shots/characters (Charge splash 1.4 GB, Cosmos Laundromat, Agent 327 Barbershop) | **CC-BY** [V] | no | small but real production rigs | https://www.blender.org/download/demo-files/ |
| Blender Studio production files | Full productions (Spring, Sprite Fright, Charge, Wing It!, …) | Content CC BY 4.0 per T&C [V] | **paid membership $17/mo** [V] | would give *real edited films with exact GT*. Excluded under current constraints; a user decision | https://studio.blender.org/terms-and-conditions/ , https://studio.blender.org/join/ |
| Anny / MHR / MakeHuman | bodies (see §4) | Apache-2.0 / CC0 [V] | no |  | §4 |

### 2.3 Procedural cinematography and virtual camera placement

Research we can implement as a "director" module:

- **Classical virtual cinematography (2010s):** toric-space camera control (Lino & Christie, SIGGRAPH 2015) [K]; the Prose Storyboard Language for shot description (Ronfard et al.), whose tree grammar is cited in the 2025 camera-trajectory survey [S]; film-editing patterns and continuity editing for 3D animation (Galvane, Ronfard, Christie) [K]. These encode the 180° rule, shot/reverse-shot, establishing shot, cut-in and match-on-action as constraints. They are exactly what a procedural shot-list generator needs.
- **Survey (2025):** "Camera Trajectory Generation: A Comprehensive Survey of Methods, Metrics, and Future Directions" [S]. https://arxiv.org/abs/2506.00974
- **Data-driven directors, 2024–26:**
  - E.T./DIRECTOR (ECCV 2024): text- and character-conditioned camera trajectories [S].
  - GenDoP (ICCV 2025): an autoregressive "director of photography" trained on DataDoP, 29K shots [V].
  - **LensCraft** (2025): models subject *volume and orientation*. Ships "a specialized simulation framework" for training data; says dataset, simulator, weights and code are public [V]. https://arxiv.org/abs/2506.00988
  - **Auteur** (arXiv 2606.01900, Jun–Sep 2026): a human-centric camera DSL where shot size, angle and composition are defined *relative to the subject*, then converted to 6-DoF. Its 34K text+motion+camera samples come partly from **procedural synthesis** and partly from CondensedMovies. Paper CC BY 4.0 [V]. https://arxiv.org/abs/2606.01900
  - LAMP (Dec 2025): an LLM writes programs in a cinematography-inspired motion DSL [S]. https://arxiv.org/abs/2512.03619
- **Recommendation.** Build our director with a subject-relative parameterisation, as Auteur and LensCraft do. Specify shot size (ECU to EWS), angle, screen position and lens, then solve for the 6-DoF pose (toric-style). Sequence shots with an edit grammar (establishing, then medium, then shot/reverse with the 180° rule, then a cut-in, then a location change), using pacing statistics measured on public-domain films by TransNetV2. Add camera-motion primitives from the CameraBench taxonomy, plus handheld noise and zoom. Every label is exact by construction.

### 2.4 Domain randomisation for sim-to-real

- Foundations: Tobin et al. 2017 (arXiv 1703.06907) [K]; Tremblay et al. 2018 (arXiv 1804.06516) [K]; structured DR (Prakash et al., arXiv 1810.10093) [K].
- **What to randomise for our problem** (beyond textures, lighting and distractors):
  - Render style: flat low-poly, PBR, toon, voxel and noisy photogrammetry-like.
  - Lens: focal length, zoom during a shot, distortion, DOF, motion blur.
  - Sensor: noise, rolling shutter, compression artefacts, film grain, colour grading and LUTs, letterboxing and aspect ratio, vertical 9:16.
  - Edit: pacing, transition type (cut, dissolve, fade, wipe), jump cuts, speed ramps.
  - Overlays: subtitles, logos and watermarks, Minecraft-style HUD, picture-in-picture.
  - Camera operators: tripod, dolly, drone, handheld, body-cam, dashcam.
- Overlays deserve special care: real edited video has them, and a 3D model will otherwise try to explain them as geometry. Add an explicit 2D overlay layer to the output and supervise it synthetically.

### 2.5 Rendering fast on the Mac (M5, 10-core GPU, Metal 4, 16 GB)

| Option | Licence | Metal / headless | Best use | Caveats | URL |
|---|---|---|---|---|---|
| Blender 5.2 via `bpy` wheel (macOS arm64, Python 3.13 only, 5.2.3 released 22 Oct 2026) | GPL-3.0, but **"What you create with Blender is your sole property"** [V] | Cycles Metal; MetalRT hardware RT defaults to "Auto" on M3+ chips (so on for M5) [S]. Forum reports M5 Cycles about 2× M4 [S]. EEVEE and Workbench rasterisers | Photoreal subset (Cycles); medium tier (EEVEE); flat low-poly with exact GT passes (Workbench / EEVEE Cryptomatte, Z, vector passes) | Seconds per frame for Cycles; heavy per-process start-up. Batch many frames per process | https://pypi.org/project/bpy/ , https://www.blender.org/about/license/ , https://devtalk.blender.org/t/cycles-apple-metal-device-feedback/21868 |
| wgpu-py / pygfx (WebGPU in Python, Metal backend) | BSD-2 [V] | lists macOS Intel and M1 support [V]; offscreen rendering [K] | **On-the-fly training renderer** for the low-poly target domain; MRT outputs (RGB, depth, instance ID, normals, flow) in one pass | Need our own shaders for GT passes | https://github.com/pygfx/wgpu-py |
| three.js WebGPURenderer in headless Chrome | MIT [K] | needs `--enable-unsafe-webgpu`; on macOS uses Metal; works headless per one guide [S]. Blank-canvas pitfalls; read back via `copyTextureToBuffer` + `mapAsync` [S] | Also a web viewer for our output scenes, so one renderer serves both viewer and data generation | Browser-process overhead | https://discourse.threejs.org/t/what-is-the-current-state-of-headless-rendering-with-webgpu/82600 , https://agent-browser.dev/webgpu |
| moderngl | MIT [V] | headless via glcontext [V]. macOS OpenGL tops out at 4.1 and is deprecated; compute shaders need GL 4.3, so none on Mac [V/K] | quick prototypes | deprecated API on macOS | https://github.com/moderngl/moderngl |
| pyrender | MIT [V] | offscreen only via EGL/OSMesa [V]. No EGL on macOS, so CPU OSMesa only [K] | not recommended on Mac | stale (Py 2.7/3.6 era CI) [V] | https://github.com/mmatl/pyrender |
| Mitsuba 3 | BSD-3 [K] | LLVM (CPU) variants on Apple Silicon; no Metal backend [K] | differentiable physically-based checks | slow on CPU | https://github.com/mitsuba-renderer/mitsuba3 [K] |
| mtldiffrast | MIT [V] | Metal; nvdiffrast-compatible rasterise/interpolate/texture/antialias; zero-copy MPS [V] | **differentiable** re-render loss for meshes | 5 stars, 11 commits, no benchmarks [V] | https://github.com/pedronaugusto/mtldiffrast |
| metal-gauss / metalsplat / mpsplat (gsplat port) | various | MPS kernels via `torch.mps.compile_shader` [S] | Gaussian-splat appearance layer if we add one | young projects | https://github.com/debsahu/metal-gauss , https://github.com/tchauffi/metalsplat |

M5 notes: Apple says each GPU core has a Neural Accelerator, giving over 4× peak GPU AI compute vs M4, programmable through Metal 4 Tensor APIs [S]. Independent analysis says MLX did not yet fully use them at the time of writing [S]. https://secure.businesswire.com/news/home/20251014240640/en/Apple-unleashes-M5-the-next-big-leap-in-AI-performance-for-Apple-silicon , https://www.creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/

**Recommendation.** Store *scenes, not pixels*: meshes, rigs, motions, cameras and the edit list. Rasterise low-poly training frames on the fly in the same process as training, either with wgpu (separate Metal queue) or, more simply, with the same differentiable Metal rasteriser used for the loss. This removes the disk bottleneck (one 20 s 256² clip with 16-bit depth, masks and flow is tens of MB stored) and gives infinite augmentation. Pre-render a smaller photoreal subset with Blender EEVEE or Cycles overnight.

---

## 3. (c) Real video sources

Hugging Face downloads of *ungated* repos need no account [K]. "Gated" means you must log in and accept terms, which violates our constraint.

### 3.1 General and edited video

| Source | Content | Size | Licence | Sign-up | URL | Use for us |
|---|---|---|---|---|---|---|
| **Wikimedia Commons** | All kinds of video incl. edited documentaries, news, nature, hikes; WebM/Ogg only | **360,000+ video files** [V] | Free licences only: CC0/PD, CC BY, CC BY-SA, … (per file) [V] | **no** (MediaWiki API; follow the User-Agent policy) [K] | https://commons.wikimedia.org/wiki/Commons:Video , https://commons.wikimedia.org/wiki/Commons:Licensing | **Core real set.** Keep per-file attribution manifest; BY-SA share-alike implications for weights are unsettled |
| **Internet Archive: Prelinger Archives** | Mid-century educational, industrial and amateur films; edited, multi-shot | **8,500+ public-domain films** [V, Wikipedia] | Public domain (per item; check rights field) [S] | no | https://archive.org/details/prelinger , https://en.wikipedia.org/wiki/Prelinger_Archives | **Core edited-video set** (old look: B&W/early colour, so domain gap) |
| Internet Archive feature films / PD | Pre-1931 US films etc. | large | PD varies by item and jurisdiction [K] | no | https://archive.org/details/feature_films [K] | edited narrative film |
| **Blender open movies** (Elephants Dream … Wing It!, Singularity 2026) | Animated, professionally edited short films | ~19 films [V] | **CC BY 4.0** (films; Spring CC BY 4.0 [V]) | no (download.blender.org / video.blender.org) | https://studio.blender.org/films/ , https://www.blender.org/download/demo-files/ | edited film with *some* GT (Spring/Sintel benchmark shots) |
| US federal government footage (NASA, NPS, …) | Space, aerial, nature, parks B-roll | large | US federal works are generally public domain (17 U.S.C. §105) [K] | NASA library no key [K] | https://images.nasa.gov/ [K] | outdoor and hiking |
| Pexels | Stock video | huge | Per-file licence is permissive [V], **but ToS §8 bans scraping and "bulk, large-scale or systematic copying ... including for machine learning"** without permission [V]; API needs a key [K] | API key | https://www.pexels.com/license/ , https://www.pexels.com/terms-of-service/ | **Do not bulk-download.** Ask Pexels for written permission if wanted |
| Pixabay | Stock video | huge | Same pattern: ToS §8 bans data mining and robots "for machine learning purposes" and bulk copying [V] | API key [K] | https://pixabay.com/service/license-summary/ , https://pixabay.com/service/terms/ | **Do not bulk-download** |
| YouTube-8M | Precomputed **features only** (frame-level 1.53 TB, video-level 31 GB) [V] | 1.53 TB | **CC BY 4.0** (features) [V] | no [V] | https://research.google.com/youtube8m/download.html | not useful: no pixels |
| Kinetics-400/700 | YouTube action clips; CVDF hosts tarballs of K400 and K700-2020 [V] | ~0.5–1 TB [K] | annotations CC BY 4.0 [K]; videos are uploader-copyrighted YouTube content | no [V] | https://github.com/cvdfoundation/kinetics-dataset | grey: YouTube ToS and copyright |
| DAVIS 2016/2017 | VOS masks, 480p and full-res | small | site states no licence [V]; commonly cited as BSD/CC [K] | no [V] | https://davischallenge.org/ | eval only |
| YouTube-VOS | VOS masks | ~4k videos [K] | annotations CC BY 4.0; data "**non-commercial research purpose only**" [V] | CodaLab [K] | https://youtube-vos.org/dataset/term/ | eval only |
| Ego4D | Egocentric video | 3,670 h [K] | custom licence agreement; ~48 h approval; AWS keys expire in 14 days [V] | **yes** [V] | https://ego4d-data.org/docs/start-here/ | excluded |
| **RealEstate10K** | Camera poses (SLAM + BA) for ~80k clips / 10M frames; file = video URL + per-frame intrinsics + 3×4 pose [V] | 720 MB [V] | **CC BY 4.0** (Google) [V] | no [V] | https://google.github.io/realestate10k/download.html | pose pseudo-GT; videos on YouTube (ToS grey) |
| DL3DV-10K | Scene-level static captures | 3.01 TB [V] | not shown; terms behind gate [V] | **gated** [V] | https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P | excluded |
| **Stereo4D** | 100K+ real 4D scenes from VR180 YouTube: camera poses, 3D tracks, YouTube IDs [V] | ~3.6 TB annotations [V] | annotations **CC BY 4.0**, code CC0 [V] | no (GCS bucket; gcloud CLI) [V] | https://github.com/Stereo4D/stereo4d-code | dynamic 3D tracks pseudo-GT; videos on YouTube |
| DynPose-100K (NVIDIA) | Camera poses for 100K dynamic internet videos; Lightspeed GT benchmark [V] | ~200 GB video, 0.7 GB annotations [V] | "NVIDIA License" (check terms) [V] | no gate seen [V] | https://huggingface.co/datasets/nvidia/dynpose-100k | camera pseudo-GT on dynamic video |
| SpatialVID-HQ | Clips with poses, depth, dynamic masks, camera-motion captions [V] | 3.53 TB [V] | **CC BY-NC-SA 4.0** [V] | **gated** [V] | https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ | excluded |
| Sekai | 5,000+ h walking, drone and game video with camera trajectories and metadata [V] | ~102 GB metadata [V] | not shown [V] | no gate seen [V] | https://huggingface.co/datasets/Lixsp11/Sekai-Project | check licence first |
| **Walking Tours (DoRA)** | 10 4K 60 fps city walks + 1 safari; faces/plates blurred [V] | ~16–18 h | **CC BY 4.0** [V] | no [V] | https://huggingface.co/datasets/shawshankvkt/Walking_Tours | outdoor/hiking-like; hosted on YouTube (ToS grey) |
| CO3D | Object-centric turntables | 1.5M frames [K] | CC BY-NC 4.0 [K] | no [K] | https://github.com/facebookresearch/co3d [K] | NC |

**YouTube caveat.** The YouTube ToS prohibit you to "access, reproduce, download ... any part of the Service or any Content except" as expressly authorised, and forbid automated access "such as robots, botnets or scrapers" [V] (https://www.youtube.com/t/terms). This applies even to CC-BY-licensed uploads (which is what FineVideo, Walking Tours and YouTube-Commons rely on). Downloading via yt-dlp is therefore a ToS issue, though not necessarily a copyright one for CC-BY videos. Treat all YouTube-sourced pixels as "research-only, user decision".

### 3.2 Minecraft

| Source | Content | Size | Licence | Sign-up | URL |
|---|---|---|---|---|---|
| **OpenAI VPT contractor data** (index files 6.x–10.x: free play, house building, diamond pickaxe) | Per segment: `.mp4` (720p recorded, 360p @20 Hz), `.jsonl` actions per tick (mouse, keyboard, GUI flags, hotbar, **yaw, pitch, xpos, ypos, zpos**, ticks, inventory, stats), `-options.json`, and a **world save `.zip` from the start of the segment** [V] | thousands of hours [K] | Code **MIT** [V]. Data: **no licence stated**; "Minecraft is the intellectual property of Microsoft, not OpenAI"; OpenAI does not purport to license Minecraft IP [V] | **no** (Azure blob `openaipublic.blob.core.windows.net`) [V] | https://github.com/openai/Video-Pre-Training |
| MineRL BASALT 2022 | FindCave, MakeWaterfall, AnimalPen, VillageHouse demos | ~150 GB per task [V] | as above | no [V] | same repo |
| MineRL-v0 (Zenodo copy) | Navigate, Treechop, ObtainIronPickaxe, ObtainDiamond | 8.6 GB [V] | Zenodo record: **MIT** [V] | no [V] | https://zenodo.org/records/12659939 , https://github.com/minerllabs/minerl |
| Replay Mod | Records sessions and replays them from any perspective [V] (packet-level, camera paths [K]) | tool | **GPL-3.0** [V] | needs a Minecraft client (paid game) | https://github.com/ReplayMod/ReplayMod |
| Mineflayer + prismarine-viewer | Bots log server-side world/entity state; viewer renders headless to MP4 [V] | tool | **MIT** [V] | needs a server (EULA acceptance) [K] | https://github.com/PrismarineJS/prismarine-viewer |
| **Luanti** (ex-Minetest) | Open-source voxel engine with Lua modding; builds on macOS [V] | engine | **LGPL-2.1** engine [V]; content CC BY-SA 3.0 per Craftium README [V] | no | https://github.com/luanti-org/luanti |
| Craftium (ICML 2025) | Gymnasium/PettingZoo RL envs on Luanti 5.12 [V] | tool | LGPL-2.1 / "MIT code, CC-BY-SA 3.0 content" (README inconsistent) [V] | no | https://github.com/mikelma/craftium . Linux wheels only [V] |

**Can we get exact Minecraft ground truth? Yes, at two levels.**

1. **From VPT, without running Minecraft.** Parse the world save (Anvil region files), voxelise the chunks around the player and render them with our own renderer, using the per-tick camera `(x, y, z, yaw, pitch)` and the FOV from `options.json`. That yields exact depth, block semantics and camera for every frame of *real gameplay video*, valid until the player edits blocks. Block edits can be partly replayed from the actions and inventory deltas. Mobs and other entities are *not* logged, so their masks come from residuals or SAM 2.
2. **From a recorded server.** Run a server with Mineflayer bots or Replay Mod and log all entity states every tick. That gives exact entity tracks too. It requires Mojang's server/client software and accepting its EULA, which is a user decision.

For a **commercially clean** Minecraft-like domain, use Luanti or our own voxel engine with CC0 (Kenney) textures and scripted agents. Then render "YouTube-style" edited montages (first person, third person, cuts between bases, HUD overlays) with complete ground truth.

**Mojang EULA position.** The official minecraft.net EULA and Usage Guidelines pages timed out three times, so this is from a summary and search snippets. Screenshots and videos are allowed "within reason", including monetised videos if conditions are met. Mojang says "do not make commercial use of anything we've made unless specifically agreed". It reserves the right to change or withdraw the guideline permissions at any time [S]. I found **no explicit AI or ML clause** [S]. Sources: https://www.tldrlegal.com/license/minecraft-terms-of-service , https://www.minecraft.net/en-us/usage-guidelines , https://www.minecraft.net/en-us/eula . Using footage for *research* training follows precedent (VPT; MineWorld from Microsoft itself [K]; Oasis [S]). A *commercial* product trained on Mojang footage would rely on fair use, which is a grey area. **Recommendation:** use real Minecraft only for evaluation and research fine-tuning, and use the Luanti/CC0 voxel domain for bulk training.

### 3.3 Driving

| Source | Content | Size | Licence | Sign-up | URL |
|---|---|---|---|---|---|
| Waymo Open | multi-sensor | large | non-commercial [given] | yes | https://waymo.com/open/ |
| nuScenes | multi-sensor | large | CC BY-NC-SA 4.0 [K] | yes [K] | https://www.nuscenes.org/ |
| KITTI / KITTI-360 | stereo, LiDAR, poses | ~180 GB [K] | CC BY-NC-SA 3.0 [K] | e-mail registration [K] | https://www.cvlibs.net/datasets/kitti/ |
| **Argoverse 2** | Sensor (1 TB), Lidar (5 TB), Motion Forecasting (58 GB), Map Change (1 TB) [V] | TB | **CC BY-NC-SA 4.0**, API MIT [V] | **no**: direct S3 HTTPS links [V] | https://www.argoverse.org/av2.html , https://www.argoverse.org/about.html |
| **comma2k19** | 33 h commute video (HEVC) + GNSS/IMU/CAN + **global camera poses (ECEF)** [V] | ~100 GB [V] | repo MIT [V]; data licence not separately stated [V] | **no** (Academic Torrents) [V] | https://github.com/commaai/comma2k19 |
| Zenseact ZOD | 100k frames, 1,473 sequences, 29 drives [V] | large | **CC BY-SA 4.0**, commercial OK [V] | **yes**: e-mail request for Dropbox link (torrents also listed) [V] | https://zod.zenseact.com/ , https://zod.zenseact.com/download/ |
| A2D2 (Audi) | camera, LiDAR, bus | ~2 TB [K] | CC BY-ND 4.0 [K] | no [K] | https://www.a2d2.audi/ (TLS error on fetch) |
| PandaSet | LiDAR + cameras | n/s | CC BY 4.0 [K] | n/s | **pandaset.org now 301-redirects to an unrelated site** (observed 2026-10-28). Use only trusted mirrors |
| OpenDV-YouTube | 1,700+ h driving video, 244 cities; mini subset 25 h / 44 GB [V] | ~3 TB raw [V] | "YouTube license"; code Apache-2.0 [V] | no [V] | https://github.com/OpenDriveLab/DriveAGI |
| MetaDrive (sim) | Procedural + replayed real scenarios, RGB/depth/semantic [V] | generator | Apache-2.0 [V] | no | https://github.com/metadriverse/metadrive |

**Best open driving choices under our constraints:** comma2k19 (MIT repo, no sign-up, real poses), Argoverse 2 (no sign-up, but non-commercial), and MetaDrive for exact-GT synthetic driving on the Mac.

### 3.4 Hiking and outdoor

- Wikimedia Commons (hiking, mountains and nature categories; CC/PD), US-government public-domain footage (NASA, NPS) [K], Walking Tours (CC BY 4.0 but YouTube-hosted) [V], Prelinger travelogues (PD) [V], and Sekai's walking and drone subset (licence unknown) [V].
- Synthetic: Infinigen nature (BSD-3) [V] and TartanAir V2 outdoor environments (CC BY 4.0) [V].

---

## 4. (d) Human and animal bodies, and motion

### 4.1 Body models

| Model | Licence | Sign-up | Key facts | URL |
|---|---|---|---|---|
| SMPL / SMPL-X / SMPL-H | non-commercial research/education/art. Explicitly "prohibits the use of the Software to train methods/algorithms/neural networks/etc. for commercial use of any kind"; commercial licence via Meshcapade [V] | yes (register to download) [K] | the de facto standard | https://smpl.is.tue.mpg.de/modellicense.html |
| STAR | non-commercial research licence; commercial via ps-license@tue.mpg.de [V] | download from MPI site [V] (registration [K]) |  | https://github.com/ahmedosman/STAR |
| SKEL | non-commercial research [V] | **yes**: create accounts on SKEL and SMPL sites [V] | biomechanical skeleton inside SMPL | https://github.com/MarilynKeller/SKEL |
| GHUM / GHUML (Google) | not stated on page [V] | **request form** [V] | 16-D shape VAE/PCA, 20-D expression, 63 joints / 124 DoF, hands, normalising-flow pose priors [V] | https://github.com/google-research/google-research/tree/master/ghum |
| **MHR, Momentum Human Rig (Meta, 2025)** | **Apache-2.0** (repo) [V] | **no** (`pixi run download-assets`) [V] | 45 identity shape, 204 pose (angles + scalings), 72 expression params; 7 LODs; neural pose correctives; **MHR↔SMPL/SMPL-X converters**; needs pymomentum; macOS not mentioned [V] | https://github.com/facebookresearch/MHR , paper arXiv 2511.15586 [V] |
| **Anny (NAVER LABS, 2025–26)** | code **Apache-2.0**; MakeHuman-derived data **CC0**; SOMA topology Apache-2.0; *the SMPL-X-compatible topology is non-commercial* [V] | no [V] | differentiable PyTorch; infants to elders; phenotype parameters + local shape + facial actions; 104-bone rig (163-bone MakeHuman rig optional); v0.1 2025-11-05 to v0.6 2026-08-06 [V] | https://github.com/naver/anny |
| MakeHuman / MPFB2 | assets CC0 (per Anny's notes) [V]; code GPL/AGPL [K] | no | the original procedural human generator, great for synthetic diversity incl. clothes | http://www.makehumancommunity.org/ [K] |
| SAM 3D Body (Meta, Nov 2025) | SAM License; **checkpoints gated** [V] | yes | single-image HMR producing MHR | https://github.com/facebookresearch/sam-3d-body |
| SMAL (animals) | licence page not retrieved [V]; MPI non-commercial pattern [K] | register [K] | quadrupeds | https://smal.is.tue.mpg.de/ |
| Animals, open alternatives | Quaternius animated animal packs (CC0 claimed [K]); Objaverse animated animals (per-object CC) [K]; Infinigen procedural creatures (BSD-3) [V] | no | build our own quadruped/bird template rig | https://quaternius.com/ |

**Recommendation.** Use **Anny** (CC0 data, child-to-elder coverage, PyTorch, no SMPL dependency, if we avoid its NC smplx topology) as the canonical human, with **MHR** as a second option and for converting any SMPL-based pseudo-labels. Our entity slot's "shape latent" for humans can be Anny's phenotype parameters. Other categories get a generic low-poly mesh latent plus a category template.

### 4.2 Motion data and priors

| Source | Content | Size | Licence | Sign-up | URL |
|---|---|---|---|---|---|
| **CMU Graphics Lab mocap** | 2,500+ clips, ASF/AMC, BVH conversions [K] | ~9 h [K] | FAQ: free for research; "You may include this data in commercially-sold products, but you may not resell this data directly, even in converted form" [K]. Fetch failed with a TLS error on 2026-10-28 | no | http://mocap.cs.cmu.edu/faqs.php |
| **100STYLE** | 100 locomotion styles; unedited BVH, 60 fps, 28 bones; retargeted FBX/BVH on GitHub [V] | 4M+ frames [V] | **CC BY 4.0** [V] | no (Zenodo 8127870) [V] | https://www.ianxmason.com/100style/ , https://zenodo.org/record/8127870 |
| LAFAN1 (Ubisoft) | 5 subjects, 77 sequences, BVH 30 fps [V] | 496,672 frames (~4.6 h) [V] | **CC BY-NC-ND 4.0** [V] | no (git LFS) [V] | https://github.com/ubisoft/ubisoft-laforge-animation-dataset |
| Bandai Namco Research Motion | stylised content × styles, BVH [V] | 36,673 + 384,931 frames [V] | **CC BY-NC 4.0** (scripts MIT) [V] | no [V] | https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset |
| AIST++ | dance; SMPL motion, 2D/3D keypoints, cameras (2.3 GB total) [V] | 2.3 GB | must accept the **AIST Dance Video DB Terms of Use** [V]; SMPL-based | ToU acceptance | https://google.github.io/aistplusplus_dataset/download.html |
| AMASS | unified mocap in SMPL-H/X | ~40 h+ [K] | non-commercial; no commercial training [V] | **yes** [K] | https://amass.is.tue.mpg.de/license.html |
| Motion-X / Motion-X++ | 81.1K clips / 15.6M frames, SMPL-X [V] | large | **non-commercial**; Google-Form approval [V] | **yes** [V] | https://github.com/IDEA-Research/Motion-X |
| Mixamo (Adobe) | rigged characters + animations | ~2.5k anims [K] | royalty-free use in projects, no raw redistribution [K]; FAQ fetch got HTTP 403 | **Adobe ID required** [K] | https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html |
| HY-Motion 1.0 (Tencent, 30 Dec 2025) | text-to-motion DiT + flow matching; 1.0B (Lite 0.46B); trained on 3,000 h + 400 h curated [V] | **min 26 GB / 24 GB VRAM** [V] | License.txt not retrieved [V] | no | https://github.com/Tencent-Hunyuan/HY-Motion-1.0 |
| MoMask | text-to-motion; outputs (T, 22, 3) joints and BVH [V] | small | code **MIT**, but trained on HumanML3D/KIT-ML and depends on SMPL/SMPL-X; "datasets each have their own respective licenses" [V] | no | https://github.com/EricGuo5513/momask-codes |
| GENMO / GEM (NVIDIA, ICCV 2025) | unified motion estimation + generation (video/text/audio), SMPL [V] | — | **NVIDIA OneWay Noncommercial** [V] | no | https://github.com/NVlabs/GENMO |

**Recommendation.** No verified permissive, small, AMASS-free text-to-motion prior exists. Train our own small motion VAE or diffusion prior on **CMU + 100STYLE**, retargeted to Anny's skeleton, plus procedural locomotion (walk/run cycles with foot-contact constraints). Keep LAFAN1 and Bandai for a non-commercial research variant only. For dance (a target domain), AIST++ is the best data but requires accepting the AIST terms, so that is a user decision.

---

## 5. (e) Evaluation

### 5.1 Existing benchmarks and metrics for monocular 4D

| Benchmark | What it measures | Metrics | Licence / access | URL |
|---|---|---|---|---|
| DyCheck iPhone (Gao et al. 2022) | dynamic NVS with honest multi-camera GT | **co-visibility-masked** PSNR/SSIM/LPIPS; PCK for correspondences [V] | code + data **Apache-2.0** [V] | https://github.com/KAIR-BAIR/dycheck , https://arxiv.org/abs/2210.13445 [K] |
| TAP-Vid (DAVIS, Kinetics, RGB-Stacking, Kubric) | 2D point tracking | AJ, <δ_avg, OA [K] | annotations **CC BY 4.0**; checkpoints (TAPNext, BootsTAPIR) Apache-2.0 [V] | https://github.com/google-deepmind/tapnet |
| TAPVid-3D | 3D point tracking | 3D-AJ, APD, OA [K] | separate licence file [V] | same repo; https://arxiv.org/abs/2407.05921 [K] |
| Sintel, Bonn RGB-D Dynamic, TUM RGB-D dynamic | video depth + camera on dynamic scenes (MonST3R/MegaSaM/CUT3R protocol) [K] | depth AbsRel, δ<1.25; camera ATE, RPE-t, RPE-r after Sim(3) [K] | various (TUM CC BY 4.0 [K]) | http://sintel.is.tue.mpg.de/ [K] |
| SpatialBench (arXiv 2605.27367, 2026) | 41 models, 19 datasets, 5 task suites | RRA/RTA/AUC; ATE/RPE after Sim(3); AbsRel/SqRel/RMSE/δ; Accuracy/Completeness/F-score [V] | paper CC BY 4.0 [V] | https://arxiv.org/abs/2605.27367 |
| Multi-THuMBS multi-shot protocol (2026) | humans across cuts | MPJPE, MPVPE, W-MPJPE, WA-MPJPE, Accel, Jitter, ATE, ID switches, PCK*, foot sliding [V] | — | https://arxiv.org/html/2607.01626 |
| CameraBench | camera-motion primitives | per-primitive AP [V] | CC BY 4.0 test [V] | https://huggingface.co/datasets/syCen/CameraBench |
| MOSEv2 | VOS incl. multi-shot, reappearance | J&F, reappearance [S] | — | https://arxiv.org/abs/2508.05630 |
| Human mesh: EMDB, 3DPW, RICH | world-frame human motion | W-MPJPE, WA-MPJPE, RTE [K] | registration / NC [K] | — |

**What's missing.** Everything above is *single-shot*. Novel-view evaluation on monocular video is famously optimistic unless co-visibility-masked (DyCheck's point). No benchmark scores *edit replication*, cross-cut camera registration, or location clustering.

### 5.2 Proposed benchmark: "CutScene" (replicate edited multi-shot video in 3D)

**Data splits** (all licence-clean):
1. **CutScene-Synth.** Held-out procedural edited videos from our generator with *disjoint* assets, locations, motions and director styles. Exact GT for everything. Five sub-domains matching the product: talking head / interview, dance, voxel game, driving, hiking.
2. **CutScene-Voxel-Real.** VPT Minecraft segments edited into multi-shot sequences (cuts between segments and in-segment jump cuts). Exact static geometry and cameras come from world saves and pose logs (research use only, per §3.2).
3. **CutScene-Film.** Public-domain films (Prelinger) and CC-BY open movies (Blender), with human annotations: shot boundaries and transition types, location IDs per shot, entity IDs per track, and ~20 sparse cross-shot 2D correspondences per location pair. Include Spring/Sintel shots, where per-shot GT depth and flow exist.

**Metric families** (report each separately, never one blended score):

| Level | Metric | Notes |
|---|---|---|
| Edit | Cut F1 at ±1/±2 frames; transition-type accuracy; gradual-transition IoU | OmniShotCut-style range F1 |
| Location | shot-to-location ARI / NMI and pairwise F1; over- and under-merge counts | handles A-B-A recurrence |
| Camera, within shot | ATE, RPE-t, RPE-r after Sim(3) **per location** (not per shot); focal error | per-location alignment forces cross-shot consistency |
| **Camera, across cuts** | **Cross-cut relative pose error**: rotation (deg), translation direction (deg), scale ratio, for shot pairs in the same location | new; the key metric for our problem |
| Cinematography | shot-size class and camera-motion primitive *derived from predicted cameras + entities* vs labels (AVE / CameraBench vocab) | tests that the 3D explains the film grammar |
| Geometry | per-location Chamfer and F-score@τ vs GT set; rendered-depth AbsRel / δ1 | low-poly fairness: F-score at coarse τ |
| Entities | IDF1 / HOTA *across cuts*; ID switches at cuts; W-MPJPE / WA-MPJPE (humans) in location frame; 3D centroid error / 3D IoU (objects) | persistence is scored explicitly |
| Re-render | playback LPIPS/PSNR on all frames; per-entity mask IoU | "does playback reproduce the video" |
| **Novel view** | (a) Synth: true unseen cameras. (b) Real: **shot-holdout NVS**: drop one shot of a location, predict its frames from the rest (given its GT or estimated camera), co-visibility-masked LPIPS | cuts give real novel views for free |
| Efficiency | wall-clock and peak memory on an M5 16 GB per minute of video | it is a product constraint |

**Fairness rules.** Disjoint assets between train and test. Report per domain and per edit type (cut-in, shot/reverse, location change, jump cut, cross-cut). Include a "no-cut" control (the same content as one continuous shot) to isolate the cost of editing. Publish alignment code (Sim(3) per location; metric scale optional with a known-height human). Provide a naive baseline: per-shot reconstruction with no cross-shot linking.

---

## 6. Critical look at the design sketch

The sketch: single network; frame, location and entity slots; decode to low-poly meshes, a rig and cameras; self-supervised re-rendering plus synthetic warm start. Points to challenge:

1. **Re-rendering against photoreal frames through low-poly meshes is a weak geometry signal.** Colour loss is dominated by appearance the low-poly output cannot represent, and texture can "explain away" geometry. Make **depth, masks, 2D/3D point tracks and wide-baseline correspondences the primary self-supervised terms**, with colour going through a per-entity or per-location appearance layer (texture atlas or a small splat set) that cannot change geometry. Most strong 2025–26 feed-forward geometry models were trained with direct 3D supervision on synthetic plus pseudo-labelled real data rather than pure photometric self-supervision [K]; we should copy that.
2. **Nothing in a per-frame re-render loss links shots.** Without explicit cross-shot evidence, the location slot collapses to one per shot (over-segmentation), or two similar rooms merge (under-segmentation). Cross-shot links need their own loss: correspondence or matching teachers across cuts, shared-entity constraints, and synthetic GT for location IDs.
3. **Use a modular teacher, a single-network student.** The 2025–26 multi-shot systems (HumanMM, Multi-THuMBS) are modular pipelines. Build that pipeline first, using only permissive models:
   - TransNetV2 / PERSIST for cuts;
   - DA3-Small/Base (Apache-2.0) or `map-anything-apache` (Apache-2.0) for per-shot depth and cameras; DA3METRIC-LARGE (Apache-2.0) for metric scale;
   - TAPNext (Apache-2.0) for tracks; SAM 2 (Apache-2.0 [K]) for masks;
   - DINOv2 matching for location graphs; Anny/MHR fitting for humans.

   Run it once over the real corpus and cache the pseudo-labels. The slot network then trains on synthetic exact GT plus filtered pseudo-labels, with re-rendering only as fine-tuning. This is also the only way training fits on 16 GB: teachers run offline, one at a time, and the student never backpropagates through them. Licences, all [V]: DA3 https://github.com/ByteDance-Seed/Depth-Anything-3 ; MapAnything https://github.com/facebookresearch/map-anything ; Video-Depth-Anything-Small https://github.com/DepthAnything/Video-Depth-Anything . VGGT's commercial checkpoint is gated [V], so avoid it.
4. **Scale and time are ambiguous across cuts.** Each shot has its own scale ambiguity. Shared scale comes from shared entities (human height prior from the body model) or metric depth. Time is also not continuous across cuts: cross-cutting, flashbacks, replays, slow motion. The frame slot needs a **story-time offset per shot**, not just a frame index.
5. **Set representation.** "Rough low-poly mesh" is under-constrained. Prefer a **layout-primitive vocabulary**: ground plane or heightfield, walls, boxes and extrusions, plus a few free meshes. It is easier to decode, regularise, evaluate and render. For voxel games, the native representation is a voxel grid, and it is exact.
6. **Camera.** Predict intrinsics per frame (zoom is common in edited video), and give the camera trajectory a CameraBench-style primitive decomposition. That makes it interpretable and checkable against shot labels.
7. **The edit needs more than cut times:** transition type and duration, plus a 2D overlay layer (subtitles, logos, HUD, picture-in-picture, split screen). Otherwise the 3D will hallucinate geometry for graphics.
8. **Continuity is a soft prior, not a hard constraint.** Real films cheat positions between shots. Let entities have a small per-shot residual transform penalised by a prior.
9. **Slot cardinality.** Use set prediction with Hungarian matching on synthetic data (known counts), plus slot birth and death with a sparsity prior on real data.

---

## 7. Original ideas: free supervision from editing

1. **Crop-cuts with exact camera GT.** A digital crop-and-resize of a frame is *exactly* a pinhole camera with a longer focal length and shifted principal point, with the same pose. Turning any continuous real clip into "wide, then cut-in close-up, then wide" gives **exact relative cameras across the cut**. That is free real-domain supervision for the cross-cut camera head. Add slight blur and noise so crops are not trivially detectable.
2. **Edit-equivariance training ("re-edit, re-solve").** Take a continuous clip (real or synthetic). Build K synthetic edits of it: jump cuts by temporal skips, crop-cuts, reordering, A-B-A insertion of shots from another video. Every edit list is exact by construction (cut times, location IDs, which frames share a camera). Require the model to produce the **same scene slots and entity tracks** for every edit as for the unedited clip. This needs no labels at all, works on Commons, Prelinger and VPT footage, and directly trains location recurrence.
3. **Re-filming synthetic worlds.** Render one 4D synthetic scene with N different procedural shot lists (different directors, pacing, coverage). Same scene, different edits, so slot embeddings should agree (contrastive or consistency). This teaches the network that the edit is a nuisance variable.
4. **Cycle consistency through our own renderer.** Prediction (low-poly scene + cameras + edit), then re-render with a *new* procedural shot list on the fast Metal rasteriser, then re-predict, should recover the same scene in canonical coordinates. It is cheap because the output domain is low-poly, and it acts as self-play for the decoder.
5. **The body as the calibration object across cuts.** In match-on-action cuts, a person's 3D pose at the last frame of shot A roughly equals the pose at the first frame of shot B. Solving the body in both shots gives the relative camera across the cut, the Multi-THuMBS boundary-alignment idea but learned and continuous. Detect when the assumption holds: a learned "continuity confidence" head, trained on synthetic data where we control match-on-action versus elliptical cuts.
6. **Film-grammar priors as soft geometric constraints.** In shot/reverse-shot, the 180° rule says both cameras lie on the same side of the line joining the two speakers, and eyelines match. Train a small **director prior** p(camera | entities, previous shot) on our procedural director, with statistics calibrated on public-domain films. Use it to break ambiguities (mirror flips, which side of the axis) at inference.
7. **Synthetic continuity-error injection.** Randomly change one prop, clothing colour or entity position across some synthetic cuts and label it. A "state-change at cut" head prevents the model from silently averaging contradictory evidence. It also gives an evaluation of whether persistence is real (does the model notice?).
8. **Minecraft as the "Rosetta stone" domain.** VPT gives real footage with exact world geometry and exact cameras. Fine-tune and evaluate there with true GT depth and segmentation of static scenery, and measure the sim-to-real gap exactly on one real domain before trusting it elsewhere.
9. **Calibrate pacing statistics without using the pixels.** Run TransNetV2 and ShotVL-7B over public-domain films, Commons and Blender films to measure shot-length distributions, shot-size transition matrices and camera-motion frequencies. Fit the procedural director to those statistics so synthetic edits look like real edits. Only aggregate statistics leave the real data.
10. **Audio as a free location and identity signal** (optional). Room tone and reverberation persist across cuts within a location, and speaker identity persists across shot/reverse-shot. Commons and Prelinger films include audio. A weak audio-embedding similarity can serve as one more edge weight in the location graph.
11. **Teacher-disagreement filtering.** Run two independent teachers (DA3 vs MapAnything cameras; TAPNext vs optical-flow chaining) and keep pseudo-labels only where they agree within tolerance. Store the disagreement as per-sample uncertainty for loss weighting.

---

## 8. Recommended data plan under the constraints

Constraints: no sign-ups or gated downloads, $0 cloud, permissive licences, M5 with 16 GB.

**Tier A: our own synthetic generator, "EditWorld" (exact GT; the backbone of training).**
- **Engine:** `bpy` 5.2 (arm64) for scene assembly and a photoreal subset; a wgpu or Metal rasteriser for on-the-fly low-poly frames during training. Store scenes, not pixels.
- **Assets:** Poly Haven (CC0, via API per its terms), Kenney / Quaternius / ambientCG (CC0, check each page), GSO (CC BY 4.0), CC-BY-filtered Objaverse with an attribution manifest, Infinigen (BSD-3) for nature and room sets.
- **Humans and motion:** Anny (Apache-2.0 / CC0 topology) as primary, MHR (Apache-2.0) as secondary. Motion: CMU + 100STYLE, retargeted, plus procedural locomotion.
- **Director:** subject-relative shot specification (shot size, angle, lens) with a toric-style solver, an edit grammar (180° rule, shot/reverse, cut-in, match-on-action, establishing shots, location changes, jump cuts), CameraBench motion primitives, transitions, overlays and HUD. Pacing comes from measured statistics.
- **Domains:** interview / talking head, dance (procedural plus CMU dance clips), voxel game (own voxel engine or Luanti with CC0 textures, HUD, first and third person), driving (MetaDrive, Apache-2.0), hiking (Infinigen nature, Poly Haven HDRIs).
- **Scale for v0:** about 50k clips of 10–30 s at 256–384 px, 3–12 shots, 1–4 locations, 1–6 entities. That is feasible because frames are rendered on the fly from compact scene files.

**Tier B: downloaded synthetic, ungated, permissive (per-shot geometry and tracking pre-training).**
- PointOdyssey (MIT on HF, 185 GB; subset first) and TartanAir V2 (CC BY 4.0, select environments).
- Spring (CC BY 4.0 assets, via DaRUS) and Kubric MOVi (Apache-2.0 code; data licence unstated, so regenerate if in doubt).
- Skip Dynamic Replica (licence click-through, NC), BEDLAM/BEDLAM2 (registration), OmniWorld (NC-SA) and Syn4D (BEDLAM2-derived humans).

**Tier C: real video with near-exact GT.**
- VPT Minecraft: world saves + per-tick poses + FOV give exact static GT. Research and evaluation use, because of Mojang IP.
- comma2k19: driving video + global poses, MIT repo, no sign-up.
- RealEstate10K poses (CC BY 4.0) and Stereo4D tracks (CC BY 4.0) **only if you accept the YouTube-ToS risk for the video pixels**.

**Tier D: real unlabelled video for self-supervision and edit-equivariance (legally clean).**
- Wikimedia Commons (CC0/PD/BY/BY-SA; per-file attribution manifest; decide whether to exclude BY-SA).
- Internet Archive Prelinger and PD feature films (edited, multi-shot).
- Blender open movies (CC BY 4.0).
- US-government PD footage for outdoor content.
- Pseudo-label everything once, offline, with the Apache/MIT teacher stack (§6.3), and train with ideas 1, 2 and 11 from §7.

**Tier E: evaluation.** CutScene-Synth, CutScene-Voxel-Real (VPT), CutScene-Film (Prelinger/Blender with our own annotations), plus standard single-shot checks: DyCheck (Apache-2.0), TAP-Vid (CC BY 4.0), and Sintel/Bonn/TUM for depth and camera.

**Explicitly excluded under current constraints:**
- Pexels/Pixabay bulk download (ToS §8).
- YouTube-pixel datasets (Kinetics, CondensedMovies, AVE, OpenDV, Walking Tours, FineVideo), unless you approve.
- Ego4D, DL3DV, SpatialVID-HQ, Waymo, nuScenes, KITTI, ZOD (sign-up or gate).
- SMPL family, AMASS, Motion-X, BEDLAM (NC + registration); Mixamo (Adobe ID).
- SAM 3, VGGT-commercial, SAM 3D Body (gated weights).
- Blender Studio production files (paid membership).

**Decisions for you.**
1. Is the YouTube-ToS risk acceptable for research-only training? It would unlock RealEstate10K, Stereo4D and Walking Tours pixels.
2. Is one month of Blender Studio membership ($17) acceptable? It would give real professionally edited films with exact production GT.
3. Should CC BY-SA data be included, given unclear share-alike implications for model weights?
4. Is real Minecraft footage only for research and evaluation, or also for a shipped model?

---

## 9. Open items I could not verify (web-search budget exhausted)

- The official Minecraft EULA and Usage Guidelines text (pages timed out); a current explicit AI clause, if any.
- Licences of MovieNet, MovieShots, AVE annotations, MovieCuts, E.T., DataDoP, HoloCine, Sekai, and the Kubric MOVi data.
- CMU mocap FAQ wording (TLS error) and the Mixamo FAQ (HTTP 403).
- HY-Motion License.txt (likely a Tencent Hunyuan community licence with territory limits [K]).
- The SMAL licence page.
- Infinigen's current macOS support, since its installation doc path moved. It was reported as Linux + macOS in the v1.4 era [S] (https://80.lv/articles/this-tool-generates-photorealistic-procedural-interiors).
- Quaternius, Kenney and ambientCG CC0 statements.
- GSO's CC BY 4.0 licence; whether DAVIS has a licence; Kinetics annotation licence; A2D2 terms (TLS error); PandaSet mirror status.
- Whether DINOv2, SAM 2 and LightGlue weights are Apache-2.0 and ungated (believed so [K]).
