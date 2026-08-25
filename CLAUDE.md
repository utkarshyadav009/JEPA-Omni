# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda activate jepa-omni   # Python 3.11; see requirements.txt for packages
export HF_TOKEN=<token>    # required for gated checkpoints (EmbeddingGemma, Llama)
```

## Commands

```bash
# Smoke tests — each file is directly runnable; exercises one module on CPU/GPU
python -m models.losses
python -m models.predictor
python -m models.vision_encoder        # downloads V-JEPA 2 ViT-L (~1.3 GB first run)
python -m models.text_target
python -m models.spine_m1              # M0 gate: prints a loss, shapes correct

# Latency / VRAM probe (run before tuning batch size)
python probe_latency.py --frames 64 --res 256 --iters 30
python probe_gpu.py
python probe_wavjepa.py                # verifies WavJEPA loads and shapes are correct

# Feature extraction (run ONCE before training; skip_encoder must be True in config)
python scripts/extract_features.py    # M1: vision features for MSR-VTT / VATEX
python scripts/extract_features_av.py # M2: AV features for VGGSound → /dev/shm/jepa_m2_cache

# M1 training and eval
python train_m1.py --config configs/m1.yaml
torchrun --nproc_per_node=2 train_m1.py --config configs/m1.yaml
python eval_m1.py --config configs/m1.yaml --ckpt checkpoints/m1/best.pt

# M2 training
python train_m2.py --config configs/m2.yaml
torchrun --nproc_per_node=2 train_m2.py --config configs/m2.yaml
python train_m2.py --config configs/m2.yaml --max-steps 200  # smoke test

# Diagnostics
python scripts/cross_modal_diagnostic.py
python scripts/assert_sigreg_grad.py
python scripts/cavmae_retrieval.py
```

## Architecture

### Milestone structure
- **M1 (done):** Offline vision-text spine. Frozen V-JEPA 2 ViT-L → trainable predictor → InfoNCE alignment against a frozen text target. Achieved V→T R@1 = 22.5 on MSR-VTT+VATEX with feature caching.
- **M2 (in progress):** Joint audio-visual predictor. Cross-modal masked latent prediction: mask one modality over a ~50% time window, predict its frozen latents from the other. World-State vector is attention-pooled output of the full (unmasked) sequence.

### Model components (`models/`)

| File | Frozen? | Role |
|---|---|---|
| `vision_encoder.py` | Yes | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`), BF16. Interface: `encode(frames_list) → (B, N, 1024)` |
| `text_target.py` | Yes | EmbeddingGemma (1536-d, faithful) or MiniLM (384-d, smoke). Interface: `encode(texts) → (B, D)` |
| `predictor.py` | Trainable | Maps encoder output to shared dim. Modes: `mlp` (best default), `transformer`, `llama_last8` |
| `spine_m1.py` | Partial | Assembles M1 spine. `forward() → (loss, metrics)`. `embed_video()` / `embed_text()` for eval |
| `audio_encoder.py` | Yes | WavJEPA base+nat wrapper. Input: `(B, 1, n_samples)` raw 16kHz PCM. Output: `(B, T, 768)` |
| `av_jepa_predictor.py` | Trainable | M2 joint predictor. `AVJepaConfig` + `AVJepaPredictor`. Produces World-State vector (never L2-norm it — SIGReg targets N(0,I)) |
| `sigreg.py` | — | Anti-collapse regularization alongside the prediction loss |
| `losses.py` | — | `info_nce`, `info_nce_with_queue`, `compute_siglip_loss` |

### Data pipeline (`data/`)
- `video_text_dataset.py` — real-time decoding via `torchcodec`; slow for training
- `cached_feature_dataset.py` — reads pre-computed BF16 tensors; **use this for M1 training**
- `av_cached_dataset.py` — AV cached features for M2; reads from `/dev/shm/jepa_m2_cache`

## Critical engineering notes

**Feature caching is mandatory for real training.** Running V-JEPA 2 live during training causes OOM across DDP workers and 2.6 min/step. Set `model.skip_encoder: true` in the config and point `data` at the cache dir. Extract once with `scripts/extract_features.py` (M1) or `scripts/extract_features_av.py` (M2).

**InfoNCE requires large batches.** Small batches (≤8) collapse or overfit even with a MoCo queue (`queue_size`). Batch 256+ with cached features is what drove M1 from R@1=16.8 to 22.5. The MoCo queue helps on single-GPU but isn't a substitute for real batch scale.

**World-State must not be L2-normalised.** `AVJepaPredictor.encode_world_state()` returns un-normalised vectors. SIGReg targets N(0,I), which is incompatible with the unit sphere.

**WavJEPA loading workaround.** The standard `AutoModel.from_pretrained` path hits a `transformers 5.x` bug (`all_tied_weights_keys` AttributeError). `audio_encoder.py` loads the model class directly from the HF snapshot with manual safetensor loading — do not change this.

**Authoritative files — do not recreate or modify:** `models/`, `configs/`, `scripts/probe_latency.py`, `requirements.txt`, `scripts/train.sbatch`. Exact class names: `VisionEncoder`, `TextTarget`, `Predictor`, `SpineM1`, `SpineConfig`, `info_nce`, `AVJepaPredictor`, `AVJepaConfig`, `sigreg`.

## Standing rule: falsifier tracking across M4 stages

After EVERY stage that touches the M3 connector or the M4b speech projector
(joint training, LoRA, M4c, anything downstream) — re-run BOTH standalone
falsifiers and append a row to `checkpoints/falsifier_tracking.md` before
considering that stage done:
  - **M3**: word-overlap F1 normal/swapped/zeroed + semantic cosine, no
    speech stream present (`scripts/m4_joint_eval.py`'s M3-standalone path).
  - **M4b**: semantic cosine normal / swapped-vs-target / swapped-vs-donor,
    no M3 stream present (same script's M4b-standalone path).

Frozen-LLM baselines (pre-any-joint-training): M3 F1 0.471/0.268/0.274,
cos 0.724. M4b cos 0.517/0.152/0.517. The risk this rule guards against is
**compounding drift across stages**, not any single stage's hit in
isolation — the first joint-exposure fine-tune already cost −9% M3 F1 and
−19% M4b cosine on its own; a few more stages each costing that much would
add up to something worth stopping for, even if no single stage looks alarming.

## Milestone gates

- **M0:** `python -m models.spine_m1` prints a finite loss and correct shapes.
- **M1:** Video→text R@1 within 5 pts of SigLIP2 baseline (32.5); abort/switch encoder if >10 pts worse. **Status: PASSED** (Run 5: R@1 = 22.5 with MLP + batch 256 + MSR-VTT+VATEX cached).
- **M2:** `effective_rank(world_state)` stays above a threshold (collapse = rank collapses); prediction loss decreases steadily.

## BMO Jetson production deployment

Real, running deployment on the Jetson Orin (7.6GB shared memory, ARM64), separate from the
M1/M2 training work above. SSH: `bmo@bmo-desktop` (Tailscale, key auth already set up).
Code lives under `~/bmo_production/` on the Jetson.

**Full operational detail — layout, mandatory load order, measured latency, streaming and
emotion voice, NvMap/compaction forensics, power and fan acoustics, deploy history — is in
the `bmo-jetson-deploy` skill** (`.claude/skills/bmo-jetson-deploy/SKILL.md`). LLM tiers,
speculative turn-taking and corpus generation are in the `bmo-llm-corpus` skill.

Prohibitions that apply whether or not those skills are loaded:
- **Do not recreate the old scattered paths** — `~/jepa_omni_transfer`, `~/gguf_models`,
  `~/bmo_stt_test`, `~/BMO-Project`, `~/bmo_fresh`. They were consolidated into
  `~/bmo_production/` and deliberately deleted.
- **`scripts/bmo_launch.sh` is the real entry point.** Never run `bmo_jetson_startup.py`
  directly in production — the wrapper does a privileged memory-compaction step first.
- **llama.cpp models MUST load before the torch/int8 perception stack.** Reversing the order
  fragments unified memory badly enough that the LLM/TTS GGUFs fail to load.
- **Never L2-normalise the World-State vector**, and never decode TTS greedily (temp=0 makes
  any neural-codec TTS loop forever — this was a testing bug, not a model defect).
- **Settled, do not re-investigate:** the NvMap ENOMEM CMA-region hypothesis (ruled out via
  live `/sys/kernel/debug/nvmap` monitoring); the ReSpeaker XMOS-DSP hypothesis (falsified,
  no on-board DSP); spectral subtraction for fan noise (measurably worse — `denoise=False`);
  ToMe on ViT-L (real structural incompatibility with the 256-token grid).

## Data locations

- MSR-VTT videos: `/home/jovyan/work/data/msrvtt/video`
- MSR-VTT annotations: `msrvtt_train_7k.json`, `msrvtt_test_1k.json`
- VATEX / VGGSound: `/home/utkarsh/data/vggsound`
- M2 AV feature cache: `/dev/shm/jepa_m2_cache` (RAM disk / tmpfs, 756GB total / ~750GB free as of 2026-07-25 — NOT ~1TiB; verify with `df -h /dev/shm` before assuming headroom, this figure drifts). Machine has 1.5TiB total RAM, so files cached on other filesystems (e.g. `/mnt/Raid-Storage-2`) still benefit from the OS page cache up to available RAM — the 750GB figure is a tmpfs *mount* cap, not a hard ceiling on how much can be kept warm in memory overall.
- CAV-MAE checkpoint: `~/models/cav-mae/audio_model.25.pth`
- **Persistent M2 VGGSound feature cache (the one that actually produced `step19000_peak.pt` / 52% R@1)**: `/mnt/Raid-Storage-2/utkarsh-data/feature_cache_vgg51k` — despite the `51k` in the dirname, it holds 199,007/199,176 clips (near-complete full corpus, ~169 missing). 764GB total, ~3.93MB/clip real (measured directly, `du`/`ls -la` on a sample file: 4,127,549 bytes) — this is vision (32,16,1024 bf16, ~1.05MB) **+ ambient_base + ambient_nat (T≈997,768 bf16 each, ~1.53MB each)**, not vision alone; don't reuse the ~1MB/clip figure for total-cache sizing. Subdirectories named `m2`/`M2` inside it are NOT "Milestone 2" — `scripts/extract_features_av.py`'s `_shard(vid) = vid[:2]` shards by the first 2 characters of the VGGSound video ID, so `m2`/`M2` just means "video IDs starting with those characters." **These shard names collide on case-insensitive filesystems** (macOS default, SMB/CIFS mounts, some container overlay/bind-mount configs) — `m2/` and `M2/` silently merge into one directory, corrupting or losing files. Never copy/rsync this cache to such a filesystem without renaming shards to something case-distinct first (e.g. prefix with a fixed marker). `/dev/shm/jepa_m2_cache` (tmpfs) is an ephemeral *working* cache location for live training runs, not where the corpus persists between sessions.
- Spatial pooling for M2's vision cache happens at **extraction time**, not load time: `scripts/extract_features_av.py`'s `_spatial_pool()` mean-pools the raw 256 spatial tokens (16×16 grid) down to 16 (4×4 grid) via `F.avg_pool2d(kernel=4, stride=4)`, producing the cached `(32,16,1024)` bf16 tensor (512 tokens total). `data/av_cached_dataset.py` reads this pooled tensor as-is — it does not pool anything itself. Do not cache unpooled/full-token features (8192 or 18432 tokens) expecting to pool later in the dataloader without checking compute cost first: an isolated `AVJepaPredictor` forward+backward at those token counts measured 9x slower / ~5.6x more peak GPU memory (8192 tokens) or outright CUDA OOM at batch=32 (18432 tokens) on a 96GB GPU (2026-07-25, `checkpoints/vjepa21_shelved/ITEM3_THROUGHPUT_PREFLIGHT.txt`) — attention cost does not scale linearly with sequence length.

## Live-pipeline defects found 2026-08-16 (do not re-break these)

Four defects survived every offline metric and were only visible in real on-device output
(`scripts/jetson_real_demo.py`). Recorded here because each is a one-line regression risk.

**`enable_thinking` must be True on the reasoning tier, False on the fast tier.**
`GGUFReasoningTier` inherited `GGUFFastTier._build_prompt_text`, which passes
`enable_thinking=False` — correct for the fast tier, fatal for the thinker. Qwen3's chat
template implements that flag by writing an **empty `<think>\n\n</think>` into the prompt**, so
the model treats deliberation as already done. Result: the "reasoning tier" emitted **zero**
`<think>` blocks and was just a second conversational model. `GGUFReasoningTier` now overrides
`_build_prompt_text`. **Real cost of thinking: 650 ms → 1,749–3,509 ms** (median ~2.4 s), which
makes the thinker the pipeline's dominant leg, ahead of perception.

**The CoT is kept, not discarded.** `generate()` used to strip `<think>…</think>` and throw it
away, so downstream only ever saw the *answer* — itself a finished BMO line. Handing a complete
utterance to the speaker and asking for an utterance leaves only paraphrase, which is exactly
what the speaker did. `FastTierResult.reasoning` now carries it. **Condition the speaker on
`.reasoning`; speak `.text`.**

**Never feed the audio branch a zero tensor.** The demo passed `torch.zeros(16000*10)`, so
WavJEPA-base (645 MiB) and M2's audio branch ran on silence and `ambient` contributed a constant
vector to *every* query answer, visual ones included. `gpt_sound_acoustic` ("What do you hear?")
is one of the six trained query fields and the candidate set has 34 `sound` tags — the audio
question was supported end-to-end and simply never asked. `MicThread` in the demo is the
reference implementation (ReSpeaker `hw:0,0`, 10 s ring, mono-mixed as `_decode_audio_raw` does).

**An empty candidate category must raise, never `continue`.** The deployed
`candidates_siglip2.pt` predated the APPEARANCE tags, so the `wearing` question silently
vanished and the run looked like it had only ever asked four questions. Use
**`candidates_siglip2_v2.pt`** (1,482 tags, `appearance` 110, 2.17 MiB) + matching
`query_vectors_siglip2_v2.pt`.

**Camera is `--rotate 180`, not 90.** The CSI module is mounted upside down in the chassis. At
90° perception confidently reported `who: a person lying down (+0.71)` for a seated person — the
highest-confidence answer in the set, and wrong. A confident wrong answer from a correct model
is a **data-orientation bug**; no retraining would have touched it.

**Mic/fan (ReSpeaker 4 Mic Array, UAC1.0, `hw:0,0`, 6ch).** ch5 is a dead loopback channel; ch0
is the *loudest*, not a beamformed output — the XMOS-DSP hypothesis was **tested and falsified**,
this board has no on-board DSP. Fan noise is **broadband** (lo<300Hz / 300-4k ratio 0.12–0.19),
so high-pass filtering is useless. **Spectral subtraction made the percept measurably worse**
("an alarm beeping" 0.451 → 0.478, "glass breaking" appearing) because a stationary tonal fan
leaves narrowband musical-noise residue — `denoise=False` by default. A wrong audio percept
propagates: it entered the thinker's reasoning and the speaker's line in all 8 rounds, hence the
`above_floor()` SNR gate. The real fix is physical distance, or an NLMS adaptive filter driven by
the fan tach now that `bmo-power` exposes RPM/PWM.

**Full-stack memory, live (TTS/STT off): 4,361 MiB used, 577 MiB free**; 115–176 MiB free in
steady state during rounds. With TTS+STT resident it is 280 MiB and the camera's NVMM allocation
fails — that constraint is unchanged.
