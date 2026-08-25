# METHODOLOGY_FALSIFIERS.md

Six specific issues this project's falsifier discipline caught before
they reached a reported result or a shipped decision. Each was found by
checking a claim against direct evidence (a re-run, a source read, a
byte-level comparison) rather than accepting a plausible description —
the pattern this project runs on throughout. One paragraph each.

**1. World-State input to the turn-taking decision head was a
presence/scale slot, not an information channel.** The original 3-class
decision head took `[World-State; speech-activity]` as input, on the
assumption that visual context helps predict when to speak. A
six-condition falsifier (`checkpoints/m4_decision_head_3class_bothpresent/
A1_PROVENANCE.txt`) tested it directly: swapping in a constant
dataset-mean World-State vector (carrying no per-example information at
all) matched or beat the real, correctly-paired World-State. A
speech-only head with the World-State branch removed entirely scored
95.00% vs. the World-State-consuming head's 93.67% — numerically better,
not worse. The head that shipped (locked in `freeze-submission-v1`) is
speech-only as a direct, measured consequence, not a simplification of
convenience.

**2. `train_m2.py`'s ambient-length cap ran after the tensor was already
on GPU.** `_cap_ambient_len` was called after `.to(device)` at all three
of its call sites, so outlier-length clips (AudioSet runs up to 244s vs.
VGGSound/Ego4D's fixed ~10s) could spike GPU memory before truncation
ever applied. This was RUN-3's proximate OOM cause, and the reason
negatives and the ambient cap were both cut under pressure — which is
what made RUN-3's result unattributable to AudioSet alone (see
`NEGATIVE_RESULTS.md`). Root-caused by reading the call order directly in
source, not by re-running and hoping; fixed by moving the cap before
device transfer and adding a `--max-ambient-t` override, though per the
freeze no retrain was run with the fix.

**3. A checkpoint-scoring script silently reused the wrong checkpoint's
cached embeddings.** `phase_ego4d_heldout_gallery_score_v2.py`'s
`--cache-path` argument defaults to a fixed path regardless of which
`--m2-ckpt` is passed. Scoring step19000 and step20000 back-to-back
without overriding `--cache-path` produced byte-identical output for two
different checkpoints — caught because identical output for two different
models is itself implausible, not because a number looked wrong. Fixed by
passing a distinct `--cache-path` per checkpoint, after which step20000
showed genuinely different (and better) Ego4D numbers than step19000.

**4. The streaming loop's World-State construction diverged from training's
in three separate ways, independently discovered.** `models/
world_state_builder.py` was built specifically to close this: (1) no
spatial pooling — `VisionEncoder.encode()` returns 8192 raw tokens,
confirmed by direct execution, while training cached 512 pooled tokens;
the streaming loop had been feeding the raw 8192. (2) Vision timestamps
used a linspace ramp (512 or 8192 distinct values) where training's own
`_ts_to_tdm_bins` produces a staircase (32 distinct group-timestamps, each
shared by 16 spatial tokens). (3) Ambient/WavJEPA features were absent
from the streaming World-State entirely — the rolling audio buffer fed
Whisper (for the decision head's speech-activity feature) but never
WavJEPA (M2's actual audio encoder), so World-State had a `vision` key and
no `ambient` key. All three were caught by direct source comparison
against `scripts/extract_features_av.py` and `data/av_cached_dataset.py`,
not by a metric looking off.

**5. Two Jetson full-stack memory figures (644MiB, 3051MiB) were
retracted for stack-composition mismatches, not superseded by a better
measurement of the same thing.** Each prior figure measured a materially
different stack than the "real" deployment target (missing Qwen2.5-1.5B
and its generation-time KV-cache growth, in one case; a different pooling
path, in another). The 854MiB figure now in `RESULTS_TABLE.md` is the
first one confirmed to include the corrected pooled World-State path
together with a real generation pass through the actual LLM — documented
in `checkpoints/m5_jetson/PHASE0_CLARIFICATION_PROVENANCE.txt` specifically
so the retraction history isn't lost.

**6. The "~51k clips → ~40% R@1" historical scale-hypothesis figure did
not exist as a logged data point.** Asked to verify it before building a
scaling plan on top of it, a direct search of `logs/` found no single run
matching that description. What DID exist was a genuine matched-step
comparison — at the identical step count (6000), 51,508 clips gave
R@1=33.46%/34.24% and 199,007 clips gave R@1=44.27%/43.95% — which
confirmed the underlying scale hypothesis while correcting the specific
number attached to it. The scaling plan proceeded on the verified
comparison, not the unverified one, and the correction was reported before
further runs were built on the wrong premise.
