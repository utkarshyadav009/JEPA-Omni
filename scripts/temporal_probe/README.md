# `scripts/temporal_probe/` — eval-time probes, audits and extraction

Everything here is **eval-time**. Nothing in this directory trains a model, and nothing writes
to a checkpoint. The one exception is `extract_epic_kitchens.py`, which *deletes source video* —
read its two-gate rule before running it.

## Naming: `world_state` is historical

Identifiers across this repo — `build_world_state_features`, `encode_world_state`,
`world_state_eff_rank`, `--mode world_state`, the `world_state` key in saved tensors — use the
name the object had when it was written.

**That name was tested and retired.** The object is an **audio-visual scene representation**.
It has no recurrence, it was never trained with a predictive term (`lam_pred = 0.0`), and
predicting its future is no easier than predicting its past (forward−backward gap −0.04 ± 0.15
for RUN-4, against a pre-registered threshold of ≥2.0). **It is not a world model.**

The identifiers are deliberately **not** renamed: they are load-bearing across a working
pipeline and saved artifacts, and renaming them would risk a real regression for zero benefit.
Prose, documents and paper text use "scene representation". See
`docs/FORWARD_INFORMATION_PROBE.md` and `docs/TEMPORAL_STRUCTURE_PROBE.md`.

## Never use `pgrep -f` in this repo

`pgrep -f` / `pkill -f` match against the **full command line**, which includes the pattern
itself when the search runs from a shell whose command line contains it. A liveness check or
cleanup written that way **matches and kills its own shell**.

This happened **four separate times** during this work. Consequences included a shell killed
mid-command (exit 143/144), a 1.68 GB partial download left behind that helped drive `/` to
100% and nearly killed RUN-4, and — worst — a **disk guard that died silently** because its own
liveness check self-matched, after which the Epic-Kitchens download filled md1 to 575 MB free
with nothing watching.

Use instead:

```bash
# match the executable name only, then confirm identity by inspecting /proc
for p in $(pgrep -x python); do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q "my_script.py" && echo "found $p"
done

# or, for a job you launched, record the PID and use kill -0
kill -0 "$PID" 2>/dev/null && echo RUNNING
```

**`$!` is not enough either.** After `nohup setsid cmd &`, `$!` is the PID of the *wrapper*;
`setsid` forks and the real process gets a different PID. The wrapper exits immediately, so a
supervisor watching `$!` reports `VANISHED` while the job runs happily. Resolve the real PID
from `/proc` after launch — this bit us once, on the Epic-Kitchens supervisor.

## Monitoring rules (P4.8)

Any long-running job gets a monitor **attached at launch**, not added afterwards. The monitor
must distinguish `RUNNING` / `COMPLETED` / `FAILED` / `VANISHED` — a process that disappears
without its completion marker must never read as still running — must survive the launching
session (`nohup` + `setsid`), and must write its own heartbeat so that **its own** death is
detectable. Exit code 0 alone is not success: require the expected output markers *and* validate
the outputs are complete and readable.

## Contents

| script | purpose |
|---|---|
| `bin_scramble_eval.py` | Phase 0/1 harness, arms A0–A10, `--fix-padding`. Verifies the RUN-2 checkpoint sha256 and aborts on mismatch. |
| `padded_eval.py` | masked eval helpers, kept out of `train_m2.py` on purpose |
| `pad_sweep.py` | uniform vs random per-clip pad controls — the decisive padding-leak control |
| `pre_pool_leak.py` | where in the stack the leak enters |
| `congruence_mask_ab.py` | ears-following A/B under the congruence mask |
| `qp_rescore_bs1.py` | query-predictor re-scoring at batch size 1 (E-6) |
| `phase2_persistence.py`, `phase3_forward.py` | Phases 2/3 at Δ ≥ 10 s |
| `p30_select_checkpoint.py` | post-hoc RUN-4 checkpoint selection by held-out R@1 (E-14) |
| `p32_stages_abcd.py`, `p32_forward_info.py` | the forward-information probe |
| `extract_scene_subset.py`, `stream_extract.py` | feature extraction helpers |
| `extract_epic_kitchens.py` | **deletes source video** — see below |

## `extract_epic_kitchens.py` — the two-gate deletion rule

One decode pass per video emits **both** outputs: scene representations for every window at
`--stride-s`, and full token features for every `--feat-every`-th window in the existing cache
layout.

A source `.MP4` is deleted **only** after *both* outputs are written, fsynced, reloaded from
disk, and checked for the expected window count. That decision lives behind a single
`both_verified` flag set in exactly one place. `--no-features` without `--keep-video` is
**refused**, because verifying one output and deleting the source before checking the other is
precisely the failure this rule exists to prevent.

Two further guards: a per-filesystem free-space floor (`--floor-gib`, default 25) checked before
each video, and an **equivalence gate** — window 0 is built both by this script's re-derivation
and by the shared `build_world_state_features`, and must match bit-for-bit, or the run aborts
before anything is written or deleted.

Deletion is recoverable in principle: `epic_kitchens/urls.txt` and `download.sh` are retained.
