"""scripts/m4_interruption_mining.py — mine REAL overlapping-speech
(interruption) events from EasyCom transcripts, and classify what the
interrupted speaker does next: RESUME (speaks again soon, similar content),
RE-PLAN (speaks again soon, different content), ABANDON (doesn't speak
again in the window). This is what answers "which EasyCom supports
supervising" for the post-halt interruption-policy state machine.

Definitions (real timing + text, no audio needed -- transcripts alone are
enough for this):
  - Interruption event: participant P's segment [s_P, e_P] and a DIFFERENT
    participant Q's segment [s_Q, e_Q] with s_P < s_Q < e_P (Q starts
    talking while P is still mid-utterance) -- P is genuinely interrupted.
  - Outcome, looked up in the SAME chunk (60s window) only:
      * no further P segment starting after e_Q within RESUME_WINDOW_SEC
        -> ABANDON
      * a further P segment exists -> compute word-overlap Jaccard
        similarity between P's interrupted segment text and P's next
        segment text; >= SIM_THRESHOLD -> RESUME (same utterance
        essentially continued/repeated), else -> RE-PLAN (new content)

HONEST CAVEAT (stated up front, not discovered after the fact): EasyCom has
no ground-truth intent label for "same utterance continued" vs "new
utterance" -- word-overlap similarity is a cheap, defensible PROXY, not a
verified label. Treat the RESUME/RE-PLAN split as approximate; the
ABANDON-vs-continues split (pure timing, no text heuristic) is on much
firmer ground.

Usage:
    python scripts/m4_interruption_mining.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

from data.m4_speech_dataset import EASYCOM_ROOT, VIDEO_FPS, _read_json_robust
from data.m4_easycom_turntaking import is_backchannel

RESUME_WINDOW_SEC = 10.0
SIM_THRESHOLD = 0.3
_WORD_RE = re.compile(r"[a-z']+")


def words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    st_root = os.path.join(EASYCOM_ROOT, "Speech_Transcriptions")
    sessions = sorted(glob.glob(os.path.join(st_root, "Session_*")), key=lambda p: int(p.rsplit("_", 1)[1]))

    outcomes = Counter()
    examples = {"RESUME": [], "RE-PLAN": [], "ABANDON": []}
    interrupter_durations = {"RESUME": [], "RE-PLAN": [], "ABANDON": []}
    n_events = 0
    n_backchannel_only_overlaps = 0
    n_chunks_with_overlap = 0

    for sess_dir in sessions:
        sess_id = int(sess_dir.rsplit("_", 1)[1])
        for cf in sorted(glob.glob(os.path.join(sess_dir, "*.json"))):
            entries = _read_json_robust(cf)
            segs = []
            for e in entries:
                pid = e.get("Participant_ID")
                text = (e.get("Transcription") or "").strip()
                if pid is None or not text:
                    continue
                s = e["Start_Frame"] / VIDEO_FPS
                en = e["End_Frame"] / VIDEO_FPS
                if en <= s:
                    continue
                segs.append({"pid": pid, "start": s, "end": en, "text": text})
            segs.sort(key=lambda x: x["start"])
            if len(segs) < 2:
                continue

            chunk_has_overlap = False
            for i, p_seg in enumerate(segs):
                # find the EARLIEST different-participant segment that starts
                # strictly inside p_seg's span -> a genuine interruption of p_seg
                # (backchannel overlaps excluded -- those are handled by the
                # 3-class decision head, not the interruption-policy state
                # machine; counting "Awesome." as an "interrupter" would
                # conflate the two mechanisms)
                interrupter = None
                saw_backchannel_overlap = False
                for q_seg in segs:
                    if q_seg["pid"] == p_seg["pid"]:
                        continue
                    if p_seg["start"] < q_seg["start"] < p_seg["end"]:
                        if is_backchannel(q_seg["text"]):
                            saw_backchannel_overlap = True
                            continue
                        if interrupter is None or q_seg["start"] < interrupter["start"]:
                            interrupter = q_seg
                if interrupter is None:
                    if saw_backchannel_overlap:
                        n_backchannel_only_overlaps += 1
                    continue

                n_events += 1
                chunk_has_overlap = True

                # find P's next segment after the interrupter ends, within window
                next_p = None
                for cand in segs:
                    if cand["pid"] != p_seg["pid"]:
                        continue
                    if cand["start"] <= interrupter["end"]:
                        continue
                    if cand["start"] - interrupter["end"] <= RESUME_WINDOW_SEC:
                        if next_p is None or cand["start"] < next_p["start"]:
                            next_p = cand

                if next_p is None:
                    outcome = "ABANDON"
                else:
                    sim = jaccard(words(p_seg["text"]), words(next_p["text"]))
                    outcome = "RESUME" if sim >= SIM_THRESHOLD else "RE-PLAN"

                outcomes[outcome] += 1
                interrupter_durations[outcome].append(interrupter["end"] - interrupter["start"])
                if len(examples[outcome]) < 4:
                    ex = {"session": sess_id, "interrupted_text": p_seg["text"],
                          "interrupter_text": interrupter["text"]}
                    if next_p is not None:
                        ex["resumed_text"] = next_p["text"]
                    examples[outcome].append(ex)

            if chunk_has_overlap:
                n_chunks_with_overlap += 1

    total = sum(outcomes.values())
    print(f"[interrupt-mine] {n_events} genuine (non-backchannel) interruption events found "
          f"across {n_chunks_with_overlap} chunks ({n_backchannel_only_overlaps} additional "
          f"overlaps were backchannel-only and excluded -- those are handled by the 3-class "
          f"decision head, not this state machine). All 12 sessions, no train/test split -- "
          f"this is a data-characterization pass, not a trained model.")
    print(f"[interrupt-mine] outcome distribution (n={total}):")
    import statistics
    for k in ["RESUME", "RE-PLAN", "ABANDON"]:
        durs = interrupter_durations[k]
        med = statistics.median(durs) if durs else float("nan")
        print(f"  {k:10s} {outcomes[k]:5d}  ({outcomes[k]/max(1,total):.1%})  "
              f"median interrupter duration={med:.2f}s")

    for k in ["RESUME", "RE-PLAN", "ABANDON"]:
        print(f"\n[interrupt-mine] example {k} cases:")
        for ex in examples[k]:
            print(f"  session={ex['session']}  interrupted=\"{ex['interrupted_text']}\"  "
                  f"interrupter=\"{ex['interrupter_text']}\"  resumed={ex.get('resumed_text', '<none>')!r}")

    out = {"n_events": n_events, "n_backchannel_only_overlaps_excluded": n_backchannel_only_overlaps,
           "n_chunks_with_overlap": n_chunks_with_overlap,
           "outcome_counts": dict(outcomes),
           "outcome_fractions": {k: outcomes[k] / max(1, total) for k in outcomes},
           "median_interrupter_duration_sec": {k: (statistics.median(v) if v else None)
                                                for k, v in interrupter_durations.items()},
           "resume_window_sec": RESUME_WINDOW_SEC, "sim_threshold": SIM_THRESHOLD,
           "examples": examples}
    os.makedirs("checkpoints/m4_interruption_policy", exist_ok=True)
    with open("checkpoints/m4_interruption_policy/interruption_mining_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[interrupt-mine] wrote checkpoints/m4_interruption_policy/interruption_mining_results.json")


if __name__ == "__main__":
    main()
