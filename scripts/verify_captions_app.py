"""scripts/verify_captions_app.py — minimal human-verification app for the
Track B AV-caption pipeline. Plays each clip (video WITH audio) and shows the
VGGSound ground-truth label alongside the generated caption(s), for tagging.

Supports three record formats (auto-detected from the input jsonl):
  - single-caption: {"clip_id", "vggsound_label", "caption"}
  - multi-granularity: {"clip_id", "vggsound_label", gpt_action_brief,
    gpt_action_detailed, gpt_summary_brief, gpt_summary_detailed,
    gpt_sound_acoustic} -- shows all five fields; gpt_sound_acoustic gets its
    own quality tag since it's the non-negotiable audio-grounded field.
  - congruence-filter: {"clip_id", "vggsound_label", "category", "reason"} --
    shows the filter's REAL_MATCH/REAL_MISMATCH/SYNTHETIC/UNCERTAIN call and
    lets a human mark AGREE/DISAGREE, to validate the filter itself.

Results save to disk as you go. This is the gate before full-corpus caption
generation.

Usage:
    streamlit run scripts/verify_captions_app.py -- \\
        --captions scripts/qwen_omni_captions_test50.jsonl \\
        --video-dir /home/utkarsh/data/vggsound \\
        --out scripts/caption_review_results.csv \\
        --sample-size 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ["good", "sound-blind", "hallucinated", "repetition"]
SOUND_TAGS = ["good", "generic/templated", "wrong", "sound-blind"]
MG_FIELDS = ["gpt_action_brief", "gpt_action_detailed", "gpt_summary_brief",
             "gpt_summary_detailed", "gpt_sound_acoustic"]
VERDICT_TAGS = ["agree", "disagree"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default=os.path.join(PROJECT_ROOT, "scripts", "qwen_omni_captions_test50.jsonl"))
    p.add_argument("--video-dir", default="/home/utkarsh/data/vggsound")
    p.add_argument("--out", default=None,
                   help="Defaults to caption_review_results.csv (single) or "
                        "mg_pilot_review_results.csv (multi-granularity).")
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # Streamlit's bootstrap already strips the "--" separator and sets
    # sys.argv = [script_path, *user_args] -- it never appears in sys.argv
    # here. (Confirmed via streamlit.web.bootstrap._fix_sys_argv source.)
    # Handle both cases defensively: a literal "--" would only show up if
    # this script is ever invoked directly (non-streamlit) with one.
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


@st.cache_data
def load_records(captions_path: str, sample_size: int, seed: int):
    with open(captions_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    random.Random(seed).shuffle(recs)
    recs = recs[:sample_size]
    if recs and "gpt_action_brief" in recs[0]:
        fmt = "multi"
    elif recs and "category" in recs[0]:
        fmt = "congruence"
    else:
        fmt = "single"
    return recs, fmt


def load_existing_results(out_path: str) -> dict:
    results = {}
    if os.path.isfile(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                results[row["clip_id"]] = row
    return results


def save_results(out_path: str, results: dict, fieldnames: list) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.values():
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="AV Caption Review", layout="centered")

    records, fmt = load_records(args.captions, args.sample_size, args.seed)
    default_out = {"multi": "mg_pilot_review_results.csv",
                   "congruence": "congruence_filter_review_results.csv",
                   "single": "caption_review_results.csv"}[fmt]
    out_path = args.out or os.path.join(PROJECT_ROOT, "scripts", default_out)
    fieldnames = {
        "multi": ["clip_id", "vggsound_label"] + MG_FIELDS + ["overall_tag", "sound_tag", "note"],
        "congruence": ["clip_id", "vggsound_label", "category", "reason", "verdict", "note"],
        "single": ["clip_id", "vggsound_label", "caption", "tag", "note"],
    }[fmt]

    if "out_path" not in st.session_state or st.session_state.out_path != out_path:
        st.session_state.results = load_existing_results(out_path)
        st.session_state.out_path = out_path
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    results = st.session_state.results
    n = len(records)
    idx = st.session_state.idx
    n_reviewed = len(results)

    title_suffix = {"multi": " — multi-granularity", "congruence": " — congruence filter", "single": ""}[fmt]
    st.title("AV Caption Review" + title_suffix)
    st.progress(n_reviewed / n if n else 0.0)
    st.caption(f"{n_reviewed}/{n} tagged so far  (saving to `{out_path}`)")

    if idx >= n:
        st.success(f"Done — reached the end of the {n}-clip sample. "
                   f"{n_reviewed}/{n} tagged. Results saved to `{out_path}`.")
        if st.button("Review again from the start"):
            st.session_state.idx = 0
            st.rerun()
        return

    rec = records[idx]
    clip_id = rec["clip_id"]
    video_path = os.path.join(args.video_dir, clip_id + ".mp4")

    st.write(f"**Clip {idx + 1}/{n}**: `{clip_id}`")

    col1, col2 = st.columns([1, 1])
    with col1:
        if os.path.isfile(video_path):
            st.video(video_path)
        else:
            st.warning(f"Video file not found: {video_path}")
        st.markdown(f"**VGGSound label:** {rec['vggsound_label']}")

    existing = results.get(clip_id)

    if fmt == "congruence":
        with col2:
            st.markdown(f"**Filter category:** {rec.get('category')}")
            st.markdown("**Filter reason:**")
            st.info(rec.get("reason"))

        default_verdict = existing["verdict"] if existing else VERDICT_TAGS[0]
        default_note = existing["note"] if existing else ""

        verdict = st.radio("Is the filter's call correct?", VERDICT_TAGS,
                            index=VERDICT_TAGS.index(default_verdict) if default_verdict in VERDICT_TAGS else 0,
                            horizontal=True, key=f"verdict_{clip_id}")
        note = st.text_input("Note (optional, e.g. what's actually audible):",
                              value=default_note, key=f"note_{clip_id}")
    elif fmt == "multi":
        with col2:
            st.markdown(f"**gpt_action_brief:** {rec.get('gpt_action_brief')}")
            st.markdown(f"**gpt_action_detailed:** {rec.get('gpt_action_detailed')}")
            st.markdown(f"**gpt_summary_brief:** {rec.get('gpt_summary_brief')}")
            st.markdown(f"**gpt_summary_detailed:** {rec.get('gpt_summary_detailed')}")
            st.markdown("**gpt_sound_acoustic** (audio-grounded, non-negotiable):")
            st.info(rec.get("gpt_sound_acoustic"))

        default_overall = existing["overall_tag"] if existing else TAGS[0]
        default_sound = existing["sound_tag"] if existing else SOUND_TAGS[0]
        default_note = existing["note"] if existing else ""

        overall_tag = st.radio("Overall tag (visual fields):", TAGS,
                                index=TAGS.index(default_overall) if default_overall in TAGS else 0,
                                horizontal=True, key=f"overall_{clip_id}")
        sound_tag = st.radio("gpt_sound_acoustic tag:", SOUND_TAGS,
                              index=SOUND_TAGS.index(default_sound) if default_sound in SOUND_TAGS else 0,
                              horizontal=True, key=f"sound_{clip_id}")
        note = st.text_input("Note (optional):", value=default_note, key=f"note_{clip_id}")
    else:
        with col2:
            st.markdown(f"**Generated caption:**")
            st.info(rec["caption"])

        default_tag = existing["tag"] if existing else TAGS[0]
        default_note = existing["note"] if existing else ""

        tag = st.radio("Tag this caption:", TAGS,
                        index=TAGS.index(default_tag) if default_tag in TAGS else 0,
                        horizontal=True, key=f"tag_{clip_id}")
        note = st.text_input("Note (optional):", value=default_note, key=f"note_{clip_id}")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        if st.button("← Previous", disabled=(idx == 0)):
            st.session_state.idx -= 1
            st.rerun()
    with nav_col2:
        if st.button("Save & Next →", type="primary"):
            if fmt == "congruence":
                row = {"clip_id": clip_id, "vggsound_label": rec["vggsound_label"],
                       "category": rec.get("category"), "reason": rec.get("reason"),
                       "verdict": verdict, "note": note}
            elif fmt == "multi":
                row = {"clip_id": clip_id, "vggsound_label": rec["vggsound_label"],
                       "overall_tag": overall_tag, "sound_tag": sound_tag, "note": note}
                for field in MG_FIELDS:
                    row[field] = rec.get(field)
            else:
                row = {"clip_id": clip_id, "vggsound_label": rec["vggsound_label"],
                       "caption": rec["caption"], "tag": tag, "note": note}
            results[clip_id] = row
            save_results(out_path, results, fieldnames)
            st.session_state.idx += 1
            st.rerun()
    with nav_col3:
        if st.button("Skip (no tag) →"):
            st.session_state.idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
