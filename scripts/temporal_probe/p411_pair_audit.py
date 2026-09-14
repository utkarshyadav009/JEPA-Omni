"""P4.11 step 2 -- audit the temporal pairs BEFORE any RUN-5 training.

The Ego4D cache already failed this test once (it was assumed to support Delta<10s and does
not), which is why this runs before code is written, not after a run disappoints.

Reported: valid files, valid consecutive pairs, pairs at each Delta, the overlap/receptive-field
relationship, train/eval FILE separation, and how many samples survive exclusions.

RECEPTIVE FIELD (measured, p4_receptive_field.json):
  V-JEPA2   full 10 s  -> a pair separated by <10 s SHARES VIDEO INPUT. Not a prediction pair.
  WavJEPA   ~2.25 s median / 4.25 s max -> audio alone would tolerate Delta >= ~4.25 s.
So Delta>=10 s is the floor for a FUSED target, and Delta>=5 s is reported separately as the
audio-only branch the measurement opened up.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import torch

WS_DIR = "/home/utkarsh/JEPA-Omni/data/epic_kitchens_ws"
ANN = "/mnt/Raid-Storage-2/utkarsh-data/epic_kitchens/annotations"
WINDOW_S = 10.0
DELTAS = [5, 10, 20, 30, 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-dir", default=WS_DIR)
    ap.add_argument("--out", default="docs/artifacts/temporal_probe/p411_pair_audit.json")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.ws_dir) if f.endswith(".pt"))
    print(f"[audit] {len(files)} extracted videos in {a.ws_dir}")

    # official EK-100 splits, by VIDEO, so train/eval separation is file-level not clip-level
    def vids_of(csvname):
        p = os.path.join(ANN, csvname)
        return {r["video_id"] for r in csv.DictReader(open(p))} if os.path.exists(p) else set()
    train_v, val_v = vids_of("EPIC_100_train.csv"), vids_of("EPIC_100_validation.csv")

    per_file, bad = {}, []
    for fn in files:
        try:
            d = torch.load(os.path.join(a.ws_dir, fn), map_location="cpu", weights_only=True)
        except Exception as e:
            bad.append((fn, repr(e))); continue
        W, st = d["world_state"].float(), d["start_s"]
        ok_finite = bool(torch.isfinite(W).all())
        # consecutive == start times exactly stride apart; gaps mean windows were dropped
        stride = float(d["stride_s"])
        diffs = (st[1:] - st[:-1])
        n_consec = int((diffs - stride).abs().lt(1e-3).sum())
        per_file[fn[:-3]] = {"n_windows": int(W.shape[0]), "dur_s": float(d.get("duration_s", 0.0)),
                             "stride_s": stride, "n_consecutive": n_consec,
                             "finite": ok_finite, "attempted": int(d.get("n_windows_attempted", 0)),
                             "starts": st}
    print(f"[audit] loaded {len(per_file)} ok, {len(bad)} unreadable")
    for fn, e in bad[:5]:
        print(f"  UNREADABLE {fn}: {e}")

    n_nonfinite = sum(1 for v in per_file.values() if not v["finite"])
    drop = sum(v["attempted"] - v["n_windows"] for v in per_file.values())
    tot_att = sum(v["attempted"] for v in per_file.values())
    tot_win = sum(v["n_windows"] for v in per_file.values())
    print(f"[audit] windows: {tot_win:,} kept of {tot_att:,} attempted "
          f"({100*drop/max(tot_att,1):.2f}% dropped), non-finite files: {n_nonfinite}")

    # ---- pairs at each Delta ------------------------------------------------------
    res_delta = {}
    for D in DELTAS:
        tot, files_with = 0, 0
        for vid, v in per_file.items():
            st = v["starts"]
            s = set(round(float(x), 3) for x in st.tolist())
            n = sum(1 for x in st.tolist() if round(float(x) + D, 3) in s)
            tot += n
            if n > 0: files_with += 1
        overlap = max(0.0, WINDOW_S - D)
        res_delta[D] = {"pairs": tot, "files": files_with,
                        "input_overlap_s": overlap,
                        "fused_valid": D >= WINDOW_S,
                        "audio_only_valid": D >= 4.25}
        tag = "VALID (fused)" if D >= WINDOW_S else ("audio-only branch" if D >= 4.25 else "INVALID")
        print(f"[audit] Delta={D:>2}s  pairs={tot:>9,}  files={files_with:>4}  "
              f"video input overlap={overlap:.1f}s  -> {tag}")

    # ---- train/eval FILE separation ------------------------------------------------
    ext = set(per_file)
    in_train = ext & train_v
    in_val = ext & val_v
    parts = defaultdict(list)
    for v in ext: parts[v.split("_")[0]].append(v)
    print(f"[audit] official split coverage: {len(in_train)} train videos, {len(in_val)} val videos, "
          f"{len(ext - train_v - val_v)} in neither")
    print(f"[audit] participants: {len(parts)}  (a participant-held-out split is available)")
    leak = in_train & in_val
    print(f"[audit] train/val video overlap: {len(leak)}  -> {'LEAK' if leak else 'CLEAN'}")

    # usable pairs under a participant-held-out split at the fused floor
    ps = sorted(parts)
    hold = set(ps[::5])                       # 20% of participants held out
    tr = [v for v in ext if v.split("_")[0] not in hold]
    ev = [v for v in ext if v.split("_")[0] in hold]
    def pairs_for(vs, D):
        t = 0
        for vid in vs:
            st = per_file[vid]["starts"]
            s = set(round(float(x), 3) for x in st.tolist())
            t += sum(1 for x in st.tolist() if round(float(x) + D, 3) in s)
        return t
    print(f"[audit] participant-held-out split: {len(tr)} train videos ({len(ps)-len(hold)} participants), "
          f"{len(ev)} eval videos ({len(hold)} participants)")
    for D in (10, 20):
        print(f"[audit]   usable pairs at Delta={D}s -> train {pairs_for(tr,D):,}  eval {pairs_for(ev,D):,}")

    out = {"n_files": len(per_file), "n_unreadable": len(bad), "n_nonfinite": n_nonfinite,
           "windows_kept": tot_win, "windows_attempted": tot_att,
           "delta": {str(k): v for k, v in res_delta.items()},
           "participants": len(parts), "held_out_participants": sorted(hold),
           "train_videos": len(tr), "eval_videos": len(ev),
           "usable_pairs_delta10": {"train": pairs_for(tr,10), "eval": pairs_for(ev,10)},
           "usable_pairs_delta20": {"train": pairs_for(tr,20), "eval": pairs_for(ev,20)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out,"w"), indent=1)
    print(f"[audit] wrote {a.out}")


if __name__ == "__main__":
    main()
