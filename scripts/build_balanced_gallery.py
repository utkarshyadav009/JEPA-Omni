"""Build the balanced 5-per-class VGGSound retrieval gallery (AV-JEPA construction).

AV-JEPA's retrieval galleries are a BALANCED 5-PER-CLASS subset of the official test
split (VGGSound N=1,545 = 309x5). Our existing data/vggsound_eval_1545.txt is NOT that:
it is an unstratified random draw that happens to total 1,545, covering 307/309 classes
with 1-12 clips each (mean 4.91 same-class distractors vs balanced's exactly 4.00).
This script builds the real thing so the comparison is protocol-matched.

Selection is seeded and sorted, so the file is reproducible byte-for-byte.
Clips must have video on disk, else they cannot be encoded; 15,341/15,446 (99.3%) do,
and all 309 classes retain >=5 candidates, so N=1,545 is reached without relaxation.
"""
import json, os, random, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = "/home/utkarsh/raid2-data/vggsound_raw/extracted/video"
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "test.csv")
OUT = os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545_balanced.txt")
PROV = os.path.join(PROJECT_ROOT, "data", "vggsound_eval_1545_balanced.provenance.json")
SEED = 0
PER_CLASS = 5


def main() -> None:
    have = {f[:-4] for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")}
    by_class = {}
    for ln in open(TEST_CSV):
        p = ln.rstrip("\n").split(",", 1)
        if len(p) != 2:
            continue
        cid = p[0].replace(".mp4", "")
        if cid in have:
            by_class.setdefault(p[1].strip(), []).append(cid)

    rng = random.Random(SEED)
    picked, shortfall = [], {}
    for cls in sorted(by_class):
        cands = sorted(by_class[cls])          # sort first so the seed is meaningful
        if len(cands) < PER_CLASS:
            shortfall[cls] = len(cands)
            picked.extend(cands)
            continue
        picked.extend(rng.sample(cands, PER_CLASS))
    picked.sort()

    assert not shortfall, f"classes short of {PER_CLASS}: {shortfall}"
    assert len(picked) == PER_CLASS * len(by_class), \
        f"expected {PER_CLASS * len(by_class)}, got {len(picked)}"
    assert len(set(picked)) == len(picked), "duplicate clip in gallery"

    with open(OUT, "w") as f:
        f.write("\n".join(picked) + "\n")
    prov = {
        "construction": "balanced %d-per-class from the OFFICIAL VGGSound test split "
                        "(AV-JEPA gallery construction)" % PER_CLASS,
        "seed": SEED, "per_class": PER_CLASS,
        "n_classes": len(by_class), "n_clips": len(picked),
        "source_split": "data/test.csv (official VGGSound test, 15,446 clips, 309 classes)",
        "video_dir": VIDEO_DIR,
        "candidates_required_video_on_disk": True,
        "test_clips_with_video": len(have & {c for v in by_class.values() for c in v}),
        "selection": "sorted(candidates) then random.Random(0).sample(...); output sorted",
    }
    json.dump(prov, open(PROV, "w"), indent=2)
    print("wrote %s  (%d clips, %d classes x %d)" % (OUT, len(picked), len(by_class), PER_CLASS))
    print("wrote %s" % PROV)


if __name__ == "__main__":
    main()
