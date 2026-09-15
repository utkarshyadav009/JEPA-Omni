"""Score the EXPLORATORY negatives x split grid.

The pre-registered pilot (cross_video, participant) FAILED. Nothing here may be presented as
that outcome -- RUN5_SPEC 2.3. This table exists to answer two questions the failure raised:
  rows    -- was the OBJECTIVE misspecified? (negative construction)
  columns -- how much of the signal is SCENE IDENTITY? (split)

Consistency gate: the (cross_video, participant) cell re-runs the failed pilot. If it does not
reproduce -1.45 vs IDENTITY, the refactor broke something and no other cell is readable.
"""
import json, os, statistics as st, sys

NEGS = ["cross_video", "mixed", "same_video"]
SPLITS = ["participant", "video"]
D = "docs/artifacts/temporal_probe"


def cell(neg, split):
    p = f"{D}/p5_exp_{neg}_{split}.json"
    if not os.path.exists(p): return None
    A = json.load(open(p))["arms"]
    seeds = sorted(int(k.split("_s")[-1]) for k in A if k.startswith("pilot_forward_s"))
    f = [A[f"pilot_forward_s{s}"]["micro_R@1"] for s in seeds]
    b = [A[f"pilot_bwdmodel_backward_s{s}"]["micro_R@1"] for s in seeds]
    sh = [A[f"pilot_shuffled_s{s}"]["micro_R@1"] for s in seeds]
    fa = [A[f"pilot_falsifier_s{s}"]["micro_R@1"] for s in seeds]
    se = lambda v: st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
    gaps = [x - y for x, y in zip(f, b)]
    return {"IDf": A["IDENTITY_forward"]["micro_R@1"], "IDb": A["IDENTITY_backward"]["micro_R@1"],
            "ridge": A["ridge_forward"]["micro_R@1"],
            "f": st.mean(f), "f_se": se(f), "gap": st.mean(gaps), "gap_se": se(gaps),
            "shuf": st.mean(sh), "fals": st.mean(fa),
            "chance": A["IDENTITY_forward"]["chance_R@1"],
            "fals_chance": A["falsifier_identity"]["chance_R@1"],
            "eff": st.mean([A[f"collapse_s{s}"]["eff_rank"] for s in seeds]),
            "eff_base": A["collapse_baseline"]["eff_rank"],
            "gallery": A["IDENTITY_forward"]["mean_gallery"],
            "nq": A["IDENTITY_forward"]["n_queries"], "seeds": len(seeds)}


if __name__ == "__main__":
    print(f"{'negatives':<12} {'split':<12} {'IDENT':>7} {'ridge':>7} {'pilot':>15} "
          f"{'vs IDENT':>14} {'fwd-bwd':>14} {'shuf/ch':>8} {'fals/ch':>8}")
    rows = {}
    for split in SPLITS:
        for neg in NEGS:
            c = cell(neg, split)
            if c is None:
                print(f"{neg:<12} {split:<12} {'-- pending --':>7}"); continue
            rows[(neg, split)] = c
            d = c["f"] - c["IDf"]
            nse = abs(d) / c["f_se"] if c["f_se"] else float("inf")
            print(f"{neg:<12} {split:<12} {c['IDf']:>7.2f} {c['ridge']:>7.2f} "
                  f"{c['f']:>8.2f} ±{c['f_se']:<5.2f} "
                  f"{d:>+7.2f} ({nse:>4.1f}σ) {c['gap']:>+7.2f} ±{c['gap_se']:<4.2f} "
                  f"{c['shuf']/c['chance']:>8.2f} {c['fals']/c['fals_chance']:>8.2f}")

    ref = rows.get(("cross_video", "participant"))
    if ref:
        d = ref["f"] - ref["IDf"]
        ok = abs(d - (-1.448)) < 0.5
        print(f"\nCONSISTENCY GATE: (cross_video, participant) reproduces {d:+.3f} "
              f"vs the pilot's -1.448 -> {'OK' if ok else '*** MISMATCH, grid not readable ***'}")

    print("\nPRE-REGISTERED GATE (applies to cross_video/participant ONLY, and it FAILED):")
    print("  1 beat IDENTITY >= +2.0 and >= 3xSE     2 fwd-bwd >= +2.0 and >= 3xSE")
    for (neg, split), c in rows.items():
        d = c["f"] - c["IDf"]
        tag = "PRE-REGISTERED" if (neg, split) == ("cross_video", "participant") else "exploratory"
        c1 = d >= 2.0 and d >= 3 * c["f_se"]
        c2 = c["gap"] >= 2.0 and c["gap"] >= 3 * c["gap_se"]
        print(f"  [{tag:<14}] {neg}/{split}: beat-identity {'PASS' if c1 else 'FAIL'}"
              f"  fwd-bwd {'PASS' if c2 else 'FAIL'}")

    print("\nSCENE-IDENTITY COLUMN TEST (participant minus video, per negative scheme):")
    for neg in NEGS:
        a, b = rows.get((neg, "participant")), rows.get((neg, "video"))
        if a and b:
            print(f"  {neg:<12} IDENTITY {a['IDf']:.2f} vs {b['IDf']:.2f} (Δ {a['IDf']-b['IDf']:+.2f})"
                  f"   pilot {a['f']:.2f} vs {b['f']:.2f} (Δ {a['f']-b['f']:+.2f})")
    if rows:
        any_c = next(iter(rows.values()))
        print(f"\ngallery {any_c['gallery']}, chance {any_c['chance']}%, "
              f"{any_c['nq']} queries, frozen-W eff_rank {any_c['eff_base']}")
