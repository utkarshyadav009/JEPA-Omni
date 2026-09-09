"""Run the IDENTICAL AudioSet probe on baseline audio-tower features (Part B5).

Imports run()/metrics() and every hyperparameter from scripts/audioset_probe.py rather than
re-declaring them, so a baseline row cannot silently differ from ours in optimiser, LR,
epochs, batch, seed or metric. Baseline feature files are produced by a separate extraction
step; this script only probes them.
"""
import argparse, csv, glob, json, os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import audioset_probe as AP


def onehot(labels, idx):
    Y = torch.zeros(len(labels), AP.NUM_CLASSES)
    for i, L in enumerate(labels):
        for m in L:
            if m in idx:
                Y[i, idx[m]] = 1.0
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="/mnt/Raid-Storage-2/utkarsh-data/audioset_feats")
    ap.add_argument("--clean-eval-ids", default="docs/artifacts/audioset_eval_clean_ids.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    idx = {r["mid"]: int(r["index"])
           for r in csv.DictReader(open("/tmp/class_labels_indices.csv"))}
    assert len(idx) == AP.NUM_CLASSES, "class map is not the canonical 527"
    clean = set(json.load(open(a.clean_eval_ids)))

    models = sorted({os.path.basename(p).rsplit("_bal_train.pt", 1)[0]
                     for p in glob.glob(os.path.join(a.feat_dir, "*_bal_train.pt"))})
    log("[probe-base] baseline models found: %s" % (models or "NONE"))
    res = []
    for m in models:
        tp = os.path.join(a.feat_dir, "%s_bal_train.pt" % m)
        ep = os.path.join(a.feat_dir, "%s_eval.pt" % m)
        if not os.path.exists(ep):
            log("  %-14s SKIP (no eval features)" % m); continue
        tr = torch.load(tp, map_location="cpu", weights_only=False)
        te = torch.load(ep, map_location="cpu", weights_only=False)
        keep = [i for i, c in enumerate(te["ids"]) if c in clean]
        Ytr = onehot(tr["labels"], idx)
        Yte = onehot([te["labels"][i] for i in keep], idx)
        log("  %s  train=%d eval=%d(kept from %d)  dim=%s  tokens=%s"
            % (m, len(tr["ids"]), len(keep), len(te["ids"]), tr.get("dim"),
               "yes" if "feat_tokens" in tr else "no"))
        # THREE variants, because "linear" is not one thing across models.
        #   linear_native  : the tower's own pooled output, exactly as its retrieval script
        #                    returns it. For CAV-MAE et al. that is an L2-NORMALISED CLS
        #                    retrieval embedding (measured norm 1.0000), which is a different
        #                    object from our raw token mean (norm 3.19) -- so this column is
        #                    NOT comparable to ours and is reported for completeness only.
        #   linear_tokmean : mean over the token sequence, matching how our own ambient_mean
        #                    is built. THIS is the comparable linear column, and it only
        #                    exists for towers that expose a real token sequence.
        #   attentive      : over the token sequence, as for ours.
        specs = [("linear", "feat_mean", "linear_native"),
                 ("linear", "feat_tokens", "linear_tokmean"),
                 ("attentive", "feat_tokens", "attentive")]
        for kind, key, tag in specs:
            if key not in tr or key not in te:
                log("    %-14s SKIP (tower exposes no token sequence)" % tag); continue
            Xtr = tr[key]; Xte = te[key][keep]
            if tag == "linear_tokmean":
                Xtr = Xtr.float().mean(1).half(); Xte = Xte.float().mean(1).half()
            r = AP.run(Xtr, Ytr, Xte, Yte, kind, log)
            r["variant"] = m; r["column"] = tag
            r["comparable_to_ours"] = tag in ("linear_tokmean", "attentive")
            r["preprocessing_source"] = tr.get("preprocessing_source")
            r["n_train"] = int(Ytr.shape[0]); r["n_eval"] = int(Yte.shape[0])
            res.append(r)
    out = {"seed": AP.SEED, "epochs": AP.EPOCHS, "batch": AP.BATCH, "lr": AP.LR,
           "weight_decay": AP.WD, "num_classes": AP.NUM_CLASSES,
           "protocol": "IDENTICAL to docs/artifacts/audioset_probe_ours.json "
                       "(hyperparameters imported from scripts/audioset_probe.py)",
           "results": res}
    json.dump(out, open(a.out, "w"), indent=2)
    log("[probe-base] WROTE %s (%d rows)" % (a.out, len(res)))


if __name__ == "__main__":
    main()
