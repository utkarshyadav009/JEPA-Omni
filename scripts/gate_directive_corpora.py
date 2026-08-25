"""Gate both directive corpora before any training, and build the thinker training file.

Counts are not enough and never were: the first speaker directive attempt passed a 150-row
count gate with 395 rows while 197 of them paired a random CoT with a line written for a
different directive. These gates check COLLAPSE, COVERAGE and RESTATEMENT -- the failure modes
actually measured on this project -- and exit non-zero so the training chain stops.
"""
import json, re, sys, collections, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_speaker_directive_rows import ALL_DIRECTIVES

TH = Path("data/bmo_thinker_directive_rows_v1.jsonl")
SP = Path("data/bmo_companion_corpus_v13.jsonl")
OUT = Path("data/bmo_thinker_directive_train.jsonl")
MAX_RESTATE = 0.75          # drop the circular tail rather than train on it

def words(s): return set(re.findall(r"[a-z]{4,}", s.lower()))
def overlap(a, b): return len(words(a) & words(b)) / max(len(words(b)), 1)

ok = True
def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)
    if not cond: ok = False

print("=== THINKER directive corpus ===", flush=True)
rows = [json.loads(l) for l in TH.open()] if TH.exists() else []
check(len(rows) >= 900, f"rows {len(rows)} >= 900")
d = collections.Counter(r["directive"] for r in rows)
check(len(d) >= 0.8 * len(ALL_DIRECTIVES),
      f"directive coverage {len(d)}/{len(ALL_DIRECTIVES)} >= 80%")
if rows:
    top, c = d.most_common(1)[0]
    check(c <= 0.25 * len(rows), f"no collapse: top directive {c/len(rows)*100:.1f}% <= 25%")
    dup = len(rows) - len({r["said"] for r in rows})
    check(dup <= 0.15 * len(rows), f"duplicate utterances {dup/len(rows)*100:.1f}% <= 15%")
    ovs = [overlap(r["reasoning"], r["directive"]) for r in rows]
    check(statistics.mean(ovs) < 0.45, f"mean restatement overlap {statistics.mean(ovs):.2f} < 0.45")
    check(all(r.get("paired") for r in rows), "every row flagged paired")

print("\n=== SPEAKER corpus v13 ===", flush=True)
srows = [json.loads(l) for l in SP.open()] if SP.exists() else []
check(len(srows) > 4144, f"rows {len(srows)} > v12's 4144")
sd = collections.Counter(r.get("directive") for r in srows if r.get("directive"))
check(len(sd) >= 0.8 * len(ALL_DIRECTIVES),
      f"speaker directive coverage {len(sd)}/{len(ALL_DIRECTIVES)} >= 80%")
missing = [x for x in ALL_DIRECTIVES if x not in sd]
if missing:
    print(f"  note: speaker missing {len(missing)}: {[m[:40] for m in missing[:4]]}", flush=True)
# the interface contract: the two halves must share a vocabulary
shared = set(d) & set(sd)
check(len(shared) >= 0.75 * len(set(d)),
      f"vocabulary overlap thinker->speaker {len(shared)}/{len(set(d))} >= 75%")

if not ok:
    print("\nGATES FAILED", flush=True); sys.exit(1)

kept = [r for r in rows if overlap(r["reasoning"], r["directive"]) <= MAX_RESTATE]
print(f"\ndropping {len(rows)-len(kept)} circular rows (overlap > {MAX_RESTATE})", flush=True)
with OUT.open("w") as f:
    for r in kept:
        # The thinker trainer builds "<think>{reasoning}</think>{answer}". The DIRECTIVE goes
        # in the answer slot deliberately: under the directive contract the thinker never
        # speaks, so its completion IS the instruction. Trained directive-only rather than
        # mixed with the old spoken-answer rows -- mixing both targets for similar prompts
        # teaches exactly the inconsistency this whole slice exists to remove.
        f.write(json.dumps({"prompt": r["prompt"], "reasoning": r["reasoning"],
                            "answer": r["directive"], "tools": []}) + "\n")
print(f"wrote {len(kept)} -> {OUT}", flush=True)
print("GATES PASSED", flush=True)
