# Quarantined artifact: EGO4D_HELDOUT_RUN3_STEP19000_RESULT.json

**Found 2026-09-10 during independent verification of `docs/ICLR_RESULTS.md` §4.**

`EGO4D_HELDOUT_RUN3_STEP19000_RESULT.json` was **byte-identical** to
`EGO4D_HELDOUT_BASELINE_V2.json` — same md5 `f8a873bf410c73c1b7679d358dd037b1`, same 1,142 bytes.
It contained the *pre-training baseline* numbers (sibling-excluded v→a 2.82 / a→v 1.78), not
RUN-3 results, despite its filename. It is the only duplicate among the 14 `EGO4D_HELDOUT_*.json`
artifacts.

**Impact if cited unknowingly:** RUN-3's Ego4D transfer would be reported as 2.82/1.78 instead of
8.75/12.91 — a 3–7× understatement, and it would look like RUN-3 performed at the untrained
baseline.

**No published number was affected.** `docs/EVIDENCE_LEDGER.md` TABLE 3 and
`docs/ICLR_RESULTS.md` §4.2 both cite RUN-3 as **8.75 / 12.91**, which is correct and comes from
`EGO4D_HELDOUT_RUN3_STEP20000_RESULT.json` (**step 20000**, not step 19000). Both documents also
already record RUN-3's step count as "20,000(?)", consistent with that source.

**What remains unknown:** whether a genuine RUN-3 *step19000* evaluation was ever run. No artifact
for it survives. Do not reconstruct one from this file.

Renamed rather than deleted so the provenance trail stays intact.
