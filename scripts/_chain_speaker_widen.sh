#!/bin/bash
# Chain the speaker widening to start when the thinker corpus finishes.
# Waits on the thinker run's LOG MARKER, never on a process name -- pgrep/pkill -f matched
# this session's own argv six separate times in this project's history and killed the wrong
# thing. A marker printed after the work completes cannot be true early and cannot false-match.
cd /home/utkarsh/JEPA-Omni
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jepa-omni

echo "[chain] waiting for THINKER_DIRECTIVE_DONE ..."
while ! grep -q "THINKER_DIRECTIVE_DONE" thinker_dir_gen.log 2>/dev/null; do sleep 30; done
echo "[chain] thinker corpus finished at $(date -Is)"

# Let the 120B fully release the GPUs before the next load; a partial release reproducibly
# causes an illegal memory access on this box.
sleep 45
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[chain] starting speaker widening (12 new directives x 8 scenes x 5)"
python scripts/generate_speaker_directive_rows.py \
    --only-new --combos 96 --per-combo 5 \
    --base data/bmo_companion_corpus_v12.jsonl \
    --out  data/bmo_companion_corpus_v13.jsonl \
    2>&1
echo "SPEAKER_WIDEN_CHAIN_DONE"
