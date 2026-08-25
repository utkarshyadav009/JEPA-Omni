#!/bin/bash
# Waits for BOTH directive corpora, GATES them, and only then trains.
# Waits on LOG MARKERS, never process names (six prior incidents in this project).
# The gates ABORT rather than train on a bad corpus: this project has twice shipped a corpus
# that passed a count check while being actively harmful, and both times it was found by
# reading rows. A monitor that trains unconditionally would reproduce that.
cd /home/utkarsh/JEPA-Omni
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jepa-omni
set -u

echo "[train-chain] waiting for THINKER_DIRECTIVE_DONE ..."
while ! grep -q "THINKER_DIRECTIVE_DONE" thinker_dir_gen.log 2>/dev/null; do sleep 60; done
echo "[train-chain] thinker corpus done $(date -Is)"

echo "[train-chain] waiting for SPEAKER_WIDEN_CHAIN_DONE ..."
while ! grep -q "SPEAKER_WIDEN_CHAIN_DONE" speaker_widen.log 2>/dev/null; do sleep 60; done
echo "[train-chain] speaker corpus done $(date -Is)"
sleep 45   # let the 120B release the GPUs fully

# TOP-UP: three canonical directives received ZERO rows because the first verifier
# false-rejected them ("gently suggest.." as not_imperative; "tell them YOUR battery.." and
# "hold YOUR ground.." as first_person). Verifier fixed; regenerate just those three and
# append, so all 26 have training data before anything trains on them.
MISSING=("gently suggest they take a break, because they have been at this a long time" \
         "tell them your battery is low and ask to be plugged in" \
         "hold your ground gently, because they are being unkind")
echo "[train-chain] === topping up 3 previously self-rejected directives ==="
python scripts/generate_thinker_directive_rows.py \
    --only "${MISSING[@]}" --append \
    --out data/bmo_thinker_directive_rows_v1.jsonl 2>&1 | tail -20
echo "[train-chain] corpus now $(wc -l < data/bmo_thinker_directive_rows_v1.jsonl) rows"

python scripts/gate_directive_corpora.py || { echo "[train-chain] GATES FAILED -- NOT TRAINING"; echo TRAIN_CHAIN_ABORTED; exit 1; }

echo "[train-chain] === training thinker v8 (directive contract) ==="
python scripts/finetune_thinker_qwen3.py \
    --corpus data/bmo_thinker_directive_train.jsonl \
    --out-dir checkpoints/bmo_thinker_qwen3_v8_directive_lora \
    --epochs 4 2>&1 | tail -40

echo "[train-chain] === training speaker v7 (26-directive vocabulary) ==="
python scripts/finetune_bmo_minicpm5_lora.py \
    --corpus data/bmo_companion_corpus_v13.jsonl \
    --real-metadata data/real_speech/metadata.csv \
    --out-dir checkpoints/bmo_lfm25_350m_v7_lora 2>&1 | tail -40

echo TRAIN_CHAIN_DONE
