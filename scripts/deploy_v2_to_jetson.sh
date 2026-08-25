#!/bin/bash
# STAGED — DO NOT RUN UNATTENDED. Morning deploy of the retrained LLMs to the
# Jetson, AFTER you've reviewed the hostility/companion sample generations in
# scratchpad/retrain_fast.log and are happy with them.
#
# Reversible: backs up the current production GGUFs first. Tokenizers are
# unchanged (same base), so only the two GGUFs move.
set -e
JETSON=bmo@bmo-desktop
GGUF_DIR=~/bmo_production/models_gguf
LOCAL=/home/utkarsh/JEPA-Omni/checkpoints
STAMP=$(date +%Y%m%d_%H%M%S)

FAST_LOCAL=$LOCAL/bmo_lfm25_350m_v2_Q8_0.gguf
THINK_LOCAL=$LOCAL/bmo_thinker_qwen3_v3_Q8_0.gguf

echo "== checking local artifacts =="
ls -la "$FAST_LOCAL" "$THINK_LOCAL"

echo "== backing up current Jetson GGUFs =="
ssh $JETSON "cd $GGUF_DIR && cp -v bmo_lfm25_350m_v1_Q8_0.gguf bmo_lfm25_350m_v1_Q8_0.gguf.bak_$STAMP && cp -v bmo_thinker_qwen3_v2_Q8_0.gguf bmo_thinker_qwen3_v2_Q8_0.gguf.bak_$STAMP"

echo "== copying new GGUFs =="
scp "$FAST_LOCAL" $JETSON:$GGUF_DIR/bmo_lfm25_350m_v2_Q8_0.gguf
scp "$THINK_LOCAL" $JETSON:$GGUF_DIR/bmo_thinker_qwen3_v3_Q8_0.gguf

echo "== repoint ~/live_bmo.py at v2/v3 (backup kept) =="
ssh $JETSON "cp ~/live_bmo.py ~/live_bmo.py.bak_$STAMP && \
  sed -i 's/bmo_lfm25_350m_v1_Q8_0.gguf/bmo_lfm25_350m_v2_Q8_0.gguf/; s/bmo_thinker_qwen3_v2_Q8_0.gguf/bmo_thinker_qwen3_v3_Q8_0.gguf/' ~/live_bmo.py && \
  grep -n 'gguf' ~/live_bmo.py"

echo "== DONE. Re-run on the Jetson:  python3 ~/live_bmo.py =="
echo "== rollback:  restore ~/live_bmo.py.bak_$STAMP and the .bak_$STAMP GGUFs =="
