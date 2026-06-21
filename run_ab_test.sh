#!/usr/bin/env bash
# run_ab_test.sh
# Automated script to run the full A/B test (Run A vs Run B) and compile results.

set -e

# Make sure we are in the workspace root
cd /home/jovyan/work/JEPA-Omni

echo "=== STAGE 0: Cleaning directories and preparing checkpoints ==="
rm -rf checkpoints/m1_lambda0 checkpoints/m1_lambda0p1
mkdir -p checkpoints/m1_lambda0 checkpoints/m1_lambda0p1

echo "=== STAGE 1: Running A/B Test - Run A (InfoNCE Control, lambda=0.0) ==="
echo "Training Run A..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_m1.py --config configs/m1_lambda0.yaml > m1_lambda0.log 2>&1 || { echo "Run A training failed! Check m1_lambda0.log"; exit 1; }

echo "Evaluating Run A..."
python eval_m1.py --config configs/m1_lambda0.yaml --checkpoint checkpoints/m1_lambda0/best.pt > m1_lambda0_eval.log 2>&1 || { echo "Run A evaluation failed! Check m1_lambda0_eval.log"; exit 1; }
echo "Run A Complete."

echo "=== STAGE 2: Running A/B Test - Run B (InfoNCE + SIGReg Treatment, lambda=0.1) ==="
echo "Training Run B..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_m1.py --config configs/m1_lambda0p1.yaml > m1_lambda0p1.log 2>&1 || { echo "Run B training failed! Check m1_lambda0p1.log"; exit 1; }

echo "Evaluating Run B..."
python eval_m1.py --config configs/m1_lambda0p1.yaml --checkpoint checkpoints/m1_lambda0p1/best.pt > m1_lambda0p1_eval.log 2>&1 || { echo "Run B evaluation failed! Check m1_lambda0p1_eval.log"; exit 1; }
echo "Run B Complete."

echo "=== STAGE 3: Compiling A/B Test Results ==="
echo ""
echo "==================== A/B TEST RESULTS SUMMARY ===================="
echo ""
echo "--- RUN A (Control: sigreg_lambda=0.0) ---"
if [ -f m1_lambda0_eval.log ]; then
    cat m1_lambda0_eval.log | grep -E "(video->text|text->video|PASS|FAIL|R@)" || cat m1_lambda0_eval.log
else
    echo "Evaluation log not found for Run A."
fi
echo ""
echo "--- RUN B (Treatment: sigreg_lambda=0.1) ---"
if [ -f m1_lambda0p1_eval.log ]; then
    cat m1_lambda0p1_eval.log | grep -E "(video->text|text->video|PASS|FAIL|R@)" || cat m1_lambda0p1_eval.log
else
    echo "Evaluation log not found for Run B."
fi
echo ""
echo "================================================================"
