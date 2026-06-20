# Handover: M1 SigReg & VATEX Scaling Experiment

## Context & Goal
The goal of this phase was to scale the M1 (Offline Vision $\rightarrow$ Text Perception Spine) by:
1. Replacing the InfoNCE loss with **SigReg** (Sketched Isotropic Gaussian Regularization, from LeJEPA 2511.08544).
2. Expanding the dataset from the 7k MSR-VTT subset to the full **VATEX** dataset (Parts 1, 2, and 3).
3. Moving training to **GPU 1** while avoiding CUDA OOM errors caused by the heavy Llama-3.2-1B predictor.

## Work Accomplished

### 1. Architectural Changes (SigReg)
*   **Implemented SigReg Loss**: Added `sigreg_loss` and `sigreg_jepa_loss` to `models/losses.py` using empirical characteristic functions (ECF) and random 1D projections (sketching).
*   **Updated Spine API**: Modified `models/spine_m1.py` `SpineConfig` to include `loss_type` (`"info_nce"` | `"sigreg"`) and `sigreg_lambda`. Updated `SpineM1.forward()` to route to the correct loss natively. 
*   **Fixed Feature Cache Support**: Updated `SpineM1.embed_video` to gracefully handle pre-computed feature tensors (`[B, N, D]`) so the pipeline skips the heavy ViT-L encoder when scaling.
*   **Cleaned Training Loop**: Removed monkey-patching in `train_m1.py`; the script now reads `loss_type` directly from the config and falls back to `sigreg_jepa_loss` for metrics/eval extraction.

### 2. Dataset Curation (VATEX Full)
*   **Downloaded Data**: Fetched VATEX Parts 1, 2, and 3 from Kaggle using user credentials.
*   **Consolidation**: Moved all `.mp4` and `.mkv` files into a single flat directory: `/home/jovyan/work/data/vatex/video`.
*   **Curation Script**: Wrote and executed `curate_final_vatex.py`. It parsed the `.arrow` files from all dataset splits (`train`, `validation`, `public_test`), cross-referenced them with the actual videos on disk, and generated a final cleaned JSON.
*   **Final Yield**: Out of 34,986 annotations and 34,372 videos on disk, successfully matched **29,045** video-caption pairs into `vatex_final_curated.json`.

### 3. Configuration & Optimization (`configs/m1.yaml`)
*   Pointed data to `vatex_final_curated.json` and `/home/jovyan/work/data/vatex/video`.
*   Changed `loss_type` to `sigreg` with `sigreg_lambda: 10.0`.
*   Increased `total_steps` to `10000`.
*   **OOM Mitigation**: The Llama-1B predictor + 64 frames (with 1024-dim tokens) causes OOM on an 80GB H100 at batch size 64. Configured `batch_size: 8` with `gradient_accumulation_steps: 8` (effective batch size 64) to fix this.

## Current Status & Next Steps
Everything is staged, curated, and configured. The previous checkpoints in `checkpoints/m1/` have been cleared to start fresh.

**Next Agent Action:**
1. Verify the current configuration matches the user's intent.
2. If `feature_cache_dir` isn't set, you must run the feature extraction script (`scripts/extract_features.py` equivalent) over the new VATEX dataset first, as real-time decoding + V-JEPA ViT-L will take days or OOM the dataloader workers.
3. Once features are cached, launch the training run specifically on GPU 1:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python train_m1.py --config configs/m1.yaml
   ```
