"""M1 Local Retrieval Demo.

Two modes:
1. Setup (--mode setup): Runs on server. Extracts video/text embeddings for a gallery of cached clips and saves them to a lightweight bank.
2. Live (--mode live): Runs on CPU (laptop or server). Loads only the text encoder and the bank. Provides a Gradio UI for Text->Video and Video->Text retrieval.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cached_feature_dataset import CachedFeatureDataset, cached_collate_fn
from models import SpineConfig, SpineM1
from utils import AttrDict, cfg_get, load_config


def run_setup(cfg: AttrDict, checkpoint_path: str, limit: int | None = None) -> None:
    """Build the embedding bank from the feature cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Running on {device}")

    feature_cache_dir = str(cfg_get(cfg, "data.feature_cache_dir"))
    if not feature_cache_dir:
        raise ValueError("demo_m1 requires data.feature_cache_dir in config")

    # 1. Load dataset (we use the eval split to build the gallery)
    dataset = CachedFeatureDataset.from_config(cfg, "eval", limit=limit)
    
    # [CRITICAL ALIGNMENT BUG FIX]
    # If a video failed to decode during extraction, CachedFeatureDataset.__getitem__ 
    # catches the FileNotFoundError and silently returns a random fallback video.
    # This is fine for training, but it scrambling the Bank metadata!
    # We must strictly filter out missing features here so __getitem__ never throws.
    valid_samples = []
    for s in dataset.samples:
        feat_path = os.path.join(feature_cache_dir, s.video_id[:2], f"{s.video_id}.pt")
        if os.path.exists(feat_path):
            valid_samples.append(s)
    print(f"[setup] Filtered {len(dataset.samples) - len(valid_samples)} missing features.")
    dataset.samples = valid_samples

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=cached_collate_fn,
        drop_last=False,
    )

    # 2. Load model
    print(f"[setup] Loading model from {checkpoint_path}...")
    with open(os.path.join(feature_cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    encoder_out_dim = manifest["hidden_size"]

    model_config = cfg.get("model", {})
    model_config["skip_encoder"] = True
    model_config["encoder_out_dim"] = encoder_out_dim

    spine = SpineM1(SpineConfig(**model_config)).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    spine.load_state_dict(state_dict, strict=False)
    spine.eval()

    # 3. Build bank
    print(f"[setup] Embedding {len(dataset)} gallery clips...")
    video_embeds = []
    text_embeds = []
    metadata = []

    with torch.no_grad():
        for batch in tqdm(loader):
            feats, captions = batch
            feats = feats.to(device)
            
            # Predictor forward
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                zv = spine.predictor(feats.float())
                zt = spine.embed_text(captions)
            
            # Normalize and move to CPU immediately to save VRAM
            zv = F.normalize(zv.float(), dim=-1).cpu()
            zt = F.normalize(zt.float(), dim=-1).cpu()
            
            video_embeds.append(zv)
            text_embeds.append(zt)

    # Reconstruct metadata (order is now 100% strictly aligned with loader)
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        metadata.append({
            "video_id": sample.video_id, 
            "caption": sample.caption,
            "video_path": sample.video_path  # Save the absolute path for UI playback!
        })

    video_bank = torch.cat(video_embeds, dim=0)
    text_bank = torch.cat(text_embeds, dim=0)

    # 4. Save bank
    os.makedirs("demo_bank", exist_ok=True)
    bank_path = "demo_bank/bank.pt"
    torch.save({
        "video_embeds": video_bank,
        "text_embeds": text_bank,
        "metadata": metadata,
        "manifest": manifest,
    }, bank_path)
    
    print(f"[setup] Saved bank to {bank_path}")
    print(f"        video_embeds: {video_bank.shape}")
    print(f"        text_embeds:  {text_bank.shape}")


def run_live(cfg: AttrDict, checkpoint_path: str) -> None:
    """Run Gradio UI for local retrieval testing (CPU only)."""
    print("[live] Loading bank...")
    bank_path = "demo_bank/bank.pt"
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"{bank_path} not found. Run --mode setup first.")
    
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    video_embeds = bank["video_embeds"]  # [N, 1536]
    text_embeds = bank["text_embeds"]    # [N, 1536]
    metadata = bank["metadata"]
    manifest = bank["manifest"]

    # 1. Load ONLY Text Encoder
    print(f"[live] Loading text encoder from {checkpoint_path}...")
    model_config = cfg.get("model", {})
    model_config["skip_encoder"] = True
    model_config["encoder_out_dim"] = manifest["hidden_size"]
    model_config["device"] = "cpu"  # Force CPU for live demo

    spine = SpineM1(SpineConfig(**model_config)).cpu()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    spine.load_state_dict(state_dict, strict=False)
    spine.eval()

    def text2video(query: str):
        if not query.strip():
            return [None, "Please enter a query."] * 5
        
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                zt = spine.embed_text([query])
            zt = F.normalize(zt.float(), dim=-1).cpu()  # [1, 1536]
        
        scores = (zt @ video_embeds.T).squeeze(0)  # [N]
        top_scores, top_indices = scores.topk(5)
        
        outputs = []
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            meta = metadata[idx.item()]
            vid_path = meta.get("video_path")
            # Gradio needs the absolute path to render the video
            info = f"Rank {i+1} | Score: {score.item():.3f} | ID: {meta['video_id']}\nCaption: {meta['caption']}"
            outputs.append(vid_path)
            outputs.append(info)
        return outputs

    def video2text(video_id: str):
        video_id = video_id.strip()
        if not video_id:
            return "Please enter a video_id.", None
        
        idx = next((i for i, m in enumerate(metadata) if m["video_id"] == video_id), None)
        if idx is None:
            return f"Error: video_id '{video_id}' not found in the demo bank.", None
            
        zv = video_embeds[idx].unsqueeze(0)  # [1, 1536]
        scores = (zv @ text_embeds.T).squeeze(0)  # [N]
        top_scores, top_indices = scores.topk(5)
        
        meta = metadata[idx]
        rank = (scores > scores[idx]).sum().item() + 1
        
        result = f"Target Video: {video_id}\nGround Truth Caption: {meta['caption']}\n\n"
        result += f"=== Self-Retrieval Rank: {rank}/{len(metadata)} ===\n\n"
        
        for i, (score, tidx) in enumerate(zip(top_scores, top_indices)):
            tmeta = metadata[tidx.item()]
            result += f"{i+1}. [Score: {score.item():.3f}] {tmeta['caption']}\n"
        
        return result, meta.get("video_path")

    # 2. Build Gradio UI
    with gr.Blocks(title="M1 Retrieval Demo") as app:
        gr.Markdown("# M1 Retrieval Demo\nRunning on CPU using pre-computed embeddings.")
        
        with gr.Tab("Text → Video Search"):
            gr.Markdown("Type a semantic query to search the video embedding bank.")
            with gr.Row():
                t2v_input = gr.Textbox(label="Text Query", placeholder="A person chopping vegetables...", scale=4)
                t2v_btn = gr.Button("Search", scale=1)
            
            # Fixed 5x Video outputs
            t2v_vid_outputs = []
            t2v_info_outputs = []
            for i in range(5):
                with gr.Row():
                    vid = gr.Video(label=f"Rank {i+1}", height=256)
                    info = gr.Textbox(label="Metadata", lines=3)
                    t2v_vid_outputs.append(vid)
                    t2v_info_outputs.append(info)
            
            # Interleave them for the return list
            t2v_all_outputs = []
            for v, i in zip(t2v_vid_outputs, t2v_info_outputs):
                t2v_all_outputs.extend([v, i])
                
            t2v_btn.click(text2video, inputs=t2v_input, outputs=t2v_all_outputs)
            t2v_input.submit(text2video, inputs=t2v_input, outputs=t2v_all_outputs)
            
        with gr.Tab("Video → Text (Self-Retrieval)"):
            gr.Markdown(
                "Enter a `video_id` from the bank to see its closest captions. "
                "If it can't retrieve its own ground-truth caption, the bank is broken."
            )
            with gr.Row():
                v2t_input = gr.Textbox(label="Video ID", placeholder="video1234", scale=4)
                v2t_btn = gr.Button("Retrieve Captions", scale=1)
            
            with gr.Row():
                v2t_vid = gr.Video(label="Source Video")
                v2t_output = gr.Textbox(label="Results", lines=15)
                
            v2t_btn.click(video2text, inputs=v2t_input, outputs=[v2t_output, v2t_vid])
            v2t_input.submit(video2text, inputs=v2t_input, outputs=[v2t_output, v2t_vid])

    print("[live] Launching Gradio server on 0.0.0.0:7860...")
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 Local Retrieval Demo")
    parser.add_argument("--mode", choices=["setup", "live"], required=True)
    parser.add_argument("--config", default="configs/m1_scale.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/m1_scale/best.pt")
    parser.add_argument("--limit", type=int, default=None, help="Max clips for the bank")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == "setup":
        run_setup(cfg, args.checkpoint, limit=args.limit)
    else:
        run_live(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
