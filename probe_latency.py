import os
import sys
import torch
from utils import load_config
from models import SpineConfig, SpineM1

def main():
    print("Loading config configs/m1.yaml...")
    cfg = load_config("configs/m1.yaml")
    
    print("Instantiating SpineM1 model on CUDA device...")
    model_config = dict(cfg["model"])
    spine_config = SpineConfig(**model_config)
    model = SpineM1(spine_config).to("cuda")
    model.train()
    
    # Set up simple optimizer over trainable parameters
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    
    # Retrieve configuration parameters for data shape
    num_frames = cfg.data.num_frames
    resolution = cfg.data.resolution
    
    print(f"Config parameters: num_frames={num_frames}, resolution={resolution}")
    
    batch_size = 2
    max_stable_batch_size = 0
    
    while True:
        print(f"\nProbing batch size {batch_size}...")
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Generate dummy video tensors of shape (B, 3, num_frames, resolution, resolution)
            # using random values in range [0, 255] as expected by the video processor
            dummy_videos_raw = torch.randint(
                0, 256, 
                (batch_size, 3, num_frames, resolution, resolution), 
                dtype=torch.uint8, 
                device="cuda"
            )
            # Permute to shape (B, num_frames, 3, resolution, resolution) expected by the vision encoder
            videos = dummy_videos_raw.permute(0, 2, 1, 3, 4)
            
            # Generate dummy text strings
            captions = ["dummy text"] * batch_size
            
            optimizer.zero_grad()
            
            # Autocast with bfloat16 to match the train_m1.py environment
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = model(videos, captions)
                
            loss.backward()
            optimizer.step()
            
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_gib = peak_vram_bytes / (1024 ** 3)
            print(f"Batch size {batch_size}: Success. Peak VRAM: {peak_vram_bytes} bytes ({peak_vram_gib:.2f} GiB)")
            
            max_stable_batch_size = batch_size
            batch_size += 2
            
            # Clean up references to free up memory for the next loop iteration
            del dummy_videos_raw, videos, captions, loss, metrics
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nCUDA Out Of Memory caught at batch size {batch_size}!")
                print(f"Maximum stable batch size found: {max_stable_batch_size}")
                # Clean up memory and exit gracefully
                torch.cuda.empty_cache()
                sys.exit(0)
            else:
                # Re-raise any other runtime errors
                raise e

if __name__ == "__main__":
    main()
