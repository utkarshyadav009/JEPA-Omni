import torch
from transformers import LlamaModel
import inspect

def main():
    print("Loading Llama Model...")
    llama = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="sdpa")
    layer = llama.layers[-8] # Get the first layer of the last 8 layers
    rotary_emb = llama.layers[0].self_attn.rotary_emb
    
    # Inputs
    B, S, H = 16, 8193, 2048
    x = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(S, dtype=torch.long, device="cuda").unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x, position_ids)
    
    print("\n--- Test 1: attention_mask=None ---")
    try:
        out = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        print(f"Output type: {type(out)}")
        print(f"Output element 0 shape: {out[0].shape}")
        print(f"Output element 0 layout: {out[0].layout}")
        print(f"Output element 0 is_nested: {getattr(out[0], 'is_nested', False)}")
    except Exception as e:
        print(f"Test 1 failed: {e}")
        
    print("\n--- Test 2: attention_mask = 4D zeros ---")
    zeros_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16, device="cuda")
    try:
        out = layer(
            x,
            attention_mask=zeros_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        print(f"Output element 0 shape: {out[0].shape}")
        print(f"Output element 0 layout: {out[0].layout}")
        print(f"Output element 0 is_nested: {getattr(out[0], 'is_nested', False)}")
    except Exception as e:
        print(f"Test 2 failed: {e}")

    print("\n--- Printing LlamaDecoderLayer.forward Source ---")
    try:
        print(inspect.getsource(layer.forward))
    except Exception as e:
        print(f"Failed to get layer source: {e}")

    print("\n--- Printing LlamaSdpaAttention.forward Source ---")
    try:
        print(inspect.getsource(layer.self_attn.forward))
    except Exception as e:
        print(f"Failed to get attn source: {e}")

if __name__ == "__main__":
    main()
