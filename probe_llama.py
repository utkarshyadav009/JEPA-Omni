import torch
from transformers import LlamaModel
import inspect

def main():
    print("Loading Llama Model...")
    llama = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="sdpa")
    layer = llama.layers[-8].to("cuda")
    rotary_emb = llama.rotary_emb if hasattr(llama, "rotary_emb") else llama.layers[0].self_attn.rotary_emb
    if isinstance(rotary_emb, torch.nn.Module):
        rotary_emb = rotary_emb.to("cuda")
    
    # Inputs
    B, S, H = 16, 8193, 2048
    x = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(S, dtype=torch.long, device="cuda").unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x, position_ids)
    
    print("\n--- Test 1: attention_mask=None ---")
    import types
    from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, eager_attention_forward, apply_rotary_pos_emb
    
    def custom_attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        print(f"  [attn] hidden_states input shape: {hidden_states.shape}")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        print(f"  [attn] input_shape: {input_shape}, hidden_shape: {hidden_shape}")

        q_proj_out = self.q_proj(hidden_states)
        print(f"  [attn] q_proj output shape: {q_proj_out.shape}")
        query_states = q_proj_out.view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        print(f"  [attn] query_states shape after transpose: {query_states.shape}")

        cos, sin = position_embeddings
        print(f"  [attn] cos shape: {cos.shape}, sin shape: {sin.shape}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        print(f"  [attn] query_states shape after RoPE: {query_states.shape}")

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        print(f"  [attn] Using attention_interface: {attention_interface.__name__ if hasattr(attention_interface, '__name__') else type(attention_interface)}")

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        print(f"  [attn] attn_output shape from interface: {attn_output.shape}")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        print(f"  [attn] attn_output shape after reshape: {attn_output.shape}")
        attn_output = self.o_proj(attn_output)
        print(f"  [attn] attn_output shape after o_proj: {attn_output.shape}")
        return attn_output, attn_weights

    layer.self_attn.forward = types.MethodType(custom_attn_forward, layer.self_attn)

    try:
        out = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        print(f"Output type: {type(out)}")
        print(f"Output shape: {out.shape if isinstance(out, torch.Tensor) else out[0].shape}")
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
