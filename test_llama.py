from transformers import LlamaConfig, LlamaModel
import torch

config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
# in newer transformers, causal mask might be controlled by is_decoder or attention_mask
model = LlamaModel(config)
x = torch.randn(1, 10, 64)

# To truly disable causal mask, we can just pass an explicit 2D attention mask of all 1s
# If that doesn't work, we can check FlashAttention / SDPA settings in LlamaConfig
mask = torch.ones(1, 10, dtype=torch.long)
out = model(inputs_embeds=x, attention_mask=mask)

# Let's inspect the weights in the causal mask if any.
# Actually, if we just use the layers:
layer = model.layers[0]
position_ids = torch.arange(10).unsqueeze(0)

# Pass with attention_mask = None and output_attentions=True to see if it's causal
out_layer = layer(x, position_ids=position_ids, output_attentions=True)
attn_weights = out_layer[1]
print("Attn weights shape:", attn_weights.shape)
print("Attn weights (first row):", attn_weights[0, 0, 0, :])
print("Attn weights (last row):", attn_weights[0, 0, -1, :])

# Now with a custom mask that is 4D: (batch, 1, seq, seq) of zeros (meaning no masking)
custom_mask = torch.zeros(1, 1, 10, 10)
out_layer2 = layer(x, attention_mask=custom_mask, position_ids=position_ids, output_attentions=True)
attn_weights2 = out_layer2[1]
print("With custom 0-mask (last row):", attn_weights2[0, 0, -1, :])
print("With custom 0-mask (first row):", attn_weights2[0, 0, 0, :])
