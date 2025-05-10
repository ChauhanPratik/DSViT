import torch

# Store intermediate features globally
INTERMEDIATE_OUTPUTS = {}

def save_hook(layer_name):
    def hook_fn(module, input, output):
        INTERMEDIATE_OUTPUTS[layer_name] = output.detach().cpu()
    return hook_fn

def register_hooks(model):
    handles = []
    for idx, block in enumerate(model.transformer_blocks):
        name = f"transformer_block_{idx}"
        handle = block.register_forward_hook(save_hook(name))
        handles.append(handle)
    return handles 

ATTENTION_WEIGHTS = {}

def save_hook(layer_name):
    def hook_fn(module, input, output):
        INTERMEDIATE_OUTPUTS[layer_name] = output.detach().cpu()
    return hook_fn

def save_attention_hook(layer_name):
    def hook_fn(module, input, output):
        # output is (attn_output, attn_weights)
        ATTENTION_WEIGHTS[layer_name] = output[1].detach().cpu()  # (B, num_heads, N, N)
    return hook_fn

def register_hooks(model):
    handles = []
    for idx, block in enumerate(model.transformer_blocks):
        block_name = f"transformer_block_{idx}"
        handles.append(block.register_forward_hook(save_hook(block_name)))
        handles.append(block.attn.register_forward_hook(save_attention_hook(f"{block_name}_attn")))
    return handles
