import torch
import yaml
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import cross_entropy
from train import evaluate_loss
import numpy as np

with open("checkpoints/tinystories_17M/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda"
model = TransformerLM(
    vocab_size=config['vocab_size'],
    context_length=config['context_length'],
    d_model=config['d_model'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    d_ff=config.get('d_ff', config['d_model'] * 4),
    rope_theta=config.get('rope_theta', 10000.0)
).to(device)

state_dict = torch.load("checkpoints/tinystories_17M/ckpt_step_16500.pt", map_location=device, weights_only=True)
if 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
else:
    model.load_state_dict(state_dict)

val_data = np.memmap(config['val_data'], dtype=np.uint16, mode='r')

from contextlib import nullcontext
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
val_loss = evaluate_loss(model, val_data, 100, 256, 256, device, 1337, ctx, None)
print(f"Final Validation Loss: {val_loss}")
