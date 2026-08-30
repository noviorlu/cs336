import torch
import yaml
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from train import evaluate_loss
import numpy as np

with open("checkpoints/tinystories_17M/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda"
model = TransformerLM.from_pretrained("checkpoints/tinystories_17M/ckpt_step_16500.pt", device=device)

val_data = np.memmap(config['val_data'], dtype=np.uint16, mode='r')

from contextlib import nullcontext
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
val_loss = evaluate_loss(model, val_data, 100, 256, 256, device, 1337, ctx, None)
print(f"Final Validation Loss: {val_loss}")
