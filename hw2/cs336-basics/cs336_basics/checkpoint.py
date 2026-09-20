import torch
import os
from typing import BinaryIO, IO, Union

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Dump all the state from the model, optimizer and iteration into the file-like object out.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """
    Load a checkpoint from src, recover the model and optimizer states,
    and return the iteration number that was saved.
    """
    checkpoint = torch.load(src, weights_only=False)
    
    # 兼容 torch.compile 产出的带 _orig_mod. 前缀的权重
    sd = checkpoint.get('model_state_dict', checkpoint)
    for k in list(sd):
        if k.startswith("_orig_mod."): 
            sd[k[len("_orig_mod."):]] = sd.pop(k)
            
    model.load_state_dict(sd)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('iteration', 0)
