import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device| None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        trunkate = - 3.0 * std, 3.0 * std
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=trunkate[0], b=trunkate[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device| None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device| None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(
            torch.empty((d_model,), device=device, dtype=dtype)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.scale
        return result.to(in_dtype)