import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        trunkate = - 3.0 * std, 3.0 * std
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=trunkate[0], b=trunkate[1])

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x: Int[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length embedding_dim"]:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(
            torch.empty((d_model,), device=device, dtype=dtype)
        )
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.scale
        return result.to(in_dtype)

class SiLU(nn.Module):
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = round(8 / 3 * d_model / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_gate = self.silu(self.w1(x))
        x_val = self.w3(x)
        return self.w2(x_gate * x_val)

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        # k: 0, 2, 4... (2d_k - 2)
        dim_range = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) # dim: [d_k // 2]
        
        # 1 / (theta ^ ((2d_k-2)/d))
        inv_freq = 1.0 / (theta ** (dim_range / d_k)) # dim: [d_k // 2]

        # i: 0, 1, 2, ..., max_seq_len - 1 
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32) # dim: [max_seq_len]
        
        # i / (theta ^ ((2d_k-2)/d)) 
        # t 是 [max_seq_len] 的列向展开，inv_freq 是 [d_k // 2] 的横向展开，外积相乘后得到二维矩阵
        freqs = einsum(t, inv_freq, 'i, j -> i j')

        cos_cached = freqs.cos()
        sin_cached = freqs.sin()

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False) 
        
    def forward(
        self, 
        x: Float[Tensor, "... sequence_length d_k"], 
        token_positions: Int[Tensor, "... sequence_length"]
    ) -> Float[Tensor, "... sequence_length d_k"]:
        
        