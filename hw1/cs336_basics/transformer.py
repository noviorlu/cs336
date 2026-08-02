import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool
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
        truncate = - 3.0 * std, 3.0 * std
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=truncate[0], b=truncate[1])

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

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len embedding_dim"]:
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
        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
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
    """
    i:    position              0 .. max_seq_len-1    \n
    k:    pair index            0 .. d_k/2-1          \n
    
    theta[i, k] = i / Theta ^ (2k / d_k)              \n
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        # 2k, k=0..d_k/2-1 → 取值 0, 2, 4, ..., d_k-2
        dim_range = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) # dim: [d_k // 2]
        
        # inv_freq[k] = 1 / Theta ** (2k / d_k)
        inv_freq = 1.0 / (theta ** (dim_range / d_k)) # dim: [d_k // 2]

        # i: 0, 1, 2, ..., max_seq_len - 1 
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32) # dim: [max_seq_len]
        
        # freqs[i, k] = theta[i, k] = i * inv_freq[k]
        freqs = einsum(t, inv_freq, 'i, k -> i k') # dim [max_seq_len, d_k // 2]

        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        rot90_sign = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer("rot90_sign", rot90_sign, persistent=False)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"], 
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        # [..., seq_len, k, 1]
        cos = rearrange(self.cos_cached[token_positions].to(x.dtype), '... k -> ... k 1') 
        sin = rearrange(self.sin_cached[token_positions].to(x.dtype), '... k -> ... k 1') 

        # [ x0, x1, x2, x3, x4, x5 ] =>
        # [
        #   [x0, x1],
        #   [x2, x3],
        #   [x4, x5]
        # ]
        x_reshaped = rearrange(x, '... (k xy) -> ... k xy', xy=2) # [..., k, 2]

        # [
        #   [x1, x0],
        #   [x3, x2],
        #   [x5, x4]
        # ] 
        # x [-1, 1]
        # =
        # [
        #   [-x1, x0],
        #   [-x3, x2],
        #   [-x5, x4]
        # ] 
        x_rotate = x_reshaped.flip(-1) * self.rot90_sign.to(x.dtype)

        # out0 = x0 * cos - x1 * sin
        # out1 = x0 * sin + x1 * cos
        # [out0 out1] = [x0, x1] * cos + [-x1, x0] * sin
        out = rearrange(x_reshaped * cos + x_rotate * sin, '... k xy -> ... (k xy)')
        return out

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:                      
    # 1. 找到指定维度 dim 上的最大值                                                         
    # keepdim=True 极其重要！它能保证找完最大值后形状不塌缩，从而可以和 x 完美相减           
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    
    # 2. 给所有的元素统一减去最大值（无损降维打击）
    x_shifted = x - x_max
    
    # 3. 算分子：对减去最大值后的张量统一求 exp
    nume = torch.exp(x_shifted)
    
    # 4. 算分母：沿着 dim 维度，把分子全部加起来（同样保持维度）
    denomi = torch.sum(nume, dim=dim, keepdim=True)
    
    # 5. 张量直接相除（分子矩阵 / 分母矩阵）
    return nume / denomi

def scaled_dot_product_attention(                                                            
    q: Float[Tensor, "b ... queries d_k"],                                                   
    k: Float[Tensor, "b ... keys d_k"],                                                   
    v: Float[Tensor, "b ... keys d_v"],                                                   
    mask: Bool[Tensor, "b ... queries keys"],
) -> Float[Tensor, "b ... queries d_v"]:                                                     
    d_k = q.shape[-1] 
    QK = einsum(q, k, '... queries d_k, ... keys d_k -> ... queries keys') / math.sqrt(d_k)
    QK = QK.masked_fill(~mask, float('-inf'))

    softQK = softmax(QK, dim=-1)
    return einsum(softQK, v, '... queries keys, ... keys d_v -> ... queries d_v')

class MultiHeadSelfAttention(nn.Module):
    '''
    In self attention   d_k = d_v = d
                        queries = keys = seq_len
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],                                                      
        mask: Bool[Tensor, "... seq_len seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        q = self.q_proj(x) # ... seq_len d_model
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_arrg = rearrange(q, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads)           
        k_arrg = rearrange(k, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads)           
        v_arrg = rearrange(v, '... seq_len (h d) -> ... h seq_len d', h=self.num_heads) 


        if self.rope is not None:
            assert token_positions is not None, "token_positions 必须由外部提供 (RoPE 启用时)"
            q_arrg = self.rope(q_arrg, token_positions)
            k_arrg = self.rope(k_arrg, token_positions)

        o_arrg = scaled_dot_product_attention(q_arrg, k_arrg, v_arrg, mask)
        o = rearrange(o_arrg, '... h seq_len d -> ... seq_len (h d)')

        return self.output_proj(o) # ... seq_len d_model

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            theta=theta, 
            device=device, 
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff, 
            device=device, 
            dtype=dtype
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        mask: Bool[Tensor, "... seq_len seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        # 1. Attention path
        x = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
        # 2. Feed-forward path
        x = x + self.ffn(self.ln2(x))
        return x

def build_attention_mask(
    seq_len: int,
    device: torch.device | str,
    x: Int[Tensor, "... seq_len"] | None = None,
    doc_sep_id: int | None = None,
) -> Bool[Tensor, "... 1 seq_len seq_len"]:
    """构造注意力 mask，True 表示允许注意。

    始终包含因果 mask；传了 x 和 doc_sep_id 就再叠加一层 document mask，
    挡住注意力跨越文档边界。

    返回值多出的那个长度为 1 的轴是给 num_heads 留的位置：MHA 的打分矩阵是
    [..., h, seq, seq]，而广播是右对齐的，不给 head 留位就会让 batch 撞上 h。
    留 1 而不是 expand 成 h，是为了让下游 masked_fill 里的 ~mask 只物化一份。
    """
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    
    if doc_sep_id is not None and x is not None:
        # Build doc_mask on the fly
        # doc_id[i] = 位置 i 属于窗口内的第几篇文档。
        # cumsum 对整型默认提升到 int64，必须显式给 dtype 才是 int32；
        # bool 可以直接 cumsum，不用先 .long()。上界是窗口内的分隔符个数
        # （≤ seq_len），int32 绰绰有余。
        is_sep = x == doc_sep_id
        doc_id = is_sep.cumsum(-1, dtype=torch.int32) - is_sep.to(torch.int32)  # 分隔符归它结束的那篇
        doc_mask = doc_id.unsqueeze(-1) == doc_id.unsqueeze(-2)     # 同篇才允许
        mask = (doc_mask & causal_mask).unsqueeze(-3) # Insert head dim
    else:
        # Construct [1, 1, seq, seq]
        mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return mask

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__() 
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                max_seq_len = context_length,
                theta = rope_theta,
                device = device,
                dtype = dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model = d_model,device=device, dtype=dtype)
        self.lm_head = Linear(in_features = d_model, out_features = vocab_size, device = device, dtype = dtype)

    def forward(
        self,
        x: Int[Tensor, "... seq_len"],
        mask: Bool[Tensor, "... seq_len seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        # mask 是必填的。这个模型本身不知道自己是自回归的——因果性完全由传进来的
        # mask 决定（用 build_attention_mask 构造）。给它一个 None 默认值的话，
        # 忘传就等于全双向注意力，而且不报错、loss 还降得特别漂亮。
        # 缺省应该是"补全"或"报错"，不能是"跳过"。
        assert x.shape[-1] <= self.context_length, f"Input sequence length {x.shape[-1]} exceeds maximum context length {self.context_length}"
        
        seq_len = x.shape[-1]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        hidden_states = self.token_embeddings(x)

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask, token_positions=token_positions)

        # Return the unnormalized logits (no softmax applied!)
        return self.lm_head(self.ln_final(hidden_states))
