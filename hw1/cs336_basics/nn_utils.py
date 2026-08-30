import torch
import math
import einx
from jaxtyping import Float, Int
from torch import Tensor



def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:                      
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


def cross_entropy(o: Float[Tensor, "... vocab_size"], t: Int[Tensor, "..."]) -> Float[Tensor, ""]:                      
    M = einx.max("... v -> ...", o)                                                                     
    shifted = einx.subtract("... v, ... -> ... v", o, M)

    # 这里不能用 einx.get_at("... [v], ... -> ...", o, t)：它的实现是先把 o 摊平成
    # 一维、再用算好的线性下标去取，而那个长度是用 int32 算的。o 的元素数一旦过
    # 2^31，摊平长度就翻负，报 "invalid shape dimension -1673527296"。
    # batch=1024 时 [1024, 256, 10000] = 26.2 亿个元素，正好越界（26.2e8 - 2^32
    # = -1673527296）。torch.gather 按维度取，不摊平，也就没有这个上限。
    o_t = torch.gather(o, -1, t.unsqueeze(-1).long()).squeeze(-1)
    log_sum_exp = torch.log(einx.sum("... v -> ...", torch.exp(shifted)))

    return (M - o_t + log_sum_exp).mean() 




def gradient_clipping(parameters, max_l2_norm: float):
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # 把各个梯度张量的 L2 范数算出来，然后堆叠成一个新的张量，最后求一次总范数。
    # 关键点 1：用 torch.linalg.vector_norm 替代 .pow(2).sum()，不产生庞大的临时张量
    # 关键点 2：全程不调用 .item()，不引发 CPU-GPU 同步阻塞
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g.detach()).to(device) for g in grads])
    total_norm = torch.linalg.vector_norm(norms)

    # 用 torch.clamp 避免 if 判断引发的隐式 CPU-GPU 同步阻塞
    clip_coef = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef.to(g.device))

