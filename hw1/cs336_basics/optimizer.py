import torch
from torch.optim.optimizer import Optimizer
import math
import torch
import math
import einx
from jaxtyping import Float, Int
from torch import Tensor

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


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # Bias correction factor
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 1. Apply weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # 2. Update first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3. Update second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4. Apply moment-adjusted weight updates
                # p = p - alpha_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss

def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

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

    # 只有这里的 if 判断会引发一次隐式的同步（为了决定是否走进分支），
    # 但相比于在循环里几百次同步，这 1 次同步是完全可以接受的（也是 PyTorch 官方做法）。
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for g in grads:
            # 乘以 clip_coef 前，也要保证 clip_coef 在当前梯度所在的设备上
            g.detach().mul_(clip_coef.to(g.device))
