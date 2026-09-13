"""NVTX 探针——只服务 nsys profile（§2.2），timeit 计时基准线不经过这里。

设计：runner 拿到一个 Probes 对象，在各阶段调用 probes.range / probes.phase；
Probes 关掉时全部返回 nullcontext，runner 不需要知道 nvtx 是否开着。
"""
import contextlib
import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import softmax

from .config import BenchConfig


class Probes:
    """一次测量里的 NVTX 开关 + 门控。

    - range(name)：无门控。warmup / step 用它——预热要单独包一个 range，
      因为 `--filter-nvtx` 默认只取该 range 的**第一个实例**，若预热步也叫 "step"，
      §2.2 (b) 抓到的就是被冷启动污染的那一步。
    - phase(name)：受 `live` 门控，且**结尾 synchronize**。forward/backward/optimizer
      和 attention 三段都用它。预热期间 live=False，探针闭嘴。
    """

    def __init__(self, cfg: BenchConfig):
        self.enabled = cfg.nvtx and cfg.is_cuda
        self.attn = self.enabled and cfg.nvtx_attn
        self.ops = self.enabled and cfg.nvtx_ops
        self.live = False   # runner 在 warmup 结束后置 True

    def range(self, name: str):
        return nvtx.range(name) if self.enabled else contextlib.nullcontext()

    @contextlib.contextmanager
    def phase(self, name: str):
        """阶段 range，**结尾带 synchronize**——§2.2 (a) 问的就是每个 pass 多久，直接读它。

        为什么必须 sync：CUDA 异步，不同步的话 range 结束时 CPU 只是"把 kernel 排完队"，
        GPU 还在算，那个时长不是阶段耗时。实测 medium@1024 前向：不同步的 range 只有
        33.7 ms，真实 GPU 耗时约 147 ms（CPU 跑在前面 4.4 倍）。

        sync 还顺带解决 backward 无法归因的问题：`loss.backward()` 的 kernel 由 autograd
        引擎的**工作线程**发起，而 NVTX range 是 per-thread 的，主线程 push 的 range
        覆盖不到（实测 nvtx_gpu_proj_sum 只归到 5 个 GPU op，前向是 6700 个）。
        sync 让时间窗口对齐后改走窗口口径，这个限制就绕开了。

        代价：掐断 CPU/GPU 流水重叠，Σ(各阶段) > 不开 nvtx 时的总步长。
        """
        if not (self.enabled and self.live):
            yield
            return
        with nvtx.range(name):
            yield
            torch.cuda.synchronize()

    def emit_ops(self):
        """§2.5 (f)：给每个 aten 算子打 NVTX range（前向 `aten::mm` 等，反向带对应前向的序号）。
        必须同时包住 forward 和 backward，所以 runner 用它包整个 step。"""
        if self.ops and self.live:
            return torch.autograd.profiler.emit_nvtx()
        return contextlib.nullcontext()

    def install_block_ranges(self, model) -> None:
        """§2.5 (f)：每个 TransformerBlock 的前向包一个 `block{i}` range，做 GUI 里的导航；
        反向没有对应 range（autograd 在工作线程跑），靠 emit_nvtx 的序号对回去。"""
        if not self.ops:
            return
        def push(i):
            def hook(module, args):          # hook 返回非 None 会替换输入/输出，所以不能写成 lambda
                if self.live:
                    nvtx.range_push(f"block{i}")
            return hook

        def pop(module, args, out):
            if self.live:
                nvtx.range_pop()

        for i, blk in enumerate(model.layers):
            blk.register_forward_pre_hook(push(i))
            blk.register_forward_hook(pop)

    def install_attention_probes(self) -> None:
        """猴补 cs336_basics.model 里的自由函数——CausalMultiHeadSelfAttention 按模块全局查它。"""
        if not self.attn:
            return
        global _ACTIVE
        _ACTIVE = self
        basics_model.scaled_dot_product_attention = _annotated_sdpa


# _annotated_sdpa 与当前 Probes 之间的活引用：模型代码里查不到 cfg，只能走模块全局。
_ACTIVE: Probes | None = None


def _annotated_sdpa(q, k, v, mask=None):
    """与 cs336_basics.model.scaled_dot_product_attention 逐行等价，多三段 phase（§2.2 (e)）。

    每段结尾带 synchronize —— (e) 问的是各段各占多久，只有同步过的 range 宽度才等于
    GPU 耗时；代价是把 attention 内部串行化了，所以默认不开。
    注意 mask 语义跟着原函数走：masked_fill 屏蔽的是 mask 为 True 的位置。
    """
    p = _ACTIVE
    d_k = q.shape[-1]

    with p.phase("attn.scores"):
        QK = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        if mask is not None:
            QK = QK.masked_fill(mask, float("-inf"))

    with p.phase("attn.softmax"):
        softQK = softmax(QK, dim=-1)

    with p.phase("attn.matmul"):
        out = einsum(softQK, v, "... queries keys, ... keys d_v -> ... queries d_v")

    return out
