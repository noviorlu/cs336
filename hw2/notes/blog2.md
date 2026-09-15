# 单卡 5090 上的 Transformer 训练（二）：attention 的显存墙与 FlashAttention

Stanford CS336《Language Modeling from Scratch》作业 2「Systems」的实验记录，续 [第一篇](blog.md)（§2 剖析与基准、§3 单卡显存）。本篇是作业 §4 GPU Kernels：先量出朴素 attention 在长序列下的显存和时间，看 `torch.compile` 能救多少，然后用 Triton 写 FlashAttention-2 的前向和反向，最后和 PyTorch 版对比。章节号沿用作业。每小节先一句话说清问题，再 1–2 句作答加一张佐证表。

> 硬件 RTX 5090 32 GB（`torch` 可用 31.3 GiB；显存一律 GiB = 2³⁰ B），torch 2.11.0+cu130，fp32 基准 `allow_tf32=False`。术语沿用第一篇 §1.1（activation / saved tensors / entry / recompute / kernel vs op）。
> 第一篇的结论：xl@2048 一层 block 为反向存 3655 MiB，其中 56% 是 attention 的 S=QKᵀ 和 P=softmax(S) 两个 `[b,h,s,s]` 矩阵；它们随 seq² 增长，checkpointing 动不了，只能在 kernel 层面解决——这就是本篇。

---

## 4.1 朴素 attention 的基准（Benchmarking PyTorch Attention）

**问题**：batch 8、单头，d_head ∈ {16, 32, 64, 128} × seq ∈ {256, 1024, 4096, 8192, 16384}，各计时 100 次前向和 100 次反向，记反向前的显存。哪些配置 OOM？对最小的 OOM 配置做显存账。saved tensors 随 seq 怎么长？怎么消掉？

TODO

## 4.2 `torch.compile`（Torch Compile）

#### (a) 编译后的 attention

**问题**：同一组配置，`torch.compile` 后的 attention 前向 / 反向快多少。

TODO

#### (b) 编译整个模型

**问题**：把整个 Transformer `torch.compile`，前向、前向+反向、full step 各快多少。

TODO

## 4.3 FlashAttention-2 前向（flash_forward）

**问题**：先用纯 PyTorch 按 tile 写一遍前向（`torch.autograd.Function`，存 L = logsumexp 而不是 P），再用 Triton 写 kernel，支持 causal mask。

TODO

## 4.4 FlashAttention-2 反向（flash_backward）

**问题**：用前向存下的 O、L 重算 P，写反向（PyTorch 版即可，Triton 版可选）。

TODO

## 4.5 FlashAttention-2 基准（flash_benchmarking）

**问题**：`triton.testing.do_bench`，batch 1、causal，seq 2⁷…2¹⁶ × d ∈ {16…128} × {bf16, fp32}，对比 Triton 版和 PyTorch 版的前向 / 反向 / 端到端延迟。作业指定 B200；本机 5090 扫到装得下为止。

TODO
