# 单卡 5090 上的 Transformer 训练（二）：attention 的显存墙与 FlashAttention

Stanford CS336《Language Modeling from Scratch》作业 2「Systems」的实验记录，续 [第一篇](blog.md)（§2 剖析与基准、§3 单卡显存）。本篇是作业 §4 GPU Kernels：先量出朴素 attention 在长序列下的显存和时间，看 `torch.compile` 能救多少，然后用 Triton 写 FlashAttention-2 的前向和反向，最后和 PyTorch 版对比。章节号沿用作业。每小节先一句话说清问题，再 1–2 句作答加一张佐证表。

> 硬件 RTX 5090 32 GB（`torch` 可用 31.3 GiB；显存一律 GiB = 2³⁰ B），torch 2.11.0+cu130，fp32 基准 `allow_tf32=False`。术语沿用第一篇 §1.1（activation / saved tensors / entry / recompute / kernel vs op）。
> 第一篇的结论：xl@2048 一层 block 为反向存 3655 MiB，其中 56% 是 attention 的 S=QKᵀ 和 P=softmax(S) 两个 `[b,h,s,s]` 矩阵；它们随 seq² 增长，checkpointing 动不了，只能在 kernel 层面解决——这就是本篇。

---

## 4.0 预备：Roofline

Roofline 回答「这段代码在这块卡上最快能跑多快」，用来判断后面每个数字合不合理。

- 硬件两个峰值：算力 P、带宽 B
- 代码一个数：算术强度 $I = \text{FLOPs}/\text{bytes}$（每读写 1 byte 显存能做多少次运算）
- 下限 $t_{\min} = \max(\text{FLOPs}/P,\ \text{bytes}/B)$；I < P/B 是 **memory-bound**（在等数据），I > P/B 是 **compute-bound**。P/B 叫 ridge point，随精度变：

| | P（FLOPS） | B（B/s） | ridge point P/B |
|---|---|---|---|
| 5090 fp32（CUDA core） | 1.05e14 | 1.79e12 | **60** ← §4.1 |
| 5090 bf16（tensor core） | 2.10e14 | 1.79e12 | 117 ← §4.5 |
| H100 fp32 | 6.7e13 | 3.35e12 | 20 |
| H100 bf16 | 9.9e14 | 3.35e12 | 295 |

**朴素 attention 的账**（fp32，batch 8，单头，seq 4096，d 64）。三个 kernel，S、P 各是 `[8, 4096, 4096]` fp32 = 512 MiB，都要落显存：

| kernel | FLOPs | bytes | I |
|---|---|---|---|
| S = QKᵀ | 1.7e10 | 写 S 512 MiB | 32 |
| P = softmax(S) | 6.7e8 | 读 S 写 P 1 GiB | < 1 |
| O = PV | 1.7e10 | 读 P 512 MiB | 32 |

三个 I 都在 60 以下 → memory-bound；总 bytes 2 GiB / B ≈ **1.2 ms** 是下限，其中一半花在几乎不算数的 softmax 上。

- **比 roofline 慢** = 实测远大于 1.2 ms，差距是实现问题（softmax 拆成多个 kernel 反复读写 S、launch 太多），要开 nsys 才知道是哪条。
- **FlashAttention** = 不把 S/P 写回显存，bytes 只剩 Q/K/V/O 共 32 MiB，I ≈ 1000，翻到 compute-bound，下限 0.3 ms。所以期望加速 3–5×（被算力封顶），不是 bytes 之比的 60×。反向用 logsumexp 重算 P，多一次 QKᵀ 换掉 P/dS 的读写，在 memory-bound 区间划算。

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
