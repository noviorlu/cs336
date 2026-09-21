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

## 4.1 朴素 attention 的基准（PyTorch Attention Benchmarking）

被测的是 hw1 的 `scaled_dot_product_attention(Q, K, V)`：`S = QKᵀ/√d → P = softmax(S) → O = PV`。输入 `[8, seq, d]`，batch 固定 8、没有 head 维；fp32；每格预热 5 步，100 次前向、100 次反向各取均值，每次后 `synchronize()`。脚本 `benchmark/attention_bench.py`，notebook `blog2.ipynb` §4.1。

**表 4.1-1** 前向 / 反向每次耗时（ms）

|   d | 256         | 1024        | 4096         | 8192          | 16384   |
|----:|:------------|:------------|:-------------|:--------------|:--------|
|  16 | 0.08 / 0.19 | 0.18 / 0.51 | 4.63 / 11.81 | 18.33 / 44.82 | OOM     |
|  32 | 0.07 / 0.20 | 0.18 / 0.51 | 4.55 / 11.24 | 18.47 / 44.56 | OOM     |
|  64 | 0.07 / 0.20 | 0.17 / 0.51 | 4.69 / 11.80 | 19.24 / 46.64 | OOM     |
| 128 | 0.07 / 0.23 | 0.26 / 0.68 | 6.00 / 14.03 | 24.08 / 55.08 | OOM     |

**主线只有一句：一切随 seq² 走，和 d 无关。** seq 翻倍，时间 ×4；d 翻 8 倍，时间只 ×1.2。seq 16384 四个 d 一起 OOM，连 d=16 都不例外。

原因是中间量 S 和 P 的形状是 `[8, seq, seq]`——里面没有 d。fp32 下一份 = `seq² × 32 B`：seq 8192 是 2 GiB，16384 是 8 GiB。一次前向+反向里同时活着几份，决定了峰值：

**表 4.1-2** d=16 同时活着的 `[8, seq, seq]` 份数与显存（GiB；seq 8192 实测，16384 按 ×4 外推）

| 时刻 | 份数 | 是哪几份 | seq 8192 | seq 16384 |
|:--|--:|:--|--:|--:|
| 前向峰值 | 4 | S、`x−max`、`exp`、P（手写 softmax 的局部变量到返回才释放） | 8.03 | **32 > 31.3 → OOM** |
| 反向前（saved tensors） | 2 | `exp` 的输出、P | 4.03 | 16 |
| 前向+反向峰值 | 6 | 上面 2 份 + dP、dS 等 | 12.04 | 48 |

- **At what size do you get out-of-memory errors?** seq = 16384，四个 d 一起。炸在前向途中：4 份 `[8,16384,16384]` 要 32 GiB，卡上只有 31.3。
- **Do the accounting for the memory usage in one of the smallest configurations that runs out of memory.** 表 4.1-2 的 seq 16384 列（d=16）。Q/K/V/O 各 8 MiB，不进账。其中真正要留给反向的只有 2 份 = 16 GiB，装得下；多出来的 2 份是手写 softmax 的临时量——换成融合的 `torch.softmax` 前向就能过（实测峰值 16.04 GiB），但反向再叠 dP、dS 后至少 4 份 = 32 GiB，还是过不去。**这一格在 32 GB 的卡上没有任何朴素实现能跑完。**
- **How does the memory saved for backward change with the sequence length?** `2 × 8 × seq² × 4 B ∝ seq²`：表 4.1-2 里 8192 → 16384 是 4.03 → 16 GiB。FLOPs 同样 ∝ seq²·d，所以时间和显存同步涨 4 倍——显存墙先到，因为它有硬上限。
- **What would you do to eliminate this memory cost?** 不让 `[seq, seq]` 落显存：Q 按行分块、K/V 按列分块，片上完成 `QKᵀ → 在线 softmax → PV`，只写回 O 和每行的 logsumexp（`[8, seq]`），反向用它重算 P。§4.0 算过：bytes 从 2 GiB 到 32 MiB。这就是 §4.3 的 FlashAttention-2。

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
