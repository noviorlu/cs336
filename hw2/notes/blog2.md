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

被测的是 hw1 的 `scaled_dot_product_attention(Q, K, V)`（`S = QKᵀ/√d → softmax → PV`，softmax 是手写的四步版），输入 `[8, seq, d]`——batch 固定 8、**没有 head 维**。每个 (d, seq) 格子：预热 5 步，100 次带图前向计时，带图前向停在反向前读 `memory_allocated()`，再 100 次反向计时（每次重新前向，秒表只夹 `backward()`）；每次前向/反向后 `synchronize()`。脚本 `benchmark/attention_bench.py`，notebook `blog2.ipynb` §4.1。

**表 4.1-1** 前向 / 反向每次耗时（ms，fp32，100 次均值；OOM 格为炸在预热那一步）

|   d | 256         | 1024        | 4096         | 8192          | 16384   |
|----:|:------------|:------------|:-------------|:--------------|:--------|
|  16 | 0.08 / 0.19 | 0.18 / 0.51 | 4.63 / 11.81 | 18.33 / 44.82 | OOM     |
|  32 | 0.07 / 0.20 | 0.18 / 0.51 | 4.55 / 11.24 | 18.47 / 44.56 | OOM     |
|  64 | 0.07 / 0.20 | 0.17 / 0.51 | 4.69 / 11.80 | 19.24 / 46.64 | OOM     |
| 128 | 0.07 / 0.23 | 0.26 / 0.68 | 6.00 / 14.03 | 24.08 / 55.08 | OOM     |

**表 4.1-2** d=16 的三条显存线（GiB，`memory_allocated()` / `max_memory_allocated()`；其余 d 只多 Q/K/V/O，≤ 3%。OOM 格括号里按「几份 `[8,s,s]`」估：反向前 2 份、前向峰值 4 份、全程 6 份）

|                      |   256 |   1024 |   4096 |   8192 | 16384       |
|:---------------------|------:|-------:|-------:|-------:|:------------|
| 反向前 memory_allocated | 0.020 |  0.080 |  1.024 |  4.032 | OOM（估 16.0） |
| 前向峰值                 | 0.024 |  0.142 |  2.022 |  8.029 | OOM（估 32.0） |
| 前向+反向峰值              | 0.028 |  0.207 |  3.030 | 12.044 | OOM（估 48.0） |

两张表一起读：d 从 16 到 128 翻 8 倍，时间只涨 1.2 倍、显存只涨 3%；seq 翻 2 倍，时间和显存都涨 4 倍。**决定一切的是 seq，不是 d**。

表 4.1-2 三条线扣掉 16 MiB 的 cuBLAS workspace 后，比例是 2 : 4 : 6 份 `[8,s,s]`（seq 256 时一份才 2 MiB，被 workspace 盖住；seq ≥ 4096 后肉眼可见）——「反向前」不是峰值，前向途中会冲到它的 2 倍，反向再冲到 3 倍。seq 16384 反向前本该只有 16 GiB、31.3 GiB 装得下，但前向峰值 32 GiB 先撞墙。这 4 份是什么，下面的表 4.1-3 逐步数：

最小的 OOM 配置是 **d=16, seq=16384**——连 d 最小的那一列都在同一格挂掉。把这一格的显存逐张量数一遍，同时对照它上一格（seq 8192，能跑）的实测：

**表 4.1-3** d=16 的显存账：一次带图前向里每个 `[8, seq, seq]` fp32 张量什么时候出现（`seq²·32 B`：seq 8192 = 2 GiB / 格，16384 = 8 GiB / 格）

| 前向步骤                 | 新增张量      |   seq 8192 活着的 GiB |   seq 16384 外推 GiB | 归到 |
|:---------------------|:----------|-------------------:|-------------------:|:--|
| S = QKᵀ/√d           | S         |              2.008 |              8.032 | 🟨 T（反向只要 Q、K，S 不存） |
| x − max              | x_shifted |              4.009 |             16.035 | 🟨 T |
| exp(x_shifted)       | nume      |              6.009 |             24.035 | 🟩 A（`/sum` 的反向要它） |
| P = nume / sum       | P         |              8.009 |             32.036 | 🟩 A（PV 的反向要它） |
| 函数返回（释放 S、x_shifted） |           |              4.013 |             16.050 |  |

`[8, seq, d]` 的 Q、K、V、O 在 seq 16384 也只有 8 MiB 一个，不进账；实测比手算多出的恒定 16 MiB 是 cuBLAS 的 workspace（`_cuda_clearCublasWorkspaces()` 后消失）。

- **At what size do you get out-of-memory errors?** seq = 16384，四个 d 一起；seq ≤ 8192 全部能跑。炸在第一次前向的 `P = nume / sum`：那一刻 S、x_shifted、nume 三份 24 GiB 已经在显存里（实测报错时 `memory_allocated()` = 24.0 GiB），第四份 8 GiB 申请失败。
- **Do the accounting for the memory usage of attention in one of the smallest configurations that runs out of memory.** 表 4.1-3。seq 8192 那一列和实测对上（峰值 8.02 GiB、反向前 4.03 GiB），把每格乘 4 就是 seq 16384：峰值 32 GiB。其中只有 2 份是反向真正需要的，另外 2 份是手写 softmax 的局部变量 `x_shifted`、`nume` 绑到函数返回才释放——同样的 d=16 / seq=16384 换成融合的 `torch.softmax`（只有 S 和 P 两份），前向就过了，峰值 16.04 GiB。
- **How does the memory saved for backward change with the sequence length?** 留给反向的是 exp 输出和 P 两份 `[8, seq, seq]`，即 `2 × 8 × seq² × 4 B ∝ seq²`：表 4.1-2 每往右一格 ×4（0.080 → 1.024 → 4.032 GiB）。而 FLOPs 也是 `∝ seq² · d`，所以时间和显存同阶增长（表 4.1-1 也是每格 ×4）——显存墙先到，因为它有硬上限。
- **What would you do to eliminate this memory cost?** 不把 `[seq, seq]` 写回显存：把 Q 按行分块、K/V 按列分块，在片上完成 `QKᵀ → 在线 softmax → PV`，只往显存写 O 和每行的 logsumexp `L`（`[8, seq]`，O(seq)）；反向用 L 重算 P。§4.0 算过这笔账：bytes 从 2 GiB 掉到 32 MiB，attention 从 memory-bound 翻到 compute-bound。这就是 §4.3 要写的 FlashAttention-2。

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
