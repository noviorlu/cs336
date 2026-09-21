# 单卡 5090 上的 Transformer 训练（二）：attention 的显存墙与 FlashAttention

Stanford CS336《Language Modeling from Scratch》作业 2「Systems」的实验记录，续 [第一篇](blog.md)（§2 剖析与基准、§3 单卡显存）。本篇是作业 §4 GPU Kernels：先量出朴素 attention 在长序列下的显存和时间，看 `torch.compile` 能救多少，然后用 Triton 写 FlashAttention-2 的前向和反向，最后和 PyTorch 版对比。章节号沿用作业。每小节先一句话说清问题，再 1–2 句作答加一张佐证表。

> 硬件 RTX 5090 32 GB（`torch` 可用 31.3 GiB；显存一律 GiB = 2³⁰ B），torch 2.11.0+cu130，fp32 基准 `allow_tf32=False`。术语沿用第一篇 §1.1（activation / saved tensors / entry / recompute / kernel vs op）。
> 第一篇的结论：xl@2048 一层 block 为反向存 3655 MiB，其中 56% 是 attention 的 S=QKᵀ 和 P=softmax(S) 两个 `[b,h,s,s]` 矩阵；它们随 seq² 增长，checkpointing 动不了，只能在 kernel 层面解决——这就是本篇。

---

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

- **从哪个规模开始 OOM？** seq = 16384，四个 d 一起。炸在前向途中：4 份 `[8,16384,16384]` 要 32 GiB，卡上只有 31.3。
- **对最小的 OOM 配置做显存账。** 表 4.1-2 的 seq 16384 列（d=16）。Q/K/V/O 各 8 MiB，不进账。其中真正要留给反向的只有 2 份 = 16 GiB，装得下；多出来的 2 份是手写 softmax 的临时量——换成融合的 `torch.softmax` 前向就能过（实测峰值 16.04 GiB），但反向再叠 dP、dS 后至少 4 份 = 32 GiB，还是过不去。**这一格在 32 GB 的卡上没有任何朴素实现能跑完。**
- **留给反向的显存随 seq 怎么变？** `2 × 8 × seq² × 4 B ∝ seq²`：表 4.1-2 里 8192 → 16384 是 4.03 → 16 GiB。FLOPs 同样 ∝ seq²·d，所以时间和显存同步涨 4 倍——显存墙先到，因为它有硬上限。
- **怎么消掉这笔显存？** 不让 `[seq, seq]` 落显存：Q 按行分块、K/V 按列分块，片上完成 `QKᵀ → 在线 softmax → PV`，只写回 O 和每行的 logsumexp（`[8, seq]`），反向用它重算 P。按第一篇 §2.1 的 roofline 算这笔账（seq 4096、d 64）：朴素版三个 kernel 要读写 S、P 共 2 GiB，算术强度 I < 60 的 ridge point，memory-bound，带宽下限 1.2 ms；不落盘后只剩 Q/K/V/O 32 MiB，I ≈ 1000，翻到 compute-bound，下限 0.3 ms——所以能期待的是 3–5× 而不是 bytes 之比的 60×。这就是 §4.3 的 FlashAttention-2。

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

bf16 走 tensor core，5090 峰值 2.10e14 FLOPS、ridge point 117（fp32 是 1.05e14 / 60，第一篇 §1.1）：同一个 kernel 在 bf16 下更容易掉回 memory-bound，对比时要分开看。

TODO
