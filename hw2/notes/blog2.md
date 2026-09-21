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

同一份 sdpa 包一层 `torch.compile(sdpa)`，网格、预热、步数和 §4.1 完全一致（编译发生在预热的第一步）。脚本同 §4.1，notebook `blog2.ipynb` §4.2。

**表 4.2-1a** 前向每次耗时：eager → compiled（ms，括号是加速比）

|   d | 256                 | 1024                | 4096                | 8192                  | 16384   |
|----:|:--------------------|:--------------------|:--------------------|:----------------------|:--------|
|  16 | 0.08 → 0.07（×1.2） | 0.17 → 0.16（×1.1） | 4.70 → 1.66（×2.8） | 18.87 → 6.54（×2.9）  | OOM     |
|  32 | 0.07 → 0.08（×0.9） | 0.18 → 0.17（×1.0） | 4.78 → 1.69（×2.8） | 19.04 → 6.78（×2.8）  | OOM     |
|  64 | 0.07 → 0.10（×0.8） | 0.19 → 0.20（×0.9） | 4.93 → 1.88（×2.6） | 19.53 → 7.48（×2.6）  | OOM     |
| 128 | 0.07 → 0.09（×0.8） | 0.26 → 0.28（×0.9） | 6.01 → 3.00（×2.0） | 23.71 → 11.99（×2.0） | OOM     |

**表 4.2-1b** 反向每次耗时：eager → compiled（ms）

|   d | 256                 | 1024                | 4096                 | 8192                  | 16384   |
|----:|:--------------------|:--------------------|:---------------------|:----------------------|:--------|
|  16 | 0.20 → 0.13（×1.5） | 0.50 → 0.31（×1.6） | 12.15 → 4.84（×2.5） | 46.53 → 18.98（×2.5） | OOM     |
|  32 | 0.21 → 0.18（×1.2） | 0.51 → 0.34（×1.5） | 11.84 → 4.98（×2.4） | 45.81 → 20.26（×2.3） | OOM     |
|  64 | 0.21 → 0.21（×1.0） | 0.51 → 0.39（×1.3） | 12.00 → 5.28（×2.3） | 46.87 → 20.09（×2.3） | OOM     |
| 128 | 0.21 → 0.17（×1.3） | 0.67 → 0.48（×1.4） | 14.02 → 7.13（×2.0） | 54.42 → 27.50（×2.0） | OOM     |

**表 4.2-2** d=16 的显存峰值，eager vs compiled（GiB；其余 d 只多 Q/K/V/O）

|                           |   256 |   1024 |   4096 |   8192 | 16384   |
|:--------------------------|------:|-------:|-------:|-------:|:--------|
| 前向峰值（eager）         | 0.024 |  0.142 |  2.022 |  8.029 | OOM     |
| 前向+反向峰值（eager）    | 0.028 |  0.207 |  3.03  | 12.044 | OOM     |
| 前向峰值（compiled）      | 0.02  |  0.08  |  1.024 |  4.032 | OOM     |
| 前向+反向峰值（compiled） | 0.025 |  0.145 |  2.032 |  8.048 | OOM     |

- **编译后的 attention 前向 / 反向快多少？** 长序列（seq ≥ 4096）前向 2.0–2.9×、反向 2.0–2.5×；短序列（seq ≤ 1024）基本不变，前向甚至慢 10–20%。原因在 §4.1 的账里：长序列时时间花在读写 `[8,seq,seq]` 上，eager 的手写 softmax 是 4 个 kernel（max、减、exp、除）各读写一遍 S，compile 把它融成一个 kernel，`[8,seq,seq]` 的读写从 ~8 遍降到 2 遍；两个矩阵乘还是交给 cuBLAS，compile 动不了它们——所以 d=128 时加速比掉到 2.0×，矩阵乘占比越大能省的越少。短序列每个 kernel 只跑几十微秒，瓶颈是 launch 而不是带宽，融合省下的读写抵不过 compile 版多出的 guard 检查。
- **显存呢？** 表 4.2-2：前向峰值从 4 份 `[8,seq,seq]` 降到 **2 份**（8.03 → 4.03 GiB）——融合 softmax 不再有 `x−max`、`exp` 两个临时量，正是 §4.1 表 4.1-2 里「多出来的那 2 份」；全程峰值 6 → 4 份。seq 16384 还是 OOM，但炸的位置从前向挪到了反向：前向 2 份 16 GiB 装得下，反向再叠 dP、dS 到 4 份 32 GiB 就不行了。**compile 消掉的是实现的浪费，消不掉 `[seq, seq]` 本身。**

#### (b) 编译整个模型

整个 Transformer `torch.compile(model)`，走第一篇 §2.1 的整模型 benchmark（`BenchConfig(compile=True)`，batch 4、seq 512）。每个配置起独立子进程，且 inductor / triton 的缓存指到临时目录——不然第二次跑命中缓存，「首步」就不是冷编译了。

**表 4.2-3** small（0.13B）full step 不预热的首步与之后各步（ms，isolate 子进程，warmup 0 / steps 5）

| | 第 1 步 | 第 2–5 步均值 |
|:--|--:|--:|
| eager | 378 | 57.9 |
| `torch.compile` | **14 477** | 46.0 |

- **首步开销多大？** eager 的 378 ms 是第一篇 §2.1(c) 那笔进程级一次性开销（kernel 懒加载、cuBLAS 句柄、显存池）；compile 的 14.5 s 是 dynamo 抓图 + inductor 生成并编译 Triton kernel，**是 eager 首步的 38 倍、稳态步长的 300 倍**。第一篇说「eager 一步就稳、compile 留到 §4.2 看」，看到的就是这个数：预热 1 步够，但那 1 步要 15 秒；有磁盘缓存时第二次启动只要零点几秒。

**表 4.2-4** 三档 × forward / fwd_bwd / full，eager vs compiled（ms，batch 4 seq 512，warmup 5 / steps 10，isolate 子进程；large full 的 eager 在这次 sweep 里 OOM，是 notebook 进程还占着几 GiB——第一篇表 2.1-1 单独跑是 372.8 ms、峰值 27.5 GiB）

|                       | eager → compiled（ms）   | 峰值显存 eager → compiled（GiB）   |
|:----------------------|:-------------------------|:-----------------------------------|
| ('small', 'forward')  | 17.9 → 15.2（×1.18）     | 3.98 → 3.26                        |
| ('small', 'fwd_bwd')  | 53.0 → 43.2（×1.23）     | 4.08 → 3.37                        |
| ('small', 'full')     | 57.2 → 46.4（×1.23）     | 5.04 → 4.33                        |
| ('medium', 'forward') | 51.1 → 42.7（×1.20）     | 10.48 → 8.59                       |
| ('medium', 'fwd_bwd') | 158.8 → 126.8（×1.25）   | 10.58 → 8.69                       |
| ('medium', 'full')    | 170.6 → 136.3（×1.25）   | 13.74 → 11.85                      |
| ('large', 'forward')  | 122.8 → 98.1（×1.25）    | 20.18 → 16.65                      |
| ('large', 'fwd_bwd')  | 360.8 → 276.5（×1.30）   | 20.28 → 16.75                      |
| ('large', 'full')     | OOM (backward) → 301.2   | 27.51 → 23.97                      |

- **前向变了多少？前向 + 反向 + optimizer 呢？** 前向 1.18–1.25×，fwd_bwd 1.23–1.30×，full 1.23–1.25×（large 按第一篇的 372.8 算是 1.24×）；模型越大加速比越高。把反向减出来（fwd_bwd − forward）：small 35.2 → 28.0、medium 107.7 → 84.1、large 238.1 → 178.5 ms，反向省 20–25%，比前向多——反向的逐元素 kernel（SwiGLU、RMSNorm、softmax 的梯度）更多，融合的余地更大。optimizer 段（full − fwd_bwd）4.2 → 3.3、11.8 → 9.5 ms，也小了一截：`torch.compile(model)` 不碰 AdamW，省的是 `zero_grad` 和梯度落地那几个小 kernel。
- **为什么整模型只有 1.2×，attention 单独能到 2.9×？** 第一篇表 2.1-5：full step 里矩阵乘占 62%，compile 动不了它们（还是 cuBLAS）；能融合的逐元素 + 归约只有 ~30%，全部消掉也就 1.4× 的上限。§4.2(a) 的 attention 长序列能 2.9× 是因为那里矩阵乘只占一小半、softmax 的读写占大头。seq 512 时 `[b,h,s,s]` 才 32 MiB × 每层 4 份，attention 的融合收益在整模型里被 FFN 的矩阵乘稀释了。
- **显存**：峰值降 0.7 / 1.9 / 3.5 GiB（forward 列），三档都是 −17%——融合 kernel 不落中间量，和 §4.2(a) 表 4.2-2 同一件事，只是这里被压缩的是 SwiGLU / RMSNorm 的 `[b,s,d_ff]`、`[b,s,d]` 临时量，不是 `[s,s]`。large full 因此从 OOM 边缘（27.5）退到 24.0 GiB。

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
