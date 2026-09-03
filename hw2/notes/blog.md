# CS336 A2 正文草稿

> 这份文件是**正文草稿**：每个 deliverable 一段 1–2 句的作答 + 佐证表格。
> 踩的坑、设计取舍、长分析都在 [todo.md](todo.md) 末尾的「博客产出规划（台账）」里。

---

## 2.1 Benchmarking Script

### (a) 脚本

`cs336_systems/benchmark.py`：按 CLI 参数初始化 `BasicsTransformerLM`、用 `torch.randint`
造随机批，跑 `w` 步预热后对 `n` 步计时，`--mode` 在 `forward` / `fwd_bwd` / `full`（含
optimizer step）之间切换；计时用 `timeit.default_timer()`，每步结束调 `torch.cuda.synchronize()`。

```sh
uv run python cs336_systems/benchmark.py --size medium --mode full --warmup 5 --steps 10
uv run python cs336_systems/benchmark.py --sweep --config default
```

### (b) 各阶段耗时

以 `medium` 为例，前向 **51.1 ms**、反向 **103.3 ms**、优化器 **12.9 ms**，反向约为前向的
两倍（三档实测 2.02 / 2.02 / 1.93）；**测量非常稳定**，所有配置的标准差都在 2.4 ms 以内
（相对值 ≤1.4%）。

`batch=4, seq_len=512, fp32`，5 步预热 + 10 步测量，单位 ms：

| Size | forward (`no_grad`) | forward | fwd_bwd | full | → Backward | → Optimizer |
|:-----|--------------------:|--------:|--------:|-----:|-----------:|------------:|
| small  |  17.03 ± 0.13 |  17.18 ± 0.38 |  51.94 ± 0.54 |  55.68 ± 0.67 |  34.76 |  3.74 |
| medium |  49.99 ± 0.52 |  51.13 ± 0.52 | 154.42 ± 1.71 | 167.36 ± 1.55 | 103.29 | 12.94 |
| large  | 117.52 ± 0.67 | 118.07 ± 0.32 | 345.91 ± 0.54 | 372.84 ± 2.36 | 227.84 | 26.93 |
| **xl** | 346.40 ± 2.00 | **OOM**（forward） | **OOM** | **OOM** | — | — |
| **10B** | **OOM**（建模型） | **OOM** | **OOM** | **OOM** | — | — |

> Backward 和 Optimizer 由相减得到（`fwd_bwd − forward`、`full − fwd_bwd`）。
> `forward` 列保留 autograd 图，与 `no_grad` 列不是同一个量。
> xl 权重 13.6 GB 装得下，是激活把它顶出了 32 GB；10B 仅 fp32 权重就 51.3 GB，建模型即 OOM。

### (c) 不预热会怎样

不预热时 10 步均值虚高 7%–59%、标准差从约 1 ms 暴涨到 87–103 ms，而波动**全部来自第 1 步**
（它是稳态的 1.7–6.9 倍），代价来自 CUDA kernel 懒加载、cuBLAS 句柄与 workspace 首次初始化、
以及显存池首次向驱动 `cudaMalloc`。**预期中的「1–2 步仍然不够」没有复现**：四个 size 在
`warmup=1` 后「第 1 步 / 第 2 步起」的比值全部落在 0.97–1.01——常被引用的 cuDNN 算法自选
只作用于卷积（本模型没有），而真正需要多步收敛的 `torch.compile` autotune 此处未启用。

每个配置跑在**独立进程**里（不预热的开销大半是进程级一次性成本，同进程连跑会让后面的配置
白捡前面的预热）。`seq_len=512, batch=4`，10 步测量：

| Size | 模式 | 稳态（第 2 步起） | 第 1 步 | 绝对开销 | 倍数 |
|:-----|:-----|------------------:|--------:|---------:|-----:|
| small  | full            |  55.68 ms | 382.71 ms | +327.0 ms | **6.87×** |
| medium | full            | 166.27 ms | 471.94 ms | +305.7 ms | **2.84×** |
| large  | full            | 373.49 ms | 647.97 ms | +274.5 ms | **1.74×** |
| xl     | forward/no_grad | 342.89 ms | 492.47 ms | +149.6 ms | 1.44× |

| Size | w=0 | w=1 | w=2 | w=5 |
|:-----|----:|----:|----:|----:|
| small  |  88.38 ± 103.42 |  55.67 ± 0.52 |  54.92 ± 0.68 |  55.65 ± 0.56 |
| medium | 196.84 ± 96.66  | 166.53 ± 0.94 | 167.04 ± 1.60 | 167.15 ± 1.81 |
| large  | 400.94 ± 86.83  | 374.43 ± 2.79 | 374.39 ± 2.04 | 375.07 ± 3.18 |
| xl     | 357.85 ± 47.34  | 351.59 ± 6.04 | 344.06 ± 3.22 | 347.91 ± 6.12 |

> 第一步的**绝对**开销几乎与模型无关（327 / 306 / 274 ms），但稳态步长从 56 涨到 373 ms，
> 所以**相对**冲击从 6.87× 缩到 1.74×——**模型越小，不预热的坑越深**。
>
> 复现：`--sweep --config warmup_by_size --isolate` 和 `--sweep --config warmup_xl --isolate`

---

## 2.2 Nsight Systems Profiling

选 **small + medium**，seq_len 取 **256 / 512 / 1024**（`large` 只跑得到 512，凑不齐三档；
small/medium 的 1024 正是各自上限）。采集 `--trace=cuda,nvtx`，warmup 5 / measure 5。

### (a) 前向总耗时，和 timeit 对得上吗

对得上：nsys 测得 10.2 / 17.5 / 52.5 ms（small）和 24.7 / 49.8 / 150.8 ms（medium），
与同机当天用 `timeit` 测得的值一致，**nsys 系统性偏高 2.3%–7.7%**，且相对开销随负载增大而下降。

| Size | Seq | nsys `forward` | `timeit`（无 profiler） | 绝对差 | 相对差 |
|:-----|----:|---------------:|------------------------:|-------:|-------:|
| small  |  256 |  10.17 ms |   9.44 ± 0.32 ms | +0.73 ms | **+7.7%** |
| small  |  512 |  17.50 ms |  16.66 ± 0.33 ms | +0.84 ms | +5.0% |
| small  | 1024 |  52.50 ms |  51.30 ± 0.70 ms | +1.20 ms | +2.3% |
| medium |  256 |  24.68 ms |  23.37 ± 0.38 ms | +1.31 ms | +5.6% |
| medium |  512 |  49.76 ms |  47.67 ± 0.64 ms | +2.09 ms | +4.4% |
| medium | 1024 | 150.78 ms | 146.52 ± 2.18 ms | +4.26 ms | **+2.9%** |

> 两组数必须当天同机测。隔几天重测同一配置会漂约 3%，足以盖住 profiler 的系统性开销。
> NVTX range 结尾必须 `synchronize()`，否则读到的是「CPU 排完队」的时间（前向只有 33.7 ms）。

### (b) 前向里最耗时的 kernel

六档中前向累计 GPU 时间最长的都是 `cutlass_80_simt_sgemm_128x256_8x4_tn_align1`（占前向
40.7%–69.6%），单次前向调用 37–169 次不等——次数随 `M = batch×seq` 增大而增加，因为 cuBLAS
会把原本用其他 tile 的 q/k/v/o 投影并入它。**加上反向后仍是它**（medium@1024 合并
forward+backward 后占 14.6%），因为反向的 GEMM 被拆到了三个 tile 变体上。

| Size | Seq | 冠军 kernel | 单次前向调用 | 占前向 | 次高的 GEMM 变体 |
|:-----|----:|:---|--:|--:|:---|
| small  |  256 | `128x256_8x4_tn` |  37 | 48.0% | `128x128_8x4_tn` × 48 |
| small  |  512 | `128x256_8x4_tn` |  85 | 69.6% | — |
| small  | 1024 | `128x256_8x4_tn` |  85 | 40.7% | — |
| medium |  256 | `128x256_8x4_tn` |  73 | 53.0% | `128x128_8x4_tn` × 96 |
| medium |  512 | `128x256_8x4_tn` |  73 | 49.2% | `256x128_8x4_tn` × 96 |
| medium | 1024 | `128x256_8x4_tn` | 169 | 44.1% | — |

> 单次前向的 matmul 总数 = `9L + 1`（`3L+1` 个 FFN/lm_head + `4L` 个 q/k/v/o + `2L` 个
> attention bmm），实测 small 109 次、medium 217 次，与手算完全一致。
>
> medium@1024 合并 forward+backward 后的排行：`128x256_tn` 62.72 ms（14.6%）>
> `256x128_nn` 54.11 ms（12.6%）> `128x128_nt` 38.12 ms。反向对每个线性层算两次 GEMM
> （`dX`、`dW`），转置模式不同，因此散在三个变体上，共 338 = 2 × 169 次。
>
> ⚠️ ③ 目前只在 medium@1024 上验证，另五档待补。

### (c) 除矩阵乘之外还有谁在吃时间

前向中逐元素 kernel（`elementwise_kernel` / `vectorized_elementwise_kernel`，来自 SwiGLU、
RoPE、mask、残差加）和归约 kernel（`reduce_kernel`，来自 RMSNorm 与 softmax）合计占
**18%–49%** 的 GPU 时间，且**随 seq_len 急剧上升**——这些算子访存受限、几乎不产生 FLOPs，
而 attention 的中间张量随序列长度平方增长。

各档**前向**的 kernel 时间构成：

| Size | Seq | GPU 总时长 | matmul | elementwise | reduce | 非 matmul 合计 |
|:-----|----:|-----------:|-------:|------------:|-------:|---------------:|
| small  |  256 |   9.3 ms | 79.3% | 17.9% | 2.2% | **20.1%** |
| small  |  512 |  16.4 ms | 76.8% | 20.4% | 2.4% | **22.8%** |
| small  | 1024 |  48.6 ms | 50.5% | 42.7% | 6.6% | **49.3%** |
| medium |  256 |  22.9 ms | 81.4% | 16.0% | 2.0% | **18.0%** |
| medium |  512 |  49.3 ms | 74.7% | 23.2% | 1.9% | **25.1%** |
| medium | 1024 | 142.2 ms | 53.2% | 40.1% | 6.6% | **46.7%** |

### (d) 完整训练步 vs 纯前向

完整训练步中矩阵乘的时间占比比纯前向**下降 6–18 个百分点**（medium@256 从 81.4% 降到
63.9%，medium@1024 从 53.2% 降到 46.7%），让出的份额几乎全被逐元素 kernel 吃掉
（16.0% → 34.4%）；原因是反向虽让 GEMM 数量翻倍，却同时产生大量梯度累加，而 AdamW
是纯逐元素运算。

| Size | Seq | matmul%（前向） | matmul%（完整步） | Δ | elementwise%（前向 → 完整步） |
|:-----|----:|---------------:|------------------:|---:|:---|
| small  |  256 | 79.3% | 64.1% | **−15.2** | 17.9% → 33.7% |
| small  |  512 | 76.8% | 61.1% | **−15.7** | 20.4% → 36.3% |
| small  | 1024 | 50.5% | 42.8% | −7.7 | 42.7% → 52.5% |
| medium |  256 | 81.4% | 63.9% | **−17.5** | 16.0% → 34.4% |
| medium |  512 | 74.7% | 60.0% | **−14.7** | 23.2% → 38.1% |
| medium | 1024 | 53.2% | 46.7% | −6.5 | 40.1% → 48.9% |

> handout 的基准是「inference (forward pass only)」。实测 `no_grad` 前向与训练步中的前向
> **kernel 调用次数完全相同、构成差 <0.3pp**（no_grad 省的是 CPU 侧建图记账，不产生额外
> GPU kernel），因此直接用 `forward` range 作基准。

### (e) attention 内 softmax vs 矩阵乘

softmax 的耗时**与它的 FLOPs 完全不成比例**：它比同在 attention 里的 `softQK·V` 矩阵乘贵 1.2–4.5 倍，
而占比最大的 `scores` 段（61%）之所以比 FLOPs 相同的 `matmul` 段慢 11.6 倍，也是因为它多带了
`/√d_k` 和 `masked_fill` 两个对 `[b, h, s, s]` 的全尺寸逐元素读写——三段都卡在带宽而非算力上。

| size | seq | forward ms | scores | softmax | matmul | attention 合计 |
|---|---|---|---|---|---|---|
| small | 256 | 11.49 | 3.14 (27%) | 0.92 (8%) | 0.75 (7%) | 4.81 (42%) |
| small | 512 | 19.66 | 7.99 (41%) | 2.03 (10%) | 1.64 (8%) | 11.66 (59%) |
| small | 1024 | 54.27 | 29.18 (54%) | 12.72 (23%) | 2.83 (5%) | 44.73 (82%) |
| medium | 256 | 27.35 | 11.06 (40%) | 1.78 (7%) | 1.69 (6%) | 14.53 (53%) |
| medium | 512 | 51.44 | 28.38 (55%) | 5.57 (11%) | 3.15 (6%) | 37.10 (72%) |
| medium | 1024 | 150.77 | 91.88 (61%) | 33.78 (22%) | 7.59 (5%) | 133.25 (88%) |

> 采法：`benchmark.py --nvtx --nvtx-attn` 把 `scaled_dot_product_attention` 换成带三段
> `_phase` 的等价实现（逐位相同，`max diff = 0.0`），每段结尾 sync，故 range 宽度即 GPU 耗时。
> 代价是 attention 被串行化，forward 比不插探针时慢约 7%，所以此表只用于**段间比占比**，
> 不能与 §2.1 的墙钟对账。数值为 5 步均值，`--mode forward`。
>
> 只覆盖前向：反向由 autograd 引擎在 `pt_autograd_0` 线程上跑 grad_fn，不会再进这个 Python
> 函数体，手插的 range 一条都不会触发（120 个 instance = 24 层 × 5 步，正好只有前向一遍）。


---

**测量条件**

- 硬件：NVIDIA GeForce RTX 5090（32 GB）｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0、Nsight Systems 2025.3.2
- 配置：`vocab=10000, batch=4`，fp32（`allow_tf32=False`，故矩阵乘走 SIMT FP32 而非 Tensor Core）
- 测法：warmup 5 / measure 10（§2.2 为 measure 5），每步 `torch.cuda.synchronize()`，
  `timeit.default_timer()` 计时；§2.1(c) 的每个配置跑在独立进程里
- 显存口径：`reset_peak_memory_stats()` 在 warmup 之后，只统计测量段；每步 `zero_grad(set_to_none=True)`
- 采集日期：2026-08-30（§2.1）、2026-08-31（§2.2）
