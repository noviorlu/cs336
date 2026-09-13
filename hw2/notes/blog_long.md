# CS336 A2 正文草稿

> 这份文件是**正文草稿**：每个 deliverable 一段 1–2 句的作答 + 佐证表格。
> 踩的坑、设计取舍、长分析都在 [todo.md](todo.md) 末尾的「博客产出规划（台账）」里。

---

## 2.1 Benchmarking Script

### (a) 脚本

`benchmark/`（`python -m benchmark`）：按 CLI 参数初始化 `BasicsTransformerLM`、用 `torch.randint`
造随机批，跑 `w` 步预热后对 `n` 步计时，`--mode` 在 `forward` / `fwd_bwd` / `full`（含
optimizer step）之间切换；计时用 `timeit.default_timer()`，每步结束调 `torch.cuda.synchronize()`。

```sh
uv run python -m benchmark --size medium --mode full --warmup 5 --steps 10
uv run python -m benchmark --sweep --config default
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

> 采法：`--nvtx --nvtx-attn` 把 `scaled_dot_product_attention` 换成带三段
> `_phase` 的等价实现（逐位相同，`max diff = 0.0`），每段结尾 sync，故 range 宽度即 GPU 耗时。
> 代价是 attention 被串行化，forward 比不插探针时慢约 7%，所以此表只用于**段间比占比**，
> 不能与 §2.1 的墙钟对账。数值为 5 步均值，`--mode forward`。
>
> 只覆盖前向：反向由 autograd 引擎在 `pt_autograd_0` 线程上跑 grad_fn，不会再进这个 Python
> 函数体，手插的 range 一条都不会触发（120 个 instance = 24 层 × 5 步，正好只有前向一遍）。

---

## 2.3 Mixed Precision Accumulation

### 累加 1000 次 0.01

**决定精度的不是加数的 dtype，而是累加器的 dtype。** 四种写法里只有「fp16 累加器」漂掉了
（9.9531），另外两种 fp32 累加器无论加数是 fp16 还是显式转成 fp32，都落在同一个值 10.0021——
加数一旦进了 fp32 累加器，它本身是从 fp16 转来的这件事就不再重要。

| 累加器 | 加数 | 结果 | 与 10 的偏差 |
|:--|:--|--:|--:|
| fp32 | fp32 | 10.0001 | +0.001% |
| **fp16** | fp16 | **9.9531** | **−0.47%** |
| fp32 | fp16（隐式提升） | 10.0021 | +0.02% |
| fp32 | fp16 → `.type(fp32)` | 10.0021 | +0.02% |
| **bf16** | bf16 | **4.0** | **−60%** |
| fp32 | bf16（隐式提升） | 10.0098 | +0.1% |

> 后两行是把 fp16 换成 bf16 再跑一遍。bf16 的失败方式和 fp16 不同：不是慢慢漂，而是
> **卡死在 4.0**。bf16 只有 7 位尾数，在 [4, 8) 这一段相邻可表示数的间距是 2² × 2⁻⁷ = 0.03125，
> 0.01 不到间距的一半，round-to-nearest 直接把它舍掉，之后每一步 `s + 0.01 == s`。
> fp16 有 10 位尾数，同一段间距是 0.0039，0.01 还能存活，只是每步都在舍入，所以是漂而不是死。
>
> 第 3、4 行相等不是巧合：`s += x` 在 `s` 是 fp32 时会先把 `x` 提升到 fp32 再加，
> 显式 `.type(torch.float32)` 做的是同一件事。它们和第 1 行的 0.002 差距来自 0.01 本身
> 在 fp16 里就不精确（fp16 的 0.01 = 0.0100021，×1000 正好 10.0021；bf16 的 0.01 = 0.0100098，×1000 = 10.0098），
> 1000 次累加把这个表示误差原样放大——fp32 累加器忠实地加出了「被舍入过的 0.01」的 1000 倍。

---

## 2.4 Benchmarking Mixed Precision

### (a) fp16 autocast 下各组件的 dtype

**autocast 不改参数，只改算子。** 参数在上下文内外都是 fp32；矩阵乘的输出（fc1、logits）是
fp16，LayerNorm 和 loss 的输出被提升回 fp32，梯度与参数同为 fp32。

| 组件 | dtype | 为什么 |
|:--|:--|:--|
| 模型参数（上下文内） | fp32 | autocast 在每次算子调用时临时转换输入，不动存储的权重 |
| `fc1` 输出 | **fp16** | `linear` 在 autocast 的「降为 fp16」列表上 |
| `ln` 输出 | fp32 | `layer_norm` 在「保留 fp32」列表上，输入 fp16 也会被提升 |
| logits（`fc2` 输出） | **fp16** | 同 `fc1`；`fc2` 收到 fp32 输入又降回 fp16 |
| loss | fp32 | `mse_loss` 在「保留 fp32」列表上 |
| 梯度 | fp32 | 与参数 dtype 一致，反向结束时转回 |

> 换成 `dtype=torch.bfloat16` 重跑，模式完全相同（fc1/logits 为 bf16，ln 仍 fp32）——
> PyTorch 的算子分类表不区分两种 16 位格式。
>
> `backward()` 放在 autocast 块外面。反向的 dtype 在前向建图时就已经定了，
> 块内调 `backward()` 不会改变它，只会让块内额外的算子（clip、optimizer）意外走 autocast。

### (b) LayerNorm 为什么要特殊对待，换 bf16 还要吗

LayerNorm 里对精度敏感的是**归约**（均值、方差在特征维上的累加）和方差里的**平方**——前者是
§2.3 的累加器问题，后者是 fp16 最大值 65504 的溢出问题。换 bf16 后溢出风险消失（指数位与 fp32
相同），但归约问题**反而更糟**：bf16 只有 7 位尾数，比 fp16 还少 3 位，§2.3 里「卡死在 4.0」
就是它。所以仍然要保留 fp32，只是理由从「怕溢出」变成「怕精度不够」；而 LayerNorm 几乎不占算力
（§2.2(c) 中 reduce kernel 只占前向 2%–7%），保留 fp32 基本免费。

### (c) bf16 混合精度 vs fp32

bf16 autocast 让前向快 **1.9–2.3×**、反向快 **1.7–1.9×**，且**模型越大加速比越高**（前向
small 1.87× → medium 2.05× → large 2.30×）；反向的加速比始终低于前向，因为反向多出的梯度累加
是往 fp32 的 `.grad` 缓冲区里写，autocast 管不到它。xl 在 bf16 下**仍然 OOM**——autocast 不缩小
fp32 权重，反而额外缓存一份 bf16 副本，参数占大头的模型从它这里省不到显存。

`batch=4, seq_len=512`，5 步预热 + 10 步测量，单位 ms：

| Size | 阶段 | fp32 | bf16 autocast | 加速比 |
|:-----|:-----|-----:|--------------:|-------:|
| small  | forward  |  17.15 ± 0.44 |   9.18 ± 0.30 | 1.87× |
| small  | backward |  33.94        |  20.13        | 1.69× |
| medium | forward  |  50.07 ± 1.01 |  24.40 ± 0.41 | 2.05× |
| medium | backward | 102.33        |  57.31        | 1.79× |
| large  | forward  | 114.83 ± 1.10 |  49.94 ± 1.11 | **2.30×** |
| large  | backward | 219.96        | 117.59        | 1.87× |
| xl     | forward  | OOM           | **OOM**       | — |

> backward = `fwd_bwd − forward`（fwd_bwd 实测：small 51.09 / 29.31，medium 152.40 / 81.71，
> large 334.79 / 167.53）。复现：`--sweep --config mixed_precision`。

峰值显存（forward 模式）：

| Size | fp32 | bf16 autocast | 降幅 |
|:-----|-----:|--------------:|-----:|
| small  |  3.98 GB |  3.16 GB | −21% |
| medium | 10.49 GB |  8.34 GB | −20% |
| large  | 20.19 GB | 16.53 GB | −18% |
| xl     | OOM @ 29.18 GB | OOM @ 29.36 GB | — |

> **为什么模型越大加速比越高**：矩阵乘的形状随 `d_model` 变大（768 → 1280），Tensor Core
> 在大 GEMM 上更接近峰值，而小 GEMM 有更多比例的时间花在 launch 和 tile 边角上；同时每层里
> 与 `d_model` 无关的固定开销（RoPE、mask、`d_k` 缩放）被摊薄。反向的差距同理，但多了一项
> 不缩水的成本：梯度累加到 fp32 `.grad`，这部分 elementwise 在 bf16 下字节数一点没少。
>
> **加速比是两个效应叠加，不能全记在「精度减半」头上**：① 权重和激活从 4 字节变 2 字节，
> 访存量减半，这是 memory-bound 算子（elementwise / reduce）的全部收益；② 本文的 fp32
> 基准是 `allow_tf32=False`，矩阵乘走 `cutlass_80_simt_sgemm`（§2.2(b)），而 bf16 矩阵乘走
> Tensor Core——这一项是硬件路径切换，与位宽无关。要拆开两项，需要补一列 `allow_tf32=True`
> 的 fp32（TF32 走 Tensor Core 但字节数不变）。**TODO：决定是否加这一列。**
>
> **显存只降 ~20% 且随模型变大而缩小**：参数仍是 fp32（(a) 已验证），autocast 省的只有激活；
> 而它默认还会在上下文内缓存一份 bf16 权重副本（`cache_enabled=True`），每个参数**多占 2 字节**。
> 模型越大参数占比越高，能省的激活份额越小、多缓存的副本越大，所以 small −21% → large −18%，
> 到 xl 时 13.6 GB fp32 权重 + 6.8 GB bf16 副本 + 优化器无关的激活，forward 就已经装不下。
> 想让 xl 跑起来，靠的是把**参数本身**变成 bf16（或 §3 的激活检查点），不是 autocast。
>
> 采法：`--autocast`，`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
> 包住前向和 loss，`backward()` 在外。loss 也放在块内是为了让 `cross_entropy` 里的
> `exp` / `sum` / `log` 被提升到 fp32——它是自写函数，autocast 只认识里面的基础算子；
> 放在块外则整个 loss 在 bf16 里算。

---

## 2.5 Memory Profiling

### (a) 显存时间线

**TODO：截图待放。** 快照已生成（`profiles/mem_xl_seq{128,2048}_{forward,full}.pickle`，
`--memory-snapshot` 开关），待从 memory_viz 截「Active Memory Timeline」。

> 采法：`python -m benchmark --size xl --seq-len 128 --mode full --warmup 1 --steps 1 --memory-snapshot x.pickle`。
> 快照从预热开始记（xl@2048 在第一个前向就 OOM，只记测量段会什么都留不下），
> dump 放在 `finally` 里，OOM 时照样落盘。
>
> xl@128 full 的时间线**没有下坡**：前向阶梯式爬 4.5 GB（32 层各存一级反向要用的激活），
> 反向每释放一层激活（~140 MB）就分配一层权重梯度（~315 MB），净值继续爬到 25.6 GB，
> 然后 optimizer 一分配 Adam 状态就 OOM。「反向释放激活」被「反向生成梯度」完全盖住。

### (b) 峰值显存

xl 的完整训练步在 32 GB 上**任何 seq 都装不下**——不是激活的问题，是 AdamW 第一次 `step()`
给 3.4B 参数分配两份 fp32 状态（2 × 13.6 GB），权重 + 梯度 + 状态 = 54 GB。前向本身只要
12.9 GB（128）/ 21.4 GB（2048）。

xl，`batch=4`，fp32，单位 GB（`max_memory_allocated`，预热后清零）：

| seq | forward（no_grad） | forward（带图） | fwd_bwd | full |
|----:|------:|------:|------:|:-----|
| 128  | 12.90 | 18.09 | 25.56 | **OOM @ optimizer**（29.35） |
| 1024 | 15.10 | OOM @ forward（28.94） | OOM | OOM |
| 2048 | 21.38 | OOM @ forward（25.96） | OOM | OOM |

> `status` 列记录死在哪个阶段，OOM 行括号里是炸掉前的水位。1024 不在题目要求里，
> 顺手测的：带图前向从 128 的 18 GB 跳到 OOM，激活随 seq 线性、attention 分数随 seq² 涨。
>
> 每参数 18 字节的账：fp32 主权重 4 + bf16 工作副本 2 + fp32 梯度 4 + Adam m/v 8。
> xl 3.4B × 18 B = 61 GB——这是 §6 优化器分片 / §7 FSDP 要拆的东西。

### (c) 混合精度对峰值显存的影响

**bf16 autocast 不但不省，反而多用 4–6 GB**：xl 的 no_grad 前向从 12.9 → 19.2 GB（128）、
21.4 → 25.3 GB（2048）。多出来的正好是一份 bf16 权重副本（3.4B × 2 B = 6.8 GB，减去激活
省下的 1–2 GB）；训练模式下（fwd_bwd）两者持平（25.56 vs 25.55），full 同样 OOM 在 optimizer。

| seq | 模式 | fp32 | bf16 autocast | Δ |
|----:|:-----|-----:|------:|----:|
| 128  | forward（no_grad） | 12.90 | **19.18** | +6.3 |
| 128  | fwd_bwd            | 25.56 | 25.55 | ±0 |
| 128  | full               | OOM @ optimizer | OOM @ optimizer | — |
| 1024 | forward（no_grad） | 15.10 | **20.60** | +5.5 |
| 2048 | forward（no_grad） | 21.38 | **25.27** | +3.9 |
| 2048 | 带图（三种模式）    | OOM @ forward | OOM @ forward | — |

> 算术核对（1024）：fp32 激活 = 15.10 − 13.6 = 1.5 GB，bf16 后 0.75；13.6 + 6.8 + 0.75 = 21.2 ≈ 实测 20.6。
> 速度倒是真快了一倍（773 → 346 ms @1024）。

**为什么会有这份副本，以及它在推理和训练里是两回事：**

Tensor Core 的 bf16 矩阵乘只吃 bf16 输入，权重存的是 fp32，所以每个 `linear` 执行前都得有
一份 bf16 的 `W`——转换本身躲不掉。问题是转出来的 `W_bf16` 什么时候能扔：

- **推理（no_grad）**：矩阵乘算完就没用了，但 autocast 把它缓存到 `with` 块结束
  （`cache_enabled=True`，假设同一块里会再用，省掉重复转换的带宽）。这时副本**纯粹是用显存换带宽**，
  `cache_enabled=False` 可以关掉；推理部署的正确做法是干脆 `model.to(bfloat16)`，
  权重只存一份 6.8 GB，没有 fp32 底座也没有副本。
- **训练**：反向要算 `dx = dy · Wᵀ`，`dy` 是 bf16，要进 Tensor Core 就得配 bf16 的 `W`。
  autograd 的规则是「前向时算子看见什么输入就存什么」，矩阵乘看见的是 `W_bf16`，
  于是它作为 saved tensor 一直活到反向用完——反向倒着走，第 1 层的 `W` 最后才用到，
  所以 32 层的副本整个反向期间都在。这时副本是**反向的输入**，关缓存也省不掉，
  只是从 autocast 的缓存变成 autograd 的 saved tensor（fwd_bwd 两者持平就是证据）。

能不能不存、反向时从 fp32 现转？能。代价是每步多读 13.6 GB + 写 6.8 GB，5090 上约 11 ms，
xl 一步 2 秒，多 0.5%——对 xl 这笔交易明显划算，PyTorch 默认不做只因为它是通用工具，
大多数模型的激活远大于权重、副本不值一提。这和 §3 gradient checkpointing 是同一个思想：
**反向需要的东西，是存下来还是现算**。大规模训练框架的做法是反过来：权重本身存 bf16，
fp32 主副本放进优化器状态，前向反向零转换，fp32 只在 `step()` 时出现，再配 §6 的分片。

### (d) 残差流张量大小

**TODO。** 提示：`[batch, seq, d_model] × 4 B / 1024²`，xl `d_model=2560`。

### (e) 最大的几笔分配

**TODO。** 待在 memory_viz 里把 Detail 拉到 10% 看 `mem_xl_seq2048_forward.pickle`。

### (f) nsys 显存 trace：单层 TransformerBlock 为反向存了多少

**TODO。** 需要 `--cuda-memory-usage=true` + `emit_nvtx`，开关待加。

---

**测量条件**

- 硬件：NVIDIA GeForce RTX 5090（32 GB）｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0、Nsight Systems 2025.3.2
- 配置：`vocab=10000, batch=4`，fp32（`allow_tf32=False`，故矩阵乘走 SIMT FP32 而非 Tensor Core）
- 测法：warmup 5 / measure 10（§2.2 为 measure 5），每步 `torch.cuda.synchronize()`，
  `timeit.default_timer()` 计时；§2.1(c) 的每个配置跑在独立进程里
- 显存口径：`reset_peak_memory_stats()` 在 warmup 之后，只统计测量段；每步 `zero_grad(set_to_none=True)`
- 采集日期：2026-08-30（§2.1）、2026-08-31（§2.2）、2026-09-12（§2.3–2.5）
