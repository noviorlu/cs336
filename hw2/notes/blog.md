# CS336 A2 正文草稿

> 这份文件是**正文草稿**，每做完一节往里攒内容，全部做完再统一整理成文。
> 素材清单、篇章规划、写作规范、坑记录在 [todo.md](todo.md) 末尾的「博客产出规划（台账）」。

---

## 2.1 Benchmarking Script（端到端性能基准测试）

### 脚本长什么样（Deliverable a）

`cs336_systems/benchmark.py`。核心是一个**闭包工厂**：`make_step_fn()` 按配置组装出一个零参数的
`step()`，计时循环只管调它。这样后面 §2.2 加 NVTX、§2.4 加 autocast、§2.5 加 memory profiler、
§4.2 换 `torch.compile`，都不用碰计时代码。

三种执行模式：`forward` / `fwd_bwd` / `full`（含 optimizer step）。计时用
`timeit.default_timer()`（系统最高分辨率时钟），**每步结束后 `torch.cuda.synchronize()`**——
不同步的话测到的只是"CPU 把 kernel 排进队列"的时间，和 GPU 真正算完没有关系。

两个值得说的设计决定：

**一、`--inference` 和 `--mode` 是正交的两个开关。** handout 里 forward-only 有两种读法：
§2.2(d) 和 §2.5(a) 把它当**推理**（该包 `no_grad`），而 §2.1 的三种模式读起来是训练步的**前缀**。
拆成两个开关就都能满足：`--mode` 决定跑到哪一步，`--inference` 决定要不要建反向图。
**默认不带 `--inference`**，所以下表里的 `forward` 是训练步的前向。
这个区分不是吹毛求疵：混着列会让 xl 那格看起来"能跑"，而且会让下面的减法失效。

**二、反向和优化器的耗时是减出来的，不是直接测的。**

```
Backward  = fwd_bwd − forward
Optimizer = full    − fwd_bwd
```

也可以在一步内部的 forward 和 backward 之间插一次 `synchronize()` 直接分段计时，但那次同步会
**掐断 CPU/GPU 的流水重叠**——正常情况下 GPU 还在算前向时，CPU 已经在排反向的 kernel 了。
测量本身会把结果抬高，所以选了相减。代价是它隐含**可加性假设**，且两个被减数接近时差值会掉进噪声。

---

### 各阶段耗时（Deliverable b）

`batch=4, seq_len=512, fp32`，5 步预热 + 10 步测量，单位 ms：

| Size | forward (`no_grad`) | forward | fwd_bwd | full | → Backward | → Optimizer |
|:-----|--------------------:|--------:|--------:|-----:|-----------:|------------:|
| small  |  17.03 ± 0.13 |  17.18 ± 0.38 |  51.94 ± 0.54 |  55.68 ± 0.67 |  34.76 |  3.74 |
| medium |  49.99 ± 0.52 |  51.13 ± 0.52 | 154.42 ± 1.71 | 167.36 ± 1.55 | 103.29 | 12.94 |
| large  | 117.52 ± 0.67 | 118.07 ± 0.32 | 345.91 ± 0.54 | 372.84 ± 2.36 | 227.84 | 26.93 |
| **xl** | 346.40 ± 2.00 | **OOM**（forward 阶段） | **OOM** | **OOM** | — | — |
| **10B** | **OOM**（建模型阶段） | **OOM** | **OOM** | **OOM** | — | — |

**前向多久、反向多久？** 以 `medium` 为例：前向 **51.1 ms**，反向 **103.3 ms**，优化器 **12.9 ms**。

**波动大不大？** 很小。所有配置的标准差都在 **2.4 ms 以内**，相对值不超过 **1.4%**。
但要强调这是**跑满 5 步预热之后**的结果——不预热完全是另一回事，见下一节。

**反向 ≈ 2 × 前向**，三档实测比值 **2.02 / 2.02 / 1.93**。粗略的解释是矩阵乘的数量：
前向一个线性层算 `Y = X·Wᵀ` 一次乘法，反向要算两次——`dX = dY·W` 用来继续往下传，
`dW = Xᵀ·dY` 用来更新参数。但**别把它当成严格的 2 倍**：Transformer 里还有 attention 的反向、
归一化、以及最底层其实不需要算 `dX`，所以实测落在 1.93–2.02 而不是正好 2。

**xl 和 10B 为什么是空的？** 两者卡的位置不一样：**10B 连模型都建不起来**
（12.83B 参数 × 4 字节 = 51.3 GB，远超 32 GB）；**xl 的权重只有 13.6 GB，装得下**，
是前向途中堆积的激活把它顶了出去。所以 xl 还能做推理前向（`no_grad` 不存激活），
只是做不了训练。**单卡 32 GB 的边界就在 large**——它的完整训练步跑得动，余量不到 5 GB。

至于激活到底占了多少、大头在哪几个算子，那是 §2.5 `memory_profiling` 的专题，
要用 PyTorch 的 memory profiler 快照来看，这里先不展开。

---

### 不预热会怎样（Deliverable c）

**先说测法**：这一节的每个配置都跑在**独立进程**里。因为不预热的开销大半是**进程级一次性成本**
（CUDA context、kernel 懒加载、显存池首次扩张），同一个进程里连着跑 small → medium，
medium 的"第一步"其实已经白捡了 small 付过的账。跨 size 比较必须隔离，否则数字没有意义。

复现命令（`--isolate` 就是上面说的每配置独立进程）：

```sh
uv run python cs336_systems/benchmark.py --sweep --config warmup_by_size --isolate
uv run python cs336_systems/benchmark.py --sweep --config warmup_xl      --isolate
```

`seq_len=512, batch=4`，10 步测量。small/medium/large 用 `full`（完整训练步），
xl 的完整训练步 OOM，退而用推理前向（**模式不同，不与前三行横向比**）：

| Size | 模式 | 稳态（第 2 步起） | **第 1 步** | 绝对开销 | 倍数 |
|:-----|:-----|------------------:|------------:|---------:|-----:|
| small  | full          |  55.68 ms | **382.71 ms** | +327.0 ms | **6.87×** |
| medium | full          | 166.27 ms | **471.94 ms** | +305.7 ms | **2.84×** |
| large  | full          | 373.49 ms | **647.97 ms** | +274.5 ms | **1.74×** |
| xl     | forward/no_grad | 342.89 ms | **492.47 ms** | +149.6 ms | 1.44× |

**1. 不预热会怎样？** 10 步均值被一根尖刺拖高，标准差直接爆掉：

| Size | w=0 | w=1 | w=2 | w=5 |
|:-----|----:|----:|----:|----:|
| small  |  88.38 ± 103.42 |  55.67 ± 0.52 |  54.92 ± 0.68 |  55.65 ± 0.56 |
| medium | 196.84 ± 96.66  | 166.53 ± 0.94 | 167.04 ± 1.60 | 167.15 ± 1.81 |
| large  | 400.94 ± 86.83  | 374.43 ± 2.79 | 374.39 ± 2.04 | 375.07 ± 3.18 |
| xl     | 357.85 ± 47.34  | 351.59 ± 6.04 | 344.06 ± 3.22 | 347.91 ± 6.12 |

**注意这里有个 size 依赖，很容易被单个模型的结论误导**：第一步的**绝对**开销几乎与模型大小无关
（full 模式下 327 / 306 / 274 ms），但稳态步长从 56 ms 涨到 373 ms——所以**相对**冲击
从 **6.87× 一路缩到 1.74×**。也就是说：**模型越小，不预热的坑越深**。
如果只拿 small 做这个实验，会以为不预热是灾难性的（均值虚高 59%）；只拿 large 做，
又会觉得没什么大不了（虚高 7%）。

绝对开销随模型变大略降（327 → 274 ms），一个可能的解释是大模型的分配次数更少、
单次更大，向驱动要内存的往返次数反而少——但这条我们没有单独验证，**先当观察记着，不当结论**。

**2. 为什么第一步这么贵？** 一堆只付一次的开销：

- **CUDA 的 kernel 懒加载**。CUDA 12 起默认开启 lazy module loading，kernel 的机器码要到
  第一次真正被调用时才装载
- **cuBLAS 句柄和 workspace 的首次初始化**
- **PyTorch caching allocator 第一次向驱动 `cudaMalloc`**。之后显存池建好了，
  后续步骤直接复用已划分的块，分配开销断崖式下跌

**3. 那为什么 1–2 步预热"仍然可能不够"？——在这台机器上，四个 size 全都够了。**

这是实测和常见说法不一致的地方，如实记下来。`warmup=1` 之后，四个 size 的
「第 1 步 / 第 2 步起」比值全部落在 **0.97–1.01**，标准差回到 0.5–6.1 ms（相对值 ≤1.8%）。
**没有任何一档表现出"需要更多步才稳定"。**

常被引用的那几个"需要多步预热"的机制，在这个负载上都不成立或没体现：

- **cuDNN 的算法自动选优**（`torch.backends.cudnn.benchmark=True` 会连续试跑多种实现）
  只作用于**卷积**——Transformer 里一个卷积也没有。cuBLAS 走的是启发式选择，不做多轮实测
- **`torch.compile` / Triton 的 autotune** 确实需要多步收敛，但 §2.1 没开编译
- **GPU 频率爬坡**理论上存在，但若它显著，`warmup=1` 的结果应当仍明显偏慢——四个 size 都没有

所以诚实的回答是：**预期中的"1–2 步不够"没有复现**，而且这不是小模型的偶然，
从 small 到 xl 都是 1 步收敛。它什么时候才成立，留到 §4.2 打开 `torch.compile`
之后再看——那时才真的有 autotune 需要收敛。

---

---

**测量条件**

- 硬件：NVIDIA GeForce RTX 5090（32 GB）｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 配置：`vocab=10000, batch=4, seq_len=512`，fp32
- 测法：warmup 5 步 / measure 10 步（Deliverable c 的扫表另见该节），每步 `torch.cuda.synchronize()`，
  `timeit.default_timer()` 计时；**Deliverable c 的每个配置跑在独立进程里**
- 显存口径：`reset_peak_memory_stats()` 在 warmup 之后，只统计测量段；每步 `zero_grad(set_to_none=True)`
- 采集日期：2026-08-30

---

## 2.2 Nsight Systems Profiling

选了 **small + medium** 两个 size，seq_len 取 **256 / 512 / 1024**。
handout 要求「三个大于 128 的 2 的幂，最大那档取显存装得下的最长」——
`large` 在 5090 上只跑得到 512（1024 就 OOM），凑不齐三档；
而 small/medium 的 **1024 正好是各自上限**，2048 两者都 OOM。

采集用轻量档 `--trace=cuda,nvtx`，warmup 5 / measure 5，每档一份完整训练步的 profile。

### (a) 前向总耗时，和 timeit 对得上吗

**两个数必须当天一起测。** 第一版我拿今天的 nsys 数去比 §2.1 几天前发表的 timeit 数，
得到「有正有负、落进噪声」的结论——错的。同一配置隔几天重测会漂 3% 左右
（`small@512` 当时 17.18 ms，重测 16.66 ms），这个漂移把系统性的 profiler 开销盖住了。
重新当天同机测一遍：

| Size | Seq | nsys `forward` range | `timeit`（无 profiler） | 绝对差 | 相对差 |
|:-----|----:|---------------------:|------------------------:|-------:|-------:|
| small  |  256 |  10.17 ms |   9.44 ± 0.32 ms | +0.73 ms | **+7.7%** |
| small  |  512 |  17.50 ms |  16.66 ± 0.33 ms | +0.84 ms | **+5.0%** |
| small  | 1024 |  52.50 ms |  51.30 ± 0.70 ms | +1.20 ms | **+2.3%** |
| medium |  256 |  24.68 ms |  23.37 ± 0.38 ms | +1.31 ms | **+5.6%** |
| medium |  512 |  49.76 ms |  47.67 ± 0.64 ms | +2.09 ms | **+4.4%** |
| medium | 1024 | 150.78 ms | 146.52 ± 2.18 ms | +4.26 ms | **+2.9%** |

**对得上，而且 nsys 一致偏高 2.3%–7.7%。** 六档符号全为正，这才是 profiler 开销该有的样子。

**相对开销随负载变大而缩小**（7.7% → 2.9%）。nsys 的拦截成本主要花在 CPU 侧——
每次 CUDA API 调用都要记一笔；而调用次数只跟层数有关（small 12 层 vs medium 24 层），
跟 seq_len 无关。序列一长，GPU 那边的活变重，同样的拦截成本就被摊薄了。

不过这个"纯按调用次数"的模型解释不了全部：同一个 size 内调用次数不变，
绝对开销却仍随 seq 增长（medium 是 1.31 / 2.09 / 4.26 ms）。
差值都在 2σ 以上，不像纯噪声。**成因没查，先记为待解释**，别当结论用。

所以 (a) 的答案不止是"对得上"，还附带一条实用结论：
**profiler 的相对开销在小配置上最明显**，想量准就别拿最小的那档去做基准。

**但这个"对得上"是有前提的，而且前提差点没成立。** 见下。

### 差点得出前向比反向快 9 倍的结论

第一版的 NVTX range 里**没有加 `synchronize()`**，读出来是这样：

```
forward    33.7 ms
backward  299   ms       ← 9 倍？
```

真值是 **145 : 320，约 1 : 2.2**。

错因是 CUDA 的异步：`model(x)` 只是把 kernel **塞进队列**就返回了，塞完 1244 个花了 33.7 ms
（约 27 µs/次，是 Python + PyTorch dispatch 的开销）。此时 GPU 才刚开始算。
CPU 进到 `backward` 之后 GPU 还在啃前向——**前向的 GPU 时间被算进了 backward 的窗口**。

把 `synchronize()` 加进 range 结尾之后：

```
forward   147.89 ms      真值 144.97  ✓
backward  292.10 ms
optimizer  11.22 ms
zero_grad   0.64 ms
──────────────────
合计      451.85 ms
step      452.82 ms      ← 阶段之和覆盖 99.8%
```

比值回到 **1.97**。而且「阶段之和 ≈ step」是个免费的自洽校验，以后每份 profile 都能拿它验一遍。

有意思的是**序列化的代价几乎为零**：452.83 vs 453.48 ms，约 0.1%。
因为 GPU 本来就 93% 满载，插 sync 只是让 CPU 别再往前跑，并没有让 GPU 闲下来。

### 那根长长的 cudaDeviceSynchronize 不是浪费

时间线上 `forward` 的 149 ms 里，有 118 ms 是 CPU 卡在 `cudaDeviceSynchronize`。
容易误读成「大部分时间在等」。把 GPU 操作按起始时刻分桶就清楚了：

```
前 30 ms（CPU 还在 launch）:   271 个 GPU op，GPU 忙  29.90 ms
30 ms 之后（CPU 在 sync）  : 1069 个 GPU op，GPU 忙 121.39 ms
```

**80% 的 kernel 是在那根 sync 长条期间执行的**，且那段时间 GPU 忙碌率接近 100%。

```
CUDA API 行(CPU):  ▓▓ launch×1244 ▓▓│░░░░░ cudaDeviceSynchronize 阻塞 ░░░░░│
CUDA HW  行(GPU):  ███████████████████████████████████████████████████████
```

**"CPU 在等" 和 "GPU 在忙" 不矛盾——它们是两个处理器。** 这正是异步执行的意义。

反过来才该担心：sync 很短说明 CPU 刚发完 GPU 就算完了，**GPU 在饿肚子**，
瓶颈在 dispatch，优化方向就变成减少 launch 次数（算子融合、CUDA Graph），
而不是优化 kernel 本身。我们这里 sync 占 78%，是健康的 GPU-bound。

### 一个手动 NVTX 拿不到的东西

`backward` 的耗时能读（因为 sync 让时间窗口对齐了），但**归因不到它的 kernel**。
nsys 的 `nvtx_gpu_proj_sum`（把 range 投影到 GPU 时间线）给出：

```
:forward    Range 150.78 ms   Proj 150.68 ms   6700 个 GPU op
:backward   Range 296.70 ms   Proj   0.0007 ms     5 个 GPU op   ← ？
```

原因在时间线上一眼可见：多了一个线程 **`[19088] pt_autograd_0`**，
反向的 kernel 全是它发起的。而 **NVTX range 是 per-thread 的**——
我们在主线程 push 的 `backward`，覆盖不到另一个线程的 launch。

所以**手动包住 `loss.backward()` 永远拿不到它的 kernel 归因**，这跟 range 放在哪无关。
要按算子看反向，得让 PyTorch 自己往引擎里插：`--pytorch=autograd-shapes-nvtx`。

