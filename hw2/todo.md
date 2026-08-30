# CS336 Assignment 2 (Systems) — 完整 TODO

> 依据 `cs336_assignment2_systems.pdf`（Version 26.1.3, Spring 2026, 48 页）逐节梳理。
> 27 个 Problem，**总分 137**。hw1 重构相关的清单已移到 [todo-hw1-refactor.md](todo-hw1-refactor.md)。
>
> **图例**：`[代码]` 有 pytest 判分 · `[写作]` 只进 writeup.pdf · `[图]` 需要截图/曲线 ·
> `⚠️多卡` 需要 >1 GPU · `⚠️大显存` 单卡 32G 放不下

## 交付物

| 文件 | 内容 |
|---|---|
| `writeup.pdf` | 所有written question 的答案，必须**排版**（typeset），不能手写 |
| `code.zip` | 跑 `./test_and_make_submission.sh` 生成 |
| leaderboard | 提交到 github.com/stanford-cs336/assignment2-systems-leaderboard |

## 分值分布

| 章节 | 主题 | 题数 | 分值 |
|---|---|---:|---:|
| §2 | Profiling & Benchmarking | 5 | 16 |
| §3 | Single-GPU Memory（激活检查点） | 1 | 4 |
| §4 | GPU Kernels（FlashAttention-2 / Triton） | 5 | **29** |
| §5 | Distributed Data Parallel | 6 | 21 |
| §6 | Optimizer State Sharding | 2 | 20 |
| §7 | Fully-Sharded Data Parallel | 2 | 20 |
| §8 | 并行策略分析（纯推导） | 5 | 17 |
| §9 | Leaderboard | 1 | 10 |
| | | **27** | **137** |

分值集中在三块：**FlashAttention (29)**、**FSDP (20)**、**Optimizer Sharding (20)**，
合计 69 分（占一半）且全部有 pytest 判分。§8 的 17 分是纯纸面推导，性价比最高，可以先做。

---

## ⚠️ 硬件现实（先读这段，会影响整份计划）

本机是**单卡 RTX 5090 / 32 GB**；题面默认 **B200（180 GB）**，多处要求 **2–6 张卡**。

**好消息：所有判分测试都能在本机跑。** 四个测试文件全部用 `mp.spawn` + **gloo 后端在 CPU 上**
起 `world_size=2`，不需要真的有两张卡：

```
tests/test_ddp.py:40                _setup_process_group(..., backend="gloo")
tests/test_sharded_optimizer.py:31  backend="gloo"
tests/test_fsdp.py:120              backend="gloo"
tests/test_attention.py             单卡即可（Triton kernel）
```

**坏消息：benchmarking 题目大量指定 xl 模型。** 实测参数量与显存需求（vocab=10000）：

| size | params | fp32 权重 | 权重+梯度+AdamW 两个动量 | 单卡 32G |
|---|---:|---:|---:|:--:|
| small | 0.13B | 0.5 G | 2.1 G | ✓ |
| medium | 0.42B | 1.7 G | 6.8 G | ✓ |
| large | 0.97B | 3.9 G | 15.5 G | ✓（激活要省着用） |
| **xl** | **3.41B** | 13.6 G | **54.5 G** | ✗ |
| 10B | 12.83B | 51.3 G | 205.3 G | ✗ |
| leaderboard | 8.13B | — | bf16 权重就 16.3 G | ✗（题面要求两张 B200） |

### 云 GPU 选项

课程给自学者列的算力来源（单张 B200 公开报价，**采集于 2026-03-28，已过去 5 个月，用前先核**）：

| provider | 单卡 B200 | 备注 |
|---|---:|---|
| **[Modal](https://modal.com/)**（课程赞助商） | $6.25/h | **每月送 $30 免费额度**；按实际计算计费，**不为空闲付费**；本地开发↔GPU 实验切换的 UX 好 |
| [Lambda Labs](https://lambda.ai/) | $6.69/h | |
| **[RunPod](https://www.runpod.io/)** | **$4.99/h** | 非抢占里最便宜 |
| **[Nebius](https://nebius.com/)** | $5.50/h（**抢占 $3.05/h**） | 抢占价是全场最低 |
| [Together](https://www.together.ai/) | $7.49/h | **最少 8 卡起**；长期承诺更便宜 |

**课程自己的建议**：先在 **CPU 上把正确性调通**，GPU 只用来跑训练（A1/A4/A5）或
**benchmark GPU 算子（A2，也就是本作业）**。这正好印证了本文档最后那张阶段表的排法。

### 每道题实际需要什么卡

**全文只有 2 处硬性要求 B200**（`grep -n B200` 核对过）：

| 题目 | 卡数 | 型号 | 原文 |
|---|---|---|---|
| §4.5 `flash_benchmarking` | 1 | **B200 强制** | "run the benchmark on a single B200"（L1560） |
| §9 `leaderboard` | 2 | **B200 强制** | "measured on two B200 GPUs"（L2593/L2661） |
| §5.1 `distributed_communication` | **最多 6** | 不限 | "Up to 6 GPUs. Each benchmarking run should take less than 5 minutes."（L1800） |
| §5.3 / §5.4 / §5.6 | 2 | 不限 | "1 node x 2 GPUs, xl"（L1887/L2006/L2011） |
| §6.2 `..._accounting` | 2 | 不限 | "1 node, 2 GPUs, xl"（L2079/L2088） |
| §7.2 `fsdp_accounting` | 2 | 不限 | "Profile the xl model on two GPUs"（L2156） |
| §2.2 nsys / §2.5 memory / §3(b) | 1 | 不限，但要 **>55 G 显存**跑 xl 完整训练步 | |

也就是说除了那两道，其余用 A100-80G / H100 / H200 都行，只要在 writeup 里写清楚跑在什么硬件上。
**唯一需要 6 张卡的是 §5.1，而且每次运行 <5 分钟** —— 这题最适合 Modal 的按秒计费。

### 粗略预算

按"1 GPU·小时"计（含调试和重跑的余量，非精确）：

| 阶段 | 内容 | 卡数 × 时长 | GPU·h | RunPod $4.99 | Nebius 抢占 $3.05 |
|---|---|---|---:|---:|---:|
| 9 | §2.2 nsys + §2.5 memory（xl） | 1 × ~3 h | 3 | $15 | $9 |
| 8′ | §3(b) checkpointing（xl） | 1 × ~1 h | 1 | $5 | $3 |
| 10a | §4.5 flash benchmark（80 组配置） | 1 × ~2 h | 2 | $10 | $6 |
| 10b | §5.1 all-reduce（2/4/6 进程） | 6 × ~0.5 h | 3 | $15 | $9 |
| 10c | §5.3/5.4/5.6 + §6.2 + §7.2（全是 2 卡 xl） | 2 × ~4 h | 8 | $40 | $24 |
| | **小计（不含 leaderboard）** | | **17** | **~$85** | **~$52** |
| 11 | §9 leaderboard（迭代优化，开放式） | 2 × ? | ? | ? | ? |

Modal 每月 $30 免费额度 ≈ **4.8 B200·小时**，够覆盖阶段 9 + 10a + 10b（8 GPU·h 里的一部分）。

**搭配建议**
- **Modal**：短平快的 benchmark（§5.1 每次 <5 分钟、§2.2 profile），按秒计费不为空闲付钱，
  免费额度先用掉
- **RunPod / Nebius 抢占**：§5.3–§7.2 那一串 2 卡 xl 的长任务，单价最低
- **Together 不合适**：8 卡起步，只有 §5.1 用得上那么多卡，其余全浪费

### 决策项（越早定越好）

- [ ] **确认当前报价**（上表采集于 2026-03-28，今天 2026-08-30，隔了 5 个月）
- [ ] **注册 Modal 拿 $30/月免费额度**，先把最短的 §5.1 和 §2.2 跑掉
- [ ] **决定 xl 相关题目怎么处理**。三个选项：
  1. **上云**（推荐）：不含 leaderboard 约 17 GPU·h / $52–85，其中只有 §4.5 必须 B200
  2. 在本机用 **large** 替代 xl，在 writeup 里显式声明替换并说明理由
     （趋势和结论通常不变，但必须写清楚，否则会被当成没做）
  3. 只做 forward-only / 不带优化器状态的部分（xl 纯前向 fp32 权重 13.6 G + 激活，
     bf16 下 6.8 G，单卡可以跑，够回答 memory_profiling 的一部分）
- [ ] **决定 leaderboard 做不做**（10 分，硬性两张 B200 + 从空 cache 起 10 分钟内跑完；
  这是唯一开放式烧钱的一题，要先设预算上限）
- [ ] 装 `nsys`（Nsight Systems CLI）——§2.1.4、§2.1.6(f)、§5.3.2(b)、§7 accounting(b) 都要它。
  **本地和云上都要装**，云镜像不一定自带
- [ ] **上云前先把代码在本机调通**（课程明说的策略）：判分测试全部能用 gloo/CPU 跑，
  上云只该用来"跑数字"，不该用来 debug

---

## §0 前置准备

- [ ] **决定用官方 basics 还是自己的 hw1 实现**
  - `pyproject.toml:35` 现在指向 `./cs336-basics`（官方参考实现）。改成自己的 hw1：
    ```toml
    cs336-basics = { path = "../hw1", editable = true }
    ```
  - **用自己的实现需要先补三处兼容**（我们刚做完的重构已经解决了接口和权重层面的问题——
    `model(x)` 单参数可调用、state_dict 与官方逐 key 一致、logits 差 6e-07——但命名还没对齐）：
    - [ ] 类名：官方 `BasicsTransformerLM` vs 我们的 `TransformerLM`
    - [ ] 模块路径：官方 `cs336_basics.model` vs 我们的 `cs336_basics.transformer`
      （§2.1.4 要求 `cs336_basics.model.scaled_dot_product_attention = annotated_版本`，
      §9 leaderboard 的测试代码也直接 import 这两个名字）
    - [ ] 最省事的做法：在 `cs336_basics/model.py` 建一个转发模块 +
      `BasicsTransformerLM = TransformerLM` 别名，而不是改动已经测试通过的代码
  - 用官方实现的好处：leaderboard 和所有题面示例代码开箱即用；坏处：hw1 的消融开关
    （norm/ffn/tie/doc-mask）用不上
- [ ] **确认 `uv` 工作流**：`uv run pytest`、`uv run nsys profile -- python ...`
- [ ] **建 `cs336_systems/` 的骨架**（现在只有一个空 `__init__.py`，题面明确说"从零随便写"）
- [ ] **`tests/adapters.py` 的 8 个 hook**，全部还是 `raise NotImplementedError`：
  `get_flashattention_autograd_function_pytorch` / `get_flashattention_autograd_function_triton` /
  `get_ddp` / `ddp_on_after_backward` / `get_fsdp` / `fsdp_on_after_backward` /
  `fsdp_gather_full_params` / `get_sharded_optimizer`
  - ⚠️ PDF 里写的是 `adapters.get_flash_autograd_function_triton`，**实际文件里叫
    `get_flashattention_autograd_function_triton`**，以文件为准
  - ⚠️ `fsdp_on_after_backward` 和 `fsdp_gather_full_params` 在 PDF 的 fsdp 题面里**没提**，
    但 adapters.py 里有，别漏
- [ ] **表格自动化**：题面强烈建议用代码生成表格（`pandas.DataFrame.to_latex()` /
  `.to_typst()`），这份作业要交的表非常多。先把这个基础设施搭好，后面每题省一大截时间

**统一模型配置**（vocab_size=10000，batch_size=4，除非特别说明 context_length=512）

| size | d_model | d_ff | num_layers | num_heads |
|---|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

---

## §2 Profiling and Benchmarking（16 分）

### 2.1 `benchmarking_script` — 4 分 `[代码基建]` `[写作]`

- [ ] **(a) 写 benchmark 脚本**（这是后面一切的地基，命令行参数要设计好）
  - 按超参初始化模型 / 造随机 batch（随机权重随机数据即可，只测速度和显存）
  - 支持三种模式：只 forward / forward+backward / forward+backward+optimizer step
  - `w` 步 warm-up 后计时 `n` 步；用 `timeit.default_timer()`（比 `time.time()` 分辨率高）
  - **每步之后 `torch.cuda.synchronize()`** —— CUDA 调用是异步的，不同步就测的是"提交耗时"
  - 后面会不断往这个脚本上加开关：混合精度、memory profiler、torch.compile、NVTX、DDP……
    **一开始就按"可组合的 flag"来设计**
- [ ] **(b)** 对 §0 表里全部 5 个 size 测 forward / backward / optimizer step，
  5 warmup + 10 measurement，报**均值和标准差**。→ 1-2 句话
- [ ] **(c)** 去掉 warm-up 重测；再试 1 步、2 步 warm-up。解释为什么结果还是不同。→ 2-3 句话
  - （提示方向：CUDA context 初始化、cuBLAS/cuDNN autotune 首次选算法、内存分配器 cache 预热）

### 2.2 `nsys_profile` — 5 分 `[写作]` `[图]` ⚠️需要 nsys

- [ ] 选 **2 个 model size × 3 个 2 的幂次 context length（>128，最大取显存能装下的极限）** 做 profile
  ```bash
  uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx --gpu-metrics-devices=0 -- python benchmark.py
  ```
  （`--cudabacktrace=all --python-backtrace=cuda` 开销很大，不需要 traceback 时关掉）
- [ ] **给代码加 NVTX 标注**：用 `@nvtx.range(...)` / `with nvtx.range(...)` 圈出
  warm-up（好用 `--nvtx-capture` 过滤掉）、forward/backward、以及 attention 内部的
  "attention scores" / "softmax" / "final matmul" 三段
  - 做法：写 `annotated_scaled_dot_product_attention`，再
    `cs336_basics.model.scaled_dot_product_attention = annotated_版本` 猴补丁
- [ ] (a) forward 总时间，和 §2.1 用 Python 标准库测的对不对得上？→ 1-2 句
- [ ] (b) forward 里累计 GPU 时间最长的 CUDA kernel 是哪个？单次 forward 调用几次？
  加上 backward 之后还是它吗？→ 1-2 句
- [ ] (c) 除了矩阵乘，还有哪些 kernel 占了不可忽略的时间？→ 1-2 句
- [ ] (d) 完整训练步（含自己的 AdamW）里矩阵乘占比相对纯推理怎么变？其它 kernel 呢？→ 1-2 句
- [ ] (e) attention 内部 softmax vs 矩阵乘的**运行时间比** 和 **FLOPs 比** 差多少？→ 1-2 句
  - （这题是后面 FlashAttention 的动机：softmax 的 FLOPs 占比极小但耗时占比不小 = memory-bound）

### 2.3 `mixed_precision_accumulation` — 1 分 `[写作]`

- [ ] 跑题面给的 4 段累加代码（fp32 累加 / fp16 累加 / fp32 累加器加 fp16 值 /
  显式 `.type(torch.float32)` 后累加），解释精度差异。→ 2-3 句
  - 纯送分题，10 分钟能做完，建议第一个做

### 2.4 `benchmarking_mixed_precision` — 2 分 `[写作]`

- [ ] **(a)** 给定 `ToyModel`（fc1 → relu → LayerNorm → fc2），在 fp16 autocast 下写出六个 dtype：
  模型参数 / fc1 输出 / LayerNorm 输出 / logits / loss / 梯度
- [ ] **(b)** LayerNorm 的哪些部分对低精度敏感？换成 **BF16** 之后还需要特殊对待吗？为什么？→ 2-3 句
  - （方向：均值/方差是累加归约，动态范围；BF16 指数位和 FP32 一样宽）
- [ ] **(c)** 给 benchmark 脚本加 `--dtype bfloat16` 开关，5 个 size 各测有/无混合精度，
  说明随规模变化的趋势。→ 2-3 句 + 计时表
  - hw1 的 `train.py` 已经有 `torch.autocast` + `nullcontext` 的写法，可以直接搬

### 2.5 `memory_profiling` — 4 分 `[写作]` `[图]` ⚠️大显存

- [ ] **(a)** 给脚本加 memory profiler 开关：
  ```python
  torch.cuda.memory._record_memory_history(max_entries=1000000)
  ...
  torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
  torch.cuda.memory._record_memory_history(enabled=None)
  ```
  拖进 pytorch.org/memory_viz 看 Active memory timeline。
  **交两张图**：xl 纯 forward / xl 完整训练步 + 2-3 句
- [ ] **(b)** context length 128 和 2048 各自的峰值显存（forward / 完整训练步）→ 2×2 表格
- [ ] **(c)** 混合精度下的峰值显存，影响大吗？→ 2-3 句
- [ ] **(d)** xl 残差流上单个激活张量多大（MiB，除 1024²）？要写推导 → 1-2 句
- [ ] **(e)** memory_viz 里把 Detail 调低，最大的那几笔分配是多大、从哪来的（看 stack trace）→ 1-2 句
- [ ] **(f)** 用 nsys 的内存 profiling flag + PyTorch 的 NVTX 标签，算出**单个 TransformerBlock
  为 backward 存了多少激活**，列出贡献最大的 5 个操作及占比；再结合 backward 时每个 block
  的显存变化，算出梯度张量占多少，和预期对得上吗？→ 截图 + 1-2 段
- ⚠️ xl 完整训练步需要 54.5 G，本机放不下。见开头的"决策项"

---

## §3 Single-GPU Memory（4 分）

> 背景：一个 xl 的 TransformerBlock 光 backward 要存的激活就 **3.6 GiB**，32 层 = **114 GiB**。
> 先用 `torch.compile` 做算子融合（RMSNorm 从存 5 个张量降到 3 个），再上激活检查点。

### 3.1 `gradient_checkpointing` — 4 分 `[写作]`

- [ ] **(a)** N 个相同 block 顺序堆叠，**忽略计算开销**时峰值激活显存最小的检查点策略是什么？
  给出策略描述 + 代码草图 + 渐近峰值显存和计算量（关于 N 的函数）→ 3-5 句 + 代码草图
  - （方向：递归/嵌套 checkpoint，经典结论是 O(log N) 显存 / O(N log N) 计算，
    或 √N 划分给出 O(√N) 显存 / O(N) 计算——要自己论证清楚）
- [ ] **(b)** xl + batch 4 + seq 2048，**只允许一层重算（不能嵌套）**时最优的 checkpoint
  块大小是多少？**实测峰值显存验证**，并和相邻的更大/更小块大小对比。→ 3-5 句 + 实测数字
- 工具：`torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)`，
  以及 `torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)` 来统计存了多少字节

---

## §4 GPU Kernels — FlashAttention-2（29 分，本作业最大头）

### 4.1 `pytorch_attention` — 2 分 `[写作]`

- [ ] 写 attention 微基准脚本：
  - batch=8，**不带 head 维**（单头）
  - `d_model ∈ {16, 32, 64, 128}` × `seq_len ∈ {256, 1024, 4096, 8192, 16384}` 全组合
  - 随机 Q/K/V；计时 100 次 forward；**记录 backward 开始前的显存**；计时 100 次 backward
  - warm-up + 每次 `torch.cuda.synchronize()`
- [ ] 报表格（含 OOM 的格子）。在哪个规模开始 OOM？对最小的那个 OOM 配置**做显存核算**
  （用 hw1 的 Transformer 显存公式）。存给 backward 的显存怎么随 seq_len 变？怎么消掉这笔开销？
  → 表 + 计算过程 + 1-2 段

### 4.2 `torch_compile` — 2 分 `[写作]`

- [ ] **(a)** 给 attention 基准加一个 `torch.compile` 版本，同配置对比 → 表
- [ ] **(b)** 编译**整个 Transformer**，对比 forward / forward+backward+optimizer → 表

### 4.3 `flash_forward` — **15 分** `[代码]`

- [ ] **(a) 纯 PyTorch 的 tiled 版本**（不用 Triton，给 Triton 版当对照）
  - `torch.autograd.Function` 子类，`def forward(ctx, Q, K, V, is_causal=False)`
  - 返回 O，同时算出 logsumexp `L`；`ctx` 里存 `L, Q, K, V, O`
  - `backward` 先 `raise NotImplementedError`
  - 这一步**可以忽略 is_causal**
  - tile 至少 16×16；测试保证所有维度是 ≥16 的 2 的幂，不用管越界
  - → `adapters.get_flashattention_autograd_function_pytorch`
  - → `uv run pytest -k test_flash_forward_pass_pytorch`
- [ ] **(b) Triton kernel 版 forward**（Algorithm 1，在线 softmax）
  - launch grid = `(T_q, batch_size)`；**只有一层循环**，遍历 key tile `j`
  - 用 `tl.make_block_ptr` + `ptr.advance(...)`（在循环末尾推进）；`tl.dot` 做矩阵乘
  - 片上 buffer `O_i, l, m` 必须 **`tl.float32`**；累加用 `acc=` 参数：`acc = tl.dot(..., acc=acc)`
  - `P̃` 要先 cast 成 V 的 dtype 再相乘；写回前把 `O_i` cast 成目标 dtype
    （`ptr.type.element_ty` 拿 dtype）
  - 函数签名按题面给的 `flash_fwd_kernel(...)` 照抄
  - → `adapters.get_flashattention_autograd_function_triton`
  - → `uv run pytest -k test_flash_forward_pass_triton`
- [ ] **(c) causal masking**
  - autograd.Function 末尾加 `is_causal` 参数，**默认 False**（否则 (a)(b) 的测试会挂）
  - Triton kernel 加 `is_causal: tl.constexpr`（这个类型标注是必需的）
  - 在 kernel 里构造 query/key 的索引向量，比较出 `B_q × B_k` 的 mask，
    被 mask 的位置**给 S 加 -1e6**（不是设 -inf）
  - `ctx.is_causal = is_causal` 存给 backward
- 调试建议：`tl.device_print`；`TRITON_INTERPRET=1` 可在 CPU 跑解释器（题面说"我们发现它有 bug"）；
  逐个算子和 (a) 的 PyTorch 版对数

### 4.4 `flash_backward` — 5 分 `[代码]`

- [ ] 用 **PyTorch + `torch.compile`**（不用 Triton）实现 backward
  - 输入 Q, K, V, O, dO, L → 输出 dQ, dK, dV
  - **先算 `D = rowsum(O ∘ dO)`**，然后按式 (13)–(19) 重算 S、P，全程**不需要 softmax**
  - 关键点：P 从 Q、K、L 重算出来，所以 forward 不用把 S/P 存进 HBM
- [ ] → `uv run pytest -k test_flash_backward`
- 注：`tests/test_attention.py` 里还有 `test_flash_backward_pytorch` 和
  `test_flash_backward_triton`（带 `is_causal` 参数化），两个都要过

### 4.5 `flash_benchmarking` — 5 分 `[写作]` ⚠️理想在 B200

- [ ] 用 `triton.testing.do_bench` 对比 Triton FA2 与普通 PyTorch attention
  - **batch=1，永远开 causal**
  - `seq_len` = 128 … 65536 的 2 的幂 × `d` = 16 … 128 的 2 的幂 × `{bfloat16, float32}`
  - 报 forward / backward / 端到端三个延迟
  - tile 大小需要随输入规模调
- [ ] → 表格

### 4.6 【可选】Triton 版 backward — 0 分，但对 leaderboard 有用

- [ ] Algorithm 2：外层循环 key tile `j`，内层循环 query tile `i`，
  **P 算两遍**（一遍给 dQ，一遍给 dK/dV），以此避开跨 thread block 的同步和 atomics

---

## §5 Distributed Data Parallel（21 分）

### 5.1 `distributed_communication_single_node` — 5 分 `[写作]` ⚠️多卡

- [ ] benchmark all-reduce 耗时：
  - 数据量 fp32 张量 **1MB / 10MB / 100MB / 1GB**
  - 进程数（GPU 数）**2 / 4 / 6**
- [ ] 注意事项：warm-up ≥5 次（NCCL 尤其需要）；每次 `torch.cuda.synchronize()`
  （**即使 `async_op=False` 也必须同步**，它只保证入队不保证完成）；
  用 `dist.all_gather_object` 聚合各 rank 的计时
- [ ] → 图/表 + 2-3 句
- ⚠️ 需要最多 6 张 GPU；本机只有 1 张。可以先用 gloo/CPU 把脚本跑通，数字留到云上补

### 5.2 `naive_ddp` — 5 分 `[代码]`

- [ ] 实现最朴素的 DDP：backward 之后对**每个参数的梯度**单独 all-reduce 求平均
  - 训练开始前用 `broadcast` 把 rank 0 的参数发给所有 rank
  - batch 切分：n 个样本切成 n/d 份（d 必须整除 n）
- [ ] → `adapters.get_ddp` +（可选）`adapters.ddp_on_after_backward`
- [ ] → `uv run pytest tests/test_ddp.py`（用 gloo/CPU，world_size=2，本机可跑）

### 5.3 `naive_ddp_benchmarking` — 3 分 `[写作]` ⚠️多卡 ⚠️大显存

- [ ] 测**每步总时间**和**通信占比**，配置：1 node × 2 GPU，**xl**
- [ ] → 描述 setup + 数字

### 5.4 `minimal_ddp_flat_benchmarking` — 2 分 `[代码]` `[写作]` ⚠️多卡

- [ ] 改成**把所有梯度拼成一个扁平张量**再做一次 all-reduce
  - 用 `torch._utils._flatten_dense_tensors` / `_unflatten_dense_tensors`
- [ ] 同条件（1×2 GPU, xl）对比逐参数 all-reduce → 数字 + 1-2 句

### 5.5 `ddp_overlap_individual_parameters` — 5 分 `[代码]`

- [ ] 写 DDP 容器类，**梯度通信与 backward 计算重叠**
  - `__init__(self, module)`：包住任意 `nn.Module`，广播初始权重
  - `forward(self, *inputs, **kwargs)`：转发给被包的 module
  - `finish_gradient_synchronization(self)`：等所有异步通信完成
  - 机制：`param.register_post_accumulate_grad_hook(...)`，在每个参数梯度算好的瞬间
    发起 `dist.all_reduce(..., async_op=True)`，把 handle 收起来；
    `optimizer.step()` 之前调 `finish_gradient_synchronization()` 逐个 `handle.wait()`
- [ ] → `uv run pytest tests/test_ddp.py`，**建议重复跑 5 次**排查竞态

### 5.6 `ddp_overlap_individual_parameters_benchmarking` — 1 分 `[写作]` `[图]` ⚠️多卡

- [ ] **(a)** 同条件（1×2 GPU, xl）和前两种 DDP 对比每步耗时 → 数字 + 1-2 句
- [ ] **(b)** 用 nsys 分别 profile 朴素版和重叠版，**两张截图**直观证明一个重叠了、一个没有

---

## §6 Optimizer State Sharding（20 分）

### 6.1 `optimizer_state_sharding` — **15 分** `[代码]`

- [ ] 写优化器状态分片的包装类（简化版 ZeRO-1）
  - `__init__(self, params, optimizer_cls, **kwargs)`：params 可以是参数列表**也可以是
    param group 列表**（要支持不同 lr）；把参数分给各 rank（约 1/world_size）；
    **必须调用 `torch.optim.Optimizer` 超类构造函数**
  - `step(self, closure, **kwargs)`：调被包优化器的 step，**之后把自己更新的那份参数
    broadcast 给其它 rank**
  - `add_param_group(self, param_group)`：超类构造时会调它，训练中也可能调
    （比如逐步解冻），所以**参数分配的逻辑要写在这里**
- [ ] → `adapters.get_sharded_optimizer`
- [ ] → `uv run pytest tests/test_sharded_optimizer.py`，**重复跑 5 次**
- ⚠️ 坑：`add_param_group` 在 `super().__init__()` 里就被调用，此时你自己的属性可能还没初始化

### 6.2 `optimizer_state_sharding_accounting` — 5 分 `[写作]` ⚠️多卡 ⚠️大显存

- [ ] **(a)** 有/无分片时的峰值显存（1×2 GPU, xl），报三个时刻：模型初始化后、
  optimizer step 之前、optimizer step 之后。**拆解**各部分（参数/梯度/优化器状态/激活）→ 2-3 句
- [ ] **(b)** 分片对训练速度的影响 → 2-3 句 + 计时
- [ ] **(c)** 我们的实现和 **ZeRO stage 1（ZeRO-DP $P_{os}$）** 有什么区别？
  重点讲**显存**和**通信量**的差异 → 2-3 句
  - （方向：我们是 all-reduce 全梯度 + broadcast 参数；ZeRO-1 是 reduce-scatter 梯度 +
    all-gather 参数，通信量更省）

---

## §7 Fully-Sharded Data Parallel（20 分）

### 7.1 `fsdp` — **15 分** `[代码]`

- [ ] 写 FSDP 容器类，包住整个模型，hook 或 wrap 其中**每个 Linear 和 Embedding**
  - `__init__(self, module, compute_dtype=None)`
  - **哪些层要分片**：Linear + Embedding。**norm 不分片**（太小，传输延迟不划算）
  - forward：权重要**提前** all-gather 好，题面明确要求
    「**只在往前数第二层完成 forward 之后**才开始 gather」（prefetch 深度 = 2，控制显存）
  - backward：同样 all-gather 拿回权重；梯度就绪后 **reduce-scatter** 到对应 rank
  - **用完立刻释放 gather 出来的完整权重**
  - `compute_dtype` 给定时：**通信前就 cast 成低精度**（省带宽），
    但 **master weights 和优化器更新保持 FP32**
  - `forward(self, *inputs, **kwargs)` / `finish_gradient_synchronization(self)`
  - 每个分片必须能配 hw1 的 AdamW 直接用
- [ ] → `adapters.get_fsdp`、`adapters.fsdp_on_after_backward`、`adapters.fsdp_gather_full_params`
- [ ] → `uv run pytest tests/test_fsdp.py`，**重复跑 5 次**抓竞态
  - 测试有 `compute_dtype` 的参数化，两种都要过

### 7.2 `fsdp_accounting` — 5 分 `[写作]` `[图]` ⚠️多卡 ⚠️大显存

- [ ] **(a)** 基于 §6 的分析，预期 FSDP 能从峰值省下多少显存？
  （可以忽略 all-gather 的预分配 buffer）→ 2-3 句
- [ ] **(b)** profile xl 在两卡上的运行，**盯着权重的 all-gather**：通信赶得上 forward 吗？
  → 2-3 句 + nsys 截图

---

## §8 并行策略分析（17 分，纯推导，无需 GPU）

> **这 17 分不需要任何硬件，建议最先做完。** 统一设定：N 个设备两两互连，
> 每设备出口带宽 W bytes/s，加速器算力 C FLOP/s，**权重和激活都是 FP16（2 字节）**，
> 矩阵乘 (A,B)(B,C) 算 2ABC FLOPs，只算 matmul 忽略逐元素算子。
> 已知：ring all-gather 和 ring reduce-scatter 都是 $\frac{N-1}{N}\frac{S}{W}$，
> ring all-reduce = 两者串联 = $2\frac{N-1}{N}\frac{S}{W}$。

### 8.1 `alternate_ring_all_reduce` — 1 分 `[写作]`

- [ ] 题面给了另一种 all-reduce 算法（每步直接传完整的 $x^{(i)}$ 而不是分块）。
  用 S、N、W 表示它的耗时 + 一句话论证
  - （提示：每步传的是整个 S 而不是 S/N，所以是 $(N-1)\frac{S}{W}$，比 ring 版差 N/2 倍）

### 8.2 `data_parallel_calcs` — 3 分 `[写作]`

FFN 前向：$x_1=xW_1$，$x_2=xW_2$，$z=f(x_1)*x_2$，$y=zW_3$；
x 是 (B,D)，$W_1,W_2$ 是 (D,D_ff)，$W_3$ 是 (D_ff,D)。反向式 (24)–(30) 题面已给。

- [ ] **(a)** $N_{DP}$ 数据并行下 backward 的 FLOPs（用 B, D, D_ff, N_DP 表示）+ 一句论证
- [ ] **(b)** backward 的通信时间（B, D, D_ff, N_DP, W 的子集）+ 一句论证
- [ ] **(c)** 其它参数固定时，$N_{DP}$ 能开到多大才不被通信卡住？给不等式 + 一句论证

### 8.3 `fsdp_calcs` — 3 分 `[写作]`

- [ ] **(a)** $N_{FSDP}$ 下 forward 和 backward 各多少 FLOPs（两个答案）
- [ ] **(b)** forward 和 backward 各多少通信时间（两个答案）
  - forward：3 次 all-gather；backward：3 次 all-gather + 3 次 reduce-scatter
- [ ] **(c)** forward 和 backward 各自的 $N_{FSDP}$ 上界（两个不等式）

### 8.4 `tp_calcs` — 4 分 `[写作]`

配置：$W_1, W_2$ **column parallel**（切输出维），$W_3$ **row parallel**（切输入维），
所以 column 之后不用 all-gather，只在最后对 y 做一次 all-reduce。

- [ ] **(a)** 写出这个 TP 配置的**完整反向传播公式**：给定 dy (B,D)，
  用分片权重、前向存下的激活、通信原语，推出 $dW_1^{(i)}, dW_2^{(i)}, dW_3^{(i)}$ 和 dx
- [ ] **(b)** forward / backward 各多少 FLOPs（两个答案）
- [ ] **(c)** forward / backward 各多少通信时间（两个答案）
- [ ] **(d)** forward / backward 各自的 $N_{TP}$ 上界（两个不等式）

### 8.5 `fsdp_tp_calcs` — **6 分** `[写作]`

2D 网格：TP rank i × FSDP rank j，$N = N_{TP} N_{FSDP}$。
每个设备持有 $W_1^{(i,j)}, W_2^{(i,j)}$ 形状 $(\frac{D}{N_{FSDP}}, \frac{D_{ff}}{N_{TP}})$，
$W_3^{(i,j)}$ 形状 $(\frac{D_{ff}}{N_{TP}}, \frac{D}{N_{FSDP}})$。

- [ ] **(a)** forward 的 FLOPs（B, D, D_ff, N_FSDP, N_TP）
- [ ] **(b)** forward 的通信时间。**假设两个轴的通信可以重叠** →
  答案应该是两个量的 **max**（FSDP 侧 vs TP 侧）
- [ ] **(c)** 最优 $N_{TP}, N_{FSDP}$ 配置下，$N$ 能开到多大？不等式 + 推导
- [ ] **(d)** 同上，但**两轴通信不能重叠**（共享网络资源）时的 $N$ 上界
  （不用管 N_TP / N_FSDP 取整）

---

## §9 Leaderboard（10 分）⚠️两张 B200

目标：**8B 模型完整训练步（forward + loss + backward + AdamW）的墙钟时间**。

```python
ctx_len=32768, vocab_size=151936, d_model=4096, d_ff=11008,
num_layers=34, num_heads=32, bfloat16, is_causal=True, batch_size=2
```
（实算 **8.13B 参数**，光 bf16 权重就 16.3 GB。题面直言"故意做得很难塞进显存"。）

**硬性约束**
- 不能改模型的输入/输出行为；要用 `cs336_basics` 里的模型；必须通过和常规实现相同的测试
- 必须自己写，不能抄现成实现
- **从空的 PyTorch/Triton cache 起算，整个 benchmark 必须 10 分钟内跑完**
  → torch.compile 和 Triton autotune 不能太激进
- 基线是 10 秒，要打赢它

**优化清单**
- [ ] Triton autotune 调 tile 大小
- [ ] 调 Triton / torch.compile 的其它配置
- [ ] **fused AdamW**
- [ ] **融合 LM head + cross-entropy**（当前实现会物化完整的
  `[batch, seq_len, vocab_size]` logits：2×32768×151936×2B = **19.9 GB**，这是最大的一块）
  甚至可以让它顺手把 backward 一起融进去
- [ ] FlashAttention 改进：
  - [ ] backward 也用 Triton 写（不只 torch.compile）
  - [ ] backward 分两趟：一趟 dQ、一趟 dK/dV，避开 atomics 和跨 block 同步
  - [ ] causal 时**提前终止** program instance，跳过必然全零的 tile
  - [ ] 把非对角 tile 和对角 tile 分开：前者完全不比较索引，后者只比一次
  - [ ] Hopper 以后的架构用 **TMA**
- [ ] 实在放不下再上激活检查点（拿速度换显存）

---

## 建议执行顺序

| 阶段 | 内容 | 分值 | 硬件 |
|---|---|---:|---|
| **1** | §8 全部推导（5 题）+ §2.3 累加精度 | **18** | 无需 GPU |
| **2** | §0 前置 + §2.1 benchmark 脚本 + §2.4 混合精度 | 6 | 单卡够 |
| **3** | §4.1 attention 微基准 + §4.2 torch.compile | 4 | 单卡够 |
| **4** | §4.3/4.4 FlashAttention forward + backward | **20** | 单卡够 |
| **5** | §5.2 naive DDP + §5.5 overlap DDP | 10 | gloo/CPU 可测 |
| **6** | §6.1 optimizer sharding | **15** | gloo/CPU 可测 |
| **7** | §7.1 FSDP | **15** | gloo/CPU 可测 |
| **8** | §3 gradient checkpointing | 4 | 单卡够（xl 要换 large） |
| **9** | §2.2 nsys + §2.5 memory profiling | 9 | 要 nsys，xl 要大显存 |
| **10** | 所有 benchmark 类题目（§4.5、§5.1/5.3/5.4/5.6、§6.2、§7.2） | 26 | **要多卡** |
| **11** | §9 leaderboard | 10 | **要两张 B200** |

阶段 1–8 合计 **92 分**，全部能在本机（单卡 5090 + gloo/CPU）完成，其中判分测试全过。
剩下 45 分几乎全卡在多卡/大显存上。按开头的预算表，**不含 leaderboard 约 17 GPU·小时、
$52–85**（Modal 每月 $30 免费额度能抵掉一部分），其中只有 §4.5 一道必须用 B200。
