# CS336 Assignment 2 (Systems) — 完整 TODO

> **来源**：依据 `cs336_assignment2_systems.pdf`（Version 26.1.3, 48 页）梳理。
> **规模**：27 个 Problem，总分 137。hw1 重构清单详见 [todo-hw1-refactor.md](todo-hw1-refactor.md)。
> 
> **图例**：`[代码]` 有 pytest 判分 · `[文字]` 产出博客正文 · `[图]` 产出截图/表格/曲线
> `⚠️多卡` 需 >1 GPU (Modal 支出) · `⚠️大显存` 5090 32G 单卡放不下
> 
> **四大基本前提（本计划的核心决策基石）：**
> 1. **不提交 Gradescope，终极产出为知乎博客**。不打 ZIP 包，直接依赖自己 hw1 的干净代码。
> 2. **多卡是花钱买的体验，不是要绕开的障碍**。§5-7 必须上 Modal 跑真实通信，不用 gloo/CPU 糊弄。
> 3. **降维只为防 OOM，不为省钱砍内容**。省钱靠挑卡型（不必非租 B200），不砍作业核心逻辑。
> 4. **花钱上云前，必须在本地 5090 跑通**。云端按秒计费的算力只用来跑数据出图，绝不用于 Debug。
> 
> **导航**：紧接着往下看 **[分值分布与执行路线]**，然后是 **[A 协作模式]** · **[B 算力规划]** · **[§0–§9 逐题清单]**。

---

## 交付物

**在家自学，不提交 Gradescope**。所以 `writeup.pdf`、`code.zip`、
`./test_and_make_submission.sh`、leaderboard 提交**全部不做**。

实际产出是两样：

| 产出 | 说明 |
|---|---|
| **知乎博客** | 记录实现步骤。取代 writeup.pdf —— 但要求不同：writeup 要的是"1-2 句作答"，博客要的是**为什么这么做、踩了什么坑、数字长什么样** |
| 代码 | `cs336_systems/` 里的实现，跑通 `uv run pytest tests/` 即可，不需要打包 |

## 分值分布与执行路线

这份作业**总分 137**。与其按章节顺序做，不如按**“先单卡开发，再上云跑数”**的物理约束重新排布。
以下路线图展示了如何**先用单卡 5090 和本地 gloo/CPU 拿下 97 分**（阶段 1–8 的 80 分 + 阶段 11 的 §8 推导 17 分），
然后再花钱上 Modal 冲刺剩下的 40 分（阶段 9 的 30 + 阶段 10 leaderboard 的 10）。

| 阶段 | 对应章节 | 内容 | 分值 | 硬件要求 |
|---|---|---|---:|---|
| | **▸ 纯单卡：跑通前向与内核** | | | |
| **1** | §0 + §2 | 环境配置、benchmark 脚本、混合精度 | 7 | 单卡 5090 |
| **2** | §4.1 + §4.2 | Attention 微基准与 `torch.compile` | 4 | 单卡 5090 |
| **3** | §4.3 + §4.4 | FlashAttention 前后向（Triton/纯 PyTorch） | **20** | 单卡 5090 |
| | **▸ 多进程：单机分布式逻辑** | | | |
| **4** | §5.2 + §5.5 | 朴素 DDP 与 通信重叠 DDP | 10 | 本机 gloo/CPU (无需 GPU) |
| **5** | §6.1 | Optimizer State Sharding | **15** | 本机 gloo/CPU (无需 GPU) |
| **6** | §7.1 | Fully-Sharded Data Parallel (FSDP) | **15** | 本机 gloo/CPU (无需 GPU) |
| | **▸ 单卡：深度性能剖析** | | | |
| **7** | §2.2 | Nsys Profile | 5 | 单卡 5090 |
| **8** | §3 | 激活检查点 (Gradient Checkpointing) | 4 | 单卡 5090 (大显存要求可降维) |
| | **▸ 上云：真实通信与极限测试** | | | **先办 §B.3 的 ⏸ 清单，过完四道闸** |
| **9a** | §5.1/5.3/5.4/5.6 + §6.2 + §7.2 | 多卡 benchmark（通信占比、显存账、nsys 重叠图） | 21 | **Modal 多卡**（计划内支出） |
| **9b** | §2.5 + §4.5 | 大显存单卡 benchmark（xl 显存 timeline、FA2 长 seq 打榜） | 9 | **单卡大显存**（可降维留本机，或顺手上云补） |
| **10** | §9 | Leaderboard 优化打榜 | 10 | 开放式 (云上或单卡) |
| | **▸ 纸面：理论收束** | | | |
| **11** | §8 | 并行策略推导分析 | **17** | 纯纸面推导 |

---

## A. 协作模式
> **🚨 核心纪律**：**没有用户的明确指令，绝不越线抢跑！只有当用户明确许可并宣布开始下一个 Section 时，AI 才能执行该 Section 的任务。**


`hw2/CLAUDE.md` 是**课程官方给选课学生的 AI 使用规范**（只讲解、只 review，不写代码、
不给解法）。用户不是选课学生、不提交、不参与评分，但**博客要记录的是"我自己的实现步骤"**，
代写会掏空文章本身。所以取折中：

| | 谁来写 | 范围 |
|---|---|---|
| **核心实现** | **用户自己写** | Triton kernel、分布式逻辑、所有数学推导 |
| **脚手架 / 非核心** | Claude 可以写 | benchmark 脚本、表格生成、nsys 封装、画图、环境配置 |
| **review & debug** | Claude | 直接指出哪行错、为什么错，但不贴成品实现 |
| **博客正文** | Claude 起草，用户审核 | 素材和结论必须来自用户实测 |

下面按这四行逐一展开。

**① 核心实现 —— 用户自己写（Claude 不给可粘贴的实现）**

- §3.1(a) 递归 checkpoint 策略推导
- §4.3 `flash_forward`（PyTorch tiled 版 + Triton kernel + causal masking）
- §4.4 `flash_backward`
- §4.6【可选】Triton 版 backward
- §5.2 `naive_ddp`、§5.4 的 flat-gradient DDP 变体、§5.5 `ddp_overlap`
- §6.1 `optimizer_state_sharding`
- §7.1 `fsdp`
- §8.1–§8.5 全部推导（DP / FSDP / TP / 2D 的 FLOPs、通信时间、瓶颈不等式）
- §9 leaderboard 的 kernel 优化（fused AdamW、融合 LM head + cross-entropy、FA 改进）

**② 脚手架 / 非核心 —— Claude 可以写**

- §1 环境配置：`pyproject` 指向、`transformer.py` → `model.py` 改名、类别名
- §2.1 benchmark 脚本骨架：CLI 参数、warmup/measure 循环、`cuda.synchronize()`、均值±标准差
- §2.2 nsys 封装、NVTX range 的包装层（**被包的 attention 实现是用户的**）
- §2.5 memory profiler 开关、snapshot dump
- §4.1 / §4.2 attention 微基准的 sweep 骨架（含 OOM 捕获）
- §4.5 `triton.testing.do_bench` 的封装（**被测的 kernel 是用户的**）
- §5.1 all-reduce 基准脚本、`mp.spawn` 样板、`all_gather_object` 聚合计时
- §5.3 / §5.4 / §5.6 / §6.2 / §7.2 的 benchmark harness：计时、通信占比统计、显存快照
- **Modal 上云脚手架**（2026-08-30 新增）：镜像定义、代码同步、Volume 存产物、
  单容器多卡的 `mp.spawn` 启动、`--backend/--world-size/--local` 开关、账单/耗时记录。
  见 §B.3——脚手架必须保证**本机四道闸和云上跑的是同一份代码**
- 所有表格生成（markdown / `to_latex` / `to_typst`）和画图
- 博客用的对照表、可复现性脚注（硬件、torch 版本、warmup/measure 步数）

**③ review & debug 的边界** —— 用户明确要了，**所以不必用苏格拉底式提问绕圈子**：

- ✅ 直接指出哪一行错了、为什么错、会以什么现象暴露
- ✅ 给验证手段：toy input、shape 断言、和参考实现对拍、profiler 检查点
- ✅ 讲清算法思路、指 PDF 的 Algorithm 1/2 和公式编号
- ❌ 不直接贴出正确的核心实现代码替换掉
- ❌ 不代填 `tests/adapters.py` 里那 8 个 hook 背后的实现

用户卡住时的降级顺序：**指出症状 → 给验证方法 → 讲清思路 → 高层伪代码**，
到伪代码为止，不落到可粘贴的成品。

**④ 博客正文 —— Claude 可以起草**（2026-08-30 补充）

**和代码不是一回事。** 代码自己写是为了学到东西；博客文段不承担这个作用，
不必也由用户从零敲。采用的模式是：

```
用户描述大致内容 / 给出素材和结论  →  Claude 写成文段  →  用户审核修改
```

- ✅ Claude 起草正文、组织段落、写过渡、润色
- ✅ Claude 做表格、图、技术准确性 review（数字对不对、术语一致、结论是否被数据支持）
- ⚠️ **事实来源必须是用户跑出来的数字和自己写的实现**。Claude 不编数字、
  不替用户"想"出结论——没有实测就标 TODO，不要填一个看起来合理的值
- ⚠️ 引用代码时贴的是**用户自己写的实现**，不是 Claude 现场另写一版
- 语气/风格在第 1 篇上校准一次，之后沿用

**这条不放宽代码的边界**：①里列的核心实现（Triton kernel、
分布式逻辑、§8 推导）仍然由用户自己写。放宽的只有"把已有的东西写成文章"这一步。



**写作时要坚守的“博客质量基线”：**
- [ ] 代码片段要**可独立运行**（作业只要答案，博客读者要能抄走就跑）
- [ ] 每张表/图注明**硬件、torch 版本、warmup/measure 步数**（可复现性）
- [ ] 中文术语统一（kernel / 算子、residual / 激活、shard / 分片……开篇定好）
- [ ] 把"踩坑"单独写出来 —— 比如已经遇到的「猴补丁打在转发模块上静默失效」，这类内容比正确答案更有阅读价值

---

## B. 算力

- 本机 **单卡 RTX 5090 / 32 GB**；题面默认 **B200（180 GB）**，多处要求 **2–6 张卡**
- **判分测试一分钱不用花**：四个测试文件全是 `mp.spawn` + gloo/CPU 起 `world_size=2`
- **本机能做完 97 分**（§4.3/4.4 的 20 + §6.1 15 + §7.1 15 + §8 17 + §5.2/5.5 10 +
  §2.1/2.3/2.4 的 7 + §2.2 的 5 + §4.1/4.2 的 4 + §3.1 的 4；其中 §3.1(b) 要降维）
- **真要花钱的只有一档**：§5.1 + §5.3/5.4/5.6 + §6.2 + §7.2，约 **11 GPU·h**，
  平台 **Modal**，属于计划内支出（理由见「读我先」前提 2）
- **可省的只有** §2.5 / §3.1(b) 的 xl（用 large 替代）和 §4.5 的长 seq（扫到哪算哪）
- **点名 B200 的只有 §4.5 和 §9**；不提交 → **型号**不再有约束力，换张卡照样出图

下面三节分别是：这些结论的依据（B.1）、钱和平台（B.2）、开机前要过的关（B.3）。

### B.1 逐题需要什么卡

**模型规模与显存**——实测参数量与显存需求（vocab=10000）：

| size | params | fp32 权重 | 权重+梯度+AdamW 两个动量 | 单卡 32G |
|---|---:|---:|---:|:--:|
| small | 0.13B | 0.5 G | 2.1 G | ✓ |
| medium | 0.42B | 1.7 G | 6.8 G | ✓ |
| large | 0.97B | 3.9 G | 15.5 G | ✓（激活要省着用） |
| **xl** | **3.41B** | 13.6 G | **54.5 G** | ✗ |
| 10B | 12.83B | 51.3 G | 205.3 G | ✗ |
| leaderboard | 8.13B | — | bf16 权重就 16.3 G | ✗（题面要求两张 B200） |

**判分测试为什么不花钱**——四个测试文件全部用 `mp.spawn` + **gloo 后端在 CPU 上**
起 `world_size=2`，不需要真的有两张卡：

```
tests/test_ddp.py:40                _setup_process_group(..., backend="gloo")
tests/test_sharded_optimizer.py:31  backend="gloo"
tests/test_fsdp.py:120              backend="gloo"
tests/test_attention.py             单卡即可（Triton kernel）
```

**逐题清单**，按章节顺序排；`本机` 一列指单卡 RTX 5090 / 32 G。**全文只有 2 处硬性指定 B200**
（`grep -n B200` 核过：L1560、L2593/L2661），其余只写"1 node x 2 GPUs"或"最多 6 张"，**不限型号**。

**§2 Profiling & Benchmarking**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 2.1 `benchmarking_script` | 1 | 不限 | 部分 | 要跑全部 5 个 size；small/medium/large ✓，xl/10B ✗ |
| 2.2 `nsys_profile` | 1 | 不限 | **✓** | 原文是"**自选**两个 size""**你显存装得下的**最长 ctx"（L252-254），按硬件自适应 |
| 2.3 `mixed_precision_accumulation` | 0 | — | ✓ | CPU 即可 |
| 2.4 `benchmarking_mixed_precision` | 1 | 不限 | 部分 | 同 2.1，xl/10B 受限 |
| 2.5 `memory_profiling` | 1 | 不限 | ✗ | **点名 xl**，ctx 128/2048，完整训练步要 >55 G |

**§3 Single-GPU Memory**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 3.1(a) 最优检查点策略 | 0 | — | ✓ | 纯推导 |
| 3.1(b) 实测验证 | 1 | 不限 | ✗ | **点名 xl** + batch 4 + seq 2048（L757） |

**§4 GPU Kernels**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 4.1 `pytorch_attention` | 1 | 不限 | **✓** | OOM 本身就是要回答的内容，小显存反而更早看到现象 |
| 4.2 `torch_compile` | 1 | 不限 | **✓** | 未点名 size |
| 4.3 `flash_forward` | 1 | 不限 | **✓** | 判分测试规模很小 |
| 4.4 `flash_backward` | 1 | 不限 | **✓** | 同上 |
| 4.5 `flash_benchmarking` | 1 | **B200 强制** | ✗ | "run the benchmark on a single B200"（L1560）；seq 扫到 65536 |

**§5 Distributed Data Parallel**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 5.1 `distributed_communication` | **2/4/6** | 不限 | ✗ | "Up to 6 GPUs. Each run < 5 minutes"（L1800）—— 唯一要 6 张的题，且每次极短 |
| 5.2 `naive_ddp` | 0 | — | **✓** | 判分测试走 gloo/CPU（test_ddp.py:40） |
| 5.3 `naive_ddp_benchmarking` | 2 | 不限 | ✗ | "1 node x 2 GPUs, xl"（L1887） |
| 5.4 `minimal_ddp_flat_benchmarking` | 2 | 不限 | ✗ | 同上条件对比（L1887） |
| 5.5 `ddp_overlap_individual_parameters` | 0 | — | **✓** | gloo/CPU 判分 |
| 5.6 `ddp_overlap_..._benchmarking` | 2 | 不限 | ✗ | "1 node, 2 GPUs, xl"（L2006/L2011）+ 要 nsys 截图 |

**§6 / §7 Sharding & FSDP**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 6.1 `optimizer_state_sharding` | 0 | — | **✓** | gloo/CPU 判分（test_sharded_optimizer.py:31） |
| 6.2 `..._accounting` | 2 | 不限 | ✗ | "1 node, 2 GPUs, xl"（L2079/L2088） |
| 7.1 `fsdp` | 0 | — | **✓** | gloo/CPU 判分（test_fsdp.py:120） |
| 7.2 `fsdp_accounting` | 2 | 不限 | ✗ | "Profile the xl model on two GPUs"（L2156）+ nsys 截图 |

**§8 / §9**

| 题目 | 卡 | 型号 | 本机 | 说明 |
|---|:--:|---|:--:|---|
| 8.1–8.5 全部 | 0 | — | **✓** | 纯纸面推导，17 分 |
| 9 `leaderboard` | 2 | **B200 强制** | ✗ | "two B200 GPUs"（L2593/L2661），且空 cache 起 10 分钟内跑完 |

### B.2 预算与平台

按"1 GPU·小时"计，含调试和重跑的余量，非精确：

| 性质 | 题目 | 卡数 × 时长 | GPU·h |
|---|---|---|---:|
| **计划内** | §5.1 all-reduce（2/4/6 进程） | 6 × ~0.5 h | 3 |
| **计划内** | §5.3 + §5.4 + §5.6 + §6.2 + §7.2（全是 2 卡 xl） | 2 × ~4 h | 8 |
| | **计划内小计** —— 多卡体验本身就是目的 | | **11** |
| 可选 | §2.5 `memory_profiling`（xl） | 1 × ~2 h | 2 |
| 可选 | §3.1(b) checkpointing（xl） | 1 × ~1 h | 1 |
| 可选 | §4.5 `flash_benchmarking`（长 seq，80 组配置） | 1 × ~2 h | 2 |
| | **合计（不含 leaderboard）** | | **16** |
| — | §9 leaderboard（开放式，不提交，可降到单卡慢慢迭代） | 2 × ? | ? |

**这些 GPU·h 值多少钱，取决于挑什么卡**。按 Modal 的 B200 $6.25/h 算是**上限**：
计划内 11 GPU·h ≈ $69，全部 16 GPU·h ≈ $100。但计划内那一档**不需要 B200**，
挑便宜档（H100 / A100 / L40S）会明显低于此，加上 $30/月免费额度，实际支出还要再降。
**具体单价到时候现核**（下面那张报价表已过期）——这也是省钱的正确做法：**挑卡型，不是砍内容**。

**为什么是 Modal**（已定，2026-08-30）

- **单容器最多 8 卡**，正好覆盖 §5.1 的"最多 6 张"和 §5.3–§7.2 的 2 卡。
  作业本身就是 **single-node**（§5.1 题名就叫 `..._single_node`），不需要跨节点，
  也就不用碰 Modal 上更麻烦的多容器组网
- **按秒计费、不为空闲付钱** → 适合"改一版跑 3 分钟看数字"的迭代节奏，
  而这正是 §5 三版 DDP 对比的工作方式
- **每月 $30 免费额度**：按 B200 $6.25/h 算 ≈ 4.8 h；但 §5/§6/§7 那 11 GPU·h
  **不需要 B200**，挑便宜档（H100 / A100 / L40S）能让免费额度覆盖掉相当大一部分。
  → **省钱的正确做法是挑卡型，不是砍内容**
- RunPod / Nebius 抢占单价更低，但要自己管机器、传数据、忍受被抢占。
  **先在 Modal 上把流程跑顺**，真觉得贵再换——省下的钱大概率抵不过折腾的时间

**其他 provider 报价（备查）**——课程给自学者列的单张 B200 公开价，
**采集于 2026-03-28，已过去 5 个月，用前先核**：

| provider | 单卡 B200 | 备注 |
|---|---:|---|
| **[Modal](https://modal.com/)**（课程赞助商） | $6.25/h | **每月送 $30 免费额度**；按实际计算计费，**不为空闲付费**；本地开发↔GPU 实验切换的 UX 好 |
| [Lambda Labs](https://lambda.ai/) | $6.69/h | |
| **[RunPod](https://www.runpod.io/)** | **$4.99/h** | 非抢占里最便宜 |
| **[Nebius](https://nebius.com/)** | $5.50/h（**抢占 $3.05/h**） | 抢占价是全场最低 |
| [Together](https://www.together.ai/) | $7.49/h | **最少 8 卡起**；长期承诺更便宜 |

### B.3 上云准入闸

课程自己的建议就是这个意思：先在 **CPU 上把正确性调通**，GPU 只用来跑训练（A1/A4/A5）或
**benchmark GPU 算子（A2，也就是本作业）**。文末那张阶段表就是照这条排的。

> **原则**：租来的每一分钟只用来**跑数字**。任何"本机也能发现"的问题，都不许留到云上发现。

**为什么这条不是废话**：`tests/` 全绿 **≠** 到了多卡 GPU 上能跑。gloo/CPU 会放过一整类问题：

| 本机 gloo/CPU 看不出来 | 到了 Modal 上会怎样 |
|---|---|
| 参数/梯度没 `.to(f"cuda:{rank}")`、没 `torch.cuda.set_device(rank)` | 直接报错或全挤在 cuda:0 |
| `mp.spawn` 前就初始化了 CUDA | 子进程 CUDA 初始化失败 |
| gloo 不支持的集合操作（`reduce_scatter_tensor` / `all_gather_into_tensor` 的部分路径、bf16） | 本机被绕过去了，NCCL 上才第一次真正执行 |
| `async_op=True` 的 handle 漏 `wait()` | gloo 偏同步，容易蒙对；NCCL 是 stream 异步，结果直接错 |
| 计时漏 `torch.cuda.synchronize()` | 本机没 CUDA 所以不报错，云上量出来的通信占比全是假的 |
| 镜像缺包 / 代码没同步 / 产物没落盘 / timeout 太短 | **最贵的一类**：机器起来了，跑了 30 秒挂掉，钱照付 |

**四道闸，从便宜到贵，按顺序过：**

- [ ] **闸 1 — `world_size=2` gloo/CPU**：`tests/` 全绿。管的是**算法正确性**（梯度平均对不对、
      分片切得对不对），和硬件无关。这一关本来就要过
- [ ] **闸 2 — `world_size=1` + NCCL + cuda:0（本机 5090）**：把**同一份脚本**用真 NCCL 后端跑一遍。
      单卡也能建 NCCL 进程组。这一关专抓"设备放置 + CUDA 初始化 + 集合操作在 NCCL 上存不存在"，
      正好是闸 1 漏掉的那一整类。**注意 NCCL 不支持两个 rank 共享同一张卡**（官方要求每 rank 独占 GPU），
      所以本机验不了"真 2 卡"，只能验"真 GPU 路径"
- [ ] **闸 3 — 目标配置在 5090 上单进程能装下**：用打算在云上跑的 batch/seq/模型档，
      先在本机跑**一步完整训练**确认不 OOM。多卡只会让**每卡**负载更小，
      本机单卡装得下 → 云上一定装得下
- [ ] **闸 4 — Modal 冒烟跑**：第一次上云**不是**跑正式实验，而是最便宜的卡 × 最短时间，
      只验证：镜像装好了、代码同步了、`nvidia-smi` 看得见 N 张卡、进程组能建、
      产物能写回 Volume 并下载到本机。**几毛钱买掉整类基础设施风险**

**脚手架要求（归 Claude，见 §A）**：Modal 入口脚本必须支持
`--backend {gloo,nccl}`、`--world-size N`、`--local`，让**闸 1/2/3 跑的是和云上完全同一份代码**，
只有启动器不同。如果本机和云上跑的是两份脚本，这四道闸就白设了。

**nsys 单独提前验**：§5.3.2(b) 和 §7 accounting(b) 要 nsys timeline，
而容器里跑 profiler 常有权限/驱动坑。**在闸 4 的冒烟跑里就把 nsys 试掉**，
别等到要出图那天才发现装不上

**⏸ 以下等真正做到 §5 再处理，现在不用想**

计划表里 §5 排在阶段 4（代码）/ 阶段 9（数字），中间隔着整个单卡 kernel 部分。
Modal 的账号、报价、镜像现在定了也会过期，到时候再一次性办：

- [ ] 核 §B.4 那张报价表（采集于 2026-03-28，早就过期），并确认 Modal 上多卡档位有哪些、
      便宜档（H100 / A100 / L40S）的价差、B200 排不排得上队
- [ ] 注册 Modal，拿 $30/月免费额度
- [ ] **镜像里怎么装 nsys**（本机是软链到 conda 的 2025.3.1，`.venv` 不跟着走，
      发行版源里的 2022 版在 Blackwell 上不可用——见 §0 Action 5）。
      这个决定拖到做镜像时再下，现在定了也会过期
- [ ] 搭 Modal 脚手架（归 Claude）：镜像、代码同步、Volume 存产物、单容器多卡的 `mp.spawn`、
      `--backend/--world-size/--local` 开关。**写一次，§5–§7 全复用**
- [ ] 第一次正式上云跑 **§5.1**：每次 <5 分钟、不依赖模型代码，**失败成本最低的多卡首跑**
- [ ] **xl 那两道单卡题（§2.5、§3.1(b)）用什么规模**——和多卡无关，排到阶段 8–9 再定：
      ① 本机用 large 替代（推荐，"单卡 32G 能做到哪"本身是博客卖点）
      ② 只做 forward-only（xl 纯前向 bf16 6.8 G，单卡能跑）
      ③ 顺手上云补图（Modal 流程那时已经为 §5–§7 搭好，边际成本很低）

---

## §0 前置准备（本文档自编号；PDF 的 §1 是 Assignment Overview）

### Action

- [x] **Action 1**: 执行重命名命令：`git mv hw1/cs336_basics/transformer.py hw1/cs336_basics/model.py`
  - **为什么必须真改名、不能留个转发模块**：§2.2 要求 `cs336_basics.model.scaled_dot_product_attention = 注解版`。
    如果 `model.py` 只是 `from .transformer import *`，这个赋值改的是 `model` 模块的全局绑定，而
    `MultiHeadSelfAttention.forward` 是在 `transformer` 的命名空间里查这个名字的——**patch 不到，实测 0 次调用，且不报错**，
    只是 NVTX range 全部消失，§2.2 的 (b)(c)(e) 直接答不出来。
- [x] **Action 2**: 批量替换引用路径。把 `hw1/` 目录下（包括 `__init__.py` 和各测试脚本）共计约 11 处的 `cs336_basics.transformer` 替换为 `cs336_basics.model`。
- [x] **Action 3**: 在改名后的 `model.py` 底部加上两行别名代码，以兼容 PDF 后续要求：
  ```python
  BasicsTransformerLM = TransformerLM
  RotaryEmbedding = RoPE
  ```
- [x] **Action 4**: 修改 `hw2/pyproject.toml:35`，把官方库替换为本地路径：
  ```toml
  cs336-basics = { path = "../hw1", editable = true }
  ```

- [x] **Action 5**: 配置 `nsys` (Nsight Systems CLI)。**别用 apt 装**——Ubuntu 源里只有 `2022.4.2`，
  比 Blackwell(sm_120) 早三年，在 5090 上会**静默失败**（能跑、能认出 5090，但收尾报
  `Importer error status`，只吐 `.qdstrm`、生不成 `.nsys-rep`）。
  - **做法**：软链 conda 里自带的 2025.3.1 到 hw2 的 venv，不污染全局：
    ```sh
    ln -sf /home/yc/miniconda3/envs/diff/nsight-compute-2025.3.1/host/target-linux-x64/nsys \
           /home/yc/projects/LLM/hw2/.venv/bin/nsys
    ```
  - **§0 的完成判据就这一行**（能不能调用到正确的二进制）：
    ```sh
    uv run which nsys && uv run nsys --version   # 要 .venv/bin/nsys + 2025.3.1
    ```
  - ⚠️ **这个方案会静默退化**：`uv run` 把 `.venv/bin` 排在 `PATH` 最前，软链在就用新版；
    **软链一旦消失**（uv 重建 venv、`diff` 那个 conda 环境被删/改名），`nsys` 会
    **悄悄退回 `/usr/bin/nsys` 那个坏版本**，不报错。而且 `.venv` 是 gitignore 的，
    **换机器 / 上 Modal 都带不过去** → 镜像里得另装，见 §B.3 的 ⏸ 清单
  - **能不能产出报告是另一回事，归 §2.2 验**（那几步需要一个带 NVTX 的脚本，
    而那本来就是 §2.2 的交付物）。2026-08-30 已预跑通，结论记在 §2.2
- [x] **Action 6**: 跑一次空转的 `uv run pytest tests/`，验证依赖全部贯通（此时 `adapters.py` 里的 8 个 hook 还是 NotImplementedError，必然报错，但只要不是 import 报错就说明环境通了）。

### Tips

1. **PDF 命名不一预警**：抄 PDF 示例时，注意官方管我们的 `rope` 参数叫 `positional_encoder`。
2. **Hook 盘点**：`tests/adapters.py` 里留了 8 个空接头（如 `get_ddp`）等着后面填。注意 PDF 里写的 `get_flash_autograd_function_triton` 在实际文件里叫 `get_flashattention_autograd_function_triton`。
3. **💡 表格自动化 Tip**：写 benchmark 时，顺手用 pandas 将结果存成 DataFrame 并调 `.to_markdown()`，方便直接贴进知乎。
4. **怎么跑 hw2 的脚本：什么都不用配，直接 `uv run`**。
   README 的安装指令藏在 Setup 末尾编号 `0.` 的代码块里，容易漏；要点是
   **`uv run` 会按 `pyproject.toml` 自动装依赖**，第一次跑就地建出 `hw2/.venv`。
   ```sh
   cd /home/yc/projects/LLM/hw2
   uv run python cs336_systems/benchmark.py --size medium
   uv run pytest
   uv run pytest -k test_flash_forward_pass_pytorch
   uv run /home/yc/miniconda3/envs/diff/nsight-compute-2025.3.1/host/target-linux-x64/nsys \
          profile -o out --force-overwrite true -t cuda,nvtx python xxx.py   # 见 Action 5
   ```
   - **从子目录跑也行**，uv 会向上找 `pyproject.toml`（实测在 `cs336_systems/` 里可用）
   - `VIRTUAL_ENV ... will be ignored` 的警告**是正常的**，意思是"忽略你的 conda，用项目 .venv"
   - 两个包都是 editable 安装，脚本放哪都能 import：
     `cs336_systems` → `hw2/cs336_systems/`，`cs336_basics` → **`hw1/cs336_basics/`**（§0 生效）
   - 要交互式 shell 或让 IDE 认：`source hw2/.venv/bin/activate`；
     IDE 解释器填 `/home/yc/projects/LLM/hw2/.venv/bin/python`
   - **别做**：① `conda activate genai`（uv 会忽略；`uv run --active` 能强制用 conda 环境，
     但那里没有 hw2 的依赖）② `pip install`（不进 `uv.lock`，换机器/上云就丢，加依赖用 `uv add`）
     ③ 以为改完 `pyproject.toml` 要手动重装（下次 `uv run` 自己重建，`uv.lock` 的 diff 就是证据）
5. **两个项目的 `.venv` 各自独立，底座还不一样**（2026-08-30 查明）：
   ```
   hw1/.venv  ← /home/yc/miniconda3/envs/genai   Python 3.12.12
   hw2/.venv  ← /home/yc/miniconda3 (base)       Python 3.13.9
   ```
   `uv run` **主动忽略当前 conda 环境**（会警告 `VIRTUAL_ENV ... will be ignored`），
   只认项目目录下的 `.venv`，所以**不需要也不该 `conda activate genai`**。
   两边 torch 都是 2.11.0+cu130 / CUDA 13.0，`cs336_basics` 都解析到 `LLM/hw1/cs336_basics/`。
   - **A2 的数字全部出自 hw2 那个 3.13.9 环境**，可复现性脚注要记 Python 版本，
     别和 hw1 笔记里 3.12 环境下的数字混账
   - **风险**：hw1 的 61 个测试在 3.12 上跑，hw2 用 3.13 导入同一份源码。纯 Python +
     editable 路径安装所以能跨版本共用，但**只有 hw2 侧才会暴露 3.13 不兼容的写法**。
     已在 3.13 下验过 `Embedding/Linear/RMSNorm/TransformerBlock/MultiHeadSelfAttention.forward`
     和猴补丁路径；**以后每引入一个新类，第一次要在 hw2 侧也跑一遍**
6. **README 的目录树对我们已经失效**：它画的是 `cs336_basics/` 在 hw2 根下，
   而我们按 README 给的另一条路子（"edit the outer pyproject.toml to point to your own
   implementation"）指向了 `../hw1`。`hw2/cs336-basics/` 那份官方参考实现还在原地，
   但**已经不被引用**——照 README 的树找文件会找错地方。留着它当对拍基准，别删。

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

### 2.1 `benchmarking_script` (4分)
**Actions:**
- [ ] 编写 `benchmark.py`，支持 CLI 参数（超参组合、三种执行模式：仅前向/前向反向/完整训练步、warm-up/measure 步数）。
- [ ] 确保在测试循环中每步结束后严格调用 `torch.cuda.synchronize()`。

**Blog:**
- [ ] 测 §0 那张模型配置表里的 5 个 size（= PDF Table 1，在 PDF §2.1.2） (5 warmup + 10 measure)，记录均值和标准差。（⚠️ 10B 本地必爆 OOM，记录报错现象或仅测前向即可）。
- [ ] 对比 0 步、1-2 步、5 步 warm-up 的结果差异并解释原因（提示：CUDA 上下文/内存池预热等）。

### 2.2 `nsys_profile` (5分)
**Actions:**
- [ ] 编写 `annotated_scaled_dot_product_attention`，用 `cs336_basics.model.scaled_dot_product_attention = ...` 替换原函数打上 NVTX 猴补丁。
- [ ] 使用 `nvtx.range` 圈出 warm-up、前向/反向、以及 Attention 内的 "scores"、"softmax"、"final matmul"。
- [ ] 选 2 个 model size × 3 个 >128 的 seq_len (选显存极限下的最大值)，跑 `nsys profile` 抓取数据。
- [ ] **先验工具链再跑正式实验**（`nsys --version` 通过不算数，旧版也能打印版本、也认得出
      5090，它是收尾时才失败的）。这四步全绿才算 nsys 可用，正好也是 (b)(e) 的取数手段：
      ```sh
      uv run which nsys && uv run nsys --version                      # 1 路径+版本≥2024
      uv run nsys profile -o /tmp/chk --force-overwrite true \
             -t cuda,nvtx python <带 NVTX 的脚本>
      ls /tmp/chk.nsys-rep                                            # 2 旧版死在这，只有 .qdstrm
      uv run nsys stats --force-export=true --report nvtx_sum /tmp/chk.nsys-rep        # 3
      uv run nsys stats --force-export=true --report cuda_gpu_kern_sum \
             --filter-nvtx="scaled dot product attention" /tmp/chk.nsys-rep            # 4
      ```
      - 第 3 步的 `nvtx_sum` 是**把 warmup 过滤掉**的手段
      - 第 4 步的 `--filter-nvtx` 把 kernel 归因到某个 range 内，**PDF 在 (b) 的 hint 里
        说的就是这个**（"filter using NVTX ranges to identify which parts of the model
        are responsible for which kernels"），(e) 比 softmax vs matmul 也靠它
      - ⚠️ `nsys stats` 复用旧的 `.sqlite` 会报 `File is older than input file` 然后
        **直接退出**（不是警告）→ 每次都带 `--force-export=true`
      - ⚠️ 旧版 nsys 不认 `--force-overwrite`，报 `option is ambiguous`——看到这个说明
        PATH 又退回 2022 版了（见 §0 Action 5）
      - **2026-08-30 已用一个 4 层 TransformerLM + annotated attention 预跑过，四步全绿**
        （patch 被调用 20 次 = 4 层 ×(2 warmup + 3 step)，同时确认改名 `model.py` 后
        猴补丁确实生效）。正式做这题时把这四步并进实现，不另留脚本

**Blog:**
- [ ] (a) 前向总时间是否与 2.1 脚本测试出的时间对齐？
- [ ] (b) 找出前向累计耗时最长的 CUDA kernel 及单次前向的调用次数；加上反向之后，最长 kernel 是否发生变化？
- [ ] (c) 指出除矩阵乘外的其他不可忽略耗时的 kernel。
- [ ] (d) 完整训练步中矩阵乘耗时占比对比纯前向有何变化？其他 kernel 呢？
- [ ] (e) 对比 Attention 内 softmax vs 矩阵乘的运行时间差距与理论 FLOPs 差距的对比关系。

### 2.3 `mixed_precision_accumulation` (1分)
**Actions:**
- [ ] 跑题面指定的 4 段累加实验代码（fp32、fp16、混合、显式 cast type）。

**Blog:**
- [ ] 用 2-3 句话解释这 4 种写法产生精度差异的原因。

### 2.4 `benchmarking_mixed_precision` (2分)
**Actions:**
- [ ] 为 `benchmark.py` 加 `--dtype bfloat16` 开关，结合 `torch.autocast` 上下文实现混合精度测试。

**Blog:**
- [ ] (a) 写出 `ToyModel` 在 fp16 autocast 下 6 个组件的实际 dtype (模型参数/fc1输出/LN输出/logits/loss/梯度)。
- [ ] (b) 解释 LayerNorm 的哪些部分对低精度敏感，以及换成 BF16 后是否还需要特殊对待及原因。
- [ ] (c) 给出 5 个 size 在有/无 BF16 混合精度下的前向+反向耗时趋势对比表并解释规律。

### 2.5 `memory_profiling` (4分)
**Actions:**
- [ ] 给脚本加入 PyTorch Memory Profiler (`_record_memory_history` 与 `_dump_snapshot`) 导出开关。

**Blog:**
- [ ] (a) 提交 xl 纯前向与完整训练步的 2 张 Active memory timeline 截图（⚠️ xl 完整步会 OOM，可留待上云跑或在此直接展示 OOM 报错）。
- [ ] (b) 用表格展示 seq=128 与 2048 时，纯前向与完整训练步的峰值显存 (共 4 个数据点)。
- [ ] (c) 测量并说明混合精度对 xl 纯前向和完整训练步峰值显存的具体影响。
- [ ] (d) 手算推导 xl 残差流上单个激活张量的大小 (以 MiB 为单位，除以 1024^2)。
- [ ] (e) 降低 pytorch.org/memory_viz 中的 Detail 级别，指出最大的那几笔显存分配分别是多大，以及它们的来源 (看 stack trace)。
- [ ] (f) 结合 nsys 内存 flag，记录单层 TransformerBlock 为反向保存了多少激活 (列出 Top 5 贡献操作及占比)；并核对反向时新产生的梯度张量内存大小是否符合理论预期。

- [ ] 📝 **收口**：把 §2 的素材填进 [blog.md](blog.md) 的「§2 素材卡」→ **第 1 篇**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：作为开篇，展示 5090 的显存极限与 nsys 读法，定下“先量再找瓶颈”的基调。

## §3 Single-GPU Memory（4 分）

### 3.1 `gradient_checkpointing` (4分)
**Actions:**
- [ ] 编写一层不嵌套重算的 checkpointing 策略代码，对 xl 模型 (batch 4, seq 2048) 进行 1 步前/反向。
- [ ] 测试相邻的更大和更小的 checkpoint 块大小的峰值显存。
- 💡 **工程陷阱**：`torch.utils.checkpoint.checkpoint` 默认重算包裹的整个 block。为准确控制“单层重算”并测量峰值显存，建议写个 `dummy_forward` 脚手架配合 `torch.cuda.max_memory_allocated()`。

**Blog:**
- [ ] (a) 纯纸面推导：N 个相同 block 堆叠时，忽略计算开销、最小化峰值显存的检查点策略是什么？(提供代码草图、关于 N 的峰值显存和计算量的渐近复杂度)。
- [ ] (b) 解释 xl 实测的最优检查点块大小，并列出实测验证的显存数字。

- [ ] 📝 **收口**：把 §3 的素材填进 [blog.md](blog.md) 的「§3 素材卡」→ **第 2 篇**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：大模型在单卡 OOM 边缘的挣扎，以及激活检查点把显存换成了什么。

---

## §4 GPU Kernels — FlashAttention-2（29 分）

### 4.1 `pytorch_attention` (2分)
**Actions:**
- [ ] 编写原生 attention 测试脚本：batch=8，单头（无 head 维）。
- [ ] 遍历 `d_model` ∈ [16, 32, 64, 128] × `seq_len` ∈ [256, 1024, 4096, 8192, 16384]。
- [ ] 对每组跑 100 次前向测速，记录反向开始前的峰值显存，再跑 100 次反向测速（需 synchronize）。

**Blog:**
- [ ] 给出全组合耗时与显存表格（包含 OOM 的格子）。指出在什么规模开始 OOM。
- [ ] 对导致 OOM 的最小规模配置进行显存使用量数学核算。
- [ ] 解释保存的反向激活显存如何随序列长度变化，以及你将如何消除这笔开销。

### 4.2 `torch_compile` (2分)
**Actions:**
- [ ] 用 `torch.compile(layer)` 包装 Attention，测速。
- [ ] 编译整个 Transformer 模型，测速。

**Blog:**
- [ ] (a) 表格对比编译与未编译的 Attention 在 4.1 配置下的耗时。
- [ ] (b) 表格对比编译与未编译的完整 Transformer 模型（前向 vs 完整训练步）的耗时。

### 4.3 `flash_forward` (15分)
**Actions:**
- [ ] (a) 纯 PyTorch 的 Tiled 版本：实现 `autograd.Function.forward`，算出 `O` 与 logsumexp `L`，
      **`ctx` 里存 L,Q,K,V,O，但只 `return O`**；`backward` 这一步先 `raise NotImplementedError`。
      接口 `def forward(ctx, Q, K, V, is_causal=False)`，本小问可忽略 `is_causal`。Tile 至少 16×16。
- **判分**：`adapters.get_flashattention_autograd_function_pytorch` → `uv run pytest -k test_flash_forward_pass_pytorch`
- [ ] (b) Triton 版 Forward：严格按 PDF Algorithm 1 实现 `flash_fwd_kernel`，launch grid `(T_q, batch_size)`，
      **单层** key tile 循环，循环末尾 advance block pointer；片上 buffer（O_i, l, m）用 `tl.float32`，
      累加进输出 buffer 时用 `tl.dot(..., acc=acc)`。
- **判分**：`adapters.get_flash_autograd_function_triton`（⚠️ 文件里实际叫 `get_flashattention_autograd_function_triton`）
      → `uv run pytest -k test_flash_forward_pass_triton`
- [ ] (c) 因果掩码：为 PyTorch 和 Triton 版加入 `is_causal` 标志（默认为 False）。Triton 中对被 mask 的元素加 `-1e6`。

### 4.4 `flash_backward` (5分)
**Actions:**
- [ ] 使用纯 PyTorch + `torch.compile`（**不用 Triton**）实现 FA2 的 backward：输入 Q,K,V,O,dO,L，
      返回 dQ,dK,dV；先算 `D = rowsum(O ∘ dO)`，再按 PDF Eq.13–19 重算 P（不需要在线技巧）。
- **判分**：`uv run pytest -k test_flash_backward`

### 4.5 `flash_benchmarking` (5分)
**Actions:**
- [ ] 编写 `triton.testing.do_bench` 脚本打榜，对比你的（部分 Triton 的）FA2 与 PyTorch 原生 Attention 的前向/反向/端到端延迟。
- ⚠️ **这是单卡题**：PDF 原文是 “run the benchmark on a **single** B200”。不提交 → 型号不再约束；
  但**别把它和 §5–§7 的多卡需求混在一起**，它要的只是一张大显存卡（或就在 5090 上扫到哪算哪）。
- [ ] 配置：batch=1，is_causal=True，`seq_len` ∈ [128...65536] (2的幂) × `d_model` ∈ [16...128] (2的幂) × [bf16, fp32]。动态调整 tile size。

**Blog:**
- [ ] 提交一张极具冲击力的表格，展示前向、反向、端到端在庞大序列下的耗时对比。

### 4.6 Triton 版 backward (可选，但防坑必做)
**Actions:**
- [ ] 💡 **排雷提醒（这是我们的判断，不是 PDF 要求）**：PDF 的 §4.5 原话就是拿
      “your (**partially**) Triton implementation” 去比，即它默认你用的是 §4.4 那个 `torch.compile` backward。
      但超长序列下它很可能被 PyTorch 原生内核打爆，出来的表不好看。**想要漂亮的 §4.5 表就先写 Triton backward**；
      只想按 PDF 走则可以跳过。
- [ ] 按 Algorithm 2 实现 Triton Backward：算两遍 `P` 以避开 atomics 同步。

- [ ] 📝 **收口**：把 §4 的素材填进 [blog.md](blog.md) 的「§4 素材卡」→ **第 3 篇（系列重点）**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：从在线 softmax 推导一路到 Triton kernel 落地，再到延迟打榜的完整链条。

---

## §5 Distributed Data Parallel（21 分）

### 5.1 `distributed_communication_single_node` (5分)
**Actions:**
- [ ] 编写 all-reduce 测速脚本：数据量 [1MB, 10MB, 100MB, 1GB] × GPU数 [2, 4, 6]（每轮不超过5分钟，需 warmup 并通过 `dist.all_gather_object` 汇总）。

**Blog:**
- [ ] 给出图表对比不同设置的耗时，用 2-3 句话评论因素间的相互作用。

### 5.2 `naive_ddp` (5分)
**Actions:**
- [ ] 实现朴素 DDP：训练前 `broadcast` rank0 的参数；每次反向后对**每个参数梯度**分别 all-reduce 求平均。
      batch 切 n/d 份（d 必须整除 n）。
- **判分**：`adapters.get_ddp` +（可选）`adapters.ddp_on_after_backward` → `uv run pytest tests/test_ddp.py`

### 5.3 `naive_ddp_benchmarking` (3分)
**Actions:**
- [ ] 测速：1 node x 2 GPUs, `xl` 模型。记录总步耗时与通信时间。

**Blog:**
- [ ] 描述跑表配置，列出总耗时与通信占比。（⚠️ 真多卡上云测试，gloo 测试出的时间无效）。

### 5.4 `minimal_ddp_flat_benchmarking` (2分)
**Actions:**
- [ ] 修改为 Flat DDP：用 `_flatten_dense_tensors` 将全模型梯度拼成一个大张量，执行 1 次 all-reduce 后反拼。

**Blog:**
- [ ] 对比 Flat DDP 与朴素 DDP 在 1x2 GPU xl 上的速度与通信时间差异（1-2句）。

### 5.5 `ddp_overlap_individual_parameters` (5分)
**Actions:**
- [ ] 实现 Overlap DDP 容器类（`__init__` 包住任意 `nn.Module` 并广播初始权重 / `forward` 转发 /
      `finish_gradient_synchronization`）：用 `register_post_accumulate_grad_hook` 在每个梯度就绪时
      异步发 `all_reduce(async_op=True)`，`optimizer.step()` 前逐个 `handle.wait()`。
- **判分**：`uv run pytest tests/test_ddp.py`，**PDF 建议重复跑 5 次**排查竞态

### 5.6 `ddp_overlap_individual_parameters_benchmarking` (1分)
**Actions:**
- [ ] 测速，并使用 Nsight profiler 抓取 Overlap 与朴素版的 timeline 对比图。

**Blog:**
- [ ] (a) 给出 Overlap 版每步耗时，与朴素版和 Flat 版横评。
- [ ] (b) 贴出 2 张 Nsight 截图，直观证明通信与计算是否成功重叠。

- [ ] 📝 **收口**：把 §5 的素材填进 [blog.md](blog.md) 的「§5 素材卡」→ **第 4 篇**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：多卡上云初体验 + Modal 租卡避坑，并用 nsys 证明通信真的藏进了 backward。

---

## §6 Optimizer State Sharding（20 分）

### 6.1 `optimizer_state_sharding` (15分)
**Actions:**
- [ ] 实现简易 ZeRO-1 包装类。重写 `__init__(params, optimizer_cls, **kwargs)`（**必须调 super 构造函数**）、
      `step(closure, **kwargs)`（更新后把自己那份参数 broadcast 出去）、
      `add_param_group(param_group)`（把参数分配到各 rank）。
- **判分**：`adapters.get_sharded_optimizer` → `uv run pytest tests/test_sharded_optimizer.py`，**跑 5 次**

### 6.2 `optimizer_state_sharding_accounting` (5分)
**Actions:**
- [ ] 在 1x2 GPU xl 跑分，精确抓取三个时刻的峰值显存：模型初始化后、optimizer step 前、optimizer step 后。测定步耗时。

**Blog:**
- [ ] (a) 报告 3 个时刻的峰值显存，并精准拆解显存构成（参数/梯度/优化器状态/激活各占多少）。
- [ ] (b) 给出有/无 sharding 的耗时对比。
- [ ] (c) 说明我们的实现与 ZeRO-DP stage 1 在显存与通信量上的区别。

- [ ] 📝 **收口**：把 §6 的素材填进 [blog.md](blog.md) 的「§6 素材卡」→ **第 5 篇（上半）**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：算一笔 optimizer sharding 的显存账，并和 ZeRO-1 对照通信量。

---

## §7 Fully-Sharded Data Parallel（20 分）

### 7.1 `fsdp` (15分)
**Actions:**
- [ ] 实现 FSDP Wrapper，接管 `Linear` 与 `Embedding`。
- [ ] 实现预取策略：前向时 all-gather。反向时 all-gather，算完立即 reduce-scatter 并释放权重。支持 `compute_dtype`。
- **判分**：`adapters.get_fsdp` → `uv run pytest tests/test_fsdp.py`，**跑 5 次**排查竞态
  - ⚠️ `adapters.py` 里还有 PDF 没提的 `fsdp_on_after_backward` 和 `fsdp_gather_full_params`，别漏
- 💡 **工程难点**：PDF 要求“只在往前数第二层完成 forward 之后才开始 gather”（prefetch depth = 2）。
  标准 PyTorch hook 无法跨层感知，需要自己做 Layer 调度。另：norm 太小不值得分片，只包 `Linear`/`Embedding`；
  master 权重留 FP32，`compute_dtype` 只作用于通信和计算

### 7.2 `fsdp_accounting` (5分)
**Actions:**
- [ ] 测速 1x2 GPU xl，用 Nsight 抓 all-gather。

**Blog:**
- [ ] (a) 基于 §6 的分析，计算 FSDP 预期能再省下多少峰值显存？
- [ ] (b) 附上 Nsight 截图，证明 all-gather 是否赶上了 forward pass。

- [ ] 📝 **收口**：把 §7 的素材填进 [blog.md](blog.md) 的「§7 素材卡」→ **第 5 篇（下半）**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：FSDP 省了多少显存，以及 all-gather 预取有没有真的把通信藏住。

---

## §8 Analyzing Parallelism Strategies（17 分）

*(纯纸面推导，无需 GPU。设带宽 W，算力 C，FP16，matmul (A,B)(B,C) 取 2ABC flops)*

### 8.1 `alternate_ring_all_reduce` (1分)
**Blog:** 评估完整全传 all-reduce 算法的时间 (S, N, W) 并附一句话解释。

### 8.2 `data_parallel_calcs` (3分)
**Blog:** (a) DP反向 FLOPs。 (b) DP反向通信时间。 (c) 给出 DP 不被通信卡脖子的最大数量不等式。

### 8.3 `fsdp_calcs` (3分)
**Blog:** (a) FSDP 前/反向 FLOPs。 (b) FSDP 前/反向通信时间。 (c) FSDP 的最大数量不等式。

### 8.4 `tp_calcs` (4分)
**Blog:** (a) 写出行列混合 TP（W1/W2 列并行、W3 行并行的 FFN）完整的反向传播公式，
要产出每个设备的 `dW1(i)`、`dW2(i)`、`dW3(i)` 和反向输出 `dx`（**不是 attention 的 dQ/dK/dV**）。 (b) TP 前/反向 FLOPs。 (c) TP 前/反向通信时间。 (d) TP 的最大数量不等式。

### 8.5 `fsdp_tp_calcs` (6分)
**Blog:** (a) FSDP+TP 前向 FLOPs。 (b) 前向通信时间 (重叠双轴)。 (c) 可重叠时最大数量 N 不等式。 (d) 不可重叠时最大数量 N 不等式。

- [ ] 📝 **收口**：把 §8 的素材填进 [blog.md](blog.md) 的「§8 素材卡」→ **第 6 篇**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：把前面攒下的多卡体感，收束成 DP/FSDP/TP 的通信量推导。

---

## §9 Leaderboard（10 分）

**Actions:**
- [ ] 在 2 张 B200（或等效）上跑 8B 模型极限。batch=2, seq=32768, bf16, causal。
      判定：`triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)` 的墙钟时间，
      **从空 PyTorch/Triton cache 起 10 分钟内要跑完**（所以别开太猛的 `torch.compile` / autotune），
      目标是**打掉 10 秒的朴素基线**。配置：vocab 151936 / d_model 4096 / d_ff 11008 / 34 层 / 32 头
- [ ] **爆改清单**：Triton autotune 调参、实现 Fused AdamW、融合 LM head + cross entropy (省显存大头)、Triton 版 Backward 替换、Causal 掩码跳过 0-tile、分离非对角线/对角线、TMA (Hopper 架构起)、上激活检查点。

**Blog:**
- [ ] 记录这极限打榜拿下的墙钟时间。

- [ ] 📝 **收口**：把 §9 的素材填进 [blog.md](blog.md) 的「§9 素材卡」→ **番外**。
      **做完这一节立刻做，不要攒**（三步流程见 blog.md 顶部）。
      角度：融合算子 + FA 优化的极限压榨日志，每一步都要有优化前/后的数字。
