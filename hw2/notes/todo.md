# CS336 Assignment 2 (Systems) — 完整 TODO

> 依据 `cs336_assignment2_systems.pdf`（Version 26.1.3, Spring 2026, 48 页）逐节梳理。
> 27 个 Problem，**总分 137**。hw1 重构相关的清单已移到同目录的 [todo-hw1-refactor.md](todo-hw1-refactor.md)。
>
> **图例**：`[代码]` 有 pytest 判分 · `[文字]` 产出博客正文 · `[图]` 产出截图/表格/曲线 ·
> `⚠️多卡` 需要 >1 GPU（→ Modal，计划内支出） · `⚠️大显存` 单卡 32G 放不下

## 协作模式（2026-08-30 定）

`hw2/CLAUDE.md` 是**课程官方给选课学生的 AI 使用规范**（只讲解、只 review，不写代码、
不给解法）。用户不是选课学生、不提交、不参与评分，但**博客要记录的是"我自己的实现步骤"**，
代写会掏空文章本身。所以取折中：

| | 谁来写 | 范围 |
|---|---|---|
| **核心实现** | **用户自己写** | Triton kernel、分布式逻辑、所有数学推导 |
| **脚手架 / 非核心** | Claude 可以写 | benchmark 脚本、表格生成、nsys 封装、画图、环境配置 |
| **review & debug** | Claude | 见下面的边界 |

### 逐题归属

**用户自己写（Claude 不给可粘贴的实现）**

- §3.1(a) 递归 checkpoint 策略推导
- §4.3 `flash_forward`（PyTorch tiled 版 + Triton kernel + causal masking）
- §4.4 `flash_backward`
- §4.6【可选】Triton 版 backward
- §5.2 `naive_ddp`、§5.4 的 flat-gradient DDP 变体、§5.5 `ddp_overlap`
- §6.1 `optimizer_state_sharding`
- §7.1 `fsdp`
- §8.1–§8.5 全部推导（DP / FSDP / TP / 2D 的 FLOPs、通信时间、瓶颈不等式）
- §9 leaderboard 的 kernel 优化（fused AdamW、融合 LM head + cross-entropy、FA 改进）

**Claude 可以写**

- §0 环境配置：`pyproject` 指向、`transformer.py` → `model.py` 改名、类别名
- §2.1 benchmark 脚本骨架：CLI 参数、warmup/measure 循环、`cuda.synchronize()`、均值±标准差
- §2.2 nsys 封装、NVTX range 的包装层（**被包的 attention 实现是用户的**）
- §2.5 memory profiler 开关、snapshot dump
- §4.1 / §4.2 attention 微基准的 sweep 骨架（含 OOM 捕获）
- §4.5 `triton.testing.do_bench` 的封装（**被测的 kernel 是用户的**）
- §5.1 all-reduce 基准脚本、`mp.spawn` 样板、`all_gather_object` 聚合计时
- §5.3 / §5.4 / §5.6 / §6.2 / §7.2 的 benchmark harness：计时、通信占比统计、显存快照
- **Modal 上云脚手架**（2026-08-30 新增）：镜像定义、代码同步、Volume 存产物、
  单容器多卡的 `mp.spawn` 启动、`--backend/--world-size/--local` 开关、账单/耗时记录。
  见「上云前的准入检查」——脚手架必须保证**本机四道闸和云上跑的是同一份代码**
- 所有表格生成（markdown / `to_latex` / `to_typst`）和画图
- 博客用的对照表、可复现性脚注（硬件、torch 版本、warmup/measure 步数）

### review & debug 的边界

用户明确要 Claude 做 review 和 debug，**所以不必用苏格拉底式提问绕圈子**：

- ✅ 直接指出哪一行错了、为什么错、会以什么现象暴露
- ✅ 给验证手段：toy input、shape 断言、和参考实现对拍、profiler 检查点
- ✅ 讲清算法思路、指 PDF 的 Algorithm 1/2 和公式编号
- ❌ 不直接贴出正确的核心实现代码替换掉
- ❌ 不代填 `tests/adapters.py` 里那 8 个 hook 背后的实现

用户卡住时的降级顺序：**指出症状 → 给验证方法 → 讲清思路 → 高层伪代码**，
到伪代码为止，不落到可粘贴的成品。

### 博客本身（2026-08-30 补充：正文可以由 Claude 起草）

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

**这条不放宽代码的边界**：上面「逐题归属」里列的核心实现（Triton kernel、
分布式逻辑、§8 推导）仍然由用户自己写。放宽的只有"把已有的东西写成文章"这一步。

---

## 交付物

**在家自学，不提交 Gradescope**。所以 `writeup.pdf`、`code.zip`、
`./test_and_make_submission.sh`、leaderboard 提交**全部不做**。

实际产出是两样：

| 产出 | 说明 |
|---|---|
| **知乎博客** | 记录实现步骤。取代 writeup.pdf —— 但要求不同：writeup 要的是"1-2 句作答"，博客要的是**为什么这么做、踩了什么坑、数字长什么样** |
| 代码 | `cs336_systems/` 里的实现，跑通 `uv run pytest tests/` 即可，不需要打包 |

**这带来的连锁影响**

- **提交打包的顾虑消失**：`test_and_make_submission.sh` 只 `zip -r .` 打包 hw2 目录，
  原本"指向 `../hw1` 会让提交里没有模型代码"的问题**不再存在**。
  → §0 的实现选型可以直接走最干净的方案（见下）
- **分值只当工作量参考**：下面的"分值"仍然有用，但读作**这题有多重多核心**，不是要拿的分
- **B200 这个「型号」的强制性没了**：§4.5 和 §9 点名 B200 是为了全班数字可比。
  不提交就不必迁就型号，用什么卡都行——**但这不等于"能省则省"**，见下条
- **多卡是花钱买的体验，不是要绕开的障碍**（2026-08-30 用户明确）：
  §5/§6/§7 的目的不是"测试能不能过"，而是**亲手感受写多卡代码的难度**——
  通信怎么和计算抢时间、hook 什么时候触发、`async_op` 的 handle 什么时候必须 wait、
  分片之后显存账怎么变。这些在 gloo/CPU 的 `world_size=2` 上**体会不到**：
  那里没有真实的通信带宽，也没有 CUDA stream，通信占比恒等于噪声。
  → **平台已定：Modal**，这部分是**计划内支出**
- **降规模只适用于"单卡装不下"的题**（§2.5 xl、§4.5 长 seq）。
  §5–§7 不在此列——那几题降的不是规模，是这门课的核心内容
- **但正因为要花钱，本机验证的地位反而更高了**（2026-08-30 用户补充）：
  gloo/CPU 从"省钱的替代品"升级成**上云的准入门槛**——见下面「上云前的准入检查」。
  租来的每一分钟都该花在"跑数字"上，不该花在"发现 import 写错了"

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
合计 69 分（占一半）且全部有 pytest 判分 —— 也就是**这三块最值得写进博客**，
既有硬核实现又有客观正确性验证。§8 的 17 分是纯纸面推导，不需要任何硬件，
可以单独成篇，适合最先动手。

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

按章节顺序排；`本机` 一列指单卡 RTX 5090 / 32 G。**全文只有 2 处硬性指定 B200**
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

**归纳**

- **本机能拿满**：§4.3/4.4 FlashAttention（20）、§6.1（15）、§7.1（15）、§5.2+5.5（10）、
  §8 全部（17）、§2.2（5）、§2.3（1）、§4.1+4.2（4）、§3.1(a) —— **合计 87 分以上**
- **点名 B200**：只有 §4.5（5 分）和 §9 leaderboard（10 分）。不提交 → **型号**不再有约束力，
  换张卡照样出图，文里写清硬件即可
- **要 2 张卡但不限型号**：§5.3、§5.4、§5.6、§6.2、§7.2（共 16 分）—— H100 / A100-80G 都行。
  **这一档要真上 Modal 跑**，"gloo 过了"不算完成（见「硬件现实」第 2 条）
- **要 6 张卡**：只有 §5.1（5 分），且每次运行 <5 分钟 —— Modal 单容器多卡正好覆盖

### 粗略预算

按"1 GPU·小时"计（含调试和重跑的余量，非精确）：

| 题目 | 卡数 × 时长 | GPU·h | RunPod $4.99 | Nebius 抢占 $3.05 |
|---|---|---:|---:|---:|
| §2.5 `memory_profiling`（xl） | 1 × ~2 h | 2 | $10 | $6 |
| §3.1(b) checkpointing（xl） | 1 × ~1 h | 1 | $5 | $3 |
| §4.5 `flash_benchmarking`（**B200**，80 组配置） | 1 × ~2 h | 2 | $10 | $6 |
| §5.1 all-reduce（2/4/6 进程） | 6 × ~0.5 h | 3 | $15 | $9 |
| §5.3 + §5.4 + §5.6 + §6.2 + §7.2（全是 2 卡 xl） | 2 × ~4 h | 8 | $40 | $24 |
| **小计（不含 leaderboard）** | | **16** | **~$80** | **~$49** |
| §9 leaderboard（迭代优化，开放式） | 2 × ? | ? | ? | ? |

**这 16 GPU·h 分两类，区别在于"是不是多卡"：**

| | 题目 | GPU·h | 性质 |
|---|---|---:|---|
| **计划内** | §5.1 + §5.3/5.4/5.6 + §6.2 + §7.2 | **11** | 多卡体验本身就是目的，这钱是要花的 |
| 可选 | §2.5、§3.1(b) 的 xl | 3 | 本机 large 也能出趋势；图不够撑文章再补 |
| 可选 | §4.5 的 B200 长 seq | 2 | 5090 扫到哪算哪，本来就是副线素材 |

### 平台：Modal（已定，2026-08-30）

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

---

## 🚧 上云前的准入检查（2026-08-30 用户要求：花钱前必须在 5090 上跑通）

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

**脚手架要求（归 Claude，见「协作模式」）**：Modal 入口脚本必须支持
`--backend {gloo,nccl}`、`--world-size N`、`--local`，让**闸 1/2/3 跑的是和云上完全同一份代码**，
只有启动器不同。如果本机和云上跑的是两份脚本，这四道闸就白设了。

**nsys 单独提前验**：§5.3.2(b) 和 §7 accounting(b) 要 nsys timeline，
而容器里跑 profiler 常有权限/驱动坑。**在闸 4 的冒烟跑里就把 nsys 试掉**，
别等到要出图那天才发现装不上

### 决策项（越早定越好）

- [x] **平台已定：Modal**（2026-08-30）
- [ ] **确认当前报价 + 卡型可用性**（上表采集于 2026-03-28，今天 2026-08-30，隔了 5 个月）。
      顺带确认：Modal 上多卡档位有哪些、B200 排不排得上队、便宜档（H100/A100/L40S）的价差
- [ ] **注册 Modal 拿 $30/月免费额度**
- [ ] **搭 Modal 脚手架**（归 Claude）：镜像定义、代码同步、Volume 存 profile 产物、
      `mp.spawn` 在单容器多卡里怎么起、结果怎么落回本机，以及上面「准入检查」要求的
      `--backend/--world-size/--local` 开关。**这一层写一次，§5–§7 全复用**
- [ ] **过完四道闸再花第一笔钱**（见上面「上云前的准入检查」）
- [ ] **第一次正式上云选 §5.1**：每次运行 <5 分钟、只需要一个 all-reduce 循环、
      不依赖模型代码——**失败成本最低的多卡首跑**，拿它把流程趟顺
- [ ] **xl 相关题目（§2.5、§3.1(b)）怎么处理**——这两道是**单卡**题，和多卡的决定无关：
  1. **本机用 large 替代 xl**（推荐）：省钱，"单卡 32G 能做到哪"本身是博客卖点。
     写清替换了什么、为什么，结论趋势不受影响
  2. 只做 forward-only（xl 纯前向 fp32 权重 13.6 G + 激活，bf16 下 6.8 G，单卡能跑），
     够写 §2.5 的一部分
  3. 上云补图：反正 Modal 流程已经为 §5–§7 搭好了，**边际成本很低**——
     真觉得某张图不够撑文章，顺手补一次即可
- [ ] **leaderboard 不提交，但值得当"优化日志"写**（见博客规划第 6 条）。
  没有 10 分钟时限和 B200 的约束，可以降到单卡规模慢慢迭代
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
  - 接口和权重层面**已经通了**（2026-08-29 的重构）：`model(x)` 单参数可调用、
    state_dict 与官方逐 key 一致、同权重同输入 logits 差 6e-07。剩下的只有命名问题。

  **判分测试实际卡在哪（`grep -rn cs336_basics tests/` 的结果）**

  hw2 四个测试文件里**只有 `test_fsdp.py` 引用 cs336_basics**，且只要三个类：
  ```
  tests/test_fsdp.py:27   from cs336_basics.model import Embedding, Linear, RMSNorm
  tests/test_fsdp.py:53   from cs336_basics.model import Embedding, Linear
  tests/test_fsdp.py:204  from cs336_basics.model import Embedding, Linear
  ```
  `test_ddp.py` / `test_sharded_optimizer.py` / `test_attention.py` **零引用**（自带 toy model）。
  这三个类我们**名字完全相同、`weight` 布局也相同**（`[out, in]`，实测 `Linear(4,8).weight` 是
  `[8, 4]`，和官方一致）。**唯一差异是模块路径**：官方在 `cs336_basics/model.py`，我们在
  `cs336_basics/transformer.py`。

  **两个方案，因为不提交，选后者**

  - [ ] ~~方案 A：建 `cs336_basics/model.py` 转发模块~~
    ```python
    from .transformer import *            # noqa: F403
    ```
    判分够用，但会踩下面那个猴补丁的坑（转发出来的名字 patch 不到）。

  - [ ] **方案 B（推荐）：`pyproject` 指向 hw1 + 把 `transformer.py` 改名成 `model.py`**
    ```toml
    # hw2/pyproject.toml:35
    cs336-basics = { path = "../hw1", editable = true }
    ```
    ```bash
    git mv hw1/cs336_basics/transformer.py hw1/cs336_basics/model.py
    # 再 sed 掉 10 处 import（decoder/interactive/scratch_eval/train/
    # tests/adapters.py ×3 / tests/test_causality.py ×2 / tests/test_rope.py）
    ```
    末尾加两行类别名（类别名不受命名空间问题影响）：
    ```python
    BasicsTransformerLM = TransformerLM
    RotaryEmbedding = RoPE
    ```
    **为什么更好**：改名后 `scaled_dot_product_attention` 和调用它的
    `MultiHeadSelfAttention.forward` 真的在同一个模块里，和官方结构一致，
    **下面那个猴补丁的坑直接消失**，PDF 的写法原样可用。
    可行性已核对：hw1 的 pyproject 本来就是 `name = "cs336_basics"` +
    `package = true` 的可安装包（规范化后 == `cs336-basics`）；
    两边都是 `torch~=2.11.0` / `python >=3.12,<3.14`，依赖不冲突；
    `notes.md` 和 `EXPERIMENTS.md` **零处**提到 `transformer.py`，没有文档要跟着改。
    README L38-40 自己也写了这条路："edit the outer `pyproject.toml` file to point to
    your own implementation"。
    **原本唯一的顾虑（提交打包时 `zip -r .` 抓不到 `../hw1`）因为不提交而消失。**

  - [ ] **⚠️ 只在选了方案 A 时才有的坑：猴补丁打在转发模块上会静默失效**

    PDF §2.1.4（L241）让你这样插 NVTX 注解：
    ```python
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    ```
    实测（转发模块 + 我们的实现）：
    ```
    patch cs336_basics.model.*        → 被调用 0 次   ✗ 完全没生效
    patch cs336_basics.transformer.*  → 被调用 1 次   ✓
    ```
    原因：官方的 `scaled_dot_product_attention` 和调用它的
    `CausalMultiHeadSelfAttention.forward` 在**同一个模块**里，改 `model` 的模块全局变量方法就能查到；
    而我们的 `MultiHeadSelfAttention.forward` 是在 **`cs336_basics.transformer` 的全局命名空间**
    里查这个名字的，转发模块里的绑定它根本不看。
    **危险在于不报错**——nsys 照跑，只是 NVTX 里永远没有 "computing softmax" 那几个 range，
    §2.2 的 (b)(c)(e) 三问答不出来，还会以为是 nsys 配置错了。
    **正确写法：`cs336_basics.transformer.scaled_dot_product_attention = 注解版`**

  - [ ] **PDF 示例代码的名字对不上（不判分，抄的时候会报错）**

    | PDF 用的 | 我们的 | 出处 |
    |---|---|---|
    | `RotaryEmbedding` | `RoPE` | §3.2 示例（L608） |
    | `BasicsTransformerLM` | `TransformerLM` | §9 leaderboard（L2618） |
    | `TransformerBlock(..., positional_encoder=...)` | `TransformerBlock(..., rope=...)` | §3.2 示例（L608） |

    前两个上面的转发模块里已经加了别名；`TransformerBlock` 的关键字名不同，抄 §3.2
    示例时手改一下即可。

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

### 2.1 `benchmarking_script` — 4 分 `[代码基建]` `[文字]`

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

### 2.2 `nsys_profile` — 5 分 `[文字]` `[图]` ⚠️需要 nsys

- [ ] 选 **2 个 model size × 3 个 2 的幂次 context length（>128，最大取显存能装下的极限）** 做 profile
  ```bash
  uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx --gpu-metrics-devices=0 -- python benchmark.py
  ```
  （`--cudabacktrace=all --python-backtrace=cuda` 开销很大，不需要 traceback 时关掉）
- [ ] **给代码加 NVTX 标注**：用 `@nvtx.range(...)` / `with nvtx.range(...)` 圈出
  warm-up（好用 `--nvtx-capture` 过滤掉）、forward/backward、以及 attention 内部的
  "attention scores" / "softmax" / "final matmul" 三段
  - 做法：写 `annotated_scaled_dot_product_attention`，再猴补丁替换掉原实现
  - **⚠️ 若 §0 选了方案 A（转发模块）**：必须 patch
    `cs336_basics.transformer.scaled_dot_product_attention`，不是 PDF 写的
    `cs336_basics.model.*` —— 打在转发模块上会静默失效（实测 0 次调用），
    NVTX range 一个都不会出现。**选方案 B（改名）则无此问题**，PDF 写法原样可用
- [ ] (a) forward 总时间，和 §2.1 用 Python 标准库测的对不对得上？→ 1-2 句
- [ ] (b) forward 里累计 GPU 时间最长的 CUDA kernel 是哪个？单次 forward 调用几次？
  加上 backward 之后还是它吗？→ 1-2 句
- [ ] (c) 除了矩阵乘，还有哪些 kernel 占了不可忽略的时间？→ 1-2 句
- [ ] (d) 完整训练步（含自己的 AdamW）里矩阵乘占比相对纯推理怎么变？其它 kernel 呢？→ 1-2 句
- [ ] (e) attention 内部 softmax vs 矩阵乘的**运行时间比** 和 **FLOPs 比** 差多少？→ 1-2 句
  - （这题是后面 FlashAttention 的动机：softmax 的 FLOPs 占比极小但耗时占比不小 = memory-bound）

### 2.3 `mixed_precision_accumulation` — 1 分 `[文字]`

- [ ] 跑题面给的 4 段累加代码（fp32 累加 / fp16 累加 / fp32 累加器加 fp16 值 /
  显式 `.type(torch.float32)` 后累加），解释精度差异。→ 2-3 句
  - 纯送分题，10 分钟能做完，建议第一个做

### 2.4 `benchmarking_mixed_precision` — 2 分 `[文字]`

- [ ] **(a)** 给定 `ToyModel`（fc1 → relu → LayerNorm → fc2），在 fp16 autocast 下写出六个 dtype：
  模型参数 / fc1 输出 / LayerNorm 输出 / logits / loss / 梯度
- [ ] **(b)** LayerNorm 的哪些部分对低精度敏感？换成 **BF16** 之后还需要特殊对待吗？为什么？→ 2-3 句
  - （方向：均值/方差是累加归约，动态范围；BF16 指数位和 FP32 一样宽）
- [ ] **(c)** 给 benchmark 脚本加 `--dtype bfloat16` 开关，5 个 size 各测有/无混合精度，
  说明随规模变化的趋势。→ 2-3 句 + 计时表
  - hw1 的 `train.py` 已经有 `torch.autocast` + `nullcontext` 的写法，可以直接搬

### 2.5 `memory_profiling` — 4 分 `[文字]` `[图]` ⚠️大显存

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

### 3.1 `gradient_checkpointing` — 4 分 `[文字]`

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

### 4.1 `pytorch_attention` — 2 分 `[文字]`

- [ ] 写 attention 微基准脚本：
  - batch=8，**不带 head 维**（单头）
  - `d_model ∈ {16, 32, 64, 128}` × `seq_len ∈ {256, 1024, 4096, 8192, 16384}` 全组合
  - 随机 Q/K/V；计时 100 次 forward；**记录 backward 开始前的显存**；计时 100 次 backward
  - warm-up + 每次 `torch.cuda.synchronize()`
- [ ] 报表格（含 OOM 的格子）。在哪个规模开始 OOM？对最小的那个 OOM 配置**做显存核算**
  （用 hw1 的 Transformer 显存公式）。存给 backward 的显存怎么随 seq_len 变？怎么消掉这笔开销？
  → 表 + 计算过程 + 1-2 段

### 4.2 `torch_compile` — 2 分 `[文字]`

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

### 4.5 `flash_benchmarking` — 5 分 `[文字]` ⚠️理想在 B200

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

### 5.1 `distributed_communication_single_node` — 5 分 `[文字]` ⚠️多卡

- [ ] benchmark all-reduce 耗时：
  - 数据量 fp32 张量 **1MB / 10MB / 100MB / 1GB**
  - 进程数（GPU 数）**2 / 4 / 6**
- [ ] 注意事项：warm-up ≥5 次（NCCL 尤其需要）；每次 `torch.cuda.synchronize()`
  （**即使 `async_op=False` 也必须同步**，它只保证入队不保证完成）；
  用 `dist.all_gather_object` 聚合各 rank 的计时
- [ ] → 图/表 + 2-3 句
- ⚠️ 需要最多 6 张 GPU；本机只有 1 张 → **这是计划里的第一次正式上云**（Modal，见「决策项」）。
  失败成本最低：不依赖模型代码、每次 <5 分钟。**先过完四道闸**（见「上云前的准入检查」），
  尤其闸 2——这题全是集合通信，NCCL 路径必须先在本机 `world_size=1` 上验过

### 5.2 `naive_ddp` — 5 分 `[代码]`

- [ ] 实现最朴素的 DDP：backward 之后对**每个参数的梯度**单独 all-reduce 求平均
  - 训练开始前用 `broadcast` 把 rank 0 的参数发给所有 rank
  - batch 切分：n 个样本切成 n/d 份（d 必须整除 n）
- [ ] → `adapters.get_ddp` +（可选）`adapters.ddp_on_after_backward`
- [ ] → `uv run pytest tests/test_ddp.py`（用 gloo/CPU，world_size=2，本机可跑）

### 5.3 `naive_ddp_benchmarking` — 3 分 `[文字]` ⚠️多卡 ⚠️大显存

- [ ] 测**每步总时间**和**通信占比**，配置：1 node × 2 GPU，**xl**
- [ ] → 描述 setup + 数字
- ⚠️ **通信占比这个数只有真多卡才有意义**——gloo/CPU 上量出来的是噪声，不能拿来写博客。
  上云前先在本机把闸 3 过掉（xl 或替代档在单卡上一步不 OOM）

### 5.4 `minimal_ddp_flat_benchmarking` — 2 分 `[代码]` `[文字]` ⚠️多卡

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

### 5.6 `ddp_overlap_individual_parameters_benchmarking` — 1 分 `[文字]` `[图]` ⚠️多卡

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

### 6.2 `optimizer_state_sharding_accounting` — 5 分 `[文字]` ⚠️多卡 ⚠️大显存

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

### 7.2 `fsdp_accounting` — 5 分 `[文字]` `[图]` ⚠️多卡 ⚠️大显存

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

### 8.1 `alternate_ring_all_reduce` — 1 分 `[文字]`

- [ ] 题面给了另一种 all-reduce 算法（每步直接传完整的 $x^{(i)}$ 而不是分块）。
  用 S、N、W 表示它的耗时 + 一句话论证
  - （提示：每步传的是整个 S 而不是 S/N，所以是 $(N-1)\frac{S}{W}$，比 ring 版差 N/2 倍）

### 8.2 `data_parallel_calcs` — 3 分 `[文字]`

FFN 前向：$x_1=xW_1$，$x_2=xW_2$，$z=f(x_1)*x_2$，$y=zW_3$；
x 是 (B,D)，$W_1,W_2$ 是 (D,D_ff)，$W_3$ 是 (D_ff,D)。反向式 (24)–(30) 题面已给。

- [ ] **(a)** $N_{DP}$ 数据并行下 backward 的 FLOPs（用 B, D, D_ff, N_DP 表示）+ 一句论证
- [ ] **(b)** backward 的通信时间（B, D, D_ff, N_DP, W 的子集）+ 一句论证
- [ ] **(c)** 其它参数固定时，$N_{DP}$ 能开到多大才不被通信卡住？给不等式 + 一句论证

### 8.3 `fsdp_calcs` — 3 分 `[文字]`

- [ ] **(a)** $N_{FSDP}$ 下 forward 和 backward 各多少 FLOPs（两个答案）
- [ ] **(b)** forward 和 backward 各多少通信时间（两个答案）
  - forward：3 次 all-gather；backward：3 次 all-gather + 3 次 reduce-scatter
- [ ] **(c)** forward 和 backward 各自的 $N_{FSDP}$ 上界（两个不等式）

### 8.4 `tp_calcs` — 4 分 `[文字]`

配置：$W_1, W_2$ **column parallel**（切输出维），$W_3$ **row parallel**（切输入维），
所以 column 之后不用 all-gather，只在最后对 y 做一次 all-reduce。

- [ ] **(a)** 写出这个 TP 配置的**完整反向传播公式**：给定 dy (B,D)，
  用分片权重、前向存下的激活、通信原语，推出 $dW_1^{(i)}, dW_2^{(i)}, dW_3^{(i)}$ 和 dx
- [ ] **(b)** forward / backward 各多少 FLOPs（两个答案）
- [ ] **(c)** forward / backward 各多少通信时间（两个答案）
- [ ] **(d)** forward / backward 各自的 $N_{TP}$ 上界（两个不等式）

### 8.5 `fsdp_tp_calcs` — **6 分** `[文字]`

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

## 博客产出规划（知乎）

作业本身的章节结构就是一条不错的叙事线：**先量，再找瓶颈，再优化，再量一次**。
按这个拆成 6 篇，每篇都能独立成文：

| # | 文章 | 覆盖 | 素材类型 | 本机能否写完 |
|---|---|---|---|:--:|
| 1 | **怎么知道时间花在哪** | §2.1–§2.5 | 计时表、nsys timeline 截图、显存 timeline | ✓（size 自选） |
| 2 | **一层激活 3.6 GiB：激活检查点** | §3 | `saved_tensors_hooks` 打印、峰值显存对比 | ✓（用 large） |
| 3 | **FlashAttention-2 从零到 Triton** ⭐ | §4 | 在线 softmax 推导、kernel 代码、延迟对比表 | ✓ |
| 4 | **把通信藏进 backward：DDP 三步走** | §5 | 通信占比、nsys 重叠对比截图 | 代码 ✓ / 数字要 2 卡 |
| 5 | **显存账本：ZeRO 与 FSDP** | §6–§7 | 分片前后显存拆解、all-gather 时序 | 代码 ✓ / 数字要 2 卡 |
| 6 | **什么时候会被通信卡住：DP/FSDP/TP 的数学** | §8 | 纯推导 + 不等式 | ✓ 不需要 GPU |

**第 3 篇是重点**：29 分、有客观正确性验证（pytest）、从数学推导到 Triton kernel 全链条，
知乎上这个题材受众最好。

**一条贯穿全系列的副线**：*「单卡 5090 + 按需租的多卡，跟着 CS336 A2 能做到哪一步」*。
课程默认 B200 + 8 卡，绝大多数读者没有。副线有两半，**别只写前一半**：

1. **单卡能做到哪**（§2.5、§3.1(b)、§4.5）：哪些能在 32G 上跑、哪些必须降规模、
   降了之后结论变没变——如实写出来，比照抄 xl 的数字更有价值
2. **要多卡时怎么办**（§5–§7）：我是**真去 Modal 上租了卡**跑的，不是用 gloo/CPU 糊过去。
   这一半对读者更实用——"我也没有多卡"的人最想知道的恰恰是
   *花多少钱、怎么起、哪些坑是容器/云特有的*。**把账单和踩的坑一起写出来**

**leaderboard（§9）虽然不提交，但仍是最好的博客素材**：
从 10 秒基线往下优化，每一步（fused AdamW、融合 LM head + cross-entropy、
Triton 版 backward、causal 提前终止）都有可量化的收益，天然是一篇"优化日志"。
配置降到单卡能跑的规模照样成立。

**写作时要额外做、但作业不要求的事**

- [ ] 代码片段要**可独立运行**（作业只要答案，博客读者要能抄走就跑）
- [ ] 每张表/图注明**硬件、torch 版本、warmup/measure 步数**（可复现性）
- [ ] 中文术语统一（kernel / 算子、residual / 激活、shard / 分片……开篇定好）
- [ ] 把"踩坑"单独写出来 —— 比如已经遇到的
      「猴补丁打在转发模块上静默失效」，这类内容比正确答案更有阅读价值

## 建议执行顺序

| 阶段 | 内容 | 分值 | 硬件 |
|---|---|---:|---|
| **1** | §0 环境配置 + §2.1 benchmark 脚本 + §2.3 累加精度 + §2.4 混合精度 | 7 | 单卡（xl/10B 那两档留到阶段 9） |
| **2** | §4.1 attention 微基准 + §4.2 torch.compile | 4 | 单卡够 |
| **3** | §4.3/4.4 FlashAttention forward + backward | **20** | 单卡够 |
| **4** | §5.2 naive DDP + §5.5 overlap DDP | 10 | 本机 gloo/CPU（闸 1） |
| **5** | §6.1 optimizer sharding | **15** | 本机 gloo/CPU（闸 1） |
| **6** | §7.1 FSDP | **15** | 本机 gloo/CPU（闸 1）｜此后搭 Modal 脚手架 |
| **7** | §2.2 nsys profile（自选两个 size，按显存自适应） | 5 | 单卡 + 要装 nsys |
| **8** | §3 gradient checkpointing | 4 | (a) 本机推导 / (b) 要大显存 |
| **9** | benchmark 类（§2.5、§4.5、§5.1/5.3/5.4/5.6、§6.2、§7.2） | 30 | **Modal 多卡**（计划内）｜过完四道闸再开机 |
| **10** | §9 leaderboard（当"优化日志"写，不提交） | 10 | 开放式 |
| **11** | **§8 全部推导（5 题）** | **17** | 无需 GPU |

**§8 为什么排到最后**（2026-08-30 用户定）：原来排第一是因为它不需要硬件。
改到最后的好处是——DP / FSDP / TP 的通信量推导，在**亲手实现过 §5 的 all-reduce、
§6 的参数分片、§7 的 all-gather + reduce-scatter 之后**再做，公式里的每一项都有对应的
代码记忆，而不是纯符号操作。博客第 6 篇也因此变成整个系列的收束，而不是开篇。

阶段 **1–7 合计 76 分**，全部在本机（单卡 5090 + gloo/CPU）完成。
阶段 8（§3，4 分）里 (a) 是本机推导、(b) 要大显存。
再加上排在最后的 §8 推导（17 分），**共 93 分不花一分钱**。

**但"不花钱"不是目标**——剩下的 44 分里，`§5.1/5.3/5.4/5.6 + §6.2 + §7.2` 那一档
（约 11 GPU·h）是**计划内支出**：多卡的手感就是要花钱买的，
gloo/CPU 上跑出来的通信占比是噪声，写进博客等于编数字。
真正可省的只有 §2.5 / §3.1(b) 的 xl（用 large 替代）和 §4.5 的 B200 长 seq。

**阶段 1–8 的真正作用因此变了**：它不只是"先把免费的分拿掉"，
更是**上云的准入准备**——每一份要在 Modal 上跑的代码，都先在本机过完四道闸
（见「上云前的准入检查」）。省钱靠的是**挑卡型 + 不在云上 debug**，不是砍内容。
