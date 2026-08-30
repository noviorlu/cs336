# CS336 A2 知乎博客 — 写作台账

> 配套：[todo.md](todo.md)（分题清单 / 硬件 / 协作模式）。
> **正文由我自己写**；Claude 负责表格、图、技术准确性 review（见 todo.md「协作模式」）。
>
> 这份文件的用途：**边做边攒素材**。每做完一道题就回来把数字、截图路径、踩的坑填进去，
> 等到动笔时素材已经齐了，不用回头翻终端记录。

---

## 系列定位

**标题方向**：《单卡 5090 跟着 CS336 A2 做系统优化》（暂定，最后再定）

**读者假设**：懂 Transformer 前向反向，写过 PyTorch，**没有 B200、没有多卡**。

**贯穿全系列的副线**：课程默认 B200(180G) + 2~6 卡，我只有一张 32G。
哪些能原样跑、哪些必须降规模、降了之后结论变没变——如实写出来。
这是这个系列区别于"照抄 handout 答案"的地方。

**统一交代（每篇开头或系列首篇声明一次）**

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5090 / 32 GB |
| torch | 2.11.0+cu130 |
| triton | 3.6.0 |
| 模型实现 | 自己的 assignment 1 实现（不是官方参考解） |
| 与官方参考实现的一致性 | state_dict 逐 key 相同，同权重同输入 logits 差 6e-07 |

---

## 六篇规划

进度图例：`○ 未开始` `◐ 素材收集中` `● 素材齐了` `✓ 已发布`

### 第 1 篇 ○ 怎么知道时间花在哪

覆盖 §2.1–§2.5 ｜ 依赖：阶段 1、7

- **要讲的**：CUDA 异步 → 为什么必须 `synchronize()`；warm-up 到底在预热什么；
  nsys 怎么读；混合精度省的是时间还是显存；显存 timeline 怎么看出在跑哪个阶段
- **素材清单**
  - [ ] 各 size 的 forward / backward / optimizer step 计时表（均值 ± 标准差）
  - [ ] 无 warm-up vs 1 步 vs 2 步 vs 5 步 的对比 —— 这是全文最有冲击力的一张表
  - [ ] nsys timeline 截图（标出 NVTX range）
  - [ ] CUDA kernel 累计耗时 Top-N
  - [ ] softmax vs matmul 的「耗时占比 ÷ FLOPs 占比」—— 埋下第 3 篇的伏笔
  - [ ] fp32 vs bf16 计时对比，以及"随规模变大差距怎么变"
  - [ ] 显存 timeline 两张图（纯 forward / 完整训练步）
- **坑记录**（发现一条填一条）
  -
- **降规模说明**：§2.5 点名 xl（完整训练步要 54.5 G），本机跑不了 →
  用 large 或只跑 forward。**写清楚换了什么、为什么、影响哪些结论**

### 第 2 篇 ○ 一层激活 3.6 GiB：激活检查点

覆盖 §3 ｜ 依赖：阶段 8

- **要讲的**：什么是 autograd 的 "residual"；为什么 RMSNorm 会存 5 个张量；
  `torch.compile` 融合之后为什么降到 3 个；checkpoint 把显存"换"成了什么；
  递归 checkpoint 的渐近分析
- **素材清单**
  - [ ] `saved_tensors_hooks` 的原始打印（融合前 / 融合后对比）
  - [ ] 单个 TransformerBlock 存了多少 MiB → 乘以层数的那个吓人数字
  - [ ] 不同 checkpoint 块大小的峰值显存实测（要有相邻的更大/更小两档做对照）
  - [ ] 递归策略的渐近推导（这是 §3.1(a)，我自己推）
- **坑记录**
  -

### 第 3 篇 ⭐ ○ FlashAttention-2：从在线 softmax 到 Triton kernel

覆盖 §4 ｜ 依赖：阶段 2、3 ｜ **系列重点**

- **为什么是重点**：29 分、有 pytest 客观验证、从数学推导一路到 GPU kernel，
  链条完整；知乎上这个题材受众最好
- **叙事线**：naive attention 在长序列上 OOM → 为什么（seq×seq 的分数矩阵）→
  在线 softmax 让 tiling 成为可能 → 纯 PyTorch tiled 版（慢但能对拍）→
  Triton kernel → causal masking → benchmark
- **素材清单**
  - [ ] §4.1 的 sweep 表：d × seq_len 的延迟 + 显存，**标出从哪一格开始 OOM**
  - [ ] OOM 那一格的显存手算核对（用 A1 的显存公式）
  - [ ] `torch.compile` 版对比 —— 说明"编译器能救多少，救不了什么"
  - [ ] 在线 softmax 的推导（running max / running sum 怎么合并两个 tile）
  - [ ] Triton kernel 代码（分段讲：block ptr 设置 → 主循环 → 归一化 → 写回）
  - [ ] FA2 vs PyTorch 的延迟对比表（forward / backward / 端到端）
  - [ ] 显存对比：seq 拉到多长仍然不 OOM
- **坑记录**（Triton 特别容易踩，重点记）
  - [ ] 片上 buffer 必须 fp32、`acc=` 怎么用
  - [ ] `P̃` 要先 cast 成 V 的 dtype
  - [ ] causal 要**加 -1e6** 而不是设 -inf（为什么）
  - [ ] `is_causal` 必须默认 False，否则前面的测试挂
  -
- **降规模说明**：§4.5 原文要求单张 B200、seq 扫到 65536。5090 上扫到哪算哪，
  如实写出上限 —— 这恰恰是副线最好的落点

### 第 4 篇 ○ 把通信藏进 backward：DDP 的三步优化

覆盖 §5 ｜ 依赖：阶段 4

- **叙事线**：逐参数 all-reduce（能跑但慢）→ 扁平化成一次（省调用开销）→
  用 `register_post_accumulate_grad_hook` 边算边传（把通信挪出关键路径）
- **素材清单**
  - [ ] all-reduce 基准：数据量 × 进程数（**要多卡，本机只能出脚本**）
  - [ ] 三种 DDP 的每步耗时 + 通信占比
  - [ ] **nsys 两张截图并排**：一张通信压在 backward 之后，一张叠在里面 —— 全文的题眼
- **坑记录**
  - [ ] `async_op=False` 也不代表通信完成（只是入队），为什么还得 synchronize
  -
- **本机限制**：判分测试走 gloo/CPU 能全过，但**通信占比的数字必须真多卡**。
  代码先写完，数字等上云或如实标注"未测"

### 第 5 篇 ○ 显存账本：从 ZeRO-1 到 FSDP

覆盖 §6–§7 ｜ 依赖：阶段 5、6

- **叙事线**：AdamW 两个动量 = 2× 权重显存 → 优化器状态分片（每个 rank 只管一份，
  step 后 broadcast）→ 连权重也分片（FSDP：用前 all-gather、用完就扔、
  梯度 reduce-scatter）→ 和 ZeRO 三个 stage 的对应关系
- **素材清单**
  - [ ] 分片前后的显存拆解表（参数 / 梯度 / 优化器状态 / 激活，三个时刻各一列）
  - [ ] 我的实现 vs ZeRO-1 的通信量差异（all-reduce+broadcast vs reduce-scatter+all-gather）
  - [ ] FSDP 的 all-gather 时序截图：通信赶不赶得上 forward
  - [ ] prefetch 深度为什么是 2
- **坑记录**
  - [ ] `add_param_group` 在超类构造函数里就被调用了
  - [ ] 哪些层不该分片（norm 太小，传输延迟不划算）
  -

### 第 6 篇 ○ 什么时候会被通信卡住：DP / FSDP / TP 的数学

覆盖 §8 ｜ 依赖：阶段 11（**最后做**）

- **为什么放最后**：亲手实现过 all-reduce、参数分片、all-gather + reduce-scatter
  之后再推公式，每一项都有代码记忆对应，而不是纯符号操作。
  这篇是整个系列的收束
- **素材清单**
  - [ ] ring all-gather / reduce-scatter / all-reduce 的耗时推导
  - [ ] 三种策略各自的 FLOPs、通信时间、瓶颈不等式
  - [ ] TP 反向传播的完整推导
  - [ ] 2D（FSDP × TP）在可重叠 / 不可重叠两种假设下的最优配置
  - [ ] 一张"什么规模该用什么并行"的总结图
- 全部是我自己推，不需要 GPU

### 番外 ○ 优化日志：把一步训练从 10 秒往下压

覆盖 §9 ｜ 不提交 leaderboard，纯当优化记录

- 每个优化点都要有**优化前 / 优化后**的数字：fused AdamW、
  融合 LM head + cross-entropy（现在光 logits 就 19.9 GB）、Triton 版 backward、
  causal 提前终止、TMA
- 配置降到单卡能跑的规模，结论照样成立

---

## 写作规范（作业不要求，但博客必须做）

- [ ] 代码片段**可独立运行** —— 作业只要答案，读者要能抄走就跑
- [ ] 每张表/图注明：**硬件、torch 版本、warmup/measure 步数、batch/seq 配置**
- [ ] 术语表（开篇定一次，全系列不换）
  - kernel（不译"内核"，避免和 OS 混）｜ 算子 = operator
  - residual = 存给反向的激活（**不是**残差连接，这个词在本作业里很容易歧义，要专门说明）
  - shard = 分片 ｜ all-gather / reduce-scatter / all-reduce 保留英文
  - tile = 分块 ｜ warp / thread block 保留英文
- [ ] **每篇都要有"我踩的坑"小节** —— 比正确答案更有阅读价值
- [ ] 降规模的地方一律显式声明：换了什么、为什么、哪些结论受影响、哪些不受
- [ ] 不贴 handout 原文（版权），公式和算法用自己的话重述

## 已经攒下的坑（还没归到具体某篇）

来自做 A2 前置准备时踩到的：

1. **猴补丁打在转发模块上会静默失效**。PDF 让你
   `cs336_basics.model.scaled_dot_product_attention = 注解版` 来插 NVTX。
   如果 `model.py` 只是 `from .transformer import *` 的转发，这个赋值改的是 `model`
   模块的全局绑定，而 `MultiHeadSelfAttention.forward` 是在 `transformer` 的命名空间里
   查这个名字的——**patch 不到，实测 0 次调用**。不报错，只是 NVTX range 全部消失。
   → 归第 1 篇。解决办法是直接把文件改名成 `model.py`，让函数和调用它的 forward 同模块
2. （继续往下加）
