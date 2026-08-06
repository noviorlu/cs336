# CS336 Assignment 1 · 实验日志

handout `experiment_log` 要的那份「a document of all the things you tried」。
每条实验的曲线在 wandb project `cs336-hw1`,run 名和下表的「run」列对得上。

**硬件**:单张 RTX 5090(32 GB,bf16 稠密峰值 209.5 TFLOPS)。handout 的预算以
B200 计,这里的墙钟时间不能直接和它比。

**噪声底线(读下面任何一个差值之前先看这个)**:同配置(lr 5e-3, batch 256)只换
随机种子跑三遍,val loss = **1.3381 / 1.3428 / 1.3323**,σ ≈ **0.0053**,极差 0.0105。
本文档里凡是小于 ~0.011(2σ)的差,一律**不作结论**。这条底线是后补的,它直接推翻了
我原先对学习率和 SwiGLU 两处的读法 —— 见 §1 和 §3。

**基线配置**(`tinystories_17M_tuned.yaml`):V=10000,T=256,d_model=512,
4 层 16 头,d_ff=1344,SwiGLU,pre-norm,RoPE θ=10000。22.70 M 参数(含两份
embedding;handout 说的 17M 是非 embedding 口径)。总 token 预算 327,680,000
= batch 256 × 5000 步 × 上下文 256。

---

## 0. 跑实验之前踩到的两个坑

这两条不是 handout 要求的实验,但它们决定了下面的数能不能信,记在最前面。

### 0.1 MFU 的两个常数都填错了,而且偏的方向相反

原来的算法是 `MFU = tok/s × 6P ÷ peak`,两处都不对:

| | 原来 | 正确 | 差 |
|:--|:--|:--|:--|
| 每 token FLOPs | `6P` = 136.18 M | `3F/T` = 111.72 M | 6P **高估 21.9%** |
| 峰值 | 340 TFLOPS | 209.5 TFLOPS | **高估 62%** |

`6P` 高估的两个来源:① 它把 embedding 查表当成算力,而查表是索引、0 FLOP —— 本
配置下 embedding 占 P 的 22.6%,凭空多出 30.72 MFLOPs/token;② 它又漏掉注意力里
两个 T² 项,少算 6.29 MFLOPs/token。净效果是高估 21.9%。

峰值那个 340 不对应 RTX 5090 任何一个公布口径:419 是带 sparsity 的,104.8 是
FP32 shader,PyTorch 的 bf16 矩阵乘走的是 FP32 累加那条,**209.5**。

两个错方向相反,合起来把 MFU 报低了:实测 578,000 tok/s,原报 23.2%,**真实
30.8%**(= 实际 64.6 TFLOPS)。对 22.7 M 这么小的模型 + 手写 attention,30% 算健康
(参考:nanoGPT 124M 在 A100 上约 37%,PaLM 540B 46%)。

### 0.2 `cross_entropy` 在 batch=1024 下 int32 溢出

`einx.get_at` 的实现是先把 logits 摊平成一维再用线性下标取,而摊平长度用 int32 算。
`[1024, 256, 10000]` = 26.2 亿元素越过 2³¹,长度翻负:

```
26.2e8 − 2³² = −1,673,527,296
RuntimeError: invalid shape dimension -1673527296 at index 0
```

换成 `torch.gather`(按维度取,不摊平)解决。**注意这和显存无关** —— 修好之后
batch=1024 仍然跑不了,原因是显存,见 §2。两件事要分开记。

---

## 1. `learning_rate` — 学习率

固定 batch=256、5000 步,只扫 lr。

| lr | val loss | run |
|:--|:--|:--|
| 5e-4 | 1.5066 | `m7lo15v4` |
| 1e-3 | 1.4186 | `x9z3zo8x` |
| 3e-3 | 1.3512 | `dn21ib3d` |
| **5e-3** | **1.3381** / 1.3428 / 1.3323(三种子) | `t8nhfslq` `seed2` `seed3` |
| **6e-3** | **1.3328** | `lr6e-3` |
| 8e-3 | 1.3539 ← 掉头 | `lr8e-3` |
| 1e-2 | 5.5256 ← 发散 | `32lczj9f` |

**最优是一段平台,不是一个点。** 6e-3 的 1.3328 和 5e-3 的三种子均值 1.3377 只差
0.0049,不到 1σ —— 这两档**分不出高下**。按 σ 过一遍整条曲线:

| 对比 | Δ | 倍数 | 判定 |
|:--|:--|:--|:--|
| 8e-3 vs 平台 | 0.016 | 3.1σ | 退化是真的 |
| 3e-3 vs 平台 | 0.014 | 2.5σ | 勉强算真 |
| 6e-3 vs 5e-3 | 0.005 | 0.9σ | **噪声,不能说 6e-3 更好** |

**搜索策略**:先按半个数量级粗扫(5e-4 → 1e-2)定出「最优在哪一档、发散在哪一档」,
再在最优和发散之间补点。粗扫结果是最优 5e-3、发散 1e-2,只差一档,所以补了 6e-3
和 8e-3 两点把边界卡细。

**deliverable ②(val loss ≤ 1.45)**:达标,最好 **1.3323**,比要求低 0.12。

**deliverable ③(edge of stability)**:最优平台 **5e-3 ~ 6e-3**、发散阈值 **1e-2**,
只差 **1.7 ~ 2 倍** —— 「最好的学习率就贴在发散边缘」这条民间智慧在这里成立。

补的两点还捞到一个粗扫看不见的结构:**8e-3 已经变差(1.3539,3.1σ),但还没发散**。
也就是说从「最优」到「炸掉」中间不是断崖,而是有一段仍然收敛、只是性能退化的窄带:

```
    ← 稳定且改善 →|← 平台 →|← 稳定但退化 →|← 发散 →
  5e-4 ········ 5e-3 ·· 6e-3 ······· 8e-3 ······· 1e-2
 1.5066        1.3377  1.3328       1.3539       5.5256
```

这和 edge-of-stability 的图像是自洽的:曲率(损失面的最大 Hessian 特征值)会自己
调整到 ≈ 2/lr 附近,lr 越大越把优化推向更平的区域,直到 lr 大过某个阈值后这个自
平衡维持不住、一步跨过去就发散。退化窄带就是「还稳得住、但已经被迫在太平的区域
里走」的那一段。

**代价**:整个 sweep 的收益(1.5066 → 1.3328 = 0.174)全部来自把 lr 从 5e-4 推到
6e-3。这个数字下面 `layer_norm_ablation` 会再用到一次。

---

## 2. `batch_size_experiment` — batch 大小

固定 lr=1e-3、5000 步(注意:步数固定意味着 batch 越小看到的 token 越少,
batch=1 只看了 1/256 的数据 —— 下面的 loss 差里有这一份贡献)。

| batch | val loss | 峰值显存 | MFU | run |
|:--|:--|:--|:--|:--|
| 1 | 2.9307 | | | `jmqgpumc` |
| 64 | 1.5452 | | | `8q3choii` |
| 128 | 1.4694 | | | `nmol1cdt` |
| 256 | 1.4188 | ~18 GB | 30.8% | `t9jnpkxj` |
| **352** | **1.4025** | 24.73 GB | **32.98%** | `bs352` |
| 384 | — | OOM | | |
| 512 | — | OOM | | |
| 1024 | — | OOM(saved tensors 就要 47.8 GB) | | |

**显存墙 = 352**(11 × 32)。384 线性外推只要 26.9 GB、卡上有 32 GB,却仍然 OOM ——
差的是 `max_memory_allocated` 和实际 `reserved` 之间的碎片、CUDA context 和显示占用。

**注意 batch=1 那条的 lr 没有重调**。handout 说 "The learning rates should be
optimized again if necessary",1e-3 对 batch=1 几乎必然过大,2.93 这个数里有多少是
「batch 小」、有多少是「lr 没调」,现在分不开。

### 讨论:batch 大不总是好,但在这个尺度上还没到拐点

**收益是递减的,而且递减得很快。** 每翻一倍 batch 换来的 val loss 改善:
64→128 得 0.076,128→256 得 0.051,256→352(只 1.375 倍)得 0.016。外推下去,
就算显存够、再翻一倍到 704 大概也只能再拿 0.01 出头。

**但「大 batch 有害」在这里还看不到。** 常说的大 batch 泛化变差,前提是**固定
epoch 数**,大 batch 因此少走很多步。这里是**固定步数**(5000),batch 越大看到的
token 越多,两个效应同向,所以曲线是单调的。要复现「批量太大反而更差」得改成固定
token 预算再扫,那是另一个实验。

**batch=1 那条掉到 2.93 有三重原因叠加,不能只归给 batch**:① 只看了
1/352 的 token;② lr 没重调;③ 单样本的梯度噪声。想拆开需要在固定 token 预算下
重扫,已记在待办。

**吞吐这一侧倒是单调受益**:batch 256 → 352,MFU 从 30.8% 涨到 32.98%。矩阵乘的
M 维从 65,536 涨到 90,112,tensor core 喂得更满。这是「为什么想要大 batch」的
真正理由 —— 是**硬件效率**,不是收敛质量。

---

## 3. 架构消融(§7.3)

全部对齐基线 lr=5e-3 / batch=256 / 5000 步。基线 val loss = **1.3381**。

| 消融 | 改了什么 | 参数量 | val loss | Δ | 相对 σ | run |
|:--|:--|:--|:--|:--|:--|:--|
| 基线 | pre-norm + SwiGLU + RoPE | 22,696,448 | **1.3377**(3 种子均值) | — | — | `t8nhfslq` 等 |
| `swiglu_ablation` | 无门控 SiLU,d_ff=2048 | 22,827,520 | 1.3404 | +0.003 | **0.5σ** | `abl_silu` |
| `pre_norm_ablation` | post-norm(式 27/28) | 22,695,936 | 1.3762 | +0.039 | 7.3σ | `abl_postnorm` |
| `no_pos_emb` | NoPE | 22,696,448 | 1.3945 | +0.057 | 10.7σ | `abl_nope` |
| `layer_norm_ablation` | 拆光所有 RMSNorm | 22,691,840 | **NaN**(同 lr)<br>1.4885(自身最优 lr 7e-4) | +0.151 | — | `abl_nonorm_*` |

**参数量对齐**:SwiGLU 三个矩阵 3·512·1344 = 2,064,384;SiLU 两个矩阵
2·512·2048 = 2,097,152,差 1.6%。消融的是「门控」这一件事,不是参数量。

**ln_final 怎么处理**:只有 pre-norm 保留。pre-norm 把归一化挪到了子层入口,残差流
层层累加没人约束,不在出口补一次的话 lm_head 拿到的尺度会失控;post-norm 每个子层
出口本就归一化过。`norm=none` 是要故意全拆,所以也去掉 —— 这是 handout
"remove all of the RMSNorms from your Transformer" 的字面读法。

一个提前能看到的信号:同一份权重下,`norm=none` 相对 pre-norm 的输出最大差是
**126.06**(post-norm 是 0.84,NoPE 是 0.53)。还没训练就差这么多,正是残差流尺度
失控的表现。

### 3.1 `layer_norm_ablation`:RMSNorm 买的不是拟合,是学习率的天花板

**在原最优 lr(5e-3)下,第 200 步 NaN**,而且死得很有指向性:

```
step 150 | loss 2.5222 | lr 3.7500e-03   ← 一路正常下降
step 200 | loss nan    | lr 5.0000e-03   ← 炸
```

warmup 正是 200 步 —— 它**恰好在 lr 第一次爬到峰值的那一步炸掉**。前 150 步(lr 还在
1.25e-3 ~ 3.75e-3)完全正常。所以不是「去掉 norm 就不能训」,是「能承受的 lr 上限被
大幅压低」。

**降低 lr 能不能救回来?能,但天花板低得多**:

| lr | 5e-4 | **7e-4** | 1e-3 | 5e-3 | 8e-3 | 1e-2 |
|:--|:--|:--|:--|:--|:--|:--|
| 拆光 RMSNorm | 1.5433 ✓ | **1.4885** ✓ 最优 | **NaN** | **NaN** | — | — |
| 基线(有 norm) | 1.5066 ✓ | — | 1.4186 ✓ | 1.3377 ✓ | 1.3539 ✓ | 5.5256 ✗ |

两条曲线各自的**发散阈值**:基线在 [8e-3, 1e-2) 之间,no-norm 在 [7e-4, 1e-3) 之间。
比值约 **10 倍** —— 拿掉 RMSNorm,可用学习率的天花板整整掉一个数量级。

**结论,以及它为什么反直觉**:同一个 lr(5e-4)下,拆掉 RMSNorm 只差 **0.037**。
RMSNorm 几乎不直接改善拟合。真正的伤害是它把你锁在低 lr 区,而 §1 已经测出这段
lr 空间值多少钱。把两件事分开算:

```
no-norm 实际能拿到的最好成绩            1.4885  (lr 7e-4)
若它能用 5e-3、且只付「同 lr 下」那份代价  1.3747  (= 1.3377 + 0.037)
基线实际拿到的                          1.3377  (lr 5e-3)

总差距  0.151 = 直接代价 0.037 (24%) + 被锁在低 lr 的代价 0.114 (76%)
```

**约 3/4 的伤害是间接的。** 这解释了为什么它是 Transformer 里少数没人敢动的组件 ——
拿掉它单看某一个 lr 的损失并不吓人(0.037,只有 7σ),吓人的是你从此够不到那段
真正出成绩的学习率。

### 3.2 `pre_norm_ablation`:位置只值 0.039,但方向明确

post-norm **1.3762**,比 pre-norm 差 0.039(7.3σ,真信号),**但完全没有不稳定**,
整条曲线稳稳下来。和 §3.1 合起来是一句干净的话:

> norm 的**存在**决定能不能训(拆掉 → 同 lr 直接 NaN);
> norm 的**位置**决定训得多好(挪到残差流上 → 稳定,但差一档)。

4 层这个深度上代价还只有 0.039。post-norm 真正的问题(深层需要 warmup 才不发散)
要更深的网络才显形,这个实验规模看不到。

### 3.3 `no_pos_emb`:NoPE 能训,但让模型自己挖位置是要付钱的

NoPE **1.3945**,比 RoPE 差 0.057(10.7σ),训练过程毫无异常。

这正好印证 handout 引的 Tsai 2019 / Kazemnejad 2023:decoder-only 模型靠因果 mask
本身就能推断位置,位置编码不是「有没有」的问题。但 0.057 说明,**让模型从 mask 里
自己挖位置,比直接把相对位置喂给它要贵** —— 而且这是三个能训起来的消融里最贵的一个。

### 3.4 `swiglu_ablation`:门控的效应小到测不出来

SiLU 无门控 **1.3404**,基线三种子是 **1.3323 / 1.3381 / 1.3428**。

```
基线 seed3   1.3323
基线 seed1337 1.3381
SiLU 无门控   1.3404   ← 夹在基线自己的种子波动里
基线 seed2   1.3428
```

**0.5σ。这不是「门控带来微小改善」,是「测不出来」。** 参数量已经对齐到 1.6% 以内
(SwiGLU 3·512·1344 = 2,064,384 vs SiLU 2·512·2048 = 2,097,152),所以差异不来自容量。

诚实的读法:Shazeer 报的增益本来就不大,而 22.7 M 参数 / 3.3 亿 token / TinyStories
这个尺度不足以让它显形。**要下「SwiGLU 不值得」的结论,需要多种子重复 + 更大规模**,
本实验只能说在这个尺度上看不到。

### 3.5 四个消融放在一起

```
拆光 RMSNorm   ████████████████████  +0.151  (同 lr 直接 NaN，须降 lr 一个数量级)
NoPE           ███████               +0.057  (10.7σ)
post-norm      █████                 +0.039  ( 7.3σ)
无门控 SiLU    ▏                     +0.003  ( 0.5σ) ← 噪声
```

顺序是:**norm 的存在 ≫ 位置编码 > norm 的位置 ≫ FFN 门控**。
值得注意的是这个顺序和「大家改哪个」正好相反 —— 门控(SwiGLU/GeGLU/ReGLU)是近年
被换来换去最多的部件,而它在这里的效应恰恰是唯一一个读不出来的。

---

## 4. `generate` — 生成文本

checkpoint:`seed3`(val 1.3323,全场最好),温度 0.8,top-p 0.9,提示词
"Once upon a time"。

```
Once upon a time, there was a little girl named Sue. She was very excited to go
to the park. She put on her hat and shoes and ran to the park with her mom.

At the park, Sue saw a boy with a big blue ball. The boy was trying to kick the
ball too. Sue wanted to play with the boy, so she asked him, "Can I play with
you?" The boy said, "Yes, let's play together!"

Sue and the boy played with the blue ball for a long time. They kicked the ball
back and forth, laughing and having fun. They became good friends and played at
the park every day. And they always remembered to share and be kind to each other.
<|endoftext|>
```

**152 token,遇 `<|endoftext|>` 自停** —— handout 允许「至少 256 token **或**到第一个
`<|endoftext|>` 为止」,这里是后者,而且模型自己学会了在故事讲完时收尾,这本身是个
好信号。

**流畅度**:完整的叙事结构(起因 → 冲突 → 解决 → 道德收尾),指代一致(Sue / the boy
全程不串),时态统一,对话引号配对。不低于 handout 给的参考样例。

**一个自洽性检查**:生成文本的压缩率 **4.072 字节/token**,训练语料是 **4.071**
(见笔记 §2.8)。模型复现出了和语料一致的 token 分布。

### 影响输出质量的因素

同一个 checkpoint、同一个提示词,只改解码参数:

**① 温度** —— 最敏感的一个。

| T | 输出 |
|:--|:--|
| 0.0(贪心) | "...She would give the frog a kiss. Lily gave the frog a kiss, and the frog turned into a prince!" —— 完全通顺,但是最安全最套路的故事,而且确定性,每次一样 |
| 0.8 | 上面那段。通顺 + 有变化,最佳区间 |
| 1.0 | 仍通顺,但开始出现语义滑坡:"The little ant and the big, dark cave became very dark"(主语粘连)、"The bee was attractive"(用词不当) |
| 1.5 | **崩坏**:"an elderly mule and special looking navy **tranquurde**"、"**oblboardingorance**"、"**tradmerAdamicking**" |

T=1.5 那些是**不存在的词**,不是用错的词。原因在 tokenizer 那一层:BPE 的词表是
子词片段,从分布尾巴上采样时模型会把几个片段拼成一个从未在语料里出现过的字符串。
字符级或词级模型不会这样坏 —— **这是子词切分特有的失效模式**。

**② top-p** —— 和温度是同一件事的两种做法(都是削尾巴),但 top-p 是自适应的。
T=1.0 + top_p=1.0(不截断)已经出语义滑坡;T=0.8 + top_p=0.9 就干净。模型有把握时
top-p 几乎不裁,没把握时裁得狠,所以它比固定温度更稳。

**③ 语料的难度(不是解码参数,但影响最大)** —— TinyStories 是 GPT-4 生成的儿童
故事,词汇、句式、主题都极窄。22.7 M 参数在这上面「流畅」是便宜的。同样的模型、
同样的算力换到 OpenWebText 上会差得多 —— 见 §5,那正是 handout 要问的问题。

**④ 上下文长度 256** —— 超过 256 token 后前文被裁掉,长故事的一致性维持不住。
上面那段 152 token 就结束了,还没撞到这个墙。

## 5. `main_experiment` — OpenWebText

架构和步数照抄 TinyStories,只有 `vocab_size` 跟着 tokenizer 变(10K → 32K)。
这一项就把模型从 22.70 M 撑到 **45.22 M**(多出来的 22.5 M 全在 embedding 和
lm_head),每 token 算力 111.7 → **179.3 MFLOPs**。

**显存挡了一道**:32K 词表让 logits `[256,256,32000]` 一份就 4.19 GB,batch 256
直接 OOM。但不能简单降 batch —— 降了就少看一半 token,两个数据集的 loss 更没法比。
所以用 **micro-batch 128 × 梯度累积 2 = 有效 batch 256**,token 预算和步数都对齐。

| lr | val loss | run |
|:--|:--|:--|
| 3e-3 | 3.9489 | `owt_lr3e-3` |
| **5e-3** | **3.9287** | `owt_lr5e-3` |

### 5.1 两个 loss 不能直接比

TinyStories 1.3323 vs OWT 3.9287,看着差 2.96 倍。**但这个比值是错的** ——
交叉熵的单位是「每 token 的 nat」,而两边的 token 根本不是一回事:词表不同
(10K vs 32K),压缩率也不同。可比的口径是 **bits per byte**:

$$\text{BPB} = \frac{\text{loss}}{\ln 2 \cdot (\text{字节/token})}$$

| 语料 | val loss | 困惑度 | 字节/token | **BPB** | 随机猜的 BPB |
|:--|:--|:--|:--|:--|:--|
| TinyStories | 1.3323 | 3.79 | 4.071 | **0.472** | 3.264 |
| OpenWebText | 3.9287 | 51.88 | 4.363 | **1.300** | 3.430 |

**换算后是 2.75 倍,不是 2.96 倍。** 差别不大,但方向说明了问题:OWT 的 token 更
「重」(4.363 字节/token),同样的 per-token loss 摊到每个字节上要除以更大的数。
跨 tokenizer 比模型,BPB 才是那个不受词表选择影响的量。

困惑度那一列更直观:TinyStories **3.79** —— 平均每步只在不到 4 个候选里挑;
OWT **51.88**,52 个候选。

### 5.2 生成的文本

`owt_lr5e-3`,温度 0.8,top-p 0.9:

```
The future of the U.S. is not yet known, but for the foreseeable future, the
U.S. will be forced to rely on the U.S. to adapt its policies to the next level
of strategic surveillance, including an unprecedented expansion of the U.S. It
is likely that any new policy will be done for such a long time. The U.S. has
been carefully deploying the U.S. in its efforts to protect its nuclear sites
and other potential nuclear weapons. [...] That is why the U.S. has already
ruled out the U.S. nuclear arsenal.
```

```
In a recent interview with AP, he said, "I'm just gonna take care of the guys
who make the show."

"They're gonna sit up and watch this show," he said. "I think that's the end of
the show. It's going to be a long time, and I think it's going to be a long
time."
[...] He is also a former director of the show's minicamp and that's why it's
been the best show in the world for so many years.
```

**流畅度:局部完美,整体空洞。** 语法、标点、引号配对、段落结构、新闻/访谈的文体
全对 —— 单看任何一个从句都像真的。但:

- **强迫性重复**:第一段 204 词里 "U.S." 出现 **19 次**,平均每 10.7 个词一次
- **自相矛盾**:"the U.S. will be forced to rely on the U.S.";"the U.S. has
  already ruled out the U.S. nuclear arsenal"
- **指代无着**:第二段的 "he" 从头到尾没有指称对象,凭空冒出一个 "Servelli"
- **没有论证推进**:两段都在原地打转,读完不知道说了什么

对照 TinyStories 那段的 type-token ratio:0.548 vs OWT 的 0.490 —— OWT 输出的
用词反而更单调,尽管它的词表大 3.2 倍。

### 5.3 为什么同样的模型、同样的算力,质量差这么多

**① 任务本身难 2.75 倍(BPB)。** TinyStories 是 GPT-4 按「3-4 岁儿童能懂」的约束
生成的,词汇、句式、主题都被刻意压窄;OWT 是真实网络爬取,主题、实体、文体、语言
全部开放。同样 45 M 的容量摊到后者上,每个方向都只够学个皮毛。

**② 学到的是表层统计,不是内容模型。** 上面那两段正是这个分界的样本:句法和文体
是**局部**规律,几百个 token 的上下文就能学会;而「不要重复」「指代要有着落」「论点
要推进」是**长程**规律,需要在 256 的上下文里维持状态,还需要世界知识来约束。模型
把便宜的那一半学到了,贵的那一半没有。

**③ 语料覆盖率差 5 倍。** 同样 3.28 亿 token 的预算:TinyStories 训练集 5.41 亿
token,跑了 **0.61 个 epoch**;OWT 训练集 27.3 亿,只跑了 **0.12 个 epoch**。更难的
数据,反而只看了五分之一的覆盖。

**④ 两边都远未训够,OWT 更甚。** 按 Chinchilla 的 20 token/参数:TinyStories 的
22.7 M 需要 4.54 亿 token(实际给了 3.28 亿,**14.4** token/参数);OWT 的 45.22 M
需要 9.04 亿(实际同样 3.28 亿,**7.2** token/参数)。**模型大了一倍,预算没变,
每参数分到的 token 直接减半。**

**⑤ 一多半算力花在了输出投影上。** V 从 10K 到 32K,`lm_head` 占前向 FLOPs 从
27.5% 涨到 **54.8%**。也就是说这次多花的 60% 算力里,绝大部分买的是「在 32000 个
候选里做 softmax」的能力,而不是更强的表示 —— 中间 4 层 Transformer 的容量一点
没变。这是「同样架构换个数据集」时最容易被忽略的一笔账。

## 6. `leaderboard`

*待填*
