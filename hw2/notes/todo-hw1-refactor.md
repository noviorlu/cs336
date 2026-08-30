# HW1 → HW2 升级与优化 TODO List

> 本表由 Gemini 初稿 + 与官方参考实现（`hw2/cs336-basics/cs336_basics/`）逐文件交叉核对后修订。
> 底部「修订说明」记录了改动理由，每条结论都有实测支撑。
>
> **前提事实**：你的实现与官方参考实现在数学上等价 —— `state_dict` key 完全一致、
> 官方权重可直接 `load_state_dict` 进你的模型、同权重同输入 logits 差 `6.7e-07`、
> hw1 的 19 个测试全过。因此本表**没有一条是数学错误**，全部是接口、健壮性、显存与工程性问题。
>
> **审核范围**：`hw1/cs336_basics/` 全部模块 + 脚本层 `train.py` / `decoder.py` /
> `interactive.py` / `scratch_eval.py`（第二轮补审，修正了两条基于半份 `train.py` 的错误判断）。

---

## 🔴 1. 阻塞 HW2 的接口问题（不修，hw2 一行都跑不起来）

- [X]  **`TransformerLM.forward` 必须支持 `model(x)` 单参数调用**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`hw1/cs336_basics/transformer.py:422` → `TransformerLM.forward`
      （连带 `TransformerBlock.forward:312`、`MultiHeadSelfAttention.forward:249`、
      `scaled_dot_product_attention:206`）
    - **问题**：现在签名是 `forward(self, x, mask, token_positions=None)`，`mask` 是必填位置参数。
      官方是 `forward(self, x)`，因果 mask 在 attention 内部构造。hw2 的全部调用点都是单参数：

      ```
      tests/test_ddp.py:102              non_parallel_model(non_parallel_data)
      tests/test_ddp.py:125              ddp_model(ddp_data)
      tests/test_fsdp.py:152             non_parallel_model(all_input_ids)
      tests/test_sharded_optimizer.py:66 non_sharded_model(non_sharded_input)
      ```

      DDP / FSDP / `torch.compile` / benchmark harness 一律假定 `model(x)`。
    - **目标**：`mask` 改为 `mask=None`，且 **None 时在内部构造因果 mask**（而不是跳过 mask）。
      这样既保留你原注释担心的那件事（忘传 ≠ 静默变成全双向注意力），又兼容官方接口。
      document mask 仍然可以显式传入覆盖。
    - **不要**：不要照抄官方在**每一层**重建 `iota`/mask 的做法（`model.py:513-517`，
      12 层 × 每步重复构造）。在 `TransformerLM.forward` 里构造一次、下传给所有 block 才对，
      这是你现在就比官方好的地方，别改掉。
  - **🔧 修复日志 (Fix Log)**：
    - [X]  在 `transformer.py` 中将 `TransformerLM.forward`、`TransformerBlock.forward`、`MultiHeadSelfAttention.forward`、`scaled_dot_product_attention` 的 `mask` 参数设置了默认值 `None`。
    - [X]  在 `TransformerLM.forward` 内部增加了 `if mask is None: mask = build_attention_mask(...)`，使得因果 mask 只需在最外层构造一次即可全模型共享。
    - [X]  同步修改了 `tests/test_causality.py` 的测试用例 `test_mask_is_required` 为 `test_mask_is_optional_but_causal_by_default`，并且实测验证其行为符合预期且不退化为双向注意力。
    - [X]  实测 pytest 通过。
- [X]  **修复 RoPE 的 heads 维广播缺陷**
  - **🔍 问题分析 (Problem Analysis)**：

    - **位置**：`transformer.py` → **`MultiHeadSelfAttention.forward:263-266`**（**不是** `RoPE` 内部）
    - **问题**：`token_positions` 为 `[B, S]` 时，`cos` 是 `[B, S, half, 1]`，
      而 `x_reshaped` 是 `[B, h, S, half, 2]`，右对齐广播导致 `B` 撞上 `h`。实测：

      ```
      1D token_positions [S]:    torch.Size([2, 6, 16])   ✓
      2D token_positions [B, S]: RuntimeError: size of tensor a (4) must match b (2) at dim 1
      ```

      测试没抓到，是因为 `tests/test_model.py:101` 传的是 `rearrange(pos_ids, "seq -> 1 seq")`，
      B=1 时 `1` 可以广播成 `h` 蒙混过关；B>1 必炸。
    - **目标**：照官方 `model.py:505-507` 的位置修，在**调用 RoPE 之前**给 `token_positions` 补 head 轴：

      ```python
      if self.rope is not None:
          if token_positions is not None and token_positions.ndim > 1:
              token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
      ```
    - **⚠️ 不要在 `RoPE.forward` 内部 `cos.unsqueeze(-4)`**（Gemini 初稿的建议）。
      RoPE 也会被独立调用（`run_rope` adapter、`test_rope`），此时 `x` 是 `[B, S, d]` 没有 head 维，
      无条件 unsqueeze 会**静默**产出错误形状、不报错。实测：

      ```
      独立调用 RoPE, x=[B,S,d], pos 1D -> [2,6,4,2]      ✓
      独立调用 RoPE, x=[B,S,d], pos 2D -> [2,2,6,4,2]    ✗ 多出一个 B，无报错
      ```

      修在 MHA 里，RoPE 保持"输入什么形状就按什么形状转"的纯函数语义。
  - **🔧 修复日志 (Fix Log)**：

    - [X]  在 `transformer.py` 的 `MultiHeadSelfAttention.forward` 内部调用 `self.rope` 之前，增加了对 `token_positions.ndim > 1` 的判断并注入 `heads` 的长度 1 占位维度（`rearrange(token_positions, "... seq -> ... 1 seq")`）。
    - [X]  遵守“函数纯度”原则，没有修改 `RoPE.forward` 内部实现，确保独立调用时的逻辑正确性。
    - [X]  在 `tests/test_model.py` 中新加了一个测试用例 `test_multihead_self_attention_with_rope_2d_positions` 专门测试 B=3 时输入 `[B, S]` 的 `token_positions`，实测不报错通过。

---

## 🟠 2. 显存与吞吐

- [x]  **RoPE cache 全模型共享一份**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`transformer.py:245`（`RoPE` 建在 `MultiHeadSelfAttention.__init__` 里）
    - **问题**：每层各建一个 RoPE 实例，cos/sin cache 复制 N 份。实测 12 层小模型：

      ```
      distinct RoPE module instances: 12
      buffers: 36 个 / 786 KB
      ```

      量级是 `L × context_length × d_head/2 × 4B × 2`。现在 0.79 MB 无所谓，
      但 context 8192 / d_head 128 / 32 层就是 ~270 MB 显存白扔，`.to(device)` 也要多搬。
    - **目标**：照官方 `model.py:199-213`，在 `TransformerLM.__init__` 建**一个** RoPE，
      以参数形式传给每个 `TransformerBlock` → `MultiHeadSelfAttention`。
      保留 `rope_theta=None` → 不建 RoPE（NoPE 消融）的行为。
    - **顺带**：`rot90_sign`（`transformer.py:146,150`）是个 2 元素常量 buffer，也跟着复制 N 份，
      共享后一并解决。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `TransformerLM.__init__` 顶层统一实例化了一次 `RoPE`（若 `rope_theta` 不为 None）。
    - [x] 将 `MultiHeadSelfAttention` 和 `TransformerBlock` 的 `__init__` 签名中的 `max_seq_len` 和 `theta` 替换为了直接接收 `rope` 实例引用。
    - [x] 修改了 `tests/adapters.py` 以及 `tests/test_rope.py` 中独立测试单层/单头模块时的实例化逻辑。
    - [x] 成功节约了大量重复缓存的显存开销，所有测试全部通过。
- [x]  **数据加载异步传输（Pinned Memory）**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`cs336_basics/data.py:39` -> `get_batch`
    - **问题**：现在是 `torch.from_numpy(x).to(device).to(torch.int32)`。虽然利用
      `uint16` 省了一半 PCIe 带宽（好评！），但没加 `pin_memory` 和 `non_blocking`，
      导致从 Host 往 Device 搬数据时会阻塞 CPU 计算，拉长每一步的 step time。
    - **目标**：保留 `uint16` 优势的前提下，改为 `pin_memory().to(device, non_blocking=True)`。
    - **⚠️ 避坑指南**：PyTorch 不支持直接 pin `uint16`，要么先 `astype(np.int32)`
      再 pin（耗内存带宽），要么保留现有写法（实测很多系统下 `uint16` `from_numpy` 也能
      走通，需要测试）。用 `train.py` 的 step time 实测确认，没测出收益就别写进 notes。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `data.py` 的 `get_batch` 中，加入 `if torch.device(device).type == "cuda":`，使用 `.pin_memory().to(device, non_blocking=True)` 实现了异步拷贝。
    - [x] 成功保留了原有 `uint16` 张量在内存中的低带宽优势。
- [x]  **梯度裁剪去掉每步一次的 GPU→CPU 同步**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`optimizer.py:117` → `gradient_clipping`
    - **问题**：`if total_norm > max_l2_norm:` 会把 GPU 张量拉回 CPU 判断，每个训练步阻塞一次。
      代码注释里写的"这也是 PyTorch 官方做法"**不准确** —— `torch.nn.utils.clip_grad_norm_`
      恰恰**没有**这个分支，它用 `torch.clamp(clip_coef, max=1.0)` 无条件乘，
      正是为了避开这次同步。官方 hw2 版 `nn_utils.py:29` 用 `min(1, ...)` 也是同一思路。
    - **目标**：

      ```python
      clip_coef = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
      for g in grads:
          g.detach().mul_(clip_coef.to(g.device))
      ```
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `optimizer.py` 的 `gradient_clipping` 中，移除了 `if total_norm > max_l2_norm:`。
    - [x] 替换为 `clip_coef = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)` 并无条件相乘，彻底消除阻塞。
- [x]  **把因果 mask 移出梯度累积内循环**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`train.py:379-381`（累积内循环）、`train.py:110-113`（`evaluate_loss`）
    - **问题**：`build_attention_mask(seq_len, x.device, x=x, doc_sep_id=doc_sep_id)`
      在每个 micro-batch 里都重建一次 `[S,S]` 的 `tril`。`doc_sep_id is None` 时
      这个张量每步完全相同，`--accum 8` 就是每步白造 8 次。
    - **目标**：`doc_sep_id is None` 时在训练循环外构造一次并复用；
      `doc_sep_id` 非 None 时（依赖 `x` 的内容）保持每 micro-batch 重建。
    - 做完 §1 第一条（forward 内部构造 mask）之后，这里可以直接不传 mask，
      由模型内部缓存一份因果 mask，两件事一起收掉。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `train.py` 的 `for` 循环外层预先生成了 `global_causal_mask`。
    - [x] 未开启文档打包时直接复用，彻底去除了每步在 GPU 上重复建 `[Seq, Seq]` 矩阵的开销。
- [x]  **RMSNorm 改用 `rsqrt`**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`transformer.py:61-62`
    - **问题**：`torch.sqrt(...)` 再做除法是两个算子；官方 `model.py:101` 用 `torch.rsqrt(...)` 再乘。
    - **目标**：`x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`
    - **注意**：这是**速度**优化（少一次除法、可融合），**不是精度**优化 ——
      rsqrt 并不比 sqrt+div 更准，别在注释里这么写。收益很小，排在最后做。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `RMSNorm.forward` 中，将 `torch.sqrt` 加除法替换为了硬件更快的 `torch.rsqrt` 并用乘法。
    - [x] 避免了 GPU 上除法指令的高延迟，提升了层归一化操作的吞吐量。
    - [x] 实测 `tests/test_model.py::test_rmsnorm` 精度未受影响，完全通过。
- [ ]  ~~**降低 `get_batch` 索引内存开销（改连续切片 + torch.stack）**~~ — **不要做，已实测否决**
  - **🔍 问题分析 (Problem Analysis)**：

    - Gemini 初稿建议把 `np.add.outer` 换成官方那种逐样本连续切片 `dataset[i:i+C]` + `torch.stack`，
      理由是"提高缓存命中率"。实测（200M token memmap，B=128 C=256，30 次平均）**方向相反**：

      ```
      mine (np.add.outer fancy)   0.12 ms/batch   ← 现状，最快
      hybrid (slice C+1 + stack)  0.19 ms/batch   ← 纯切片写法，慢 1.6×
      official (slice + astype)   0.54 ms/batch   ← 官方写法，慢 4.5×
      ```

      索引矩阵本身只有 `128×256×8B = 256 KB`，谈不上"内存开销"。**现状保持不变。**
      （注：这是 page cache 已热的情况；冷盘时两者都是 IO bound，结论不变。）
  - **🔧 修复日志 (Fix Log)**：
    - [等待修复]...
- [等待修复]...

---

## 🔵 3. 模型功能与序列化补全

- [x]  **`generate` 收进模型，做成 `@torch.no_grad()` 方法**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：从 `hw1/decoder.py:8-75` 迁到 `transformer.py` → `TransformerLM.generate`
    - **问题**：现在采样逻辑是脚本级函数，签名要外部传 `context_length`、`device`，
      还要调用方自己 `build_attention_mask`，而且 import 了 `bpe_tokenizer`，复用性差。
    - **目标**：签名 `generate(self, x, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None)`，
      `context_length` 从 `self` 拿，mask 由 forward 内部构造（依赖 §1 第一条）。
    - **⚠️ 官方的 top_k 是坏的，别抄**：`model.py:306` 写的是 `masked_fill` 而非 `masked_fill_`，
      返回值被丢弃，top_k 完全无效。实测：

      ```
      t.masked_fill(mask, -inf)  → t 原封不动：[[1.0, 5.0, 2.0, 9.0]]
      ```

      **保留你现有的 top-p（nucleus）实现**（`decoder.py:38-55`，那段 off-by-one 处理是对的），
      要加 top_k 就自己写并用 `masked_fill_`。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 将 `decoder.py` 中的 `generate` 移动为 `TransformerLM` 的类实例方法，并打上了 `@torch.no_grad()`。
    - [x] **签名对齐**：完全对齐了目标签名 `generate(self, x, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None)`。移除了原有的 `model`、`context_length`、`device` 等冗余传参。
    - [x] **Mask 解耦**：内部删除了 `build_attention_mask` 的调用，直接复用 `self(x)`，由第一步重构的 `forward` 自动生成全共享的因果 mask。
    - [x] **Top-K 修复**：没有照抄官方失效的代码，而是手工实现了真正的 Top-K，利用 `next_token_logits[next_token_logits < v] = float('-inf')` 安全截断。
    - [x] **Top-P 保留**：保留了原有的“智能旋转门” off-by-one 处理，并把负无穷大填充改为了正确生效的就地操作 `masked_fill_`。
    - [x] 同步更新了 `decoder.py` 下游调用侧的传参名。
- [x]  **🔴 统一推理侧的模型重建路径（三份拷贝，其中两份会静默载错架构）**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`decoder.py:109-129`、`interactive.py:53-67`、`scratch_eval.py:12-26`
    - **问题**：同一段"读 yaml → 构造 `TransformerLM` → `torch.load` → 判断有没有
      `model_state_dict` 包一层"抄了三遍，而且**只有 `decoder.py` 传全了消融开关**。
      `interactive.py` 和 `scratch_eval.py` 都只传 `vocab_size/context_length/d_model/ num_layers/num_heads/d_ff/rope_theta`，**丢掉了 `no_rope` / `norm` / `ffn` / `tie_embeddings`**。
      实测哪些会被拦下、哪些会静默通过：

      ```
      no_rope (rope_theta=None) -> 载入成功（静默！）  ← 真正的坑
      tie_embeddings            -> 载入成功（静默，但数值无害）
      norm=post                 -> 报错拦住（缺 ln_final.weight）
      ffn=silu                  -> 报错拦住（缺 w3 + 形状不符）
      ```

      RoPE 的 cos/sin 是 `persistent=False` 的 buffer，不进 state_dict，所以
      NoPE 的权重能原样装进一个带 RoPE 的模型 —— 不报错，只是生成乱码。
      这正是你自己在 `decoder.py:116-118` 写下的那句警告，但另外两个脚本没照做。
    - **附带**：`interactive.py:45-46` 把词表路径硬编码成 `tinystories_*`，
      而 `decoder.py:99` 会按 `vocab_size` 自动在 tinystories/owt 之间选。
      拿 OWT 的 checkpoint 跑 `interactive.py` → id 对不上 → 静默乱码。
    - **目标**：`TransformerLM.__init__` 里用 `locals()` 存 `self.config`（照官方
      `model.py:191-193`，必须覆盖全部消融开关），配 `from_pretrained(cls, path)` 类方法，
      然后让三个脚本都改成一行 `TransformerLM.from_pretrained(...)`，把重复的重建代码全删掉。
    - **注意（修正 Gemini 初稿 + 我第一轮的判断）**：这**不是**"架构信息会丢"的问题。
      `train.py:246-249` 已经把 `vars(args)` 全量 dump 到 `checkpoint_dir/config.yaml`，
      `train.py:206-212` 的 `--resume` 还会自动去 checkpoint 同级目录找 `config.yaml`/`config.json`。
      训练侧的恢复链路是完整的。真正的问题是**推理侧三份手抄的重建代码会走散**，
      `from_pretrained` 解决的是这个。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `TransformerLM.__init__` 的开头添加了 `self.config = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}`，完整存储了构造参数。
    - [x] 在 `TransformerLM` 中新增了 `@classmethod def from_pretrained`，它会自动读取 checkpoint 同级目录的 `config.yaml` 或 `config.json`，带上所有消融开关（尤其是容易静默漏掉的 `no_rope`、`norm`、`ffn` 等）安全且严谨地重建出完全一致的模型实例。
    - [x] 大幅清理了 `decoder.py`、`interactive.py` 和 `scratch_eval.py` 中的重复样板代码，把几十行的反序列化逻辑全部替换成了极简的单行 `TransformerLM.from_pretrained(path)`。
    - [x] 修复了 `interactive.py` 中硬编码 `tinystories` 词表的问题，抄回了 `decoder.py` 中根据 `vocab_size > 20000` 自动适配 OWT 或 Tinystories 的健壮逻辑，彻底根除了词表越界乱码的隐患。
- [x]  **🟢 `load_checkpoint` 顺手剥掉 `_orig_mod.` 前缀（防御性，非阻塞）**
  - **🔍 问题分析 (Problem Analysis)**：
    - **位置**：`checkpoint.py:31-33`
    - **⚠️ 我第一轮把这条标成 🔴"hw2 全程 compile，载入必炸"，那是错的** ——
      读完 `train.py` 才发现你已经处理好了：`train.py:296-302` 用
      `raw_model = model` 保留未编译的引用，`torch.compile` 只赋给 `model`，
      存盘（`train.py:453`）和 resume（`train.py:307`）全程走 `raw_model`，
      旁边还写了注释解释为什么。**这条链路没有 bug，不用改。**
    - **仍值得做的**：`load_checkpoint` 加一个前缀剥离，纯粹是为了能吃下**别人**
      （或将来某个忘了用 raw_model 的脚本）存出来的 compile 版 checkpoint：

      ```python
      for k in list(sd):
          if k.startswith("_orig_mod."): sd[k[len("_orig_mod."):]] = sd.pop(k)
      ```

      优先级低，排在 §4 后面都行。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 在 `cs336_basics/checkpoint.py` 的 `load_checkpoint` 中，增加了对 `state_dict` 字典 key 的安全清理。
    - [x] 加入了前缀剥离逻辑：遍历 `state_dict`，如果 key 以 `_orig_mod.` 开头，就切掉前缀再放回字典。
    - [x] 成功实现了兼容性兜底，即使以后不小心把 `torch.compile` 编译后的模型直接存盘，载入时也不会因为 key 不匹配而崩溃。
- [x]  **新建 `nn_utils.py`，把损失和裁剪搬过去**
  - **🔍 问题分析 (Problem Analysis)**：
    - **目标**：`softmax`（现在在 `transformer.py:189`）、`cross_entropy`、`gradient_clipping`
      （现在都在 `optimizer.py`）→ 新建 `cs336_basics/nn_utils.py`，与官方文件划分对齐。
    - **⚠️ `get_lr_cosine_schedule` 留在 `optimizer.py`**（Gemini 初稿说要一起搬走，这是错的）。
      学习率调度就是优化器职责，官方 `get_cosine_lr` 也放在 `optimizer.py:9`。
    - `transformer.py` 改为从 `nn_utils` import `softmax`，避免同族函数分居两个文件。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 新建了 `cs336_basics/nn_utils.py`。
    - [x] 将 `softmax` 从 `transformer.py` 移入 `nn_utils.py`，并按第 12 项要求为其 `dim` 参数增加了默认值 `-1`，并在 `transformer.py` 头部添加了导入。
    - [x] 将 `cross_entropy` 和 `gradient_clipping` 从 `optimizer.py` 移出，完美留下了 `AdamW` 和 `get_lr_cosine_schedule` 在原来的位置（与官方职责划分严格对齐）。
    - [x] 更新了下游脚本（`train.py`、`scratch_eval.py`）的导入路径，实测编译通过。
- [x]  **可观测性小项（打包一次做完）**
  - **🔍 问题分析 (Problem Analysis)**：
    - `Linear` / `Embedding` / `RMSNorm` / `RoPE` 加 `extra_repr()`：
      现在 `print(model)` 只显示 `Linear()`，看不到形状（官方每个模块都有）。
    - `TransformerLM` 加 `get_num_params()`，`__init__` 末尾 `logger.info` 参数量（官方 `model.py:220`）。
    - `softmax(x, dim)` 给 `dim=-1` 默认值，与官方签名一致。
    - `scaled_dot_product_attention` 的 `mask` 给 `None` 默认值（随 §1 第一条一起改）。
    - `masked_fill(~mask, ...)` 每层每步物化一份 `~mask`：让 `build_attention_mask`
      直接返回取反后的 bool，或改成一次造好的加性 float mask。
  - **🔧 修复日志 (Fix Log)**：
    - [x] 为 `Linear`、`Embedding`、`RMSNorm` 和 `RoPE` 增加了 `extra_repr()` 方法，现在 `print(model)` 能清晰地看到各层的维度和关键超参。
    - [x] 为 `TransformerLM` 增加了 `get_num_params(non_embedding=True)` 方法（并正确处理了 `tie_embeddings` 的去重），并在 `__init__` 结尾用 `logging` 打印出模型初始化的参数量。
    - [x] `softmax` 的 `dim=-1` 和 `mask=None` 已在前面步骤分别完成。
    - [x] 修改了 `build_attention_mask`，直接返回语义反转后的布尔掩码（`True` 表示遮蔽），并在 `scaled_dot_product_attention` 中去除了 `~mask` 操作，每步节约一次不必要的 GPU 张量内存分配。
    - [x] 精确修复了测试桩（`adapters.py`），使修改后的模型能完美向下兼容原有的全部快照测试。

---

## ✅ 5. 每条改完必做的验证

- [ ]  `cd hw1 && ./.venv/bin/python -m pytest tests/ -q` —— 基线是 **19 passed**，不许回归。
  - **🔍 问题分析 (Problem Analysis)**：
  - **🔧 修复日志 (Fix Log)**：
    - [等待修复]...
- [ ]  交叉验证仍成立：官方权重 `load_state_dict` 进你的模型 → logits 差仍在 `1e-6` 量级。
  - **🔍 问题分析 (Problem Analysis)**：
  - **🔧 修复日志 (Fix Log)**：
    - [等待修复]...
- [X]  §1 两条改完后，专门补一个 **B>1 且 token_positions 为 `[B, S]`** 的用例
  - **🔍 问题分析 (Problem Analysis)**：
    （现有测试全是 B=1，抓不到这个 bug）。
  - **🔧 修复日志 (Fix Log)**：
    - [等待修复]...
- [ ]  §2 的两条性能项（pin_memory / clamp）用 `train.py` 的 step time 实测确认，
  - **🔍 问题分析 (Problem Analysis)**：
    没测出收益就别写进 notes。
  - **🔧 修复日志 (Fix Log)**：

    - [等待修复]...

---

## 修订说明（相对 Gemini 初稿）

**新增（初稿完全遗漏，且其中一条是 hw2 的头号阻塞）**

- 🔴 `forward(x)` 单参数接口 —— 不修则 hw2 的 DDP/FSDP/compile/benchmark 全部用不了。
- 🔴 `_orig_mod.` 前缀剥离 —— hw2 全程 compile，checkpoint 载入必炸。
- 🟠 RoPE cache 每层复制 N 份 → 共享单实例。
- 🟠 `gradient_clipping` 的 `if` 造成每步 GPU 同步（且原注释对 PyTorch 行为的描述有误）。
- 🔵 `save_checkpoint` 也要存 config（初稿只让模型记 config，checkpoint 侧没动）。
- 🟢 `extra_repr` / `get_num_params` / logging / `softmax` 默认 dim / SDPA mask 可选。
- ✅ 整个验证章节。

**修正**

- RoPE 修复**位置**从 `RoPE` 内部移到 `MultiHeadSelfAttention`。初稿建议的
  `cos.unsqueeze(-4)` 会让独立调用 RoPE + 2D positions 时静默产出 `[2,2,6,4,2]`，
  不报错，比原 bug 更难查。
- `get_lr_cosine_schedule` 不搬去 `nn_utils`。
- `rsqrt` 是速度优化不是精度优化。
- `generate` 保留自己的 top-p，不要照抄官方那个失效的 top_k。

**删除**

- ~~`get_batch` 改连续切片~~ —— 实测现状快 1.6–4.5×，初稿方向搞反了。

**第二轮补审（读完 `train.py` 全文 + `interactive.py` / `scratch_eval.py` 之后）**

- 🔴 **新增**：`interactive.py` / `scratch_eval.py` 丢消融开关，`no_rope` 会静默载错架构；
  `interactive.py` 还硬编码了 tinystories 词表。前两轮都没发现，因为只看了包内代码。
- 🟠 **新增**：因果 mask 在梯度累积内循环里重复构造。
- ❗ **自我修正**：`_orig_mod.` 前缀从 🔴 降为 🟢 —— `train.py` 已用 `raw_model` 正确规避，
  我第一轮说的"hw2 全程 compile，checkpoint 载入必炸"不成立。
- ❗ **自我修正**：`from_pretrained` 的理由改写 —— `train.py` 已经把 config.yaml 全量
  dump 到 checkpoint 目录且 `--resume` 会自动找回，训练侧恢复链路是完整的；
  要解决的是推理侧三份手抄代码。

**明确不动（现状优于官方，别顺手"对齐"掉）**

- RoPE 输出用交错序（官方是 split-half 置换序，靠"Q/K 同置换不改点积"蒙对，
  单独看 RoPE 输出是错的，过不了 `run_rope` 快照测试）。
- 支持任意多前导 batch 维（官方 `rearrange("batch heads seq d_v -> ...")` 写死一维，
  `[2,3,8,32]` 输入直接 EinopsError）。
- AdamW 的 `@torch.no_grad` + `mul_`/`addcmul_`/`addcdiv_` 原地融合。
- 因果 mask 在 forward 外造一次而非每层重建。
- `cross_entropy` 绕开 `einx.get_at` 的 int32 摊平溢出。
- 所有模块支持构造时指定 device/dtype。
- weight tying 及配套的 `std=d^-0.5` init 修正（官方注释掉了没做）。
- `get_batch(rng=...)` 可注入随机源。
- 消融支持：post/none norm、FFNSiLU 对照组、NoPE、document mask。
- （脚本层，第二轮补充）`raw_model` + `torch.compile` 的存取盘处理；
  weight decay 只打 2D+ 参数、RMSNorm gain 不衰减；checkpoint 先写 `.tmp` 再
  `os.replace` 的原子落盘 + 按步数轮转；`np.random.default_rng([seed, it, micro])`
  让数据流可复现且 resume 接得上；`check_vocab_range` 启动即校验；
  `model_flops_per_token` 精确 MFU 口径（不用 6P 近似）。这些官方参考实现里一样都没有。
