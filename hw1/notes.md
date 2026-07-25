# CS336 Assignment 1 — 学习笔记

> 记录做作业时遇到的问题、概念理解、坑与解决方法。
> 公式用 LaTeX：行内 `$...$`，独立 `$$...$$`（VSCode 装 Markdown+Math 预览或 GitHub 均可渲染）。

---

## 0. 整体地图

作业目标：从零搭一个完整的 Transformer LM 训练流程。四大块：

| Handout 章节 | 实现内容 | adapters 函数 | 测试文件 |
|---|---|---|---|
| §2 BPE Tokenizer | BPE 训练 + Tokenizer(encode/decode) | `run_train_bpe`, `get_tokenizer` | `test_train_bpe.py`, `test_tokenizer.py` |
| §3 Transformer LM | Linear/Embedding/RMSNorm/SwiGLU/Attention/RoPE/Block/LM | `run_linear` … `run_transformer_lm` | `test_model.py`, `test_nn_utils.py` |
| §4 Loss & Optimizer | cross-entropy / AdamW / grad clip / cosine LR | `run_cross_entropy` … `run_get_lr_cosine_schedule` | `test_nn_utils.py`, `test_optimizer.py` |
| §5 Training loop | get_batch / checkpoint 存取 | `run_get_batch`, `run_save/load_checkpoint` | `test_data.py`, `test_serialization.py` |

**工作流程**：在 `cs336_basics/` 里从零写实现 → 在 `tests/adapters.py` 里把 `run_xxx` 接到自己的实现（只做胶水，无实质逻辑）→ `uv run pytest` 验证。

**跑测试命令**：`uv run pytest`（不是 `uv run tests` —— `tests` 是目录不是命令）。

---

## 1. Unicode / 字符 / 字节 概念梳理

### 1.1 str vs bytes 是两个层面

```
"café"          ← str：字符（Unicode 码点）的序列
   ↕  encode / decode
b'caf\xc3\xa9'  ← bytes：0–255 的字节序列
```

- **encode**：`str → bytes`，按编码规则（UTF-8）把字符翻译成字节。
- **decode**：`bytes → str`，反向。二者互逆。
- 计算机底层只存/传字节（0–255），但字符有 15 万+ 个，所以需要编码规则做映射。

### 1.2 `len()` 数的是什么，取决于对象类型

```python
len("café")               # 4  —— str 的元素是「字符」
len("café".encode("utf-8"))  # 5  —— bytes 的元素是「字节」
```

`len` 只是忠实地数当前对象有几个元素，没有偏爱谁。

### 1.3 UTF-8：几个字节由「码点范围」决定

| 码点范围 | 字节数 | 例子 |
|---|---|---|
| U+0000–U+007F | 1 | ASCII (`c`, `a`, `f`) |
| U+0080–U+07FF | 2 | `é`, `ñ`, 希腊/西里尔 |
| U+0800–U+FFFF | 3 | **绝大多数常用汉字**、假名、韩文 |
| U+10000–U+10FFFF | 4 | emoji、扩展区生僻字（如 `𠮷`） |

- 「café」= 4 字符 = 5 字节，因为 `c/a/f` 各 1 字节、`é` 占 2 字节 → `[99, 97, 102, 195, 169]`。
- **中文不都是 3 字节**：常用字在 CJK 基本区（3 字节），生僻字/扩展区在 U+20000 以上是 4 字节。
- 判断口诀：不要记"中文=3字节"，要记「**字节数由 `hex(ord(c))` 落在哪个范围决定**」。

### 1.4 byte-level BPE 的好处

所有文本先 `encode("utf-8")` 变成 0–255 字节，BPE 只在**字节层**操作：
- 初始词表大小固定 = **256**（所有可能字节值）。
- 永远不会 out-of-vocabulary（任何字符拆到字节都逃不出 0–255）。
- "某字符占几字节"的复杂度被 UTF-8 这一步完全吸收，训练算法对此透明。

### 1.5 repr vs str/print

| | 面向 | 目标 | 例子 `chr(0)` |
|---|---|---|---|
| `repr()` / 交互式回显 | 调试 | 精确、无歧义 | `'\x00'`（可见转义） |
| `str()` / `print()` | 用户 | 干净好看 | 空白（终端渲染不出控制字符） |

- 容器（list/dict/tuple）的 `str` 会对**每个元素调用 `repr`** → 所以 `print(["a\nb"])` 显示 `['a\nb']`（带引号、`\n` 不换行）。
- **坑**：`print(b'caf\xc3\xa9')` 仍显示 `b'...'`，不会解码成 `café`。bytes 没有"给人看"的 str 形式，要看字得显式 `.decode("utf-8")`。
- **坑**：decode 不能逐字节做——多字节字符（如 `é` = `\xc3\xa9`）的字节必须凑齐才能解码。

### 1.6 UTF-8 / UTF-16 / UTF-32 三者对比

都是"把 Unicode 码点变成字节"的编码方案，区别在**编码单位宽度**：

| 编码 | 基本单位 | 一个字符占多少字节 |
|---|---|---|
| UTF-8 | 1 字节 | **变长** 1~4 |
| UTF-16 | 2 字节 | **变长** 2 或 4（最少 2） |
| UTF-32 | 4 字节 | **定长** 永远 4 |

- UTF-32：每字符固定 4 字节，整齐但极浪费（`a` 也占 4 字节，塞一堆 `\x00`）。
- UTF-16：至少 2 字节，基本区 2 字节 / 扩展区 4 字节。
- UTF-8：ASCII 只 1 字节且兼容 ASCII，其余按需 2~4 字节，最省。
- 同一字符 `a`：utf-8 → `[97]`；utf-16 → `[97,0]`（+BOM）；utf-32 → `[97,0,0,0]`（+BOM）。
- 实测 "hello"：utf-8 `[104,101,108,108,111]`（5 字节，无 0）；utf-16 每字符后跟 1 个 0（12 字节）；utf-32 跟 3 个 0（24 字节）。→ 这是 unicode2(a) 的核心证据。

### 1.7 为什么历史上会出现 UTF-16/32（含 ASCII vs 字节容量）

先厘清 **ASCII vs 字节容量**（常见混淆）：
- **ASCII 只有 128 个字符（0–127），7 位就够**（$2^7=128$），因为当年只需覆盖英文。含控制字符 0–31、可打印 32–126、DEL 127。
- **256 是「一个字节」的容量**（8 位 = $2^8$，值 0–255），是**存储单位**容量，不是 ASCII 的。ASCII 只用掉字节前一半，最高位恒为 0。
- 128–255 历史上被各国拿去塞自己的字符（Latin-1、Windows-1252、GBK 低区…），互不兼容 → 催生 Unicode 统一。

演化时间线（历史包袱 + 定长思路的产物）：
1. 起点只有 ASCII（7 位，只能表示英文）。
2. Unicode 诞生时**天真以为 2 字节(65536)够装所有字符** → 催生 **UTF-16** 雏形（定长 2 字节，`第 n 字符 = 第 2n 字节`，好算）。
3. 后来字符远超 6 万（现 15 万+，上限 U+10FFFF）→ UTF-16 被迫用"代理对"补丁变成变长；**UTF-32** 出现用 4 字节回归真·定长（代价：巨浪费）。
4. **UTF-8**（Ken Thompson, 1992）换思路：兼容 ASCII + 省空间 + 避开 `\x00` 坑 → 后来居上成事实标准（>98% 网页）。

| 编码 | 动机 | 现定位 |
|---|---|---|
| UTF-16 | 早期以为 2 字节够，图定长 | 遗留：Windows/Java/JS 内部字符串 |
| UTF-32 | 想要真定长、装下所有码点 | 极少用于存储，太浪费 |
| UTF-8 | 兼容 ASCII + 省 + 避 `\x00` | 事实标准 |

**关联作业**：① UTF-8 单字节模式 `0xxxxxxx` 正好吻合 ASCII 的 0–127 → 所以兼容。② BPE 初始词表是 **256**（所有字节值 0–255），不是 128——因为处理的是**字节**层面，UTF-8 流里字节会取到 128–255（多字节字符的组成部分）。

### 1.8 UTF-8 如何"变长"：前缀自解析机制

机制：**每个字节的高位前缀自带角色标记**（固定写死在标准里，`x` 是承载码点的有效位）：

```
1 字节  U+0000–007F     0xxxxxxx
2 字节  U+0080–07FF     110xxxxx 10xxxxxx
3 字节  U+0800–FFFF     1110xxxx 10xxxxxx 10xxxxxx
4 字节  U+10000–10FFFF  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

- `0...` = 单字节(ASCII)；`110...`=2字节头；`1110...`=3字节头；`11110...`=4字节头；`10...`=**续接字节**。
- 解码器从左往右扫，看首字节前缀就知道"后面还要再读几个 `10...` 字节" → **自解析、无歧义**。
- 实例 `é`(U+00E9=233)：填 2 字节模板 → `11000011 10101001` = `\xc3\xa9` = `[195,169]`。验证：`195=0b11000011`(110开头=2字节头)，`169=0b10101001`(10开头=续接)。
- **关联 unicode2(b)(c)**：`\xc3` 单独 decode 会报错（声明后面还有续接字节，但没了）；构造非法序列的思路：`10...` 打头但前面无合法开头，或 `110...` 后面跟的不是 `10...`。

### 1.9 延伸：LLM 中文说着变韩/日文 ≠ 字节接近

（这条是对"字节 vs 表示层"的重要澄清，不是字节层现象）
- LLM 不在**字节层**预测，而是在 **token ID**（离散整数）上预测。中文 token 和日文 token 是词表里两个不同 ID，ID 相邻也无语义/形近关系 → "字节接近导致混淆"不成立。
- 中日韩虽在 Unicode 里码点区间相邻（CJK 区），但每个字的具体字节组合完全不同，并不"接近"。
- **真正原因**：① 训练数据 CJK 语料混杂 → **embedding 空间**里三者互为邻居；② 中文高质量数据稀疏，采样易漂移；③ 采样随机性 + 上下文漂移；④ tokenizer 不匹配（中文被切碎，token 效率低）—— 正是 `tokenizer_experiments` 那题的方向。
- **直觉纠正**：不是**字节**接近，而是**embedding 空间**里 CJK 向量互为邻居。字节只是最底层存储，模型不在那层"思考"。

---

## 2. 书面题（计分，自己作答）

> 下面只放题目和我的作答，答案自己写。相关观察素材见 §1。

### Problem `unicode1` — Understanding Unicode (1 pt)

**(a)** `chr(0)` 返回什么 Unicode 字符？
- `\x00` 是十六进制表示的**空字符（Null character，缩写 NUL）**，ASCII 码值为 0。它渲染不出任何字形（不占显示空间），但仍占一个字符位置；C 语言等用它作字符串结束标志。

**(b)** 它的 `__repr__()` 表示 和 打印出来 有何不同？
- `repr` 显示成 `'\x00'`（可见转义），`print` 显示为无可见字符。
- 原因：repr 面向调试，要精确无歧义，故用可见转义把控制字符"显示出来"；print 输出字符本身，终端无法渲染控制字符。

**(c)** 它出现在文本中会怎样？
- 字符**存在但渲染不出来**。它没有截断字符串（不像 C 里的 `\0`），`len` 仍把它计入一个位置——只是终端显示为空白。

### Problem `unicode2` — Unicode Encodings (3 pts)

**(a)** 为什么在 UTF-8 字节上训练 tokenizer 优于 UTF-16/UTF-32？
- UTF-8 对常见字符只用 1 字节，字节序列比 UTF-16/32 **更短**；而且 UTF-16/32 会插入大量填充的 **`\x00`**，这些 `\x00` 会成为**高频字节对**，让 BPE 把有限的合并名额浪费在无意义的填充上，学不到真正有用的子词。
- （证据：`list("hello".encode(...))` → utf-8 `[104,101,108,108,111]`；utf-16 每字符后跟 1 个 0；utf-32 跟 3 个 0，见 §1.6）
- ⚠️ 措辞坑：不要说 "UTF-8 是压缩/最优压缩"（它是编码规则不是压缩算法）；不要说 BPE 是"聚类"（BPE 是频率驱动的**贪心合并**，不是 clustering）。

**(b)** 为什么 `decode_utf8_bytes_to_str_wrong`（逐字节 decode）是错的？给一个出错的输入例子。
- 因为非 ASCII 字符（如 `é`）在 UTF-8 里由**多个字节共同编码**（`é` = `\xc3\xa9`），逐字节单独 decode 会把这些字节切断。
- 例子：`"café".encode("utf-8")` = `[99,97,102,195,169]`。前三个 ASCII 字节能各自解码，到 `\xc3`(195) 时——它是 2 字节序列的**首字节**（`0b11000011`，`110` 开头声明后面还有 1 个续接字节），单独解码时续接字节 `\xa9` 缺失 → 报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 ... unexpected end of data`。

**(c)** 给一个不能 decode 成任何 Unicode 字符的双字节序列。
- 核心：违反 UTF-8 前缀规则（合法双字节须为 `110xxxxx 10xxxxxx`）的序列都无法 decode。三种非法方式：

| 例子 | 非法原因 | 报错关键词 |
|---|---|---|
| `bytes([0xff, 0xff])` | `0xff=11111111`，首字节前缀 `11111` 不属于任何合法类（只有 `0`/`110`/`1110`/`11110`） | invalid start byte |
| `bytes([0x80, 0x80])` | `0x80=10000000`，`10` 开头是**续接字节**标记，不能当开头 | invalid start byte |
| `bytes([0xc3, 0x28])` | 首字节 `0xc3=11000011`(`110`合法头,声明后面须跟续接字节)，但 `0x28=00101000` 是 ASCII(`0`开头)，不是 `10` 续接字节 | invalid continuation byte |

- 推荐答 `[0xc3, 0x28]` 或 `[0x80, 0x80]`：能体现理解了"续接字节"概念。

---

## 3. BPE Tokenizer 训练（§2.4–2.5）

> 三步：**① Vocab init → ② Pre-tokenization → ③ BPE Merge loop**。
> 产出两样（`run_train_bpe` 返回值）：**vocab** `dict[int,bytes]`（ID→字节串）、**merges** `list[tuple[bytes,bytes]]`（按创建顺序）。
> 一句话：**从 256 字节出发，反复把"最常相邻的 token 对"合并成新 token，直到词表达到 `vocab_size`。**
>
> 全局数据流：
> ```
> ① 256 字节 + 特殊 token → 初始 vocab
> ② 语料 → split特殊token → 正则切 → 转字节 → {pre-token字节序列: 次数}
> ③ 循环{数对(加权) → 选最高频(平局取字典序大) → 加vocab+记merges → 替换} → 最终 vocab+merges
> ```

---

## 3.1 Vocab init（词表初始化）

### 3.1.1 初始化 & 词表大小

- 初始词表 = 全部 256 字节值（ID 0–255：`b'\x00'`…`b'\xff'`）+ 特殊 token。
- 之后每合并 1 次词表 +1，直到 `vocab_size`。

**词表大小公式**：
$$\text{vocab\_size} = \underbrace{256}_{\text{初始字节}} + (\text{合并次数}) + \underbrace{\text{len(special\_tokens)}}_{\text{特殊 token 数}}$$
- `vocab_size` 是给定目标 → **合并次数 = vocab_size − 256 − len(special_tokens)**。
- 例：TinyStories `vocab_size=10000`, 1 个特殊 token → 合并 9743 次。

**三类词表来源**：

| | 来源 | 参与合并 |
|---|---|---|
| 256 字节 | 初始化 | 是（合并原料） |
| 合并出的 token | 训练算出 | 是 |
| 特殊 token | 人为加入 | **否**（硬边界，不进 merge 统计，但占一个 ID） |

### 3.1.2 特殊 token

**类别**（`len(special_tokens)` = "特殊 token 数"；作业里只用 `<|endoftext|>`）：

> ⚠️ ID **不通用**！每个 tokenizer 自己分配，同一 token 在不同模型 ID 不同。下表 ID 是**具体某模型的真实值**（标了来源），不是"标准编号"。

| 类别 | token | 真实 ID（来源） |
|---|---|---|
| 文档/序列边界 | `<\|endoftext\|>` | **50256**（GPT-2，vocab 50257 的最后一个） |
| | `<s>` (BOS) / `</s>` (EOS) | **1 / 2**（Llama 2 SentencePiece） |
| | `<\|begin_of_text\|>` / `<\|end_of_text\|>` | **128000 / 128001**（Llama 3） |
| 填充/未知 | `<unk>` | **0**（Llama 2）；byte-level BPE 无 OOV → **不需要** `<unk>`（见 §1.4） |
| | `<pad>` | 无统一值，很多 tokenizer 甚至没有，需手动加（batch 对齐用） |
| 对话/角色 | `<\|eot_id\|>` / `<\|start_header_id\|>` / `<\|end_header_id\|>` | **128009 / 128006 / 128007**（Llama 3） |
| | `<\|im_start\|>` / `<\|im_end\|>`（ChatML） | 因 tokenizer 而异；后面对齐/RLHF 作业会遇到 |
| 工具/结构 | 标记工具调用、代码块等 | 因模型而异 |

- 规律：GPT-2 放**末尾**（ID=vocab-1）；Llama 放**开头**（0/1/2）；Llama 3 放**基础词表之后**（128000+）→ 印证"放开头/末尾"是设计选择。
- **共同本质**：人为定义、**永不被拆开**（永远一个 ID）、占固定 ID 但不参与 merge。
- **实现提示**：接口是 `list[str]` → **别写死"只有一个"**，对列表每个都同样处理（`test_train_bpe_special_tokens` 测多个）。特殊 token ID 放开头还是结尾？→ 看 `tests/fixtures/train-bpe-reference-vocab.json` 的排法确定。

---

## 3.2 Pre-tokenization（预分词）

**不合并，只准备"片段→次数"频率表。** 数据流（顺序重要）：
```
原始语料
→ 先按 special_tokens split（re.escape，因 token 含 | 等正则元字符；多个用 "|".join(map(re.escape, special_tokens))）
→ 每段用 GPT-2 正则 re.finditer 切 pre-token（不用 findall，省内存）
→ 每个 pre-token 转 UTF-8 字节序列
→ 统计每个 pre-token 出现次数
```
产物（例）：`(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,s,t):3, (n,e,w,e,s,t):6`（实际是字节 tuple）。

### 3.2.1 为什么需要预分词

思想实验：**不**预分词、直接把文本当连续字节流，会数到**跨词边界**的对（如 `the cat` 里 `(e,␣)(␣,c)`）→ 三个坏处：
1. **语义割裂/依赖标点的怪 token**：`dog` `dog!` `dog.` `dog?` 变成 4 个毫不相干的 ID，语义几乎一样却各学一份，浪费词表。→ 预分词把标点切开，`dog` 始终复用。
2. **跨词无意义组合**：`e␣t`（横跨两词）这种统计噪声占词表却学不到有用东西。→ 预分词限定"只在片段内合并"，学到的都是词内子词（`th`/`ing`/`tion`）。
3. **效率灾难**：不预分词 = 每轮数"语料字节数"个对；预分词后 `the` 出现 100 万次只**存一份 + 记次数**，`(t,h)` 数**一次** ×权重 → 工作量从"语料字节数"降到"**不同词的个数**"，小几个数量级。

**注意**：切的边界不只是标点，主要是**空格**（+数字/字母切换+缩写）；切出的片段**不是最终 token**，只是**合并的作用范围**，片段内部仍被 BPE 继续切成子词。

### 3.2.2 GPT-2 预分词正则详解

```
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
- 需 **`regex` 包**（`import regex as re`），标准库 `re` 不支持 `\p{L}`。用 **`re.finditer`**（省内存）。直接照用即可（handout 允许）。
- `\p{L}`=字母(任何语言)，`\p{N}`=数字，`\s`=空白，`[^\s\p{L}\p{N}]`=标点/符号。
- 结构 = 6 个用 `|`(或) 连的选项，从左到右优先匹配：

| 选项 | 模式 | 匹配 | 例 |
|---|---|---|---|
| 1 | `'(?:[sdmt]\|ll\|ve\|re)` | 英文缩写后缀 | `it's`→`'s`, `i'll`→`'ll`, `they're`→`'re` |
| 2 | ` ?\p{L}+` | (可选前导空格+)字母串=词 | `some`, ` text` |
| 3 | ` ?\p{N}+` | (可选空格+)数字串 | ` 100`；`abc123`→`abc`+`123` |
| 4 | ` ?[^\s\p{L}\p{N}]+` | (可选空格+)标点/符号串 | `!!`, ` ---`, ` $` |
| 5 | `\s+(?!\S)` | 尾随空白(后面无非空白) | 行尾/文末空白 |
| 6 | `\s+` | 兜底：其余连续空白 | |

- **前导空格吸附**（选项2的 ` ?`）：空格被吸到词**前面** → `some text` 的 `text` 切成 ` text`（带空格），句首 `text` 切成 `text`（不带）。二者字节序列不同 → **不同 token**。这就是为什么 GPT-2 系 ` the`(带空格) 和 `the` 是两个 token。
- **选项5 必须在选项6 前**：`|` 从左到右优先，具体规则放前、兜底放后，否则 `\s+` 先吃光空白。
- 完整例：`re.findall(PAT, "I'll pay $100 now!!")` → `['I', "'ll", ' pay', ' $', '100', ' now', '!!']`

### 3.2.3 现代 tokenizer 演进 & 经典正则

**GPT-2 正则范式至今仍是主流**，后续模型改的是参数/细节。经典正则对比：

GPT-2/GPT-3（本作业用，5万）：
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
GPT-3.5/GPT-4 = `cl100k_base`（10万）：
```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```
改动：① **数字限 `\p{N}{1,3}`**（≤3位，`12345`→`123`+`45`）；② 缩写 `(?i:...)` 忽略大小写；③ 换行处理更细。
GPT-4o = `o200k_base`（20万）：更精细，范式不变。

| 模型 | tokenizer | 词表 | 范式 |
|---|---|---|---|
| GPT-2/3 | 原始 BPE | 5万 | GPT正则 |
| GPT-4 / GPT-4o | cl100k / o200k | 10万 / 20万 | GPT正则 |
| Llama 1/2 | SentencePiece | 3.2万 | SP |
| Llama 3 | tiktoken风格 | 12.8万 | GPT正则（倒戈） |
| Mistral | SP→Tekken | 3.2万→13万 | 转GPT正则 |
| Qwen 2.5 | tiktoken风格 | ~15万 | GPT正则（中文强） |
| Gemma | SentencePiece | 25.6万 | SP（少数没转） |
| DeepSeek | tiktoken风格 | ~10万 | GPT正则 |

**两趋势**：① 向 tiktoken 收敛（编码快、生态好）；② 词表越做越大（多语言覆盖，否则小语种被切碎，见 §1.9）。数字限位让 `2024/12345` 不各占 token → 算术更稳、词表不被撑爆。
**Takeaway**：本作业的 GPT-2 正则 + byte-level BPE = 现在开源主流(tiktoken)的基础骨架。

---

## 3.3 BPE Merge loop（合并循环）

### 3.3.1 每轮 4 步 + 规则

循环直到 vocab 满，每轮：
1. **数所有相邻对（加权）**：遍历频率表，数每个 pre-token 内相邻对 × 该 pre-token 次数。
2. **选最高频对**：平局选**字典序更大**的（`max` 对 `tuple[bytes,bytes]` 的默认行为；验证 `max([(b's',b't'),(b'e',b's')])`）。
3. **记录**：新 token 加入 vocab（下一个 ID）+ 该对追加进 merges（顺序重要，encode 按此顺序）。
4. **替换**：所有 pre-token 内相邻的该对 → 合并成单 token。回到步骤 1。

### 3.3.2 逐帧 loop example（务必看懂）

输入频率表（bpe_example 简化，演示做 3 次合并）：
```
(l,o,w):5   (l,o,w,e,r):2   (w,i,d,e,s,t):3   (n,e,w,e,s,t):6
初始: vocab=256字节, merges=[]
```

**第 1 轮**
- A 数对（加权）：`(l,o):7 (o,w):7 (w,e):8 (e,r):2 (w,i):3 (i,d):3 (d,e):3 (e,s):9 (s,t):9 (n,e):6 (e,w):6`
- B 选：`(e,s)`=9 与 `(s,t)`=9 平局 → 字典序 `s`(0x73)>`e`(0x65) → **选 `(s,t)`**
- C 记：`vocab[256]=b'st'`, `merges=[(s,t)]`
- D 替换：
  ```
  (l,o,w):5   (l,o,w,e,r):2   (w,i,d,e,st):3   (n,e,w,e,st):6   ← s,t 合并
  ```

**第 2 轮**（基于第1轮改动后的表）
- A 数对：`(l,o):7 (o,w):7 (w,e):8 (e,r):2 (w,i):3 (i,d):3 (d,e):3 (e,st):9 (n,e):6 (e,w):6`
  - ⚠️ 第1轮的 `(e,s)`/`(s,t)` **消失**，冒出新对 **`(e,st)`=9**——它是第1轮合并**产生**的，第1轮之前根本不存在！
- B 选：`(e,st)`=9 → **选 `(e,st)`**
- C 记：`vocab[257]=b'est'`, `merges=[(s,t),(e,st)]`
- D 替换：`(w,i,d,est):3   (n,e,w,est):6`

**第 3 轮**
- A 数对：…`(l,o):7 (o,w):7`… 平局 → 字典序 `o`>`l` → **选 `(o,w)`**
- C 记：`vocab[258]=b'ow'`, `merges=[(s,t),(e,st),(o,w)]`
- D 替换：`(l,ow):5   (l,ow,e,r):2`

**结束** → 返回 `vocab(256字节+st,est,ow)` + `merges=[(s,t),(e,st),(o,w)]`

**核心理解**：loop 就是**反复修改同一张频率表**，每轮扫表→选最高频→记录→就地合并→用改动后的表再来一遍。终止：`len(vocab)==vocab_size`。
**为什么必须串行**：第2轮的 `(e,st)` 是第1轮合并的产物，第1轮之前看不到 → 无法提前知道第2轮选谁 → 必须先跑完第1轮。这条依赖链无法打破。

### 3.3.3 效率 & 并行（重要结论）

| 阶段 | 能并行 | 怎么加速 |
|---|---|---|
| Pre-tokenization | ✅ 数据并行 | `multiprocessing` 多进程（绕开 GIL）+ `find_chunk_boundaries` 按 `<\|endoftext\|>` 切块；只汇总 1 次 |
| Merge loop | ❌ 顺序依赖 | 算法优化：**增量更新** pair 计数（一次合并只影响相邻的几个对，不全量重扫） |
| GPU | ❌ 基本无用 | 符号处理(查字典/找max/替换)不适合 GPU 的稠密数值并行 |

- **为什么 merge 难并行**：不是数据分不开，而是**每轮都要全局决策（选全局最高频对），且这轮决策依赖上轮执行结果**。9 核各管各内存也没用——每轮仍需"汇总局部计数→选→广播"这个同步点，×上万次，通信开销 >> 单步计算（单步只动几百万 unique 词的一小部分，毫秒级）。
- **数据 TB 级也不改变结论**：预分词后是"unique pre-token→次数"频率表，unique 词数由**词汇种类**决定（Heaps 定律，几百万封顶），**不随原始数据线性增长**。GB 和 TB 的合并阶段面对同量级频率表 → 单步计算仍毫秒级 → 并行仍不划算。（随数据线性增长的是**预分词**，所以并行价值全在那。）
- **判据**：并行只在"任务粒度大 + 同步次数少"时划算。合并 = 粒度小 + 同步上万次 = 最差场景。极致加速靠换 **Rust/C++**（消除解释器开销），非并行/GPU。
- Profiling：`cProfile` / `py-spy` 先测瓶颈再优化。

### 3.4 优化历程

**✅ 里程碑：train_bpe 全 3 测试通过（2026-07-24）— 这题 15 分完整拿下**
- `cs336_basics/bpe_tokenizer.py` 实现完成，接到 `adapters.run_train_bpe`；`uv run pytest tests/test_train_bpe.py` **3/3 PASS**。
- 实测 valid+vocab1000 的前几个 merge：`(b' ',b't')`, `(b'h',b'e')`, `(b' ',b'a')`, `(b' t',b'he')`(→` the`), `(b' a',b'nd')`(→` and`)——渐进合并正确，符合 §3.3.2。

#### 3.4.1 总表：merge loop 耗时（valid 22.5MB + vocab_size=10000）

跑法：`python cs336_basics/train_tinystories.py`，每版跑 3 次取中位数。**每一步的 merge 序列都逐条比对过，9743 条完全一致**（见 §3.4.4）。

| # | 版本 | Merge 耗时 | vs 上一步 | vs 起点 |
|---|---|---|---|---|
| 0 | 增量更新，每轮 `max` 全扫 pair_counts | 14.3s※ | — | 1× |
| 1 | + 懒删除堆（min-heap + `neg_bytes` 哨兵） | 1.46s | **9.8×** | 9.8× |
| 2 | + `neg_bytes` 按 token id 缓存 | 0.89s | **1.64×** | 16× |
| 3 | + 只更新计数真正变化的 pair（delta 更新） | 0.54s | **1.66×** | 27× |
| 4 | + 堆定期重建，清掉 stale 条目 | **0.42s** | **1.28×** | **34×** |

※ #0 是早期记录的值，未在本轮重测；#1~#4 是同一天同一台机器连续测的。

**分阶段的定性收益**（更早期，pytest 全套 `uv run pytest tests/test_train_bpe.py`）：

| 版本 | 全套耗时 | 关键改动 |
|---|---|---|
| 朴素串行 | ~16s | 每轮全量重数 pairs + 重建所有词 |
| + 并行预分词、原地改、跳过不含该 pair 的词 | ~5s | `Pool` 多进程；`len` 缓存；`if id_a not in token_id: continue` |
| + 增量更新合并 loop | **0.87s** | `pair_counts` / `pair_to_words` 持久增量 |

⚠️ **小数据看不出堆的优势**：corpus.en/vocab500 上堆版才 0.3s，和无堆差不多甚至略慢（堆维护开销 ≈ 省下的扫描）。**对比优化效果必须在目标规模上测。**

#### 3.4.1b 懒删除堆（lazy-deletion heap）—— 通用技巧，值得单独记

**问题**：`heapq` 只保证「堆顶最小」，**不支持「高效修改/删除堆里某个已存在元素」**——要找到它得 O(n) 扫描。但 BPE 里 pair 的计数**频繁变化**，天然需要「改堆里某个 pair 的优先级」。

**懒删除的解法（核心思想）**：**不改旧的，只 push 新的；旧的留着变 stale，弹出时验证 + 丢弃。**

```python
# 计数变化时：不去堆里找旧条目改，直接 push 新值（O(log n)，不扫描）
def push(self, pair, count):
    heapq.heappush(self.heap, (-count, tie_break_key, pair))

# 取最优时：弹出，和「真相源」pair_counts 核对，不一致就是 stale，丢弃继续弹
def pop_best(self, pair_counts):
    while self.heap:
        neg_count, _, _, pair = heapq.heappop(self.heap)
        if -neg_count == pair_counts.get(pair, 0) and -neg_count > 0:
            return pair            # 堆里的值 == 真值 → 有效
        # 否则过时，丢弃，继续
    return None
```

**关键理解（为什么这样反而快）**：
- **push 从不扫描堆**——它只做上浮（sift-up），O(log n)。push 一个已存在的 pair 会留下**重复条目**（旧的 stale + 新的有效），这是**故意的**，不是 bug。
- **真相源在别处**：`pair_counts`（dict）才是「某 pair 当前计数」的权威。堆只是个「大概按优先级排」的加速结构，靠 pop 时和 `pair_counts` 核对来纠正。
- **权衡**：用「允许 stale 垃圾 + pop 时验证」换掉了「O(n) 堆内查找修改」。每个操作都 O(log n)，代价是堆膨胀。
- **膨胀治理**：见 #4 `maybe_rebuild`——堆长 > 2×存活+1024 时重建，清 stale。

**这个模式的适用场景（记住它）**：任何「需要优先队列 + 元素优先级频繁变化」的场合（Dijkstra 变体、事件调度、任务队列……）。标准库堆不支持 decrease-key，懒删除是最常用的绕过法。判据：**修改频繁、且有一个独立的「真相源」能在 pop 时验证有效性**。

#### 3.4.2 第 2~4 步：每条优化为什么快

**#2 `neg_bytes` 按 token id 缓存**

```python
def _neg(self, token_id):                      # PairHeap 里加一层缓存
    key = self._neg_cache.get(token_id)
    if key is None:
        key = self._neg_cache[token_id] = neg_bytes(self.vocab[token_id])
    return key
```
- 原来每次 `push` 都对两个 id 各跑一遍 `tuple(-x for x in b) + (1,)`——生成器 + 建元组，纯分配开销。
- merge 到后期 token 越来越长（十几字节），而 push 次数是万级 × 每次多个 pair → 这是白给的重复计算。
- id→bytes 只增不改，所以缓存**永不失效**，最多算 `vocab_size` 次。

**#3 只更新计数真正变化的 pair**

原写法是「整词 remove-then-add」：先把该词**所有**相邻对减掉、再把新序列**所有**相邻对加回。问题是一个词里大部分 pair 根本没变，却白白经历「减→push→加→push」。

**具体例子**：词 `[a, b, c, d, e]`（count=5），合并 `(c,d) → X`：
```
旧序列: [a, b, c, d, e]   相邻对: (a,b) (b,c) (c,d) (d,e)
新序列: [a, b, X, e]      相邻对: (a,b) (b,X) (X,e)
```
| 对 | 旧出现 | 新出现 | delta | 要动吗 |
|---|---|---|---|---|
| `(a,b)` | 1 | 1 | **0** | ❌ 跳过（远离合并点，没受影响） |
| `(b,c)` | 1 | 0 | −1 | ✅ 计数 −5、push |
| `(c,d)` | 1 | 0 | −1 | ✅ 计数 −5、push（就是 best_pair） |
| `(d,e)` | 1 | 0 | −1 | ✅ 计数 −5、push |
| `(b,X)` | 0 | 1 | +1 | ✅ 计数 +5、push |
| `(X,e)` | 0 | 1 | +1 | ✅ 计数 +5、push |

关键：`(a,b)` 旧新都出现 1 次 → delta=0。**整词 remove-then-add 会对它「−5 再 +5」（净 0）+ push 两次，纯白干**；delta 法直接跳过。词越长，合并点两侧不变的对越多，省得越狠——合并一个对只影响它**左右相邻**的 3~5 个对，与词长无关。

```python
old_pairs = Counter(zip(seq, seq[1:]))          # zip(seq,seq[1:]) 得所有相邻对；Counter 数重数
new_pairs = Counter(zip(new_seq, new_seq[1:]))
for pair in old_pairs.keys() | new_pairs.keys():
    delta = new_pairs[pair] - old_pairs[pair]
    if delta:                                   # 只有真变了才动 counts + push
        ...global_pair_counts[pair] += delta * count; push...
```
- **为什么 delta 一定对**：一个对的全局计数 = 各词贡献之和。这个词合并前贡献 `old*count`、合并后 `new*count` → 净变化 `(new−old)*count = delta*count`。只加这个 delta，全局就对。没变的对 delta=0 不影响。
- `delta == 0` 的 pair：计数没变 ⇒ 堆里那条旧条目**依然有效** ⇒ 不用 push。不变量不破。
- 验证 `A B A B` 合并 `(A,B)`：old=`{(A,B):2,(B,A):1}`, new=`{(X,X):1}` → `(A,B)`−2、`(B,A)`−1、`(X,X)`+1，全对上。
- 实测（tinystories_sample）：总 push 次数 **4549 → 2274**，直接砍半。push 是 `O(log n)` 且比较的是长元组，是热点中的热点。
- 附带好处：`pair_to_words` 的 `discard`/`add` churn 也一起没了。
- **通用性**：这是「局部改动、大部分不变」类增量维护的通法——算 delta 只更新变化部分。以后 Rust 重写、别的增量算法都能复用。

**#4 堆定期重建**

```python
def maybe_rebuild(self, pair_counts):
    if len(self.heap) > 2 * len(pair_counts) + 1024:
        self.heap = [self._key(p, c) for p, c in pair_counts.items() if c > 0]
        heapq.heapify(self.heap)
```
- 懒删除的代价是堆**只增不减**，stale 条目越堆越多 → `push`/`pop` 的 `log n` 里那个 n 是**堆长度**而不是存活 pair 数。
- 实测（tinystories_sample）：峰值堆长 **2505 → 1272**。
- 重建后每个存活 pair 都带当前 count，**不变量天然成立**，所以这步不影响正确性。
- 阈值 `2×存活 + 1024`：重建是 O(n)，摊还下来可忽略；`+1024` 避免小规模下反复重建。

#### 3.4.3 顺手修的两个正确性 bug

| bug | 触发条件 | 后果 | 修法 |
|---|---|---|---|
| `re.split("", chunk)` | `special_tokens=[]` | pattern 为空串，**在每个字符之间都切一刀**，pre-token 全被打成单字符 | `re.split(pat, chunk) if pat else [chunk]` |
| 分块硬编码 `b"<\|endoftext\|>"` | 特殊 token 不叫这个名字 | 找不到切点 → 所有边界退化到 EOF → **32 进程变 1 个**，并行全失效 | 用 `special_tokens[0].encode()`；无特殊 token 时直接 `[0, filesize]` 单块 |

实测第二条：0.7MB、特殊 token 为 `<|sep|>` 的语料，**修前 1 块 / 修后 32 块**。

另外清理了 `global_pair_to_words` 的空 set 泄漏（`defaultdict` 改普通 dict + `setdefault`，空了就 `del`），只是内存问题，不影响结果。

#### 3.4.4 怎么验证「优化没改坏结果」（方法论，比结论更重要）

计数漂移这类 bug **静态看不出来**，唯一可靠办法是对拍。这次用了三层：

1. **差分测试（对拍朴素实现）**：写一个每轮全量重算 `pair_counts`、用 `max(..., key=(count,(bytes_a,bytes_b)))` 挑选的 brute-force 版，和优化版逐条比 merges + vocab。
   - 语料要**专门造刁钻的**：高度撞车的 tie 语料（大量 pair 同频，专测 tie-break）、`aaaa` 重叠串（测贪心从左到右 + 重叠对记账）、中文/带音标拉丁（测多字节）、随机串。
   - 覆盖：5 组语料 × 3 个 vocab_size 全 match；tinystories_sample 582 条 merge 全 match；5MB 语料 2743 条 merge 全 match。
2. **不变量断言**：把 merge loop 抄一份出来，**每步 merge 后**从 `global_vocab_ids` 从零重算真值，校验三件事——
   - `global_pair_counts` 没漂移；
   - `global_pair_to_words` 和真值完全相等（且没有空 set 残留）；
   - **每个存活 pair 都有一条携带当前 count 的堆条目**（这条是懒删除正确性的命根子）。
   - 582 步全部通过、零漂移。
3. **官方测试**：`pytest tests/test_train_bpe.py` 3/3。

#### 3.4.5 懒删除堆：为什么是对的（两条不变量）

| 不变量 | 内容 | 靠什么保证 |
|---|---|---|
| 1 无遗漏 | 任何 count>0 的 pair，堆里至少有一条**携带它当前 count** 的条目 | 每次 count 变动都紧跟一次 `push`；重建也是按当前 counts 建的 |
| 2 可鉴别 | `pop` 出来的条目能判断是否过期 | key 是 `(pair, count)` 的纯函数，`count != pair_counts.get(pair,0)` 就是 stale |

推论：设真正最优 pair 是 `p*`，由不变量 1，`(p*, C*)` 在堆里；比它先弹出的要么过期被丢，要么**更优或等价**（而更优不可能，因为 p\* 已最优）⇒ 第一个通过校验的弹出结果就是真正最优。✅

**merge 不会凭空造出 `(id_a, id_b)`**：`new_id` 永远是全新 id，所以 best_pair 处理完必然归零并被删除，循环外的 `pop(best_pair, None)` 只是保险。

#### 3.4.6 踩过的坑清单

*A. 计数不同步类（增量更新）*
- **`i`/`j` 变量冲突**：外层词下标用 `j`、内层扫描用 `i`，一开始都用 `i` → 内层 `i=0` 覆盖外层下标 → 写错位置。**内外循环变量必须不同名。**
- **`ord(c)` vs `encode`**：`[ord(c) for c in token]` 是**码点**不是 UTF-8 字节！byte-level 必须 `list(token.encode("utf-8"))`。
- **减 / 加 必须严格对称**：任何不对称 → 计数漂移 → tie-break 选错 → 测试挂。合并 `A B` 会连带改变 `(前,A)→(前,X)`、`(B,后)→(X,后)`，不只是 `(A,B)` 消失。
- **清理 0 计数对**：`if pair_counts[pair] <= 0: del`，否则会选到 count=0 的僵尸对。
- **`affected` 必须拷贝**：`list(pair_to_words[best_pair])`，循环体里会改这个 set。
- **`.discard()` 不用 `.remove()`**：同词可能多次贡献同一 pair（`A B A B`）。
- **`pop(k, None)` 不用 `del`**：可能已在减计数时删过。

*B. tie-break 类（堆）*
- **`neg_bytes` 的长度陷阱**（最大的坑）：`tuple(-x for x in b)` 反转了字节值，但**元组比较里「前缀相同时短的更小」这条长度规则没被反转** → `b'b'` vs `b'bc'` 选错。
  - 修复：**加正数哨兵** `tuple(-x for x in b) + (1,)`。哨兵 `1` > 任何 `0~-255`，让更长的串在对应位补负数 → min 里更小 → 复现「长的更大」。
  - 哨兵必须是**正数**：`0` 会和 `neg_bytes(b'\x00')` 撞；`b"\xff"` 补码方案也会撞（`\x00` 的补码就是 `\xff`）。
  - 穷举验证（含 `\x00`/`\xff` 的 1~3 字节串，66306 组比较）：naive 版 **936 组违反**反序，哨兵版 **0 违反**。
- **int 版 tie-break**：不能比 ID 数值！必须 `(count, (vocab[id_a], vocab[id_b]))` 转回 bytes 比字典序。
- 备选方案（未用）：堆只存 `-count`，pop 时收集所有并列最高 count 的候选再正常 `max`——避开反转，但 `pop_best` 更复杂。

*C. 一般实现类*
- 序列用 **int(token ID) list** 表示（非 bytes）；`list(word.encode())` 直接得 int list。
- **merges 存 bytes 对**：`(vocab[id_a], vocab[id_b])`，不是 id 对。
- 替换用「扫描重建新 list」（`i+2`/`i+1`），非原地 pop（避下标错位）；`new_id = len(vocab)` 别用 `len-1`。
- 特殊 token 最后加，终止条件留余量 `vocab_size - len(special_tokens)`。
- `bytes([i])`（方括号！）造单字节；`bytes(65)` 是 65 个 0，`bytes('A')` 报错。
- `if __name__ == "__main__":` 包住底部测试调用，避免 import 时乱跑十几秒。
- **优化2 的位置坑**：`if id_a not in token_id: continue` 必须放在重建循环**前**，放后面等于白重建。

#### 3.4.7 如何 profile

```python
import cProfile, pstats
cProfile.run('train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])', 'prof')
pstats.Stats('prof').sort_stats('cumulative').print_stats(20)
```
- 看 **tottime**（函数自身耗时，找瓶颈看它）、**cumtime**（含子调用）、**ncalls**（几百万次=热点）。
- 命令行版：`python -m cProfile -s cumtime <script>`；可视化：`snakeviz`。
- 实战战果：profile 增量版（valid+vocab10000）显示 `max` 7.3s + `rank` 6.6s（被调 **1 亿次**）= 几乎全部时间 → 直接指向「用堆替代每轮全扫」。而 a/c 增量操作才 0.05s。**先 profile 再优化，别猜。**

### 3.5 优化 TODO（先 Python 跑通再做）
- **顺序铁律**：Python 写对 → 过 `test_train_bpe` → profile 找瓶颈 → 先试 `multiprocessing`（handout 说并行后 TinyStories BPE 可压到 2 分钟内，可能纯 Python 就达标）→ 还不够才上 C/Rust。
- **C/Rust 重写热点**（handout 认可：`cppyy`/`nanobind`/PyO3）：快在无解释器开销、无对象封装开销（回忆 `grep` 数 27630 个 EOF 秒出的原因）。
  - 只对**预分词**（CPU 密集、随数据线性增长）有用；**合并 loop 是串行依赖**，C 只能让每步更快、消不掉顺序（见 §3.3.3）。
  - 打算借此**学 Rust + PyO3**（用作业当实战）。
- **数据探查用命令行**（比写 Python 快）：`grep -c/-oF`、`wc -l/-c`、`stat -c %s`、`head/tail`。
  - 实测：TinyStories valid 有 **27630** 个 `<|endoftext|>`、22.5MB；train 2.23GB → 倒推 ~**2.73M** 文档（对得上 handout 的 2.12M 量级）；平均 ~816 字节/文档。

---

## 4. Embedding & 表示学习（讨论）

> §2 tokenizer 和 §3 Transformer 之间的桥梁概念，讨论中梳理。

### 4.1 token ID → 向量：embedding matrix 查表
- Embedding matrix 形状 `(vocab_size, d_model)`：**行数=词表大小**（每个 ID 一行），**列数=d_model**（每行是该 token 的向量）。
- **token ID 就是行号**：给 ID=256 → 取矩阵第 256 行的 `d_model` 维向量。本质是 **lookup（按行号取行）**，不是真矩阵乘法。
  - 理论上等价于 one-hot 向量 × 矩阵，但 one-hot 里绝大部分是 0，乘出来就是取那一行 → 直接取行快得多。
- 这就是 `run_embedding` 要做的事（`weights=(vocab_size,d_model)`，`token_ids` 是行号）。
- **特殊 token 在这层无区别**：`<|endoftext|>`(如 ID 10000) 就是矩阵第 10000 行，和普通 token 一样是一个可学习向量。tokenizer 层它"特殊"（不被拆），embedding 层它就是普通一行。
- 呼应：ID 空间是编号、可无限往上加；256 字节只占 ID 0–255，合并 token 从 256 起，特殊 token 再往后 → 词表没被"占满"。

### 4.2 embedding matrix 如何训练
- **它就是普通可学习参数**，没有特殊训练方式 = 整个模型怎么训练它就怎么训练。
- 训练大循环（§5 要实现）：forward（ID→查行→Transformer→logits→loss）→ backward（求梯度）→ step（AdamW 更新）→ zero_grad。
- **embedding 特有洞察**：一个 batch 只查了出现过的 token 的行 → 反向时**梯度只流回被用到的行**，没出现的 token 那一步梯度=0、不更新。
  - ⟹ 高频 token 更新多、学得充分；低频/生僻 token 更新少、学得差。
  - **这解释了 §1.9**"中文稀疏→掌握不锐利"：中文 token 出现少 → embedding 行更新少 → 没训充分 → 采样易漂移。
- **方向由 loss 驱动**：LM 的 loss 是预测下一词的 cross-entropy（§4 `run_cross_entropy`）。梯度告诉每个参数"往哪调能让正确词概率更高"。经海量更新，常出现在相似上下文的 token 向量被"推"到相近位置 → "猫≈狗""CJK 互为邻居"是 loss + 上下文**塑造**出来的，非人为设计。

### 4.3 embedding 参与 pretraining（端到端）
- **参与，且从随机初始化一起练**。pretraining = 在海量文本上预测下一词、反复更新**所有参数**。
- 全部可学习参数：embedding matrix + 各层 attn(q/k/v/o) + FFN(w1/w2/w3) + RMSNorm gain + LM head。**一起端到端训练**，没有谁先谁后。
- LM head 也是 `(vocab_size, d_model)`，做反向的事（向量→各 token 的 logits）；有些模型让输入 embedding 和 LM head **共享矩阵**（weight tying）。
- 对比"不参与训练"的情况帮理解：① 用冻结的预训练词向量（word2vec/GloVe，见 §4.4）；② 微调时 freeze embedding。CS336 做的是 **from scratch pretraining，embedding 跟着一起学**。
- 术语：**Pretraining**=大规模自监督从零训整个模型（assignment 1 在 TinyStories/OWT 上做的）；**Fine-tuning**=之后小数据针对性再调。

### 4.4 历史：word2vec / GloVe（两阶段 → 端到端的演进）
- **word2vec**（2013, Mikolov）：专门训词向量的独立模型，思想"上下文决定词义"。CBOW（上下文→中心词）/ Skip-gram（中心词→上下文）。
- 著名性质：向量类比 `vec(king)-vec(man)+vec(woman)≈vec(queen)` → 向量空间编码语义关系。

| | word2vec 时代（两阶段） | 现代 LLM（端到端） |
|---|---|---|
| embedding 怎么来 | 单独训好 | 跟整个模型一起 pretraining |
| 用时 | 塞进下游、常冻结 | 全程被 loss 更新 |
| 训练目标 | 通用上下文预测 | 直接是最终任务（预测下一词） |

- **为何弃用两阶段**：① 端到端让 embedding 被最终任务塑造，更贴合；② word2vec **静态**——一词一个固定向量，"bank"(银行/河岸)分不开；而 Transformer 里 embedding 只是起点，过 attention 后同词在不同上下文得**不同**表示（contextualized）。
- **思想活下来了**："上下文决定意义"一脉相承——word2vec 显式用上下文训向量；LM 预测下一词时 loss 把相似上下文的 token 向量推近。§1.9 的"CJK 互为邻居"与 word2vec 类比性质**同源**，只是融进了端到端训练。
- 作业里**不用** word2vec，但理解它能明白"为什么 embedding 能从零学出有意义结构"——因为预测任务本身就蕴含"上下文决定意义"的信号。

---

## 5. 待续（§3 Transformer 实现时继续记）
