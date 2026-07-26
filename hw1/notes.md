# CS336 Assignment 1 — 学习笔记

> 记录做作业时遇到的问题、概念理解、坑与解决方法。
> 公式用 LaTeX：行内 `$...$`，独立 `$$...$$`。

---

## 0. 整体地图

### 0.1 四大块 & 工作流程

作业目标：从零搭一个完整的 Transformer LM 训练流程。四大块：

| Handout 章节 | 实现内容 | adapters 函数 | 测试文件 |
|---|---|---|---|
| §2 BPE Tokenizer | BPE 训练 + Tokenizer(encode/decode) | `run_train_bpe`, `get_tokenizer` | `test_train_bpe.py`, `test_tokenizer.py` |
| §3 Transformer LM | Linear/Embedding/RMSNorm/SwiGLU/Attention/RoPE/Block/LM | `run_linear` … `run_transformer_lm` | `test_model.py`, `test_nn_utils.py` |
| §4 Loss & Optimizer | cross-entropy / AdamW / grad clip / cosine LR | `run_cross_entropy` … `run_get_lr_cosine_schedule` | `test_nn_utils.py`, `test_optimizer.py` |
| handout §5 Training loop | get_batch / checkpoint 存取 | `run_get_batch`, `run_save/load_checkpoint` | `test_data.py`, `test_serialization.py` |


**工作流程**：在 `cs336_basics/` 里从零写实现 → 在 `tests/adapters.py` 里把 `run_xxx` 接到自己的实现（只做胶水，无实质逻辑）→ `uv run pytest` 验证。


**Leaderboard（handout §7.5，6 分）**：在 OWT 上训模型、最小化 validation loss。规则只有两条——单次跑 **≤45 分钟 B200**、只能用课程给的 OWT 数据；其余随意。及格线是打败 loss 5.0 的朴素基线。提 PR 到 `stanford-cs336/assignment1-basics-leaderboard`。**排的是模型质量，不是 tokenizer 速度。**


### 0.2 语料

| | 大小 | 是什么 | 特点 |
|---|---|---|---|
| **TinyStories** | train 2.23 GB / valid 22.5 MB | GPT-3.5/4 **合成**的儿童故事（Eldan & Li, 2023），词汇刻意限制在 3–4 岁小孩能懂的范围 | 极干净、极同质。用途是让小模型也能训出流利英语，便于快速迭代 |
| **OpenWebText** | train 11.92 GB / valid 290 MB | Reddit 高赞外链的网页正文，GPT-2 训练语料 WebText 的开源复现 | 真实、嘈杂、多样：乱码、排版分隔线、多语言混杂 |
| **The Pile**（不下载，只作参照） | 825 GB | EleutherAI (2020)，**22 个来源混合**——arXiv、GitHub 代码、PubMed、专利、Stack Exchange、维基、书籍、字幕、网页 | 卖点是**成分公开且刻意多样**（对比 OWT 的纯网页抓取）。GPT-Neo/GPT-J/Pythia 用它训练；现已被 RedPajama、Dolma 取代，但仍是**"真实预训练语料有多大"的标准参照单位** |

前两个是作业实际用的（`data/`，handout 提供）；Pile 只出现在 handout §2.7(c) 的估算题里——*你的 tokenizer 拿到真实规模语料上要跑多久*（见 §2.10）。

**TinyStories 和 OWT 的差异贯穿整份笔记**：unique pre-token 6 万 vs 660 万、最长 token 是真词 vs 乱码、缓存命中率 99.75% vs 96.93%——**同一份代码在它们身上常常给出相反的结论**（§2.7 教训 1）。这不是巧合：一个是合成的干净同质语料，一个是真实的嘈杂多样语料。

---

## 1. Unicode / 字符 / 字节

### 1.1 str vs bytes 是两个层面

```
"café"          ← str：字符（Unicode 码点）的序列
   ↕  encode / decode
b'caf\xc3\xa9'  ← bytes：0–255 的字节序列
```

- **encode**：`str → bytes`（按 UTF-8 规则）；**decode**：反向。二者互逆。
- 计算机底层只存字节（0–255），但字符有 15 万+ 个，所以需要编码规则做映射。
- `len()` 数的是当前对象的元素：`len("café")=4`（字符），`len("café".encode())=5`（字节）。
- **坑**：`print(b'caf\xc3\xa9')` 仍显示 `b'...'`，不会解码成 `café`——bytes 没有"给人看"的形式，要显式 `.decode()`。

### 1.2 UTF-8 的变长机制

**几个字节由码点范围决定**：

| 码点范围 | 字节数 | 例子 |
|---|---|---|
| U+0000–U+007F | 1 | ASCII (`c`, `a`, `f`) |
| U+0080–U+07FF | 2 | `é`, `ñ`, 希腊/西里尔 |
| U+0800–U+FFFF | 3 | **绝大多数常用汉字**、假名、韩文 |
| U+10000–U+10FFFF | 4 | emoji、扩展区生僻字（如 `𠮷`） |

> ⚠️ **中文不都是 3 字节**：生僻字/扩展区在 U+20000 以上是 4 字节。口诀：不要记"中文=3字节"，要记「字节数由 `hex(ord(c))` 落在哪个范围决定」。

**怎么做到变长而无歧义**：每个字节的高位前缀自带角色标记（`x` 是承载码点的有效位）：

```
1 字节  U+0000–007F     0xxxxxxx
2 字节  U+0080–07FF     110xxxxx 10xxxxxx
3 字节  U+0800–FFFF     1110xxxx 10xxxxxx 10xxxxxx
4 字节  U+10000–10FFFF  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

`0...`=单字节(ASCII)，`110/1110/11110...`=2/3/4 字节头，`10...`=**续接字节**。解码器从左往右扫，看首字节前缀就知道后面还要读几个续接字节 → **自解析**。

实例 `é`(U+00E9)：填 2 字节模板 → `11000011 10101001` = `\xc3\xa9` = `[195,169]`。

### 1.3 UTF-8 vs UTF-16/32，以及为什么会有后两者

三者都是"码点→字节"的编码方案，区别在**编码单位宽度**：

| 编码 | 基本单位 | 一字符占几字节 | 动机 | 现定位 |
|---|---|---|---|---|
| UTF-8 | 1 字节 | **变长 1~4** | 兼容 ASCII + 省 + 避 `\x00` | 事实标准（>98% 网页） |
| UTF-16 | 2 字节 | 变长 2 或 4 | 早期以为 2 字节(65536)够，图定长好算 | 遗留：Windows/Java/JS 内部字符串 |
| UTF-32 | 4 字节 | **定长 4** | 想要真定长、装下所有码点 | 极少用，太浪费 |

实测 "hello"：utf-8 `[104,101,108,108,111]`（5 字节，无 0）；utf-16 每字符后跟 1 个 0（12 字节）；utf-32 跟 3 个 0（24 字节）。→ 这是 unicode2(a) 的核心证据。

### 1.4 byte-level BPE 的好处

所有文本先 `encode("utf-8")` 变成 0–255 字节，BPE 只在**字节层**操作：
- 初始词表大小固定 = **256**（不是 128——UTF-8 流里字节会取到 128–255）。
- 永远不会 OOV（任何字符拆到字节都逃不出 0–255）⇒ **不需要 `<unk>`**。
- "某字符占几字节"的复杂度被 UTF-8 这步完全吸收，训练算法对此透明。

### 1.5 延伸：LLM 中文说着变韩/日文 ≠ 字节接近

- LLM 不在**字节层**预测，而是在 **token ID** 上预测。中文和日文 token 是词表里两个不同 ID，ID 相邻也无语义关系 → "字节接近导致混淆"不成立。
- **真正原因**：① 训练数据 CJK 混杂 → **embedding 空间**里三者互为邻居；② 中文高质量数据稀疏，采样易漂移；③ 采样随机性；④ tokenizer 不匹配（中文被切碎）。
- **直觉纠正**：不是字节接近，而是 **embedding 空间**里 CJK 向量互为邻居。字节只是最底层存储，模型不在那层"思考"。（与 §3.2 的"高频 token 学得充分"呼应）

---

### 1.6 书面题

#### Problem `unicode1` — Understanding Unicode (1 pt)

**(a)** `chr(0)` 返回什么 Unicode 字符？
- `\x00` 是**空字符（Null character，NUL）**，ASCII 码值 0。渲染不出任何字形（不占显示空间），但仍占一个字符位置；C 语言等用它作字符串结束标志。

**(b)** 它的 `__repr__()` 表示 和 打印出来 有何不同？
- `repr` 显示成 `'\x00'`（可见转义），`print` 显示为无可见字符。
- 原因：repr 面向调试，要精确无歧义，故用可见转义把控制字符"显示出来"；print 输出字符本身，终端无法渲染控制字符。

**(c)** 它出现在文本中会怎样？
- 字符**存在但渲染不出来**。它没有截断字符串（不像 C 里的 `\0`），`len` 仍把它计入一个位置——只是终端显示为空白。

#### Problem `unicode2` — Unicode Encodings (3 pts)

**(a)** 为什么在 UTF-8 字节上训练 tokenizer 优于 UTF-16/UTF-32？
- UTF-8 对常见字符只用 1 字节，字节序列比 UTF-16/32 **更短**；而且 UTF-16/32 会插入大量填充的 **`\x00`**，这些 `\x00` 会成为**高频字节对**，让 BPE 把有限的合并名额浪费在无意义的填充上，学不到真正有用的子词。
- （证据见 §1.3 的 "hello" 实测）
- ⚠️ 措辞坑：不要说 "UTF-8 是压缩/最优压缩"（它是编码规则不是压缩算法）；不要说 BPE 是"聚类"（BPE 是频率驱动的**贪心合并**，不是 clustering）。

**(b)** 为什么 `decode_utf8_bytes_to_str_wrong`（逐字节 decode）是错的？给一个出错的输入例子。
- 因为非 ASCII 字符（如 `é`）在 UTF-8 里由**多个字节共同编码**（`é` = `\xc3\xa9`），逐字节单独 decode 会把这些字节切断。
- 例子：`"café".encode("utf-8")` = `[99,97,102,195,169]`。前三个 ASCII 字节能各自解码，到 `\xc3`(195) 时——它是 2 字节序列的**首字节**（`0b11000011`，`110` 开头声明后面还有 1 个续接字节），单独解码时续接字节 `\xa9` 缺失 → 报 `UnicodeDecodeError: ... unexpected end of data`。

**(c)** 给一个不能 decode 成任何 Unicode 字符的双字节序列。
- 核心：违反 UTF-8 前缀规则（合法双字节须为 `110xxxxx 10xxxxxx`）的序列都无法 decode。三种非法方式：

| 例子 | 非法原因 | 报错关键词 |
|---|---|---|
| `bytes([0xff, 0xff])` | `0xff=11111111`，首字节前缀 `11111` 不属于任何合法类 | invalid start byte |
| `bytes([0x80, 0x80])` | `0x80=10000000`，`10` 开头是**续接字节**标记，不能当开头 | invalid start byte |
| `bytes([0xc3, 0x28])` | 首字节 `0xc3` 是合法 2 字节头，但 `0x28` 是 ASCII(`0`开头)，不是 `10` 续接字节 | invalid continuation byte |

- 推荐答 `[0xc3, 0x28]` 或 `[0x80, 0x80]`：能体现理解了"续接字节"概念。

---

## 2. BPE Tokenizer（handout §2.4–2.6）

> 三步：**① Vocab init → ② Pre-tokenization → ③ Merge loop**。
> 产出（`run_train_bpe` 返回值）：**vocab** `dict[int,bytes]`、**merges** `list[tuple[bytes,bytes]]`（按创建顺序）。
> 一句话：**从 256 字节出发，反复把"最常相邻的 token 对"合并成新 token，直到词表达到 `vocab_size`。**
>
> ```
> ① 256 字节 + 特殊 token → 初始 vocab
> ② 语料 → split特殊token → 正则切 → 转字节 → {pre-token: 次数}
> ③ 循环{数对(加权) → 选最高频(平局取字典序大) → 加vocab+记merges → 替换} → vocab + merges
> ```

### 2.1 Vocab init

初始词表 = 全部 256 字节值（ID 0–255）+ 特殊 token；之后每合并 1 次词表 +1。

$$\text{vocab\_size} = \underbrace{256}_{\text{初始字节}} + (\text{合并次数}) + \text{len(special\_tokens)}$$

⇒ **合并次数 = vocab_size − 256 − len(special_tokens)**。例：TinyStories `vocab_size=10000` + 1 个特殊 token → 合并 9743 次。

| 词表来源 | 参与合并 |
|---|---|
| 256 字节（初始化） | 是（合并原料） |
| 合并出的 token（训练算出） | 是 |
| 特殊 token（人为加入） | **否**（硬边界，不进 merge 统计，但占一个 ID） |

> ⚠️ 造字节的两个坑：`bytes([i])`（**方括号**）才是造单字节，`bytes(65)` 是 65 个 `\x00`、`bytes('A')` 直接报错；把词转字节要用 `list(token.encode("utf-8"))`，**`[ord(c) for c in token]` 得到的是码点不是 UTF-8 字节**（非 ASCII 就错，见 §1.1）。

**特殊 token**：作业里只用 `<|endoftext|>`。共同本质是「人为定义、**永不被拆开**、占固定 ID 但不参与 merge」。

> ⚠️ **ID 不通用**（GPT-2 放末尾、Llama 放开头），放哪是设计选择——看 `tests/fixtures/train-bpe-reference-vocab.json` 的排法定。
> **实现提示**：接口是 `list[str]`，**别写死"只有一个"**（`test_train_bpe_special_tokens` 测多个）。

### 2.2 Pre-tokenization

**不合并，只准备"片段→次数"频率表。** 数据流（顺序重要）：
```
原始语料
→ 先按 special_tokens split（re.escape，因 token 含 | 等正则元字符）
→ 每段用 GPT-2 正则切 pre-token
→ 每个 pre-token 转 UTF-8 字节序列 → 统计次数
```
产物（例）：`(l,o,w):5, (l,o,w,e,r):2, (w,i,d,e,s,t):3, (n,e,w,e,s,t):6`

**为什么需要**（思想实验：不预分词直接把文本当连续字节流，会数到跨词边界的对）：
1. **语义割裂**：`dog` `dog!` `dog.` `dog?` 变成 4 个毫不相干的 ID，语义几乎一样却各学一份，浪费词表。
2. **跨词无意义组合**：`e␣t`（横跨两词）这种统计噪声占词表却学不到东西。
3. **效率灾难**：不预分词 = 每轮数"语料字节数"个对；预分词后 `the` 出现 100 万次只**存一份 + 记次数** → 工作量从"语料字节数"降到"**不同词的个数**"，小几个数量级。

> 切的边界主要是**空格**（+标点+数字/字母切换+缩写）；切出的片段**不是最终 token**，只是**合并的作用范围**，片段内部仍被 BPE 继续切成子词。

**GPT-2 正则**（需 `regex` 包，标准库 `re` 不支持 `\p{L}`）：
```
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
6 个用 `|` 连的选项，**从左到右优先匹配**：① 英文缩写后缀(`'s/'ll/'ve/'re`) ② ` ?\p{L}+` 词 ③ ` ?\p{N}+` 数字 ④ ` ?[^\s\p{L}\p{N}]+` 标点 ⑤ `\s+(?!\S)` 尾随空白 ⑥ `\s+` 兜底。

- **前导空格吸附**（选项②的 ` ?`）：空格被吸到词**前面** → `some text` 的 `text` 切成 ` text`。这就是为什么 GPT-2 系 ` the` 和 `the` 是两个 token。
- **选项⑤必须在⑥前**：`|` 从左到右优先，具体规则放前、兜底放后。
- 例：`re.findall(PAT, "I'll pay $100 now!!")` → `['I', "'ll", ' pay', ' $', '100', ' now', '!!']`

**预分词里修过的两个 bug —— 关键是 `pytest` 全绿也抓不到它们**：

| bug | 触发条件 | 后果 | 修法 |
|---|---|---|---|
| `re.split("", chunk)` | `special_tokens=[]` | pattern 为空串 ⇒ **在每个字符之间都切一刀**，pre-token 全被打成单字符 | `re.split(pat, chunk) if pat else [chunk]` |
| 分块硬编码 `b"<\|endoftext\|>"` | 特殊 token 不叫这个名字 | 找不到切点 ⇒ 边界全退化到 EOF ⇒ **32 进程变 1 个**，并行全失效（实测 `<\|sep\|>` 语料：修前 1 块 / 修后 32 块） | 用 `special_tokens[0].encode()`；无特殊 token 时单块 |

第一条后来在 `encode` 里又以另一种形式出现（`special_split_pat` 为空时要置 `None`，见 §2.9）——**同一个空集合边界，两处都要防。**

**演进**：这套正则至今是主流骨架。GPT-4 的 `cl100k_base` 最重要的改动是**数字限 `\p{N}{1,3}`**（`12345`→`123`+`45`，算术更稳、词表不被撑爆）。各家的分词算法对比见 §2.11.1。

### 2.3 Merge loop 算法

每轮 4 步，循环到词表满：
1. **数所有相邻对（加权）**：遍历频率表，数每个 pre-token 内相邻对 × 该 pre-token 次数。
2. **选最高频对**：平局选**字典序更大**的（`max` 对 `tuple[bytes,bytes]` 的默认行为）。
3. **记录**：新 token 加入 vocab（下一个 ID）+ 该对追加进 merges（**顺序重要**，encode 按此顺序）。
4. **替换**：所有 pre-token 内该对 → 合并成单 token。回到 1。

**逐帧例子**（`(l,o,w):5 (l,o,w,e,r):2 (w,i,d,e,s,t):3 (n,e,w,e,s,t):6`）：

| 轮 | 数对（关键项） | 选谁 | 为什么 | 替换后 |
|---|---|---|---|---|
| 1 | `(e,s):9 (s,t):9 (w,e):8 (l,o):7 (o,w):7` | `(s,t)` | 和 `(e,s)` 平局 → 字典序 `s`(0x73)>`e`(0x65) | `(w,i,d,e,st) (n,e,w,e,st)` |
| 2 | **`(e,st):9`** ← 新冒出来的 | `(e,st)` | 最高频 | `(w,i,d,est) (n,e,w,est)` |
| 3 | `(l,o):7 (o,w):7` | `(o,w)` | 平局 → `o`>`l` | `(l,ow) (l,ow,e,r)` |

⇒ `merges=[(s,t),(e,st),(o,w)]`，`vocab` 加了 `b'st'`, `b'est'`, `b'ow'`。

**两个核心理解**：
- loop 就是**反复修改同一张频率表**：扫表→选最高频→记录→就地合并→用改动后的表再来一遍。
- **为什么必须串行**：第 2 轮的 `(e,st)` 是第 1 轮合并的**产物**，第 1 轮之前根本不存在 → 无法提前知道第 2 轮选谁。**这条依赖链无法打破**（并行的讨论见 §2.6）。

### 2.4 高效实现：四个结构 + 三项技术

朴素实现每轮全量重数 pairs、重建所有词，是 $O(\text{轮数} \times \text{语料})$。当前实现靠四个持久结构协同做**增量维护**：

| 结构 | 类型 | 作用 |
|---|---|---|
| `global_vocab_ids` | `list[list[int]]` | 每词的 id 序列，**原地改**（下标=词标识，必须稳定） |
| `global_vocab_counts` | `list[int]` | 每词出现次数（下标对齐） |
| `global_pair_counts` | `Counter` | 每个 pair 的全局加权计数（**真相源**） |
| `global_pair_to_words` | `dict[pair, set[int]]` | pair → 含它的词下标集合 ← **增量的关键索引** |
| `global_word_pairs` | `list[dict[pair,int]]` | 每词内部 pair→次数（常驻，判断成员关系用） |

每轮：从堆取 best_pair → **只遍历 `pair_to_words[best_pair]` 的词**（不是所有词）→ 每词重建序列并只更新变化的 pair。

**技术 1：delta 更新（只动计数真正变化的 pair）**

词 `[a,b,c,d,e]`（count=5），合并 `(c,d)→X`：

| 对 | 旧 | 新 | delta | 要动吗 |
|---|---|---|---|---|
| `(a,b)` | 1 | 1 | **0** | ❌ 跳过（远离合并点） |
| `(b,c)` `(c,d)` `(d,e)` | 1 | 0 | −1 | ✅ 计数 −5 + push |
| `(b,X)` `(X,e)` | 0 | 1 | +1 | ✅ 计数 +5 + push |

- **为什么 delta 一定对**：一个对的全局计数 = 各词贡献之和。这词合并前贡献 `old*count`、合并后 `new*count` → 净变化 `(new−old)*count`。只加这个 delta，全局就对。
- `delta==0` 的 pair 计数没变 ⇒ 堆里那条旧条目**依然有效** ⇒ 不用 push，不变量不破。
- 验证 `A B A B` 合并 `(A,B)`：old=`{(A,B):2,(B,A):1}`, new=`{(X,X):1}` → 全对上。
- **通用性**：这是「局部改动、大部分不变」类增量维护的通法。

> **增量维护的记账纪律**（四条都是被 bug 逼出来的，静态查不出来）：
> `if pair_counts[pair] <= 0: del` 清 0 计数对，否则会选到僵尸对 ｜ `affected` 必须 `list(...)` 拷贝，循环体里会改这个 set ｜ 用 `.discard()` 不用 `.remove()`，同词可能多次贡献同一 pair（`A B A B`）｜ 用 `pop(k, None)` 不用 `del`，key 可能已在减计数时被删过。

**技术 2：懒删除堆（通用技巧，值得单独记）**

问题：`heapq` **不支持高效修改堆里已存在元素的优先级**（要 O(n) 扫描找到它），但 BPE 里 pair 计数频繁变化。

解法：**不改旧的，只 push 新的；旧的留着变 stale，弹出时验证 + 丢弃。**

```python
def push(self, pair, count):                    # 从不扫描堆，只做 sift-up，O(log n)
    heapq.heappush(self.heap, (-count, tie_break_key, pair))

def pop_best(self, pair_counts):                # 和「真相源」核对
    while self.heap:
        neg_count, _, _, pair = heapq.heappop(self.heap)
        if -neg_count == pair_counts.get(pair, 0) and -neg_count > 0:
            return pair                          # 堆值 == 真值 → 有效
    return None                                  # 否则是 stale，丢弃继续弹
```

- push 一个已存在的 pair 会留下**重复条目**（旧 stale + 新有效），这是**故意的**，不是 bug。
- **真相源在别处**：`pair_counts` 才是权威，堆只是「大概按优先级排」的加速结构。
- **正确性靠两条不变量**：① **无遗漏**——任何 count>0 的 pair 堆里至少有一条携带其当前 count 的条目（靠"每次计数变动都紧跟一次 push"保证）；② **可鉴别**——key 是 `(pair,count)` 的纯函数，对不上就是 stale。⇒ 第一个通过校验的弹出结果必是真正最优。
- **膨胀治理**：堆只增不减，`log n` 的 n 是**堆长度**。`maybe_rebuild` 在堆长 > `2×存活+1024` 时按当前 counts 重建。
- **适用场景（记住它）**：任何「优先队列 + 元素优先级频繁变化」的场合（Dijkstra 变体、事件调度、任务队列）。标准库堆不支持 decrease-key，懒删除是最常用的绕过法。**判据：修改频繁 + 有独立的「真相源」能在 pop 时验证。**

**技术 3：min-heap 模拟 max + tie-break 反转**

规则是「频率高优先，平局取字节字典序**大**」，但 `heapq` 是 min-heap：
- 频率：存 `-count`。
- 字节：需要反转字典序 → `neg_bytes(b) = tuple(-x for x in b) + (1,)`。

> ⚠️ **哨兵 `(1,)` 是整个实现最大的坑**。`tuple(-x for x in b)` 反转了字节值，但**元组比较里「前缀相同时短的更小」这条长度规则没被反转** → `b'b'` vs `b'bc'` 选错。加正数哨兵后，更长的串在对应位补负数 → min 里更小 → 复现「长的更大」。
> 哨兵必须是**正数**：`0` 会和 `neg_bytes(b'\x00')` 撞；`b"\xff"` 补码方案也撞（`\x00` 的补码就是 `\xff`）。
> 穷举验证（含 `\x00`/`\xff` 的 1~3 字节串，66306 组比较）：naive 版 **936 组违反**反序，哨兵版 **0 违反**。

### 2.5 优化记录（一张表）

| # | 改动 | 做了什么 | 测量场景 | 耗时 | 加速 |
|---|---|---|---|---|---|
| — | 朴素串行 | 每轮全量重数 pairs + 重建所有词 | pytest 全套 | ~16s | 1× |
| 1 | 并行预分词 + 微优化 | `Pool` 多进程切块；`len` 提出循环；`if id_a not in token_id: continue` 跳过不含该 pair 的词 | pytest 全套 | ~5s | 3.2× |
| 2 | 增量更新 merge loop | 引入 `pair_counts` + `pair_to_words` 持久结构，只遍历含 best_pair 的词，不再全量重数 | pytest 全套 | **0.87s** | **18×** |
| 3 | 懒删除堆 | `max()` 每轮全扫几万个 pair → `heapq` O(log n) 弹出 + stale 校验 | TS valid, vocab 10k | 14.3s → **1.46s** | **9.8×** |
| 4 | `neg_bytes` 按 id 缓存 | id→bytes 只增不改，缓存永不失效，最多算 vocab_size 次 | 同上 | **0.89s** | 1.64× |
| 5 | delta 更新 | 只动计数真正变化的 pair（见 §2.4 技术 1）；push 次数 4549→2274 | 同上 | **0.54s** | 1.66× |
| 6 | 堆定期重建 | 堆长 > `2×存活+1024` 时按当前 counts 重建，清 stale；峰值堆长 2505→1272 | 同上 | **0.42s** | 1.28× |
| 7 | `findall` + `Counter.update` | 预分词计数从 Python 循环 `+= 1` 换成 C 里的 `_count_elements`；`PAT` 模块级预编译 | TS train 2.23GB | 12.58s → **7.55s** | 1.67× |
| 8 | 分块数 = 进程数 × 4 | 文档长度不均导致 worker 空转（实测最慢 79.6ms vs 平均 58ms），多切块让快 worker 多领任务 | 同上 | **7.17s** | 1.05× |
| 9 | `gc.disable()` | merge loop 期间关分代 GC（`try/finally` 恢复），见 §2.7 教训 1 | owt_valid, vocab 32k | 42.75s → **28.21s** | **1.52×** |
| 10 | 邻域增量 + per-word 计数常驻 | 一次走完序列就地记录 delta，不再重建两个整词 `Counter`；常驻的 per-word pair 计数用来判断成员关系 | 同上 | **19.93s** | **1.42×** |

**目标语料确认**（owt_train 11.9GB + vocab 32000）：merge **544.95s → 236.54s = 2.30×**，端到端 585.8s → 278.1s。

> ⚠️ **这 2.30× 只归因于 #9 + #10**，不是 7~10 的合力：
> - #7/#8 是**预分词**的改动，物理上碰不到 merge 时间，而且在 544.95s 那次跑之前就已生效（两次跑的 pretok 分别是 40.84s / 41.53s，基本没动）。
> - 即基线代码状态 = #1–#8 已应用，优化后 = #1–#10。
> - 旁证：同样这两条在迭代语料 owt_valid 上是 42.75 → 28.21 → 19.93 = **2.15×**，与 2.30× 对得上（**兑现率 107%**）。
>
> 各阶段的加速倍数不能直接相乘——每条都是在**当时的代码状态**上测的，而且前面的优化会削弱后面的收益（见 §2.7 教训 4）。

**被否决的尝试**（同样测了、同样对拍通过，但不采纳）：

| 改动 | 结果 | 为什么 |
|---|---|---|
| heap key 后缀按 pair 缓存 | 20.02s（0%） | 第 10 条已经大幅减少 push，heap 不再是热点 |
| rebuild 阈值放宽到 `8×+1M` | 27.79s（−28%） | 堆变长直接推高每次 push/pop 的比较成本 |
| 完全不 rebuild | 51.82s（−62%） | 同上，极端情形 |
| rebuild 阈值收紧到 `1.25×` | 22.63s（−13%） | 重建太频繁，每次 O(存活) |
| rebuild 阈值 `3×`（扫出的最优） | 19.57s（+1.8%） | 低于 5% 采纳阈值；曲线在 2~4× 间很平，现值已在最优区间 |

### 2.6 并行性分析（结论随规模翻转）

| 阶段 | 能并行 | 说明 |
|---|---|---|
| Pre-tokenization | ✅ 数据并行 | `multiprocessing` 绕开 GIL，按 special token 切块。**价值全在这**——它随数据量线性增长 |
| Merge loop | ⚠️ **看规模** | 每轮要「全局决策 + 依赖上轮结果」，只能在**一轮之内**并行（把 affected 的词分给多个 worker） |
| GPU / 线程 | ❌ | GPU：符号处理不适合稠密数值并行。线程：CPython 3.12 有 GIL，循环体是 100% Python 字节码 |

**划不划算完全取决于每轮粒度**（对照：派发 32 个空任务 ≈ 300µs）：

| 规模 | unique pre-token | 每轮 word 循环（中位数） | 结论 |
|---|---|---|---|
| TinyStories valid + vocab 10k | 6 万 | **3.5 µs** — 开销贵 85 倍 | **绝无可能** |
| owt_train + vocab 32k | **660 万** | 0.3 ms（均值 20.5ms）— 开销占 1.5% | **值得算** |

**正确形状是永久分片**：worker k 常驻持有 `words[k::N]` 和自己那份局部 `pair_to_words`，父进程每轮只广播 `(id_a, id_b, new_id)`（几十字节），worker 回传局部 delta。而不是每轮把 `ids[wi]` pickle 进 worker——实测一轮最多要搬 **714 万个 id**，搬运量正比于工作量。

**但上限只有 2.82×**（优化后重测：worker 侧 67% / parent 侧 `pair_counts`+heap **28%** ← 硬地板 / `pop_best` 5%）。优化前是 3.92×——**#10 砍掉的正是可并行的那部分，上限跟着掉**。要破 28% 的地板得连堆也分片，复杂度和 tie-break 风险都上一档。

**通信开销固定 ≈ 9.5s**（31743 轮 × 一次 round-trip）：owt_valid（loop 才 20.8s）净收益仅 1.2× **不值得**；owt_train（loop 236.5s、轮数相同）只占 4% ⇒ 约 2.5×。
⇒ **同一份并行代码在两个语料上一个赚一个赔。轮数固定而每轮工作量随语料涨，是并行在这里能成立的唯一原因。**

### 2.7 优化方法论（五条教训）

**流程**（每条优化都走一遍，不跳步）：定基线（3 次中位数 + 存下 merges 作正确性基准）→ profile 找热点 → **一次只改一条** → 对拍 merges 逐条相同（不同就直接回滚，不 debug 性能）→ 测时 → 记账 → 收益 <5% 就回滚 → 全做完在**目标语料**确认。

**1. 测量规模决定结论，甚至能让结论反转。**
同一个改动：懒删除堆在 corpus.en/vocab500 上和无堆**差不多甚至略慢**（维护开销 ≈ 省下的扫描），在 valid/vocab10000 上是 9.8×。预分词的 `findall` 改动单进程 1.54×、32 进程只有 1.29×（固定开销占比变大）、2.23GB 上才回到 1.67×。
⇒ **两级语料**：迭代语料要快（几十秒一轮）**且保留目标语料的特征**。owt_valid(290MB) 合格；TinyStories **不合格**——它 `affected` 中位数是 1，所有和"每轮粒度"相关的优化在上面全测不出来。

**2. GC 是隐藏的大头，而且会把正确的优化伪装成负收益。**
第 10 条优化单独上、GC 开着时是 **42.75 → 54.16s，看起来是 −27% 的负优化**，差点直接回滚。关掉 GC 才发现它其实是 28.21 → 19.22 = **+47%**。
- 原因：merge loop 每轮分配几百万个小容器（新序列、per-word dict、heap key 元组），它们**不构成引用环**、靠引用计数就能回收，但分代 GC 仍会在每次 gen-2 扫描时遍历整个存活集合——而这个集合正好被常驻 dict 撑大了。**分配得越多，GC 罚得越狠。**
- ⇒ **任何"多分配换少计算"的优化，评估前必须先把 GC 排除掉**，否则测到的是 GC 的账不是你的账。

**3. "空间换时间"不是无条件成立的，要看空间进了哪个数据结构。**
同一轮里两条都是花内存，结果相反：per-word 计数常驻（+2~3GB）**快 1.42×**；让堆无限增长（+1.3GB）**慢 2.6×**。
区别在于：dict 是 **O(1)** 查找，多占内存不影响单次操作成本；堆是 **O(log n)**，n 就是堆长度 ⇒ **给对数级结构塞空间是自伤**。

**4. 每改一条就要重新 profile，优化清单会被上一条优化作废。**
"heap key 缓存"是照着改第 10 条**之前**的 profile 定的（那时 `_neg` 被调 4080 万次）。第 10 条减少了 touched pair、连带减少 push，heap 早已不是热点 → 实测 0 收益。

**5. 微基准和整体跑可能给出相反的结论。**
第 10 条的微基准是**每词快 3.5×**，整体却慢 27%——差异全在 GC 这种全局效应上。**微基准只能解释局部，不能拍板。**

**profile**：`cProfile.run(..., 'prof')` + `pstats.Stats('prof').sort_stats('tottime')`。看 **tottime**（函数自身耗时，找瓶颈看它）和 **ncalls**（几百万次=热点）。
实战：profile 直接指出 `max` 7.3s + `rank` 6.6s（被调 **1 亿次**）= 几乎全部时间 → 指向"用堆替代每轮全扫"；后来又指出 667 万次 `Counter` 构造 → 指向邻域增量。**先 profile 再优化，别猜。**

**怎么验证优化没改坏结果**（计数漂移这类 bug 静态看不出来，唯一可靠办法是对拍）：
1. **差分测试**：写一个每轮全量重算的 brute-force 版，逐条比 merges + vocab。语料要**专门造刁钻的**——大量 pair 同频的 tie 语料（测 tie-break）、`aaaa` 重叠串（测贪心从左到右 + 重叠记账）、中文/带音标拉丁（测多字节）、随机串。
2. **不变量断言**：每步 merge 后从 `vocab_ids` 从零重算真值，校验 `pair_counts` 没漂移、`pair_to_words` 与真值相等、**每个存活 pair 都有一条携带当前 count 的堆条目**（这条是懒删除正确性的命根子）。
3. **官方测试** `pytest tests/test_train_bpe.py`。

### 2.8 训练产物：两个 tokenizer（2026-07-25）

跑法：`python cs336_basics/main_bpe_train.py [tinystories|owt]`，产物落在 `data/<name>_{vocab,merges}.json`。

| | **TinyStories** | **OpenWebText** |
|---|---|---|
| 语料 | `TinyStoriesV2-GPT4-train.txt` 2.23 GB | `owt_train.txt` 11.92 GB |
| `vocab_size` / merges | 10,000 / 9,743 | 32,000 / 31,743 |
| 分块边界 | 0.0003 s | 0.0006 s |
| 预分词 | 6.99 s | 41.37 s |
| **merge loop** | **1.07 s** | **213.72 s** |
| **总耗时** | **8.3 s** | **295.6 s（4.9 min）** |
| 峰值内存 | 0.2 GB | 14.0 GB |
| handout 限制 | ≤30 min / ≤30 GB ✅ | ≤12 h / ≤100 GB ✅ |
| unique pre-token | 6 万 | **660 万** |
| **compression ratio**（训练语料） | **4.071 bytes/token** | **4.363 bytes/token** |
| 平均 token 长度 | 5.79 B（中位 6） | 6.34 B（中位 6） |
| **最长 token** | `b' accomplishment'`（15 B） | 64 B 的**乱码**（见下） |
| 落盘大小 | 0.2 + 0.2 MB | 0.6 + 0.5 MB |

**读数**：owt 的 4.363 比 TinyStories 的 4.071 高 7%——词表大 3.2 倍只换来这点压缩率。原因有两层：① 边际收益递减，BPE 先合并的都是最高频模式，后面 2.2 万个格子吃的是长尾；② owt 的长尾里有相当比例是乱码和分隔线（见下），这些 token 覆盖的文本占比极低，对整体压缩率几乎没贡献。**词表翻倍 ≠ 压缩率翻倍**，这是 leaderboard 上选 vocab_size 时要记住的。

> ⚠️ 这个 compression ratio 是**训练语料全量**的，不是 handout §2.7(a) 要交的那个（那个是各采样 10 篇文档，且 (b) 还要交叉编码），也不含被 split 掉的 `<|endoftext|>` 分隔符字节。
> **但它是 `encode` 的现成验收标准**：写完 `encode` 后在训练语料上跑，bytes/token 应该和这个数很接近，对不上就是 `encode` 有 bug。免费的对拍手段，不用另写 brute force。

**它为什么是免费的**：`_merge_loop` 结束时，`word_ids[i]` 是 pre-token i 依次施加 merge 1..N 之后的序列——而 `Tokenizer.encode` 的 Step 2 做的也是依次施加 merge 1..N。**训练结束时整个语料已经被编码过一遍了**，只需在建索引时额外记下每词的原始字节数（`word_nbytes`，一个 int）。注意要用 `count` 加权：频率表里 `the` 只存一份但语料里出现上百万次。

#### handout §2.5 的「最长 token 合理吗」——答案是不合理

owt 最长的 10 个 token **一个真词都没有**：

```
 64B  'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'
 64B  '----------------------------------------------------------------'
 48B  '————————————————'
 32B  '________________________________'   '================================'
 32B  '................................'   '********************************'
```

那个 64 字节的是 `b'\xc3\x83\xc3\x82'` 重复 16 次，即 **double-encoded UTF-8 mojibake**（重复编码乱码）：一段 UTF-8 字节被误当 latin-1 解码、再按 UTF-8 编回去，每错一轮就多一层 `ÃÂ`。比如不换行空格 U+00A0 的 UTF-8 是 `\xc2\xa0` → 误读成 latin-1 得 `'Â '` → 再编码成 `\xc3\x82\xc2\xa0`。64 字节说明同一段文本在爬取管线里被**错误转码了很多轮**。

对照 TinyStories 最长的 10 个，全是规矩英文词：` accomplishment` / ` disappointment` / ` responsibility` / ` uncomfortable` / ` compassionate` …

**结论**：
- TinyStories 最长 token **合理**——GPT-4 生成的干净儿童故事，最长 token 自然是最长的常用词。
- OWT 最长 token **不合理**——它反映的是网页语料的编码损坏和排版分隔线，不是语言结构。**算法没错，是数据脏**：这些字节序列确实高频，BPE 忠实地学了。
- 这就是 handout §2.5(b)「对比两个 tokenizer」的答案：干净合成语料 vs 脏爬取语料。延伸问题——**词表里有多少格子被垃圾占掉了？**这直接吃掉 compression ratio 和 leaderboard 的 token 预算。真实管线会先做数据清洗（去重、编码修复、过滤），正是后面 data 那个 assignment 的主题。

#### 序列化：为什么是 JSON + latin-1

`store_tokenizer` / `load_tokenizer` 落盘 `vocab.json` + `merges.json`。**测试不碰 `from_files`，格式完全自选**，但这几个决定值得记：

**bytes 塞不进文本格式**——实测 `vocab_size=300` 的词表里 **128/300 个 token 不是合法 UTF-8**（多字节字符的续接字节，单独拿出来非法），`b.decode("utf-8")` 直接崩。三种解法：

| 方案 | 谁在用 | 做法 |
|---|---|---|
| byte→unicode 映射 | GPT-2 的 `vocab.json`/`merges.txt` | 256 个字节各映射到一个**可打印**字符（空格→`Ġ`、换行→`Ċ`）。**这个 trick 是为了迁就 `merges.txt` 用空格分隔两个 token 的文本格式**——不映射的话 token 本身是空格就歧义了 |
| **latin-1**（本项目） | — | 字节 0–255 一一映射到码点 U+0000–U+00FF ⇒ `b.decode("latin-1").encode("latin-1") == b` 对**任意**字节串成立。merges 存 JSON 数组而非空格分隔，所以不需要 GPT-2 那套 trick |
| base64 | tiktoken / Llama 3 | 最保险，代价是不可读 |

**三层别搞混**（以 id=195 为例）：token 是 1 个字节(值 195) → JSON 里是 1 个字符(码点 195, `'Ã'`) → **磁盘上是 2 个字节**(`C3 83`，因为文件按 UTF-8 写)。所以 ≥128 的字节在磁盘上都膨胀一倍。

**为什么不用 `.pt` / pickle**——实测反而是 pickle 最省最快：

| 格式 | 大小 | 写 | 读 |
|---|---|---|---|
| `vocab.json`(latin-1) | 172.7 KB | 2.8 ms | 1.7 ms |
| **`vocab.pkl`** | **114.7 KB** | **0.4 ms** | **0.3 ms** |
| `vocab.pt`(torch.save) | 340.0 KB | 16.6 ms | 4.3 ms |

⇒ **选 JSON 是在效率上主动认输**。理由是别的三条：① **安全**——pickle 反序列化即执行任意代码，而 tokenizer 是从网上下载的产物（PyTorch 2.6 把 `torch.load` 默认改成 `weights_only=True` 就是为这个）；② **可移植**——读它的常常不是 Python（HF `tokenizers` 是 Rust、llama.cpp 是 C++、transformers.js 是 JS）；③ **可读可 diff**，词表是要人眼检查的东西。

> **先做尺度检查**：172 KB、一个进程只加载一次 ⇒ 大小和速度这两个维度上**所有选项都远低于痛阈**，那就不该拿它们做决策依据。
> **通则：配置类产物用文本格式，二进制只留给真正大的数值数据。** 同一个作业里两种数据两种答案——vocab/merges 用 JSON（KB 级、要人看），token 数组用 uint16 裸二进制（GB 级、机器随机读，见 §2.10）。

### 2.9 Tokenizer：encode / decode（handout §2.6）✅ 15 分

`pytest tests/test_tokenizer.py` → **24 passed, 1 xfailed**（xfail 是测试自己标的：`encode` 不要求省内存）。

> ⚠️ **绿灯 ≠ 证明**：`test_encode_iterable_memory_usage` 的 `@memory_limit(1MB)` 装饰器在 `finally` 里就把 rlimit 还原了，而被它装饰的 `_encode_iterable` 是**生成器函数**——调用只创建生成器、不执行，等真正迭代时限制早撤了（实测：普通函数里 rlimit=14MB 生效，生成器迭代时是「无限制」）。所以那个 PASSED 的证明力比看起来弱；真正的依据是**结构本身**（`yield` + `buf` 只留未结算的尾巴）和自建对拍。**测试测的是它写下的那个条件，不一定是你以为的那个条件。**

**`__init__` 建四张表**——传进来的 `vocab`/`merges` 都答不了 encode 要问的问题：

| 表 | 类型 | 为什么 |
|---|---|---|
| `inverse_vocab` | `dict[bytes, int]` | `vocab` 方向反了；合并操作 bytes，输出要 ID |
| `merge_rank` | `dict[pair, int]` | list 查"排第几"是 O(n)；rank = 在 `merges` 里的下标 |
| `cached_word_ids` | `dict[str, list[int]]` | pre-token 记忆化。key 用 `str`，命中时省一次 `.encode()` |
| `special_split_pat` | 编译的正则 | 依赖 `self.special_tokens`，不能放模块级 |

坑：special token 要追加进 vocab，且**必须同步更新 `inverse_vocab`**；`special_tokens` 参数即使 vocab 全有也得传——`vocab` 只是 `id→bytes`，**没标记谁是特殊的**。

**`encode` 三段**：① `special_split_pat` 切开并保留 → special 直接查表；② 普通段过 `PAT`；③ 每个 pre-token 单独合并。

> ⚠️ ① 必须独立，**不能把 special 塞进 `PAT` 当 `or` 分支**（tiktoken 裁判实测）：
> `'x <|endoftext|> y'` 合进 PAT 得 `['x', ' <|', 'endoftext', '|>', ' y']` —— special token 被切碎。
> 因为交替是逐位置试：扫到 `<` 前的**空格**时 special 分支不匹配，` ?[^\s\p{L}\p{N}]+` 贪心吃掉 `' <|'`。**`or` 救不了。**
> `special_split_pat` 三要点：`re.escape` + 外层捕获组（`re.split` 才保留分隔符）+ **按长度降序**（交替是从左到右优先，不是最长匹配）。

**`_merge_pretoken`**：拆成单字节 → 循环{挑 `merge_rank` 最小的相邻对，合并其所有出现} → 挑不到退出。
- 与"从头遍历所有 merge"**等价**（实测 277 个 pre-token 零差异），但**快 119 倍**。等价依据：训练时 `new_id = len(vocab)` ⇒ 用到 `X` 的 merge，rank 必大于造出 `X` 的那条；而合并后新冒出的对只可能含 `X` ⇒ 都在前方。
- 成本：遍历法固定 5 万次，挑最小法 ≤ 词长−1 ≈ **3 次**（加权平均 pre-token 4.22 字节）。
- **不要用堆**：候选只 ~3 个且每轮只取一次最小值，实测 `min()` 在 3~2000 各规模都比 `heapify+pop` 快 2 倍多。对照训练那边堆赢 9.8×——**同一优化规模不同结论相反**，见 §2.7 教训 1。

**`encode_iterable` 三条不变量**（5MB 文件 / 1MB 内存，且须与整体 `encode` 逐 token 相同）：

| # | 危险 | 守法 |
|---|---|---|
| 1 | special token 横跨切点，`split` 只看到半截 | 结算上限 `limit = len(buf) − (S−1)`（S=最长 special 长度）；若仍落在某个 special 匹配内部，**再退到它的 `start()`** |
| 2 | 最后一个 PAT 匹配可能还没吃完，`\s+(?!\S)` 要看后一字符 | **扣住最后一个匹配**。只扣最后一个文本段——被 special 夹住的段是硬边界 |
| 3 | ⚠️ **吐已算好的 token，不能拿结算前缀重跑 PAT** | 最短反例 `'\n\n!!'`：整体 `['\n','\n','!!']`，但 `findall('\n\n')` 单独跑时前瞻在 EOS 成功 → `['\n\n']`。**我在这栽了两次** |

实现上用 `itertools.chain(iterable, [None])` 加哨兵、延迟一拍，才知道当前块是不是最后一块。
「只在遇到 special token 时才结算」不行：roundtrip 测试**没传 special_tokens**，没有刷新点，缓冲区会涨满整个文件。
已知局限：一个巨大的连续空白串整个是一个匹配，会全堆在 `buf` 里。

**`decode`**：`b"".join(...).decode("utf-8", errors="replace")`。**先拼接再一次性解码**——merge 可能落在多字节字符中间，逐 token 解会各自失败变成两个 `�`。

**验证**：官方测试（含一批 `matches_tiktoken`）+ 自建对拍器（1011 刁钻用例 × 3 种 special 配置 × 5 种切法，含 chunk=1 和逐行）。**三条不变量是被反例逐个逼出来的**，每错一次就往语料里加一个。

### 2.10 §2.7 实验 & 数据落盘（4 分）✅

三个脚本：`main_bpe_train.py`（训 tokenizer）→ `main_tokenizer_experiments.py`（(a)(b)(c)）→ `main_tokenize_dataset.py`（(d) 落盘）。

**(a)(b) compression ratio（bytes/token，越大越好）**

| 样本 | 用哪个 tokenizer | bytes | tokens | bytes/token |
|---|---|---|---|---|
| TinyStories 10 篇 | TinyStories (10k) | 7,633 | 1,861 | 4.102 |
| OWT 10 篇 | OWT (32k) | 60,084 | 13,586 | **4.422** |
| OWT 10 篇 | **TinyStories (10k)** ← 交叉 | 60,084 | **17,952** | **3.347** |

只有后两行可比（同一份 OWT 样本）；第一行是另一份样本，只能横向看比值。
⚠️ 样本量差很多：10 篇儿童故事才 **7.6 KB**，10 篇网页有 **60 KB**——小样本的 ratio 方差大，能和全量的 4.071 对上有运气成分。真要下结论看 §2.10 (d) 的全量数字（4.116 / 4.371）。

⇒ **用错 tokenizer，压缩率掉 24.3%、token 数多 32.1%。** TinyStories 的词表学自 GPT-4 生成的儿童故事，没有网页文本的词汇（URL、技术术语、非英文、排版符号），OWT 只能被切得更碎。
**这条有真实代价**：leaderboard 是固定 45 分钟算力预算，token 效率低 32% ≈ 模型少看 32% 的文本；推理时 context window 也更快耗尽。**和 §2.11.3「多语言不公平」是同一个机制**——小语种被切碎，本质就是拿一个没覆盖这类文本的词表去编码。

**(c) 吞吐（单进程）**

handout 拿 **The Pile**（825 GB，见 §0.2）当尺度参照物——不是真去跑，是算一笔账：*这个 tokenizer 拿到真实规模的语料上要多久？* 慢 100 倍的实现（比如 `encode` 遍历全部 merge）同样的账就是 90 天，预处理直接变成工程阻塞。

| | 单进程 MB/s | 缓存命中率 | Pile 825GB 单进程 |
|---|---|---|---|
| TinyStories | **20.7** | 99.75%（每 401 次出现才真算 1 次） | 11.1 h |
| OWT | **10.6** | 96.93%（每 32.5 次） | 21.5 h |

⚠️ **别拿单进程 ÷ 32 估并行**。实测 32 进程是 **187.2 MB/s**（见下 (d) 的 owt_train），加速比只有 **17.6×** 不是 32×（效率 55%，损耗在 IO、pickle 回传、父进程拼接、负载不均）。⇒ Pile 的实际估算是 **1.2 小时**，不是 ÷32 得到的 0.67 小时。

⇒ **吞吐由缓存命中率决定，不由词表大小决定。** 同样 20MB，OWT 有 13.5 万个不同 pre-token（TinyStories 才 1.3 万），`_merge_pretoken` 被真正调用的次数多 10 倍。直觉会归因于"词表大 3 倍所以慢"，实际是**语料多样性**。

**(d) 落盘产物**（`data/<split>.npy`，32 进程）

| split | 语料 | 耗时 | token 数 | bytes/token | 文件 |
|---|---|---|---|---|---|
| tinystories_train | 2.23 GB | 8.5 s (261 MB/s) | 5.41 亿 | 4.116 | 1.08 GB |
| tinystories_valid | 0.02 GB | 0.2 s | 546 万 | 4.117 | 0.01 GB |
| **owt_train** | **11.92 GB** | **63.7 s (187 MB/s)** | **27.27 亿** | 4.371 | **5.45 GB** |
| owt_valid | 0.29 GB | 2.5 s | 6640 万 | 4.367 | 0.13 GB |

**为什么 uint16**：范围 0–65535，装得下 32000 和 10000。换 uint32 文件直接大一倍（5.45→10.9 GB），而训练是用 `np.memmap` 随机采样的，**文件越小页缓存命中越好**。
⚠️ **条件是 `vocab_size ≤ 65536`**，落盘时应 assert。Llama 3（128k）、Qwen（15 万）这类词表就必须用 uint32。

**并行怎么做的**：`find_chunk_boundaries` 按 `<|endoftext|>` 切 `进程数×4` 块 → 各 worker `encode` 自己那块 → 父进程按块序拼接。
- **正确性**：边界落在 special token 起点，而它是硬边界，pre-token 和 merge 都不跨越 ⇒ 分块编码 ≡ 整体编码。**实测验证**：valid 上 546 万 token 与整体 `encode` 逐个相同，`decode` 回原文一字不差。
- **worker 回传 `np.uint16` 数组而不是 `list[int]`**：pickle 的是 2 字节/token 的连续 buffer，不是几千万个 Python int 对象。
- **tokenizer 建在模块级**：Linux 的 `Pool` 走 fork，全局对象 copy-on-write 免费继承；且 worker 跨任务复用同一对象，`cached_word_ids` 能累积（块数取 `×4` 就是为了摊薄冷启动）。

> **三条独立路径互相印证**：训练期 `word_ids` 算出 4.071 / 采样 10 篇 `encode` 算出 4.102 / 全量落盘算出 4.116。完全不同的代码路径收敛到同一个数 ⇒ **免费的正确性交叉验证**。方法论上比数字本身值钱：任何时候一个量能用两条路算，就该都算一遍。

### 2.11 分词的代价 & 去 tokenizer 化（延伸）

> 背景与延伸，和 §2 的实现无关。后续读「替换/去掉 tokenizer」方向的论文，笔记记在 §3.4。

#### 2.11.1 三种主流分词算法

| | **BPE** | **WordPiece** | **Unigram LM** |
|---|---|---|---|
| 代表 | GPT 系、Llama 3、Qwen | BERT | SentencePiece：Llama 1/2、Gemma |
| 训练时选谁 | 频率最高的对 `count(AB)` | 似然提升最大，实践上 ≈ `count(AB)/(count(A)·count(B))` | 反向：从大词表剪枝，去掉损失最小的 piece |
| **推理怎么切** | **重放 merge 序列**（有序） | **词表内最长前缀匹配**（trie） | **Viterbi 求全局最可能切分** |
| 交付物 | vocab + **merges** | **只要 vocab** | vocab + 每个 piece 的 score |
| 贪心？ | 是 | 是 | 否，全局最优 |
| OOV | byte-level 下不存在 | 需要 `[UNK]` | 需要回退 |

**为什么 WordPiece 能贪心而 BPE 不能**：两者把「编码」定义成了不同的东西——
- BPE：「**复现训练时发生过的那串操作**」⇒ 必然依赖顺序，必然要带着 merges 走。
- WordPiece：「**在词表里取最长**」⇒ 只依赖词表这个集合，与训练怎么得到它无关。

这是设计取舍，不是谁更聪明。BERT 的具体做法：按空白/标点切词 → 每个词取词表里最长前缀 → 剩余部分加 `##` 前缀继续 → 匹配不上就 `[UNK]`。

> ⚠️ **所以不能把 BPE 的 `encode` 改写成 trie 贪心**。反例：merges 只有 `rank0:(b,c)→bc`、`rank1:(a,b)→ab`，编码 `"abc"`——BPE 挑 rank 最小的 `(b,c)` 得 `[a, bc]`，trie 最长前缀得 `[ab, c]`。**结果不同。** 不是那个算法不好，是它和你训出来的这套 merges 不配套（见 §3.3 末条）。

#### 2.11.2 分词不是 Transformer 的缺陷，但有一条真实因果链

**分词发生在模型之外**：Transformer 拿到的就是一串整数 ID，不知道也不关心它怎么来的。换成 RNN / LSTM / Mamba，问题一模一样。所以这不是 Transformer 架构的缺陷。

但有一条因果链值得记住：

> attention 是 **O(n²)**，embedding 和最后的 softmax 是 **O(vocab)** ⇒ 分词是一道**压缩工序**，把序列长度压下来好让 O(n²) 付得起。

**它本身不产生任何语义价值。** 如果 attention 免费，直接喂字节即可——这正是 §3.4 那批工作的出发点。

#### 2.11.3 分词制造的真问题

| 问题 | 表现 |
|---|---|
| **算术差** | `12345` 的切法不稳定，学不到位值关系。GPT-4 的 `\p{N}{1,3}`（见 §2.2）就是为此打的补丁 |
| **字符级失明** | 「strawberry 里有几个 r」——模型没见过字符，只见过几个 token ID |
| **多语言不公平** | 小语种被切得稀碎，同一句话花几倍 token，又贵又差 |
| **提示词边界效应** | 末尾多一个空格就换一套 token，行为跟着变（前导空格吸附，见 §2.2） |
| **glitch token** | `SolidGoldMagikarp` 那类：在词表里但训练语料几乎没出现 ⇒ embedding 行从没被更新过 ⇒ 模型碰到就发疯（对应 §3.2「梯度只流回被用到的行」） |
| **词表格子被垃圾占** | owt 词表里最长的 10 个 token 全是乱码和分隔线（见 §2.8）；词表翻倍 ≠ 压缩率翻倍 |

**最要命的一条不是"问题"而是"约束"**：**encode 必须和训练时完全一致**。模型学的是「ID 47 后面常跟 ID 892」这种统计；训练用 BPE 的 rank 顺序切、推理改用贪心切，同一句话会变成一串不同的 ID，分布一偏模型立刻退化。这就是 `test_tokenizer.py` 里一堆 `test_*_matches_tiktoken` 要求逐 token 对上的原因——**tokenizer 不是"差不多就行"的组件，它是模型的一部分**。

#### 2.11.4 去 tokenizer 化的尝试（论文笔记待补）

思路都是「把压缩这一步交给模型自己学，而不是外挂一个固定词表」。SSM / 线性注意力让长序列变便宜之后，这条路更有希望。

| 工作 | 一句话 | 笔记 |
|---|---|---|
| **CANINE**（2021, Google） | 字符级编码器，下采样后再进 Transformer | 待补 |
| **ByT5**（2021, Google） | 直接吃 UTF-8 字节的 T5，无词表 | 待补 |
| **Charformer**（2021） | 学习式的子词切分（GBST），端到端可微 | 待补 |
| **MegaByte**（2023, Meta） | 固定大小 patch + 局部/全局两级 Transformer，让字节级序列可扩展 | 待补 |
| **Byte Latent Transformer**（2024, Meta） | 按**熵**动态划分 patch——信息量大的地方切细，平淡处切粗 | 待补 |

要盯的共同问题：① 序列变长后算力怎么摊（这是分词存在的唯一理由）；② 去掉词表后，前面那张表里的问题解决了几条、又新增了什么；③ 和固定词表相比，同算力下的实际质量。

### 2.12 进度 & 下一步

| handout | 分数 | 状态 |
|---|---|---|
| §2.4–2.5 `train_bpe` | 15 | ✅ 3/3（§2.5、§2.9） |
| §2.6 `Tokenizer` | 15 | ✅ 24 passed, 1 xfailed（§2.10） |
| §2.7 实验 | 4 | ✅ 见 §2.10；四个 `.npy` 已落盘 |

**§2 全部 34 分拿下。** 下一步是 handout §3 Transformer——`data/*.npy` 就是 handout §5 训练时 `np.memmap` 读的输入。

**记住 tokenizer 速度和 LM 训练速度无关**：训练读的是已落盘的 memmap 数组，tokenizer 不在那个循环里。影响训练的是它的**输出**——`vocab_size`（决定 embedding/LM head 参数量和 softmax 开销）和 **compression ratio**（决定同样 token 预算下模型看到多少文本，对 45 分钟的 leaderboard 是实打实的杠杆）。所有优化都保持了 merges 逐条相同 ⇒ 没改变输出 ⇒ 没影响训练。

**Rust + PyO3**（handout 认可 `cppyy`/`nanobind`/PyO3）：性能边际价值已经不大（训练 5 分钟一次性），但学习价值高——算法已调对可原样搬、接口窄、有现成对拍工具，后续 assignment（Triton kernel、数据管线）一直用得上。

**数据探查用命令行**（比写 Python 快）：`grep -c/-oF`、`wc -l/-c`、`stat -c %s`。实测 TinyStories valid 有 **27630** 个 `<|endoftext|>`、22.5MB；train 2.23GB → 倒推 ~**2.73M** 文档；平均 ~816 字节/文档。


---

## 3. Embedding & 表示学习（讨论）

> §2 tokenizer 和 §3 Transformer 之间的桥梁概念，讨论中梳理。

### 3.1 token ID → 向量：embedding matrix 查表
- Embedding matrix 形状 `(vocab_size, d_model)`：**行数=词表大小**（每个 ID 一行），**列数=d_model**（每行是该 token 的向量）。
- **token ID 就是行号**：给 ID=256 → 取矩阵第 256 行的 `d_model` 维向量。本质是 **lookup（按行号取行）**，不是真矩阵乘法。
  - 理论上等价于 one-hot 向量 × 矩阵，但 one-hot 里绝大部分是 0，乘出来就是取那一行 → 直接取行快得多。
- 这就是 `run_embedding` 要做的事（`weights=(vocab_size,d_model)`，`token_ids` 是行号）。
- **特殊 token 在这层无区别**：`<|endoftext|>`(如 ID 10000) 就是矩阵第 10000 行，和普通 token 一样是一个可学习向量。tokenizer 层它"特殊"（不被拆），embedding 层它就是普通一行。
- 呼应：ID 空间是编号、可无限往上加；256 字节只占 ID 0–255，合并 token 从 256 起，特殊 token 再往后 → 词表没被"占满"。

### 3.2 embedding matrix 如何训练
- **它就是普通可学习参数**，没有特殊训练方式 = 整个模型怎么训练它就怎么训练。
- 训练大循环（handout §5 要实现）：forward（ID→查行→Transformer→logits→loss）→ backward（求梯度）→ step（AdamW 更新）→ zero_grad。
- **embedding 特有洞察**：一个 batch 只查了出现过的 token 的行 → 反向时**梯度只流回被用到的行**，没出现的 token 那一步梯度=0、不更新。
  - ⟹ 高频 token 更新多、学得充分；低频/生僻 token 更新少、学得差。
  - **这解释了 §1.5**"中文稀疏→掌握不锐利"：中文 token 出现少 → embedding 行更新少 → 没训充分 → 采样易漂移。
- **方向由 loss 驱动**：LM 的 loss 是预测下一词的 cross-entropy（§4 `run_cross_entropy`）。梯度告诉每个参数"往哪调能让正确词概率更高"。经海量更新，常出现在相似上下文的 token 向量被"推"到相近位置 → "猫≈狗""CJK 互为邻居"是 loss + 上下文**塑造**出来的，非人为设计。

### 3.3 embedding 参与 pretraining（端到端）
- **参与，且从随机初始化一起练**。pretraining = 在海量文本上预测下一词、反复更新**所有参数**。
- 全部可学习参数：embedding matrix + 各层 attn(q/k/v/o) + FFN(w1/w2/w3) + RMSNorm gain + LM head。**一起端到端训练**，没有谁先谁后。
- LM head 也是 `(vocab_size, d_model)`，做反向的事（向量→各 token 的 logits）；有些模型让输入 embedding 和 LM head **共享矩阵**（weight tying）。
- 对比"不参与训练"的情况帮理解：① 用冻结的预训练词向量（word2vec/GloVe，见 §3.4）；② 微调时 freeze embedding。CS336 做的是 **from scratch pretraining，embedding 跟着一起学**。

### 3.4 历史：word2vec / GloVe（两阶段 → 端到端的演进）
- **word2vec**（2013, Mikolov）：专门训词向量的独立模型，思想"上下文决定词义"。CBOW（上下文→中心词）/ Skip-gram（中心词→上下文）。
- 著名性质：向量类比 `vec(king)-vec(man)+vec(woman)≈vec(queen)` → 向量空间编码语义关系。

| | word2vec 时代（两阶段） | 现代 LLM（端到端） |
|---|---|---|
| embedding 怎么来 | 单独训好 | 跟整个模型一起 pretraining |
| 用时 | 塞进下游、常冻结 | 全程被 loss 更新 |
| 训练目标 | 通用上下文预测 | 直接是最终任务（预测下一词） |

- **为何弃用两阶段**：① 端到端让 embedding 被最终任务塑造，更贴合；② word2vec **静态**——一词一个固定向量，"bank"(银行/河岸)分不开；而 Transformer 里 embedding 只是起点，过 attention 后同词在不同上下文得**不同**表示（contextualized）。
- **思想活下来了**："上下文决定意义"一脉相承——word2vec 显式用上下文训向量；LM 预测下一词时 loss 把相似上下文的 token 向量推近。§1.5 的"CJK 互为邻居"与 word2vec 类比性质**同源**，只是融进了端到端训练。

---

## 4. 待续（§3 Transformer 实现时继续记）
