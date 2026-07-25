# CS336 Assignment 1 — 学习笔记

> 记录做作业时遇到的问题、概念理解、坑与解决方法。
> 公式用 LaTeX：行内 `$...$`，独立 `$$...$$`。

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

**跑测试**：`uv run pytest`（不是 `uv run tests`——`tests` 是目录不是命令）。

**Leaderboard（§7.5，6 分）**：在 OWT 上训模型、最小化 validation loss。规则只有两条——单次跑 **≤45 分钟 B200**、只能用课程给的 OWT 数据；其余随意。及格线是打败 loss 5.0 的朴素基线。提 PR 到 `stanford-cs336/assignment1-basics-leaderboard`。**排的是模型质量，不是 tokenizer 速度。**

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
- `len()` 忠实地数当前对象有几个元素：`len("café")=4`（字符），`len("café".encode())=5`（字节）。

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

演化：ASCII(7 位) → Unicode 诞生时天真以为 2 字节够 → UTF-16 → 字符超 6 万（现 15 万+，上限 U+10FFFF）→ UTF-16 被迫用代理对变成变长、UTF-32 出来回归定长 → UTF-8（Ken Thompson, 1992）换思路后来居上。

**顺带厘清 ASCII vs 字节容量**（常见混淆）：ASCII 只有 **128** 个字符（0–127，7 位够），因为当年只需覆盖英文；**256 是「一个字节」的容量**（8 位），是存储单位容量。ASCII 只用掉字节前一半，最高位恒为 0；128–255 历史上被各国拿去塞自己的字符（Latin-1、GBK 低区…）互不兼容，催生了 Unicode。

实测 "hello"：utf-8 `[104,101,108,108,111]`（5 字节，无 0）；utf-16 每字符后跟 1 个 0（12 字节）；utf-32 跟 3 个 0（24 字节）。→ 这是 unicode2(a) 的核心证据。

### 1.4 byte-level BPE 的好处

所有文本先 `encode("utf-8")` 变成 0–255 字节，BPE 只在**字节层**操作：
- 初始词表大小固定 = **256**（不是 128——UTF-8 流里字节会取到 128–255）。
- 永远不会 OOV（任何字符拆到字节都逃不出 0–255）⇒ **不需要 `<unk>`**。
- "某字符占几字节"的复杂度被 UTF-8 这步完全吸收，训练算法对此透明。

### 1.5 repr vs str/print

| | 面向 | 目标 | `chr(0)` |
|---|---|---|---|
| `repr()` / 交互式回显 | 调试 | 精确、无歧义 | `'\x00'`（可见转义） |
| `str()` / `print()` | 用户 | 干净好看 | 空白（终端渲染不出控制字符） |

- 容器（list/dict/tuple）的 `str` 会对**每个元素调用 `repr`** → `print(["a\nb"])` 显示 `['a\nb']`。
- **坑**：`print(b'caf\xc3\xa9')` 仍显示 `b'...'`，不会解码成 `café`。bytes 没有"给人看"的 str 形式，要看字得显式 `.decode()`。

### 1.6 延伸：LLM 中文说着变韩/日文 ≠ 字节接近

- LLM 不在**字节层**预测，而是在 **token ID** 上预测。中文和日文 token 是词表里两个不同 ID，ID 相邻也无语义关系 → "字节接近导致混淆"不成立。
- **真正原因**：① 训练数据 CJK 混杂 → **embedding 空间**里三者互为邻居；② 中文高质量数据稀疏，采样易漂移；③ 采样随机性；④ tokenizer 不匹配（中文被切碎）。
- **直觉纠正**：不是字节接近，而是 **embedding 空间**里 CJK 向量互为邻居。字节只是最底层存储，模型不在那层"思考"。（与 §4.2 的"高频 token 学得充分"呼应）

---

### 1.7 书面题

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

## 3. BPE Tokenizer 训练（§2.4–2.5）

> 三步：**① Vocab init → ② Pre-tokenization → ③ Merge loop**。
> 产出（`run_train_bpe` 返回值）：**vocab** `dict[int,bytes]`、**merges** `list[tuple[bytes,bytes]]`（按创建顺序）。
> 一句话：**从 256 字节出发，反复把"最常相邻的 token 对"合并成新 token，直到词表达到 `vocab_size`。**
>
> ```
> ① 256 字节 + 特殊 token → 初始 vocab
> ② 语料 → split特殊token → 正则切 → 转字节 → {pre-token: 次数}
> ③ 循环{数对(加权) → 选最高频(平局取字典序大) → 加vocab+记merges → 替换} → vocab + merges
> ```

### 3.1 Vocab init

初始词表 = 全部 256 字节值（ID 0–255）+ 特殊 token；之后每合并 1 次词表 +1。

$$\text{vocab\_size} = \underbrace{256}_{\text{初始字节}} + (\text{合并次数}) + \text{len(special\_tokens)}$$

⇒ **合并次数 = vocab_size − 256 − len(special_tokens)**。例：TinyStories `vocab_size=10000` + 1 个特殊 token → 合并 9743 次。

| 词表来源 | 参与合并 |
|---|---|
| 256 字节（初始化） | 是（合并原料） |
| 合并出的 token（训练算出） | 是 |
| 特殊 token（人为加入） | **否**（硬边界，不进 merge 统计，但占一个 ID） |

**特殊 token**：作业里只用 `<|endoftext|>`。共同本质是「人为定义、**永不被拆开**、占固定 ID 但不参与 merge」。

> ⚠️ **ID 不通用**，每个 tokenizer 自己分配：GPT-2 的 `<|endoftext|>`=50256（放**末尾**）、Llama 2 的 `<s>/</s>`=1/2（放**开头**）、Llama 3 的 `<|begin_of_text|>`=128000（放基础词表**之后**）。→ 印证"放开头/末尾"是设计选择。
>
> **实现提示**：接口是 `list[str]`，**别写死"只有一个"**（`test_train_bpe_special_tokens` 测多个）。放开头还是结尾看 `tests/fixtures/train-bpe-reference-vocab.json` 的排法。

### 3.2 Pre-tokenization

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

**演进**：GPT-2 正则范式至今仍是主流。GPT-4 的 `cl100k_base` 改了三处——**数字限 `\p{N}{1,3}`**（`12345`→`123`+`45`，算术更稳、词表不被撑爆）、缩写忽略大小写、换行处理更细。两个趋势：① 各家向 tiktoken 风格收敛（Llama 3、Mistral、Qwen 2.5、DeepSeek 都从 SentencePiece 倒戈，只剩 Gemma 还用 SP）；② 词表越做越大（5万→10万→20万，多语言覆盖，否则小语种被切碎）。**本作业的 GPT-2 正则 + byte-level BPE = 现在开源主流的基础骨架。**

### 3.3 Merge loop 算法

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
- **为什么必须串行**：第 2 轮的 `(e,st)` 是第 1 轮合并的**产物**，第 1 轮之前根本不存在 → 无法提前知道第 2 轮选谁。**这条依赖链无法打破**（并行的讨论见 §3.6）。

### 3.4 高效实现：四个结构 + 三项技术

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

### 3.5 优化记录（一张表）

| # | 改动 | 做了什么 | 测量场景 | 耗时 | 加速 |
|---|---|---|---|---|---|
| — | 朴素串行 | 每轮全量重数 pairs + 重建所有词 | pytest 全套 | ~16s | 1× |
| 1 | 并行预分词 + 微优化 | `Pool` 多进程切块；`len` 提出循环；`if id_a not in token_id: continue` 跳过不含该 pair 的词 | pytest 全套 | ~5s | 3.2× |
| 2 | 增量更新 merge loop | 引入 `pair_counts` + `pair_to_words` 持久结构，只遍历含 best_pair 的词，不再全量重数 | pytest 全套 | **0.87s** | **18×** |
| 3 | 懒删除堆 | `max()` 每轮全扫几万个 pair → `heapq` O(log n) 弹出 + stale 校验 | TS valid, vocab 10k | 14.3s → **1.46s** | **9.8×** |
| 4 | `neg_bytes` 按 id 缓存 | id→bytes 只增不改，缓存永不失效，最多算 vocab_size 次 | 同上 | **0.89s** | 1.64× |
| 5 | delta 更新 | 只动计数真正变化的 pair（见 §3.4 技术 1）；push 次数 4549→2274 | 同上 | **0.54s** | 1.66× |
| 6 | 堆定期重建 | 堆长 > `2×存活+1024` 时按当前 counts 重建，清 stale；峰值堆长 2505→1272 | 同上 | **0.42s** | 1.28× |
| 7 | `findall` + `Counter.update` | 预分词计数从 Python 循环 `+= 1` 换成 C 里的 `_count_elements`；`PAT` 模块级预编译 | TS train 2.23GB | 12.58s → **7.55s** | 1.67× |
| 8 | 分块数 = 进程数 × 4 | 文档长度不均导致 worker 空转（实测最慢 79.6ms vs 平均 58ms），多切块让快 worker 多领任务 | 同上 | **7.17s** | 1.05× |
| 9 | `gc.disable()` | merge loop 期间关分代 GC（`try/finally` 恢复），见 §3.7 教训 1 | owt_valid, vocab 32k | 42.75s → **28.21s** | **1.52×** |
| 10 | 邻域增量 + per-word 计数常驻 | 一次走完序列就地记录 delta，不再重建两个整词 `Counter`；常驻的 per-word pair 计数用来判断成员关系 | 同上 | **19.93s** | **1.42×** |

**目标语料确认**（owt_train 11.9GB + vocab 32000）：merge **544.95s → 236.54s = 2.30×**，端到端 585.8s → 278.1s。

> ⚠️ **这 2.30× 只归因于 #9 + #10**，不是 7~10 的合力：
> - #7/#8 是**预分词**的改动，物理上碰不到 merge 时间，而且在 544.95s 那次跑之前就已生效（两次跑的 pretok 分别是 40.84s / 41.53s，基本没动）。
> - 即基线代码状态 = #1–#8 已应用，优化后 = #1–#10。
> - 旁证：同样这两条在迭代语料 owt_valid 上是 42.75 → 28.21 → 19.93 = **2.15×**，与 2.30× 对得上（**兑现率 107%**）。
>
> 各阶段的加速倍数不能直接相乘——每条都是在**当时的代码状态**上测的，而且前面的优化会削弱后面的收益（见 §3.7 教训 4）。

**被否决的尝试**（同样测了、同样对拍通过，但不采纳）：

| 改动 | 结果 | 为什么 |
|---|---|---|
| heap key 后缀按 pair 缓存 | 20.02s（0%） | 第 10 条已经大幅减少 push，heap 不再是热点 |
| rebuild 阈值放宽到 `8×+1M` | 27.79s（−28%） | 堆变长直接推高每次 push/pop 的比较成本 |
| 完全不 rebuild | 51.82s（−62%） | 同上，极端情形 |
| rebuild 阈值收紧到 `1.25×` | 22.63s（−13%） | 重建太频繁，每次 O(存活) |
| rebuild 阈值 `3×`（扫出的最优） | 19.57s（+1.8%） | 低于 5% 采纳阈值；曲线在 2~4× 间很平，现值已在最优区间 |

### 3.6 并行性分析（结论随规模翻转）

| 阶段 | 能并行 | 说明 |
|---|---|---|
| Pre-tokenization | ✅ 数据并行 | `multiprocessing` 绕开 GIL，按 special token 切块，只汇总 1 次。**价值全在这**——它随数据量线性增长 |
| Merge loop | ⚠️ **看规模** | 见下 |
| GPU | ❌ 基本无用 | 符号处理（查字典/找 max/替换）不适合稠密数值并行 |
| 线程 | ❌ 一律排除 | CPython 3.12 有 GIL，循环体是 100% Python 字节码操作 dict/tuple |

merge loop 每轮要「全局决策 + 依赖上轮结果」，所以只能在**一轮之内**并行（把 affected 的词分给多个 worker）。**能否划算完全取决于每轮的粒度**：

| 规模 | unique pre-token | 每轮 word 循环（中位数） | vs 派发 32 个空任务（进程 300µs） | 结论 |
|---|---|---|---|---|
| TinyStories valid + vocab 10k | 6 万 | **3.5 µs** | 开销贵 85 倍 | **绝无可能** |
| owt_train + vocab 32k | **660 万** | 0.3 ms（均值 20.5ms） | 开销占 1.5% | **值得算** |

**正确的形状是永久分片，不是并行 `for word_idx in affected`**：后者每轮都要把 `ids[wi]` pickle 进 worker、结果 pickle 回来（实测一轮最多搬 714 万个 id），搬运量正比于工作量。永久分片则是——worker k 常驻持有 `words[k::N]` 及自己那份局部 `pair_to_words`，父进程每轮只广播 `(id_a, id_b, new_id)`（几十字节），worker 返回局部 delta 字典，父进程归并后更新 `pair_counts` + heap。

**当前代码的 Amdahl 上限**（优化后重测）：

| 环节 | 占比 |
|---|---|
| worker 侧（可分片并行） | 67% |
| parent 侧（`pair_counts` + heap push） | 28% ← **硬地板** |
| `pop_best` | 5% |

⇒ 完美 32× 并行的上限只有 **2.82×**（优化前是 3.92×——**O1 砍掉的正是可并行的那部分，上限跟着掉**）。要突破 28% 的地板得连堆也分片（worker 维护局部 top-k、父进程归并候选），复杂度和 tie-break 正确性风险都上一档。

**通信开销是固定的、与语料无关**：31743 轮 × 一次 round-trip(~300µs) ≈ **9.5s**。
- owt_valid（loop 才 20.8s）：净收益约 1.2×，**不值得**。
- owt_train（loop 236.5s，轮数相同）：9.5s 只占 4% ⇒ 估算 → ~95s，约 2.5×。
- ⇒ **同一份并行代码在两个语料上一个赚一个赔。轮数固定而每轮工作量随语料涨，是并行在这里能成立的唯一原因。**

### 3.7 优化方法论（五条教训）

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

**profile 怎么做**：
```python
import cProfile, pstats
cProfile.run('train_bpe(...)', 'prof')
pstats.Stats('prof').sort_stats('tottime').print_stats(20)
```
看 **tottime**（函数自身耗时，找瓶颈看它）、**cumtime**（含子调用）、**ncalls**（几百万次=热点）。命令行版 `python -m cProfile -s cumtime`；可视化 `snakeviz`。
实战：profile 直接指出 `max` 7.3s + `rank` 6.6s（被调 **1 亿次**）= 几乎全部时间 → 指向"用堆替代每轮全扫"；后来又指出 667 万次 `Counter` 构造 → 指向邻域增量。**先 profile 再优化，别猜。**

**怎么验证优化没改坏结果**（计数漂移这类 bug 静态看不出来，唯一可靠办法是对拍）：
1. **差分测试**：写一个每轮全量重算的 brute-force 版，逐条比 merges + vocab。语料要**专门造刁钻的**——大量 pair 同频的 tie 语料（测 tie-break）、`aaaa` 重叠串（测贪心从左到右 + 重叠记账）、中文/带音标拉丁（测多字节）、随机串。
2. **不变量断言**：每步 merge 后从 `vocab_ids` 从零重算真值，校验 `pair_counts` 没漂移、`pair_to_words` 与真值相等、**每个存活 pair 都有一条携带当前 count 的堆条目**（这条是懒删除正确性的命根子）。
3. **官方测试** `pytest tests/test_train_bpe.py`。

### 3.8 踩过的坑

*A. 计数不同步类（增量更新）*
- **`i`/`j` 变量冲突**：外层词下标用 `j`、内层扫描用 `i`，一开始都用 `i` → 内层 `i=0` 覆盖外层下标 → 写错位置。**内外循环变量必须不同名。**
- **`ord(c)` vs `encode`**：`[ord(c) for c in token]` 是**码点**不是 UTF-8 字节！byte-level 必须 `list(token.encode("utf-8"))`。
- **减/加必须严格对称**：任何不对称 → 计数漂移 → tie-break 选错 → 测试挂。合并 `A B` 会连带改变 `(前,A)→(前,X)`、`(B,后)→(X,后)`，不只是 `(A,B)` 消失。
- **清理 0 计数对**：`if pair_counts[pair] <= 0: del`，否则会选到 count=0 的僵尸对。
- **`affected` 必须拷贝**：`list(pair_to_words[best_pair])`，循环体里会改这个 set。
- **`.discard()` 不用 `.remove()`**：同词可能多次贡献同一 pair（`A B A B`）。
- **`pop(k, None)` 不用 `del`**：可能已在减计数时删过。

*B. 一般实现类*
- **tie-break 不能比 ID 数值**！必须 `(count, (vocab[id_a], vocab[id_b]))` 转回 bytes 比字典序。
- **merges 存 bytes 对** `(vocab[id_a], vocab[id_b])`，不是 id 对。
- 替换用「扫描重建新 list」（`i+2`/`i+1`），非原地 pop（避下标错位）；`new_id = len(vocab)` 别用 `len-1`。
- 特殊 token 最后加，终止条件留余量 `vocab_size - len(special_tokens)`。
- `bytes([i])`（方括号！）造单字节；`bytes(65)` 是 65 个 0，`bytes('A')` 报错。
- `if __name__ == "__main__":` 包住底部测试调用，避免 import 时乱跑十几秒。
- 「跳过不含该 pair 的词」的 `continue` 必须放在重建循环**前**，放后面等于白重建。

*C. 修过的两个正确性 bug*

| bug | 触发条件 | 后果 | 修法 |
|---|---|---|---|
| `re.split("", chunk)` | `special_tokens=[]` | pattern 为空串，**在每个字符之间都切一刀**，pre-token 全被打成单字符 | `re.split(pat, chunk) if pat else [chunk]` |
| 分块硬编码 `b"<\|endoftext\|>"` | 特殊 token 不叫这个名字 | 找不到切点 → 边界全退化到 EOF → **32 进程变 1 个**，并行全失效（实测 `<\|sep\|>` 语料：修前 1 块 / 修后 32 块） | 用 `special_tokens[0].encode()`；无特殊 token 时单块 |

### 3.9 下一步

- **性能已经够用**：tokenizer **只训练一次**，owt 全量 5 分钟跑完就存盘。继续优化 merge loop 对完成作业的边际价值接近零。
- **真正该优化的是 `Tokenizer.encode`**（§2.6）：§2.7(c) 专门让你估吞吐并推算「tokenize 825GB 的 Pile 要多久」，(d) 要把 OWT train/valid **全部编码落盘成 uint16 数组**——这才是你会实际等的时间，而且是纯数据并行（文档间独立），比 merge loop 友好得多。
- **注意**：BPE 训练的速度对 **LM 训练速度零影响**——训练时读的是已落盘的 `np.memmap` 数组，tokenizer 不在那个循环里。影响训练的是 tokenizer 的**输出**：`vocab_size`（决定 embedding/LM head 参数量和 softmax 开销）和 **compression ratio**（bytes/token，决定同样 token 预算下模型看到多少文本——对 45 分钟的 leaderboard 是实打实的杠杆）。而所有优化都严格保持了 merges 逐条相同，**没改变输出，也就没影响训练**。
- **Rust + PyO3**（handout 认可 `cppyy`/`nanobind`/PyO3）：性能边际价值不大，但学习价值高——算法已调对可原样搬、接口窄、有现成对拍工具，且后续 assignment（Triton kernel、数据管线）会一直用得上。
- **数据探查用命令行**（比写 Python 快）：`grep -c/-oF`、`wc -l/-c`、`stat -c %s`。实测 TinyStories valid 有 **27630** 个 `<|endoftext|>`、22.5MB；train 2.23GB → 倒推 ~**2.73M** 文档（对得上 handout 的 2.12M 量级）；平均 ~816 字节/文档。

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
