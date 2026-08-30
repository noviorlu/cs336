"""RoPE 的性质测试。

官方 `test_rope` 只做两件事：和参考快照比数值、位置用 1-D 的 `arange(seq)`。
所以下面三条路径它一条都碰不到，而其中两条在实现过程中真的出过问题：

  - positions 带 batch 维（§3.12 静态扫描第 4 条：右对齐时 batch 撞上 num_heads）
  - bf16 输入被静默提升成 fp32（§3.9：符号向量忘了跟着转 dtype，
    而官方测试跑的是 fp32，两边碰巧一致所以照样绿）

还有一条是 RoPE 的**定义性质**（内积只依赖相对距离）——快照只能证明"和以前一样"，
证明不了"对"。

有一件事要说清楚，免得高估这些断言：**相对位置性质和保范数都钉不住"配对方式"**。
"前半段配后半段"那种配法同样是合法旋转、同样满足这两条，只是和作业规定的相邻配对
不同（§3.9 第 2 点）。实测给一个对半配的实现跑上面的断言，内积极差只有 2.4e-07，
根本抓不到。真正钉住配对的是最后那个手算用例（而且它必须用 d_k=4，见那条注释）和官方的快照比对。

写这些的时候全部通过，所以它们是回归保护，不是在抓 bug。
"""

import math

import pytest
import torch

from cs336_basics.model import MultiHeadSelfAttention, RoPE

D_K, MAX_SEQ, THETA = 16, 32, 10000.0


@pytest.fixture
def rope():
    return RoPE(theta=THETA, d_k=D_K, max_seq_len=MAX_SEQ)


def test_inner_product_depends_only_on_relative_distance(rope):
    """RoPE 存在的理由：<R_m q, R_n k> 只依赖 m-n，与绝对位置无关。"""
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, D_K), torch.randn(1, 1, D_K)

    def dot(m, n):
        return (rope(q, torch.tensor([m])) * rope(k, torch.tensor([n]))).sum()

    for gap in (0, 1, 5):
        vals = torch.stack([dot(m, m + gap) for m in range(6)])
        assert torch.allclose(vals, vals[0], atol=1e-5), (
            f"间距 {gap} 时内积随绝对位置漂移了：{vals.tolist()}"
        )


def test_rotation_preserves_norm(rope):
    """旋转是正交变换，不能改变向量长度。"""
    torch.manual_seed(0)
    x = torch.randn(2, 7, D_K)
    out = rope(x, torch.arange(7))
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_position_zero_is_identity(rope):
    """位置 0 的旋转角是 0，应当原样返回。"""
    torch.manual_seed(0)
    x = torch.randn(1, 1, D_K)
    assert torch.allclose(rope(x, torch.tensor([0])), x, atol=1e-6)


def test_positions_with_batch_dim_match_1d(rope):
    """positions 传 [seq] 和 [batch, seq] 必须等价。

    §3.12 第 4 条记的就是这里：MHA 会在 seq 前面插一个 head 维，
    positions 没有对应的轴时，右对齐会让 batch 撞上 num_heads。
    conftest 里 batch = heads = 4，所以那种情况下不报错、只是静默算错。
    """
    torch.manual_seed(0)
    mha = MultiHeadSelfAttention(d_model=D_K, num_heads=4, rope=RoPE(theta=THETA, d_k=D_K//4, max_seq_len=MAX_SEQ))
    x = torch.randn(4, 6, D_K)                      # batch 故意和 heads 一样是 4
    # mask 语义已改为 True=阻断，因果 mask 是上三角（不含对角线）
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)

    out_1d = mha(x, mask=mask, token_positions=torch.arange(6))
    out_2d = mha(x, mask=mask, token_positions=torch.arange(6).expand(4, 6))
    assert torch.allclose(out_1d, out_2d, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_output_dtype_follows_input(rope, dtype):
    """输出 dtype 必须跟随输入。

    角度表按 §3.9 固定存 fp32（bf16 只有 8 位有效精度，位置超过 256 就开始丢值），
    但用的时候要把 cos/sin 降到输入的 dtype，而不是把 x 升上去——方向反了会让
    整条残差流被拖成 fp32，混合精度就白做了。
    """
    x = torch.randn(1, 6, D_K, dtype=dtype)
    assert rope(x, torch.arange(6)).dtype == dtype


def test_matches_hand_computed_rotation():
    """手算一个最小用例，钉住配对方式和角度约定。

    必须取 d_k=4：d_k=2 时只有一对，"相邻配对"和"前半段配后半段"退化成同一件事，
    区分不出来（我第一版就写成了 d_k=2，跑注入配对错误的实现照样通过）。

    d_k=4 时 k=0,1，inv_freq[k] = 1/Θ^(2k/4)，Θ=10000 → [1, 1/100]，
    所以位置 1 上两对的旋转角分别是 1 和 0.01 弧度。相邻配对把
    [a,b,c,d] 分成 (a,b) 和 (c,d)，各转各的角。
    """
    rope_4d = RoPE(theta=THETA, d_k=4, max_seq_len=4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    t0, t1 = 1.0, 0.01
    expected = torch.tensor([[[
        1 * math.cos(t0) - 2 * math.sin(t0), 1 * math.sin(t0) + 2 * math.cos(t0),
        3 * math.cos(t1) - 4 * math.sin(t1), 3 * math.sin(t1) + 4 * math.cos(t1),
    ]]])
    assert torch.allclose(rope_4d(x, torch.tensor([1])), expected, atol=1e-5)
