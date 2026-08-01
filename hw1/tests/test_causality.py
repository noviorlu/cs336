"""因果性与 mask 的回归测试。

这不是作业自带的测试，是自己补的。动机：`TransformerLM` 自己不判断因果性，
完全依赖外部传进来的 mask。这个洞在重构中被打开过两次，而官方测试全都显式
构造 mask，所以一次都没抓到——绿灯只证明了"传了 mask 的那条路"是对的。
"""

import pytest
import torch

from cs336_basics.transformer import TransformerLM, build_attention_mask


def _model(vocab_size=50, seq=8, heads=4):
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=vocab_size, context_length=seq * 2, d_model=32,
        num_layers=2, num_heads=heads, d_ff=64, rope_theta=1e4,
    ).eval()


def test_mask_is_required():
    """忘了传 mask 必须当场报错，而不是静默退化成全双向注意力。"""
    m = _model()
    x = torch.randint(0, 50, (2, 8))
    with pytest.raises(TypeError):
        m(x)  # type: ignore[call-arg]


def test_causal_mask_blocks_future():
    """改动未来的 token，不能影响任何过去位置的 logits。"""
    m = _model()
    x = torch.randint(0, 50, (2, 8))
    x_future = x.clone()
    x_future[:, -1] = (x_future[:, -1] + 1) % 50

    mask = build_attention_mask(x.shape[-1], x.device)
    with torch.no_grad():
        a = m(x, mask=mask)[:, :-1]
        b = m(x_future, mask=mask)[:, :-1]
    assert torch.equal(a, b), "未来的 token 影响了过去的位置——因果性泄漏"


def test_document_mask_blocks_across_boundary():
    """跨文档时，分隔符之后的位置不能受分隔符之前内容的影响。"""
    sep = 49
    m = _model()
    #            ┌── 文档 A ──┐  sep  ┌── 文档 B ──┐
    x = torch.tensor([[1, 2, 3, sep, 10, 11, 12, 13]])
    x_other_doc = x.clone()
    x_other_doc[0, :3] = torch.tensor([7, 8, 9])  # 只改文档 A 的内容

    mask = build_attention_mask(x.shape[-1], x.device, x=x, doc_sep_id=sep)
    mask_other = build_attention_mask(
        x_other_doc.shape[-1], x_other_doc.device, x=x_other_doc, doc_sep_id=sep
    )
    with torch.no_grad():
        a = m(x, mask=mask)[:, 4:]              # 文档 B 的位置
        b = m(x_other_doc, mask=mask_other)[:, 4:]
    assert torch.equal(a, b), "文档 A 的内容影响了文档 B——跨文档污染没被挡住"


def test_mask_keeps_head_axis_unexpanded():
    """mask 要给 head 留一个长度 1 的轴，且不要 expand 成物理全尺寸。

    expand 本身零拷贝，但下游 masked_fill 里的 ~mask 会按 expand 后的形状物化。
    """
    x = torch.randint(0, 50, (4, 8))
    causal = build_attention_mask(8, x.device)
    assert causal.shape == (1, 1, 8, 8)

    doc = build_attention_mask(8, x.device, x=x, doc_sep_id=49)
    assert doc.shape == (4, 1, 8, 8)
    assert doc.untyped_storage().nbytes() == 4 * 8 * 8, "mask 不该被 expand 成 [B, h, s, s]"
