"""§4.1 的三张表，从 attention_fp32.json 和一次逐步实测生成。blog2.ipynb 里同样的代码。"""
import json, math
import pandas as pd
import torch
from einops import einsum

OUT = "notes/assets/s4"
GiB = 2**30


def pivot_tables(df):
    """表 4.1-1（fwd / bwd ms）与表 4.1-2（反向前 GiB），d 为行、seq 为列。"""
    ok = df.status == "ok"
    t = df.assign(cell=[f"{f:.2f} / {b:.2f}" if o else "OOM" for f, b, o in zip(df.fwd_ms, df.bwd_ms, ok)])
    t1 = t.pivot(index="d", columns="seq", values="cell")
    m = df.assign(cell=[f"{v:.3f}" if o else "OOM" for v, o in zip(df.mem_before_bwd_gib, ok)])
    t2 = m.pivot(index="d", columns="seq", values="cell")
    return t1, t2


def forward_steps(seq, d=16, batch=8):
    """表 4.1-3：逐步跑一遍 sdpa 的前向，每步读 memory_allocated()。和 hw1 的实现逐行对应。"""
    Q, K, V = (torch.randn(batch, seq, d, device="cuda", requires_grad=True) for _ in range(3))
    base = torch.cuda.memory_allocated()
    rows, live = [], lambda: (torch.cuda.memory_allocated() - base) / GiB
    S = einsum(Q, K, "b q d, b k d -> b q k") / math.sqrt(d);  rows.append(("S = QKᵀ/√d", "S", live()))
    mx = S.max(dim=-1, keepdim=True)[0]
    xs = S - mx;                                                rows.append(("x − max", "x_shifted", live()))
    nu = torch.exp(xs);                                         rows.append(("exp(x_shifted)", "nume", live()))
    P = nu / nu.sum(-1, keepdim=True);                          rows.append(("P = nume / sum", "P", live()))
    O = einsum(P, V, "b q k, b k d -> b q d")
    del S, mx, xs, nu;                                          rows.append(("函数返回（释放 S、x_shifted）", "", live()))
    return pd.DataFrame(rows, columns=["前向步骤", "新增张量", f"seq {seq} 活着的 GiB"])


if __name__ == "__main__":
    df = pd.DataFrame(json.load(open(f"{OUT}/attention_fp32.json")))
    t1, t2 = pivot_tables(df)
    steps = forward_steps(8192)
    steps["seq 16384 外推 GiB"] = steps.iloc[:, 2] * 4
    with open(f"{OUT}/tables_4_1.md", "w") as f:
        for name, t in [("表 4.1-1 fwd / bwd ms", t1), ("表 4.1-2 反向前 GiB", t2), ("表 4.1-3 逐步显存", steps)]:
            f.write(f"{name}\n\n{t.to_markdown(index=t is not steps, floatfmt='.3f')}\n\n")
    print(open(f"{OUT}/tables_4_1.md").read())
