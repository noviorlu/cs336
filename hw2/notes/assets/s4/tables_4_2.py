"""§4.2 的表，从 attention_fp32.json / attention_compiled.json 生成。"""
import pandas as pd

OUT = "notes/assets/s4"


def speedup_table(e, c):
    """表 4.2-1：每格「eager → compiled ms（×加速）」，前向和反向各一张。"""
    m = e.merge(c, on=["d", "seq"], suffixes=("_e", "_c"))
    tabs = {}
    for k in ("fwd", "bwd"):
        cell = [f"{a:.2f} → {b:.2f}（×{a/b:.1f}）" if s == "ok" else "OOM"
                for a, b, s in zip(m[f"{k}_ms_e"], m[f"{k}_ms_c"], m.status_c)]
        tabs[k] = m.assign(cell=cell).pivot(index="d", columns="seq", values="cell")
    return tabs["fwd"], tabs["bwd"]


def memory_table(e, c):
    """表 4.2-2：d=16 前向峰值 / 全程峰值，eager vs compiled（GiB）。"""
    rows = []
    for name, df in (("eager", e), ("compiled", c)):
        d16 = df[df.d == 16].set_index("seq")
        for col, label in (("peak_fwd_gib", "前向峰值"), ("peak_gib", "前向+反向峰值")):
            rows.append(pd.Series([f"{v:.3f}" if s == "ok" else "OOM" for v, s in zip(d16[col], d16.status)],
                                  index=d16.index, name=f"{label}（{name}）"))
    return pd.DataFrame(rows)


def model_table(m):
    """表 4.2-4：三档 × 三模式，eager → compiled ms（×加速），另给峰值显存。"""
    rows = []
    for (size, mode), g in m.groupby(["size", "mode"], sort=False):
        e, c = g[~g["compile"]].iloc[0], g[g["compile"]].iloc[0]
        t = (f"{e.avg_ms:.1f} → {c.avg_ms:.1f}（×{e.avg_ms / c.avg_ms:.2f}）" if e.status == "OK" and c.status == "OK"
             else f"{e.status if e.status != 'OK' else e.avg_ms} → {c.avg_ms:.1f}")
        rows.append(dict(size=size, mode=mode, **{"eager → compiled（ms）": t,
                                                 "峰值显存 eager → compiled（GiB）": f"{e.peak_mem_gib:.2f} → {c.peak_mem_gib:.2f}"}))
    return pd.DataFrame(rows).set_index(["size", "mode"]).rename_axis([None, None])


if __name__ == "__main__":
    e = pd.read_json(f"{OUT}/attention_fp32.json")
    c = pd.read_json(f"{OUT}/attention_compiled.json")
    f, b = speedup_table(e, c)
    mem = memory_table(e, c)
    model = model_table(pd.read_json(f"{OUT}/model_eager_vs_compiled.json"))
    with open(f"{OUT}/tables_4_2.md", "w") as fh:
        for name, t in (("表 4.2-1a 前向 ms", f), ("表 4.2-1b 反向 ms", b), ("表 4.2-2 显存峰值 GiB", mem), ("表 4.2-4 整模型", model)):
            fh.write(f"{name}\n\n{t.to_markdown()}\n\n")
    print(open(f"{OUT}/tables_4_2.md").read())
