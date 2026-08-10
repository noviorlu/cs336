#!/usr/bin/env python3
"""从训练日志画 §8/§9 要用的 loss 曲线。

数据源两处：scratchpad/runs/*.log（本轮的 run）和 wandb/run-*/files/output.log
（更早的 lr / batch sweep）。两边格式相同，都是 train.py 打的：
    step N | loss X | lr ...
    Step N: train_loss X, val_loss Y
"""
import re, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/tmp/claude-1000/-home-yc-projects/fcb5053f-7478-48ae-8ee5-81ed6e291baf/scratchpad/runs"
WB   = "/home/yc/projects/LLM/hw1/wandb"
OUT  = "/home/yc/projects/LLM/hw1/notes_img"
os.makedirs(OUT, exist_ok=True)

# 系统里有 Noto Sans CJK，不注册的话中文全是豆腐块
from matplotlib import font_manager as fm
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"figure.dpi": 200, "font.size": 11, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})

def curve(path):
    """返回 (steps, train_loss)。NaN 会被保留成 float('nan')，画出来自然断开。"""
    if not os.path.exists(path):
        return [], []
    txt = open(path, errors="ignore").read()
    xs, ys = [], []
    for m in re.finditer(r"^step (\d+) \| loss ([\d.]+|nan)", txt, re.M):
        xs.append(int(m.group(1)))
        ys.append(float("nan") if m.group(2) == "nan" else float(m.group(2)))
    return xs, ys

def wb(rid):
    g = glob.glob(f"{WB}/run-*-{rid}/files/output.log")
    return g[0] if g else ""

def save(fig, name, title):
    fig.suptitle(title, fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}.png")

# ── 图 1：TinyStories 的 lr sweep —— edge of stability ────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for lbl, src, c in [("5e-4", wb("m7lo15v4"), "#9ecae1"), ("1e-3", wb("x9z3zo8x"), "#6baed6"),
                    ("3e-3", wb("dn21ib3d"), "#3182bd"), ("5e-3", wb("t8nhfslq"), "#08519c"),
                    ("6e-3", f"{RUNS}/lr6e-3.log", "#238b45"),
                    ("8e-3", f"{RUNS}/lr8e-3.log", "#e6550d"),
                    ("1e-2", wb("32lczj9f"), "#a50f15")]:
    x, y = curve(src)
    if x: ax.plot(x, y, label=f"lr {lbl}", color=c, lw=1.3)
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(1.2, 7)
ax.legend(fontsize=9.5, ncol=2, frameon=False)
save(fig, "01_lr_sweep", "TinyStories 学习率扫描：最优贴在发散边缘，中间有一段「稳定但退化」")

# ── 图 2：消融，TinyStories vs OWT ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
for ax, (tag, runs, ylim, uni) in zip(axes, [
    ("TinyStories (lr 5e-3)", [("基线", "seed3", "#08519c"), ("无门控 SiLU", "abl_silu", "#238b45"),
        ("post-norm", "abl_postnorm", "#e6550d"), ("NoPE", "abl_nope", "#756bb1"),
        ("拆光 RMSNorm", "abl_nonorm_lr5e-3", "#a50f15")], (1.2, 7), None),
    ("OpenWebText (lr 5e-3)", [("基线", "owt_lr5e-3", "#08519c"), ("无门控 SiLU", "owt_abl_silu", "#238b45"),
        ("post-norm", "owt_abl_postnorm", "#e6550d"), ("NoPE", "owt_abl_nope", "#756bb1"),
        ("拆光 RMSNorm", "owt_abl_nonorm", "#a50f15")], (3.5, 11), 7.5597)]):
    for lbl, r, c in runs:
        x, y = curve(f"{RUNS}/{r}.log")
        if x: ax.plot(x, y, label=lbl, color=c, lw=1.3)
    if uni:
        ax.axhline(uni, ls="--", lw=1, color="#666")
        ax.text(ax.get_xlim()[1]*.55, uni+.12, f"unigram 熵 {uni:.2f}", fontsize=9, color="#666")
    ax.set_title(tag, fontsize=11); ax.set_xlabel("step"); ax.set_ylim(*ylim)
    ax.legend(fontsize=9, frameon=False)
axes[0].set_ylabel("train loss")
save(fig, "02_ablation", "同样四个消融：TinyStories 上差百分之几，OWT 上 post-norm 直接退化成词频表")

# ── 图 3：lr 选错的代价 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for lbl, r, c in [("lb_A1  lr 3e-3（炸）", "lb_A1", "#a50f15"),
                  ("mA1    lr 1.5e-3", "mA1", "#08519c")]:
    x, y = curve(f"{RUNS}/{r}.log")
    if x: ax.plot(x, y, label=lbl, color=c, lw=1.4)
x, y = curve(f"{RUNS}/lb_A1.log")
if x:
    # 要标的是**前段那个局部低点**和它之后的反弹，不是全局最小——lb_A1 爬回来
    # 之后末尾才是全局最小，用 min() 会标错地方。
    n = len(y)
    lo = min(range(n // 8), key=lambda i: y[i])                  # 前 12.5% 里的最低点
    hi = max(range(lo, n // 3), key=lambda i: y[i])              # 它之后到 1/3 处的最高点
    ax.annotate(f"第 {x[lo]} 步触底 {y[lo]:.2f}", xy=(x[lo], y[lo]),
                xytext=(x[lo] + 900, y[lo] - .75), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1, color="#a50f15"))
    ax.annotate(f"反弹到 {y[hi]:.2f}\n之后 4500 步都在还债", xy=(x[hi], y[hi]),
                xytext=(x[hi] + 1100, y[hi] + .9), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1, color="#a50f15"))
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(3.2, 8)
ax.legend(fontsize=10, frameon=False)
save(fig, "03_lr_blowup", "同一个架构（d768/L8），lr 从 3e-3 改成 1.5e-3 换来 1.46 的 loss")

# ── 图 4：宽深比（固定算力、参数量差 6% 以内）────────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for lbl, r, c in [("d/L=40   d640/L16  99.8M", "mC1", "#756bb1"),
                  ("d/L=64   d768/L12 109.5M", "mB1", "#08519c"),
                  ("d/L=100  d896/L9  114.9M", "mA2", "#238b45"),
                  ("d/L=171  d1024/L6 108.7M", "mB2", "#e6550d")]:
    x, y = curve(f"{RUNS}/{r}.log")
    if x: ax.plot(x, y, label=lbl, color=c, lw=1.3)
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(3.4, 5.2)
ax.legend(fontsize=9.5, frameon=False, title="宽深比", title_fontsize=9.5)
save(fig, "04_aspect_ratio", "宽深比是个碗，底在 64（步数不同是因为固定的是 FLOPs 不是步数）")

# ── 图 5：最终 run ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.8))
x, y = curve(f"{RUNS}/final_d768L12.log")
ax.plot(x, y, color="#08519c", lw=1.2, label="train loss")
txt = open(f"{RUNS}/final_d768L12.log", errors="ignore").read()
ev = [(int(a), float(b)) for a, b in re.findall(r"^Step (\d+): train_loss [\d.]+, val_loss ([\d.]+)", txt, re.M)]
if ev:
    ax.plot([e[0] for e in ev], [e[1] for e in ev], "o-", ms=3, color="#a50f15", lw=1.2, label="val loss")
    ax.annotate(f"最终 val {ev[-1][1]:.4f}", xy=ev[-1], xytext=(ev[-1][0]*.62, ev[-1][1]+.6),
                fontsize=10, arrowprops=dict(arrowstyle="->", lw=.8))
ax.axhline(7.5597, ls="--", lw=1, color="#666")
ax.text(1200, 7.72, "OWT unigram 熵 7.56", fontsize=9, color="#666")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_ylim(3.0, 8.5)
ax.legend(fontsize=10, frameon=False)
save(fig, "05_final_run", "leaderboard 最终 run：109.5M，26943 步跑满 1.21e18 FLOPs")

print("done")
