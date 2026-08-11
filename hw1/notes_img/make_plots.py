#!/usr/bin/env python3
"""给 EXPERIMENTS.md 的每个实验出一张图。

数据源是 wandb（先跑 fetch_wandb.py 缓存到 wandb_cache.json）。少数几张图
（BPE 优化的耗时、玩具 SGD 的发散）不在 wandb 里，数据直接写在下面——
它们是一次性的计时/解析结果，本来就没进训练日志。

早期几条 run 没起名字（wandb 自动生成的 blooming-shape-1 之类），所以按
config 认：pick() 支持用超参匹配。
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
DB = json.load(open(os.path.join(HERE, "wandb_cache.json")))

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"figure.dpi": 200, "font.size": 11, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})

C = dict(blue="#08519c", mid="#3182bd", light="#9ecae1", green="#238b45",
         orange="#e6550d", red="#a50f15", purple="#756bb1", gray="#666666",
         teal="#3f7f8c", olive="#7f9e3d")

# ── 取数 ────────────────────────────────────────────────────────────────
def pick(name=None, **cfg):
    """按 run 名或 config 找一条 run。config 全部匹配才算命中。"""
    for rid, r in DB.items():
        if name is not None and r["name"] != name:
            continue
        if all(r["config"].get(k) == v for k, v in cfg.items()):
            return r
    return None

def series(run, key="train/loss"):
    """返回 (steps, values)，按 step 排好序。"""
    if not run: return [], []
    pts = [(row["_step"], row[key]) for row in run["history"] if key in row]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]

def save(fig, name, title):
    fig.suptitle(title, fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}.png")

def line(ax, run, label, color, key="train/loss", **kw):
    x, y = series(run, key)
    if x: ax.plot(x, y, label=label, color=color, lw=1.4, **kw)
    return bool(x)

# ═══ 阶段一 · Tokenizer（不在 wandb，数据来自 §2/§3/§4 的表）═══════════
# 01 BPE 优化的累计效果
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
steps = ["朴素", "①并行\n预分词", "②增量\nmerge", "③懒删除堆", "④neg_bytes\n缓存",
         "⑤delta\n更新", "⑥堆定期\n重建"]
t_small = [16.0, 5.0, 0.87, None, None, None, None]          # pytest 全套
t_mid   = [None, None, None, 1.46, 0.89, 0.54, 0.42]         # TinyStories valid 10k
ax = axes[0]
xs = range(len(steps))
ax.plot([i for i, v in enumerate(t_small) if v], [v for v in t_small if v],
        "o-", color=C["blue"], lw=1.6, ms=6, label="pytest 全套")
ax.plot([i for i, v in enumerate(t_mid) if v], [v for v in t_mid if v],
        "s-", color=C["orange"], lw=1.6, ms=6, label="TinyStories valid, vocab 10k")
ax.set_yscale("log"); ax.set_xticks(list(xs)); ax.set_xticklabels(steps, fontsize=8)
ax.set_ylabel("耗时 (s，对数轴)"); ax.legend(fontsize=9, frameon=False)
ax.set_title("逐条优化的耗时（两个测量场景各自的基线不同）", fontsize=10)
ax = axes[1]
rej = ["现值\n(rebuild 2×)", "heap key\n缓存", "rebuild 3×", "rebuild\n1.25×", "rebuild\n8×+1M", "完全不\nrebuild"]
val = [19.93, 20.02, 19.57, 22.63, 27.79, 51.82]
cols = [C["green"]] + [C["gray"]]*2 + [C["orange"]]*3
ax.bar(range(len(rej)), val, color=cols)
ax.axhline(19.93*1.05, ls="--", lw=1, color=C["red"])
ax.text(3.1, 19.93*1.05+1.2, "5% 采纳阈值", fontsize=9, color=C["red"])
ax.set_xticks(range(len(rej))); ax.set_xticklabels(rej, fontsize=8)
ax.set_ylabel("owt valid, vocab 32k 耗时 (s)")
ax.set_title("被否决的尝试：曲线在 2~4× 之间很平", fontsize=10)
save(fig, "01_bpe_opt", "BPE 训练优化：10 轮把 merge loop 从 16 s 压到 0.42 s")

# 02 压缩率对比
fig, ax = plt.subplots(figsize=(8.6, 4.4))
labels = ["TinyStories 语料\nTS 词表(10k)", "OpenWebText 语料\nOWT 词表(32k)", "OpenWebText 语料\n用错→TS 词表(10k)"]
vals = [4.071, 4.363, 3.302]
cols = [C["blue"], C["green"], C["red"]]
b = ax.bar(range(3), vals, color=cols, width=.55)
for i, v in enumerate(vals):
    ax.text(i, v + .06, f"{v:.3f}", ha="center", fontsize=11)
ax.annotate("", xy=(2, 3.302), xytext=(1, 4.363),
            arrowprops=dict(arrowstyle="->", lw=1.6, color=C["red"]))
ax.text(1.5, 4.05, "压缩率 −24.3%\ntoken 数 +32.1%", ha="center", fontsize=10, color=C["red"])
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("字节 / token（越大越好）"); ax.set_ylim(0, 5)
save(fig, "02_compression", "词表大 3.2 倍只换来 7% 压缩率；但用错词表要赔 24%")

# ═══ 阶段二 · 优化器（解析计算，不在 wandb）═════════════════════════════
fig, ax = plt.subplots(figsize=(8.6, 4.8))
data = {
    "lr = 1e1": ([25.0, 22.2, 19.8, 17.7, 15.8, 14.1, 12.6, 11.3, 10.1, 9.0], C["blue"]),
    "lr = 1e2": ([25.0, 6.25, .69, .02, 1e-4, 1e-7, 1e-11, 1e-15, 1e-20, 1e-25], C["green"]),
    "lr = 1e3": ([25.0, 2.03e3, 4.11e5, 1.11e8, 3.75e10, 1.52e13, 7.2e15, 3.9e18, 2.4e21, 1.6e24], C["red"]),
}
for lbl, (ys, c) in data.items():
    ax.plot(range(1, 11), [max(y, 1e-30) for y in ys], "o-", label=lbl, color=c, lw=1.6, ms=4)
ax.set_yscale("log"); ax.set_xlabel("步"); ax.set_ylabel("loss（对数轴）")
ax.legend(fontsize=10, frameon=False); ax.set_ylim(1e-28, 1e26)
ax.text(5.2, 1e-20, "1e2 第 4 步就到机器零", fontsize=9, color=C["green"])
ax.text(1.4, 1e20, "1e3 指数发散——\n1/√t 的衰减救不回来", fontsize=9, color=C["red"])
save(fig, "03_lr_toy", "玩具二次函数：最优 lr 贴着发散边界，中间没有平滑过渡")

# ═══ 阶段三 · 语言模型 ═══════════════════════════════════════════════════
# 04 噪声底线：三个种子
fig, ax = plt.subplots(figsize=(8.6, 4.8))
seeds = [("seed 1337", pick(name=None, seed=1337, learning_rate=0.005, batch_size=256), C["blue"]),
         ("seed 2", pick(name="seed2"), C["orange"]),
         ("seed 3", pick(name="seed3"), C["green"])]
for lbl, r, c in seeds:
    line(ax, r, lbl, c)
ax.set_xlim(3000, 5100); ax.set_ylim(1.30, 1.48)
ax.set_xlabel("step"); ax.set_ylabel("train loss")
ax.legend(fontsize=10, frameon=False)
ax.text(3100, 1.455, "三条只差随机种子。末端 val = 1.3381 / 1.3428 / 1.3323\n"
                     "σ = 0.0053 —— 小于 2σ (0.011) 的差都不能下结论",
        fontsize=9.5, va="top")
save(fig, "04_seed_noise", "噪声底线：同配置换种子跑三遍（放大末段）")

# 05 lr sweep
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for lr, c in [(5e-4, C["light"]), (1e-3, "#6baed6"), (3e-3, C["mid"]), (5e-3, C["blue"]),
              (6e-3, C["green"]), (8e-3, C["orange"]), (1e-2, C["red"])]:
    r = pick(name=f"lr{lr:g}".replace("0.006", "6e-3").replace("0.008", "8e-3")) \
        or pick(learning_rate=lr, batch_size=256, d_model=512, num_layers=4, max_iters=5000)
    line(ax, r, f"lr {lr:g}", c)
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(1.2, 7)
ax.legend(fontsize=9, ncol=2, frameon=False)
save(fig, "05_lr_sweep", "TinyStories 学习率扫描：1e-2 在第 300 步起飞后再也回不来")

# 06 batch：loss 和显存墙
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
bs = [1, 64, 128, 256, 352]; loss = [2.9307, 1.5452, 1.4694, 1.4188, 1.4025]
axes[0].plot(bs, loss, "o-", color=C["blue"], lw=1.6, ms=7)
for x, y in zip(bs, loss): axes[0].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                                            xytext=(6, 6), fontsize=9)
axes[0].set_xscale("log", base=2); axes[0].set_xlabel("batch size"); axes[0].set_ylabel("val loss")
axes[0].set_title("固定 5000 步——batch 越大看到的 token 越多，两个效应同向", fontsize=10)
mb = [320, 352, 384, 512, 1024]; mem = [22.51, 24.72, None, None, None]
axes[1].bar([str(b) for b in mb], [m if m else 32 for m in mem],
            color=[C["green"], C["green"], C["red"], C["red"], C["red"]])
for i, m in enumerate(mem):
    axes[1].text(i, (m if m else 32) + .4, f"{m:.2f} GB" if m else "OOM", ha="center", fontsize=10)
axes[1].axhline(31.34, ls="--", lw=1.2, color=C["gray"])
axes[1].text(.1, 31.9, "卡上可用 31.34 GB", fontsize=9, color=C["gray"])
axes[1].set_ylabel("峰值显存 (GB)"); axes[1].set_ylim(0, 36)
axes[1].set_title("384 线性外推只要 26.9 GB 却仍 OOM（碎片+context）", fontsize=10)
save(fig, "06_batch", "Batch 与显存墙：墙在 352")

# 07 消融 TinyStories vs OWT
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
for ax, (tag, runs, ylim, uni) in zip(axes, [
    ("TinyStories (lr 5e-3)",
     [("基线", "seed3", C["blue"]), ("无门控 SiLU", "abl_silu", C["green"]),
      ("post-norm", "abl_postnorm", C["orange"]), ("NoPE", "abl_nope", C["purple"]),
      ("拆光 RMSNorm", "abl_nonorm_lr5e-3", C["red"])], (1.2, 7), None),
    ("OpenWebText (lr 5e-3)",
     [("基线", "owt_lr5e-3", C["blue"]), ("无门控 SiLU", "owt_abl_silu", C["green"]),
      ("post-norm", "owt_abl_postnorm", C["orange"]), ("NoPE", "owt_abl_nope", C["purple"]),
      ("拆光 RMSNorm", "owt_abl_nonorm", C["red"])], (3.5, 11), 7.5597)]):
    for lbl, n, c in runs:
        line(ax, pick(name=n), lbl, c)
    if uni:
        ax.axhline(uni, ls="--", lw=1.2, color=C["gray"])
        ax.text(2600, uni + .15, f"unigram 熵 {uni:.2f}（只学会词频）", fontsize=9, color=C["gray"])
    ax.set_title(tag, fontsize=10); ax.set_xlabel("step"); ax.set_ylim(*ylim)
    ax.legend(fontsize=9, frameon=False)
axes[0].set_ylabel("train loss")
axes[0].text(1500, 6.2, "五条几乎重叠\n（拆光 RMSNorm 是 NaN，画不出来）", fontsize=9.5)
save(fig, "07_ablation", "同样四个消融：TinyStories 上差百分之几，OWT 上 post-norm 整条压在 unigram 熵线上")

# 08 BPB：跨语料唯一可比的口径
fig, ax = plt.subplots(figsize=(8.6, 4.6))
grp = ["TinyStories\n(22.7M)", "OpenWebText\n(45.2M)"]
raw = [1.3323, 3.9287]; bpb = [0.472, 1.300]; rnd = [3.264, 3.430]
x = range(2); w = .35
ax.bar([i - w/2 for i in x], raw, w, label="原始 val loss (nats/token)", color=C["light"])
ax.bar([i + w/2 for i in x], bpb, w, label="BPB (bits/byte)", color=C["blue"])
for i in x:
    ax.text(i - w/2, raw[i] + .06, f"{raw[i]:.3f}", ha="center", fontsize=10)
    ax.text(i + w/2, bpb[i] + .06, f"{bpb[i]:.3f}", ha="center", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(grp, fontsize=10); ax.set_ylabel("loss")
ax.legend(fontsize=9.5, frameon=False)
ax.text(.5, 3.3, f"原始比 {3.9287/1.3323:.2f}×  →  BPB 比 {1.300/0.472:.2f}×\n"
                 "交叉熵的单位是「每 token 的 nat」，两边 token 不是一回事",
        ha="center", fontsize=10)
save(fig, "08_bpb", "跨 tokenizer 比模型只能看 BPB")

# ═══ 阶段四 · leaderboard ═══════════════════════════════════════════════
# 09 lr 炸掉
fig, ax = plt.subplots(figsize=(8.6, 4.8))
r1, r2 = pick(name="lb_A1"), pick(name="mA1")
line(ax, r1, "lb_A1  lr 3e-3（炸）", C["red"])
line(ax, r2, "mA1    lr 1.5e-3", C["blue"])
x, y = series(r1)
if x:
    n = len(y)
    lo = min(range(n // 8), key=lambda i: y[i])
    hi = max(range(lo, n // 3), key=lambda i: y[i])
    ax.annotate(f"第 {x[lo]} 步触底 {y[lo]:.2f}", xy=(x[lo], y[lo]),
                xytext=(x[lo] + 900, y[lo] - .75), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1, color=C["red"]))
    ax.annotate(f"反弹到 {y[hi]:.2f}\n之后 4500 步都在还债", xy=(x[hi], y[hi]),
                xytext=(x[hi] + 1100, y[hi] + .9), fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1, color=C["red"]))
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(3.2, 8)
ax.legend(fontsize=10, frameon=False)
save(fig, "09_lr_blowup", "同一个架构（d768/L8），lr 从 3e-3 改成 1.5e-3 换来 1.46 的 loss")

# 10 lr 缩放律
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
ax = axes[0]
probes = {768: [(1.0, 4.2028), (1.5, 4.1615), (2.0, 4.2395), (3.0, 5.0228)],
          896: [(0.86, 4.3813), (1.29, 4.3308), (1.93, 4.5140)],
          1024: [(0.75, 4.5931), (1.13, 4.5081), (1.69, 4.6383)]}
for (d, pts), c in zip(probes.items(), [C["blue"], C["green"], C["orange"]]):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o-", color=c, lw=1.5, ms=7, label=f"d={d}")
    b = min(pts, key=lambda p: p[1])
    ax.plot([b[0]], [b[1]], "*", color=c, ms=18)
ax.set_xscale("log"); ax.set_xlabel("lr (×10⁻³，对数轴)"); ax.set_ylabel("末 10 点均值 loss")
ax.legend(fontsize=10, frameon=False); ax.set_ylim(4.1, 5.1)
ax.set_title("每个宽度各扫三档，★ 是各自最优", fontsize=10)
ax = axes[1]
ds = [768, 896, 1024]; best = [1.50, 1.29, 1.13]
ax.plot(ds, best, "o", color=C["red"], ms=11, label="实测最优", zorder=3)
dd = [700 + i * 10 for i in range(40)]
ax.plot(dd, [1.5 * 768 / d for d in dd], "--", color=C["gray"], lw=1.5,
        label=r"$1.5\times10^{-3}\cdot 768/d$")
for d, b in zip(ds, best): ax.annotate(f"{b:.2f}e-3", (d, b), textcoords="offset points",
                                       xytext=(8, 6), fontsize=10)
ax.set_xlabel("d_model"); ax.set_ylabel("最优 lr (×10⁻³)")
ax.legend(fontsize=10, frameon=False)
ax.set_title("三个宽度全部命中；L=8 和 L=12 同宽时最优 lr 相同", fontsize=10)
save(fig, "10_lr_scaling", r"lr* ∝ 1/d，与深度无关（准到 1.5 倍以内）")

# 11 宽深比
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
ax = axes[0]
for lbl, n, c in [("d/L=40   d640/L16  99.8M", "mC1", C["purple"]),
                  ("d/L=64   d768/L12 109.5M", "mB1", C["blue"]),
                  ("d/L=100  d896/L9  114.9M", "mA2", C["green"]),
                  ("d/L=171  d1024/L6 108.7M", "mB2", C["orange"])]:
    line(ax, pick(name=n), lbl, c)
ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_ylim(3.4, 5.2)
ax.legend(fontsize=9, frameon=False, title="宽深比", title_fontsize=9)
ax.set_title("步数不同是因为固定的是 FLOPs 不是步数", fontsize=10)
ax = axes[1]
ar = [40, 64, 100, 171]
mid = [3.5684, 3.5533, 3.5666, 3.5864]
pr = [None, 4.3326, 4.3308, 4.2818]
ax.plot(ar, mid, "o-", color=C["blue"], lw=1.8, ms=9, label="中间预算 C/4（~6500 步）")
ax2 = ax.twinx()
xs = [a for a, v in zip(ar, pr) if v]; ys = [v for v in pr if v]
ax2.plot(xs, ys, "s--", color=C["orange"], lw=1.6, ms=8, label="探针（~1100 步）")
ax.set_xscale("log"); ax.set_xticks(ar); ax.set_xticklabels([str(a) for a in ar])
# 对数轴默认还会画次要刻度标签（6×10¹ 之类），和我们手设的 40/64/100/171 叠在一起
ax.minorticks_off()
ax.set_xlabel("宽深比 d/L"); ax.set_ylabel("val loss（C/4）", color=C["blue"])
ax2.set_ylabel("loss（探针）", color=C["orange"])
ax.plot([64], [3.5533], "*", color=C["red"], ms=22, zorder=5)
ax.annotate("碗底", xy=(64, 3.5533), xytext=(46, 3.5565), fontsize=10, color=C["red"])
ax.set_ylim(3.550, 3.590)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="lower right")
ax.set_title("排序随预算翻转：短程偏袒浅模型", fontsize=10)
save(fig, "11_aspect_ratio", "宽深比是个碗，底在 64（不是 Kaplan 的 100）")

# 12 最终 run
fig, ax = plt.subplots(figsize=(8.6, 4.8))
r = pick(name="final_d768L12")
line(ax, r, "train loss", C["blue"])
x, y = series(r, "eval/val_loss")
if x:
    ax.plot(x, y, "o-", ms=4, color=C["red"], lw=1.4, label="val loss")
    ax.annotate(f"最终 val {y[-1]:.4f}", xy=(x[-1], y[-1]), xytext=(x[-1] * .62, y[-1] + .6),
                fontsize=11, arrowprops=dict(arrowstyle="->", lw=1))
ax.axhline(7.5597, ls="--", lw=1.2, color=C["gray"])
ax.text(1200, 7.75, "OWT unigram 熵 7.56", fontsize=9, color=C["gray"])
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_ylim(3.0, 8.5)
ax.legend(fontsize=10, frameon=False)
save(fig, "12_final_run", "leaderboard 最终 run：109.5M，26943 步跑满 1.21e18 FLOPs")

# 13 全链条 + MFU
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
ax = axes[0]
names = ["OWT 基线\n45M", "lb_A1\n81M lr错", "mA1\n81M C/4", "mB1\n109M C/4", "final\n109M 跑满"]
vals = [3.9287, 4.9853, 3.5300, 3.5533, 3.2786]
cols = [C["gray"], C["red"], C["mid"], C["mid"], C["green"]]
ax.bar(range(5), vals, color=cols, width=.6)
for i, v in enumerate(vals):
    # lb_A1 那根顶到 4.99，数值标签要躲开 5.0 那条及格线，放到柱子里面
    inside = v > 4.5
    ax.text(i, v - .28 if inside else v + .08, f"{v:.4f}", ha="center", fontsize=10,
            color="white" if inside else "black")
ax.axhline(5.0, ls="--", lw=1.2, color=C["red"])
ax.text(4.45, 5.12, "leaderboard 及格线 5.0", fontsize=9, color=C["red"], ha="right")
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("OWT val loss"); ax.set_ylim(0, 5.6)
ax.set_title("全链条：最大的一跳来自把 lr 改对（4.99 → 3.53）", fontsize=10)
ax = axes[1]
P = [45.2, 81.2, 109.5, 570.5]; mfu = [26.2, 43.0, 49.3, 61.4]
ax.plot(P, mfu, "o-", color=C["blue"], lw=1.8, ms=9)
for p, m in zip(P, mfu):
    ax.annotate(f"{m:.1f}%", (p, m), textcoords="offset points", xytext=(8, -12), fontsize=10)
ax.set_xscale("log"); ax.set_xlabel("参数量 (M，对数轴)"); ax.set_ylabel("MFU (%)")
# 同上：对数轴的次要刻度标签（4×10¹ 之类）没意义，直接标实际的参数量
ax.set_xticks(P); ax.set_xticklabels([f"{p:g}" for p in P]); ax.minorticks_off()
ax.set_ylim(20, 68)
ax.set_title("矩阵乘越大，tensor core 喂得越满", fontsize=10)
save(fig, "13_summary", "从 45M 到 109.5M：loss 降 0.65，MFU 从 26% 涨到 49%")

print("done")
