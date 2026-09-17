"""§2.2：一步 fwd_bwd 的显存曲线，按 W / G / A / T 四项堆叠着色（fp32），bf16 autocast 画成线。
前向第 i 层结束：W + A·i/L（末尾加 T）；反向走完 j 层：W + G·j/L + A·(L−j)/L + T。
数字来自 memory_xl_peak.md / autocast_saved_tensors.txt。颜色与正文 🟦W 🟥G 🟩A 🟨T 一致。"""
import json
import matplotlib.pyplot as plt

MEAS = {(r["size"], r["autocast"]): r for r in json.load(open("notes/assets/s2/peak_moment_measured.json"))}   # measure_peak_moment.py

C = {"W": "#4A90D9", "G": "#E4572E", "A": "#3CB371", "T": "#F2C14E"}
cases = [  # name, L, W, G, A_fp32, A_bf16(含副本), T, 实测峰值 fp32/bf16
    ("xl @ seq 128, batch 4  (G > A)",    32, 12.7, 12.7, 5.3,  3.5 + 6.35, 0.2,  25.56, 25.55),
    ("small @ seq 512, batch 4  (A > G)", 12, 0.48, 0.48, 3.41, 2.28 + 0.23, 0.17, 4.08, 3.18),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (name, L, W, G, A, A16, T, p32, p16) in zip(axes, cases):
    xs = list(range(2 * L + 1))
    # 四项随 x 的值：前向 i=0..L，反向 j=1..L
    w = [W] * (2 * L + 1)
    a = [A * i / L for i in range(L + 1)] + [A * (L - j) / L for j in range(1, L + 1)]
    g = [0] * (L + 1) + [G * j / L for j in range(1, L + 1)]
    t = [0] * L + [T] * (L + 1)
    ax.stackplot(xs, w, a, g, t, colors=[C["W"], C["A"], C["G"], C["T"]], labels=["W weights", "A saved tensors", "G gradients (.grad)", "T backward temporaries"], alpha=.85)
    tot = [w[i] + a[i] + g[i] + t[i] for i in xs]
    kp = max(xs, key=lambda x: tot[x])
    ax.annotate(f"fp32 peak {tot[kp]:.2f} (measured {p32})", (kp, tot[kp]), textcoords="offset points",
                xytext=(-8, 6) if kp > L else (8, 6), ha="right" if kp > L else "left", fontsize=8)
    # bf16 autocast：A 换成 A16（含副本），其余不变
    m16 = [W + A16 * i / L for i in range(L)] + [W + A16 + T] + [W + G * j / L + A16 * (L - j) / L + T for j in range(1, L + 1)]
    ax.plot(xs, m16, c="k", ls="--", lw=1.2, label=f"bf16 autocast (A = {A16:.2f})")
    # 实测点：每层前向结束 / 反向结束时的 memory_allocated()（measure_peak_moment.py，hook 采样）
    size = name.split()[0]
    for ac, c, lab in [(False, "k", "measured fp32"), (True, "gray", "measured bf16")]:
        r = MEAS.get((size, ac))
        if r:
            pts = r["fwd"] + r["bwd"]
            ax.scatter(range(1, len(pts) + 1), pts, s=10, c=c, marker="x", zorder=5, label=lab)
    kp16 = max(xs, key=lambda x: m16[x])
    ax.annotate(f"bf16 peak {m16[kp16]:.2f} (measured {p16})", (kp16, m16[kp16]), textcoords="offset points",
                xytext=(-8, -14) if kp16 > L else (8, -14), ha="right" if kp16 > L else "left", fontsize=8)
    ax.axvline(L, c="gray", lw=.8, ls=":")
    ax.set_title(f"{name}\nW = G = {W} GiB, A = {A} GiB, L = {L}", fontsize=10)
    ax.set_xticks([0, L, 2 * L]); ax.set_xticklabels(["start", "forward ends\nbackward starts (j=0)", "backward ends (j=L)"], fontsize=8)
    ax.set_ylabel("active memory  GiB"); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("one fwd_bwd step, stacked by term: forward W + A·i/L, backward M(j) = W + G·j/L + A·(L−j)/L + T", fontsize=9.5)
fig.tight_layout()
fig.savefig("notes/assets/s2/peak_moment.png", dpi=140)
