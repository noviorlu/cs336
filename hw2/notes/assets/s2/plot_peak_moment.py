"""§2.3(d)(e)：一步 fwd_bwd 的显存曲线。前向第 j 层结束：W + A·j/L（末尾加 T）；反向走完 k 层：W + G·k/L + A·(L−k)/L + T。
xl@128 与 small@512，fp32 vs bf16 autocast。数字来自 memory_xl_peak.md / autocast_saved_tensors.txt。"""
import matplotlib.pyplot as plt

cases = [  # name, L, W, G, A_fp32, A_bf16(含副本), T, 实测峰值 fp32/bf16
    ("xl @ seq 128, batch 4  (G > A)",    32, 12.7, 12.7, 5.3,  3.5 + 6.35, 0.2,  25.56, 25.55),
    ("small @ seq 512, batch 4  (A > G)", 12, 0.48, 0.48, 3.41, 2.28 + 0.23, 0.17, 4.08, 3.18),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, L, W, G, A32, A16, T, p32, p16) in zip(axes, cases):
    xs = list(range(2 * L + 1))                       # 0..L 前向逐层，L..2L 反向逐层
    for j, (A, lab, c, meas) in enumerate([(A32, "fp32", "C0", p32), (A16, "bf16 autocast", "C1", p16)]):
        fwd = [W + A * i / L for i in range(L)] + [W + A + T]      # 前向末尾那一点已在算第一层反向，带上 T
        bwd = [W + G * k / L + A * (L - k) / L + T for k in range(1, L + 1)]
        M = fwd + bwd
        ax.plot(xs, M, marker="o", ms=2.5, c=c, label=f"{lab}: A = {A:.2f} GiB")
        kp = max(xs, key=lambda x: M[x])
        ax.annotate(f"{lab} peak {M[kp]:.2f} (measured {meas})", (kp, M[kp]), textcoords="offset points",
                    xytext=(-8, -14 - 16 * j) if kp > L else (8, 6 + 14 * j), ha="right" if kp > L else "left", fontsize=8, c=c)
    ax.axvline(L, c="red", lw=.8, ls="--")
    y0 = ax.get_ylim()[0]
    ax.text(L / 2, y0, "forward: +A/L per layer", fontsize=8, c="gray", ha="center", va="bottom")
    ax.text(1.5 * L, y0, "backward: +G/L − A/L per layer", fontsize=8, c="gray", ha="center", va="bottom")
    ax.set_title(f"{name}\nW = G = {W} GiB, L = {L}", fontsize=10)
    ax.set_xticks([0, L, 2 * L]); ax.set_xticklabels(["start", "forward ends\nbackward starts", "backward ends"], fontsize=8)
    ax.set_ylabel("active memory  GiB"); ax.legend(fontsize=8, loc="upper left" if G > A32 else "lower center")
fig.suptitle("one fwd_bwd step: forward W + A·j/L, backward W + G·k/L + A·(L−k)/L + T   — backward slope (G−A)/L decides where the peak is", fontsize=9.5)
fig.tight_layout()
fig.savefig("notes/assets/s2/peak_moment.png", dpi=140)
