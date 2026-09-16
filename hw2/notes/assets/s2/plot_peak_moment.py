"""§2.3(d)(e)：M(k) = W + G·k/L + A·(L−k)/L + T，xl@128 与 small@512，fp32 vs bf16 autocast。数字来自 memory_xl_peak.md / autocast_saved_tensors.txt。"""
import matplotlib.pyplot as plt

cases = [  # name, L, W, G, A_fp32, A_bf16(含副本), T, 实测峰值 fp32/bf16
    ("xl @ seq 128, batch 4  (G > A)",    32, 12.7, 12.7, 5.3,  3.5 + 6.35, 0.2,  25.56, 25.55),
    ("small @ seq 512, batch 4  (A > G)", 12, 0.48, 0.48, 3.41, 2.28 + 0.23, 0.17, 4.08, 3.18),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (name, L, W, G, A32, A16, T, p32, p16) in zip(axes, cases):
    ks = list(range(L + 1))
    for j, (A, lab, c, meas) in enumerate([(A32, "fp32", "C0", p32), (A16, "bf16 autocast", "C1", p16)]):
        M = [W + G * k / L + A * (L - k) / L + T for k in ks]
        ax.plot(ks, M, marker="o", ms=3, c=c, label=f"{lab}: A = {A:.2f} GiB")
        kp = max(ks, key=lambda k: M[k])
        ax.annotate(f"{lab} peak {M[kp]:.2f} (measured {meas})", (kp, M[kp]), textcoords="offset points",
                    xytext=(-10 if kp else 10, -14 - 12 * j if kp else 8 + 12 * j), ha="right" if kp else "left", fontsize=8, c=c)
    ax.axvline(0, c="gray", lw=.6, ls=":"); ax.axvline(L, c="gray", lw=.6, ls=":")
    ax.text(0, ax.get_ylim()[0], " k=0: forward ends", fontsize=7, c="gray", va="bottom")
    ax.text(L, ax.get_ylim()[0], "k=L: backward ends ", fontsize=7, c="gray", va="bottom", ha="right")
    ax.set_title(f"{name}\nW = G = {W} GiB, L = {L}", fontsize=10)
    ax.set_xlabel("k = layers finished in backward"); ax.set_ylabel("M(k)  GiB"); ax.legend(fontsize=8, loc="lower right" if G > A32 else "upper right")
fig.suptitle("M(k) = W + G·k/L + A·(L−k)/L + T   — slope (G−A)/L decides where the peak is", fontsize=10)
fig.tight_layout()
fig.savefig("notes/assets/s2/peak_moment.png", dpi=140)
