"""显存快照——只服务 §2.5 memory_profiling，timeit 计时基准线不经过这里。

挂在 model_bench.run_model() 里，从预热开始记到测量结束。为什么不只记测量段：xl@2048 的 full step
在第一个前向就 OOM，那样什么都留不下。代价是时间线第一步带着 cuBLAS workspace 等一次性
初始化，看图时看第二步（或者 --warmup 0 --steps 1，接受第一步的噪声）。

产出 .pickle 拖进 pytorch.org/memory_viz 看「Active Memory Timeline」：
每笔分配的大小、存活区间、以及分配它的 Python 调用栈。
"""
import contextlib

import torch


@contextlib.contextmanager
def record_snapshot(path: str | None, max_entries: int = 1_000_000):
    """path 为 None 时什么都不做。

    dump 放在 finally：xl@2048 的 full step 预期 OOM，OOM 时更需要这份快照
    （它会显示炸掉前的时间线，能看到是哪一步把显存顶上去的）。
    """
    if path is None or not torch.cuda.is_available():
        yield
        return

    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"显存快照已写出：{path}（拖进 https://pytorch.org/memory_viz 查看）")


# ---- §2.5 (f)：从 nsys 的 sqlite 里算「一层 TransformerBlock 为反向存了多少」 -------------
#
# 采集：PYTORCH_NO_CUDA_MEMORY_CACHING=1 nsys profile --cuda-memory-usage=true --trace=cuda,nvtx -- \
#         python -m benchmark --size xl --seq-len 128 --mode fwd_bwd --nvtx --nvtx-ops --warmup 0 --steps 1
#       （不关 caching allocator 的话 nsys 只看到显存池增长，看不到每个张量）
# 分析：python -m benchmark.memory profiles/mem_trace_xl_seq128_fwd_bwd.nsys-rep [--block 5]
#
# 归属规则：
#   residual  = block{i} 前向 range 内 cudaMalloc、且到 range 结束还没 cudaFree 的那些地址
#   来源算子  = 包着该 cudaMalloc 时刻的最内层 aten::* range（emit_nvtx 打的，带 seq = N）
#   反向窗口  = 反向线程上 seq 落在 block{i} 前向 seq 区间内的所有 range 的时间并集
#   梯度字节  = 反向窗口内的净分配 + 该窗口内释放掉的 residual（先加回去再看新增的是什么）

import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MiB = 2 ** 20
_SEQ = re.compile(r"seq = (\d+)")


def _sqlite(rep: Path) -> sqlite3.Connection:
    db = rep.with_suffix(".sqlite")
    if not db.exists() or db.stat().st_mtime < rep.stat().st_mtime:
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", str(db), str(rep)], check=True, capture_output=True)
    return sqlite3.connect(db)


def block_residuals(rep: Path | str, block: int = 5, top: int = 5):
    rep = Path(rep)
    c = _sqlite(rep)
    (b_start, b_end, tid), = c.execute(
        "select start, end, globalTid from NVTX_EVENTS where text = ?", (f"block{block}",)).fetchall()

    # 前向 range 内的 aten 算子（同线程），用来做归属和拿 seq 区间
    ops = c.execute(
        "select text, start, end from NVTX_EVENTS where globalTid = ? and text like 'aten::%' "
        "and start >= ? and end <= ? order by start", (tid, b_start, b_end)).fetchall()
    seqs = [int(m.group(1)) for t, _, _ in ops if (m := _SEQ.search(t))]
    seq_lo, seq_hi = min(seqs), max(seqs)

    def innermost_op(t):
        cands = [o for o in ops if o[1] <= t <= o[2]]
        return max(cands, key=lambda o: o[1])[0].split(",")[0] if cands else "(non-aten)"

    # 前向 range 内的分配；到 range 结束还没释放的 = residual。
    # 关了 caching 之后 cudaFree 过的地址会被复用，所以要按「分配之后的第一次 free」配对，不能只看地址。
    allocs = c.execute(
        "select start, address, bytes from CUDA_GPU_MEMORY_USAGE_EVENTS "
        "where memoryOperationType = 0 and start between ? and ?", (b_start, b_end)).fetchall()
    frees_by_addr = defaultdict(list)
    for a, t in c.execute("select address, start from CUDA_GPU_MEMORY_USAGE_EVENTS "
                          "where memoryOperationType = 1 order by start"):
        frees_by_addr[a].append(t)

    def freed_at(t_alloc, addr):
        return next((t for t in frees_by_addr[addr] if t > t_alloc), None)

    residual = [(t, a, n, freed_at(t, a)) for t, a, n in allocs if (f := freed_at(t, a)) is None or f > b_end]
    res_bytes = sum(n for _, _, n, _ in residual)
    by_op = defaultdict(int)
    for t, _, n, _ in residual:
        by_op[innermost_op(t)] += n

    # 反向：seq 落在区间内的 range（反向在另一条线程，用 seq 对回来）
    bw = [(s, e) for txt, s, e in c.execute(
        "select text, start, end from NVTX_EVENTS where text like '%seq = %' and globalTid != ?", (tid,))
          if seq_lo <= int(_SEQ.search(txt).group(1)) <= seq_hi]
    bw_start, bw_end = min(s for s, _ in bw), max(e for _, e in bw)
    bw_alloc = c.execute("select coalesce(sum(bytes),0) from CUDA_GPU_MEMORY_USAGE_EVENTS "
                         "where memoryOperationType = 0 and start between ? and ?", (bw_start, bw_end)).fetchone()[0]
    bw_free = c.execute("select coalesce(sum(bytes),0) from CUDA_GPU_MEMORY_USAGE_EVENTS "
                        "where memoryOperationType = 1 and start between ? and ?", (bw_start, bw_end)).fetchone()[0]
    res_freed_in_bw = sum(n for _, _, n, f in residual if f is not None and bw_start <= f <= bw_end)

    print(f"block{block}  前向 {(b_end - b_start) / 1e6:.1f} ms，aten 算子 {len(ops)} 个，seq {seq_lo}–{seq_hi}")
    print(f"  前向分配 {sum(n for *_, n in allocs) / MiB:.1f} MiB，其中活到 range 结束的（residual）"
          f"{res_bytes / MiB:.1f} MiB / {len(residual)} 个张量")
    print(f"  top {top} 来源：")
    for op, n in sorted(by_op.items(), key=lambda kv: -kv[1])[:top]:
        print(f"    {n / MiB:8.1f} MiB  {n / res_bytes:6.1%}  {op}")
    print(f"  反向窗口 {(bw_end - bw_start) / 1e6:.1f} ms：分配 {bw_alloc / MiB:.1f}，释放 {bw_free / MiB:.1f}"
          f"（其中 residual {res_freed_in_bw / MiB:.1f}），净 {(bw_alloc - bw_free) / MiB:+.1f} MiB")
    print(f"  ⇒ 反向新产生的张量（梯度等）≈ 净变化 + 释放的 residual = "
          f"{(bw_alloc - bw_free + res_freed_in_bw) / MiB:.1f} MiB")
    return res_bytes, by_op


def plot_block(rep: Path | str, block: int = 5, out: Path | str | None = None):
    """把 nsys GUI 里那两行画出来：上图整步的 GPU 显存曲线 + 每层 block range；
    下图放大到 block{block} 的前向，标出为反向保存下来的分配（residual）和它们的来源算子。
    数据全部来自 nsys 的 sqlite（CUDA_GPU_MEMORY_USAGE_EVENTS + NVTX_EVENTS），与 GUI 同源。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = Path(rep)
    c = _sqlite(rep)
    ev = c.execute("select start, bytes, memoryOperationType, address from CUDA_GPU_MEMORY_USAGE_EVENTS "
                   "where memoryOperationType in (0, 1) order by start").fetchall()
    t0 = ev[0][0]
    ts, ys, live = [], [], 0
    for t, n, op, _ in ev:
        live += n if op == 0 else -n
        ts.append((t - t0) / 1e6); ys.append(live / MiB)

    blocks = c.execute("select text, start, end from NVTX_EVENTS where text like 'block%' order by start").fetchall()
    fwd = c.execute("select start, end from NVTX_EVENTS where text = 'forward'").fetchone()
    bwd = c.execute("select start, end from NVTX_EVENTS where text = 'backward'").fetchone()
    (b_start, b_end, tid), = c.execute("select start, end, globalTid from NVTX_EVENTS where text = ?",
                                       (f"block{block}",)).fetchall()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 11), height_ratios=[1, 1.4, .8],
                                        gridspec_kw={"hspace": .5})
    ax3.sharex(ax2)

    # ---- 上图：整步 ----
    ax1.plot(ts, ys, lw=.8, drawstyle="steps-post")
    for name, s, e in blocks:
        ax1.axvspan((s - t0) / 1e6, (e - t0) / 1e6, color="tab:orange" if name == f"block{block}" else "tab:gray",
                    alpha=.35 if name == f"block{block}" else .12, lw=0)
    for lab, (s, e) in (("forward", fwd), ("backward", bwd)):
        ax1.annotate(lab, ((s + e) / 2 / 1e6 - t0 / 1e6, max(ys) * .97), ha="center", fontsize=10)
        ax1.axvline((e - t0) / 1e6, ls="--", lw=.8, c="tab:red", alpha=.6)
    ax1.annotate(f"block{block} forward\n{(b_start - t0) / 1e6:.0f}–{(b_end - t0) / 1e6:.0f} ms, zoomed below ↓",
                 xy=((b_start + b_end) / 2 / 1e6 - t0 / 1e6, ys[0] + max(ys) * .05),
                 xytext=(0, -55), textcoords="offset points", ha="center", fontsize=9, c="tab:orange",
                 arrowprops=dict(arrowstyle="->", color="tab:orange"))
    ax1.set_ylabel("GPU memory (MiB, since trace start)")
    ax1.set_title(f"{rep.stem}: cudaMalloc/cudaFree timeline (caching allocator off); "
                  f"gray = block ranges, orange = block{block}")
    ax1.grid(alpha=.3)

    # ---- 下两行：放大 block 前向，显存曲线 + 算子 range 共用时间轴（就是 nsys GUI 的排版）----
    ops = c.execute("select text, start, end from NVTX_EVENTS where globalTid = ? and text like 'aten::%' "
                    "and start >= ? and end <= ? order by start", (tid, b_start, b_end)).fetchall()
    frees_by_addr = defaultdict(list)
    for a, t in c.execute("select address, start from CUDA_GPU_MEMORY_USAGE_EVENTS "
                          "where memoryOperationType = 1 order by start"):
        frees_by_addr[a].append(t)
    ms = lambda t: (t - t0) / 1e6

    # 显存曲线（block 内）
    seg = [(ms(tt), y) for (tt, *_), y in zip(ev, ys) if b_start <= tt <= b_end]
    ax2.step([t for t, _ in seg], [y for _, y in seg], where="post", lw=1.2, c="tab:blue")
    alloc_marks = []                              # (t_ms, y, bytes, saved, op_text)
    for t, n, op, a in ev:
        if op != 0 or not (b_start <= t <= b_end):
            continue
        f = next((x for x in frees_by_addr[a] if x > t), None)
        saved = f is None or f > b_end
        y = next(yy for tt, yy in seg if tt >= ms(t))
        inner = [o for o in ops if o[1] <= t <= o[2]]
        opname = max(inner, key=lambda o: o[1])[0].split(",")[0].replace("aten::", "") if inner else "?"
        alloc_marks.append((ms(t), y, n, saved, opname))
        ax2.plot(ms(t), y, "o", c="tab:red" if saved else "tab:gray", ms=4)
        if n >= 5 * MiB:
            ax2.annotate(f"+{n / MiB:.0f}M", (ms(t), y), xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=7, c="tab:red" if saved else "tab:gray")
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="tab:red", label="cudaMalloc, saved for backward (residual)"),
                        Patch(color="tab:gray", label="cudaMalloc, freed within the block")],
               loc="upper left", fontsize=9)
    ax2.set_ylabel("GPU memory (MiB)")
    ax2.set_title(f"block{block} forward ({ms(b_start):.1f}–{ms(b_end):.1f} ms): memory (top) aligned with "
                  f"aten ops (bottom), same time axis — like the nsys GUI")
    ax2.grid(alpha=.3, axis="y")
    ax2.tick_params(labelbottom=False)

    # 算子 range 行：只画顶层 aten（不被别的 aten 包住的），交错两行防重叠，
    # 标签只给「时长 ≥ 1% block」或「里面发生过 ≥5 MiB 分配」的
    top_ops = [o for o in ops if not any(p is not o and p[1] <= o[1] and o[2] <= p[2] for p in ops)]
    big_alloc_t = {t for t, _, n, _, _ in alloc_marks if n >= 5 * MiB}
    for i, (txt, s0, e0) in enumerate(top_ops):
        name = txt.split(",")[0].replace("aten::", "")
        row = i % 2
        has_alloc = any(ms(s0) <= t <= ms(e0) for t in big_alloc_t)
        saved_here = any(ms(s0) <= t <= ms(e0) and sv for t, _, _, sv, _ in alloc_marks)
        color = "tab:red" if saved_here else ("tab:gray" if has_alloc else "lightgray")
        ax3.barh(row, ms(e0) - ms(s0), left=ms(s0), height=.8, color=color, alpha=.7, edgecolor="white", lw=.3)
        if has_alloc or (e0 - s0) >= 0.01 * (b_end - b_start):
            ax3.text((ms(s0) + ms(e0)) / 2, row, name, ha="center", va="center", fontsize=8, rotation=90)
    ax3.set_ylim(-.6, 1.6); ax3.set_yticks([]); ax3.set_ylabel("aten ops\n(NVTX)")
    ax3.set_xlabel("time (ms, same clock as the top panel)")
    x_hi = max(ms(o[2]) for o in ops) + 0.05
    ax3.set_xlim(ms(b_start) - 0.02, x_hi); ax2.set_xlim(ms(b_start) - 0.02, x_hi)
    # 竖线把每笔 ≥5 MiB 的分配连到它下面的算子
    for t, y, n, saved, _ in alloc_marks:
        if n >= 5 * MiB:
            ax2.axvline(t, ymin=0, ymax=1, c="tab:red" if saved else "tab:gray", lw=.4, alpha=.5)
            ax3.axvline(t, c="tab:red" if saved else "tab:gray", lw=.4, alpha=.5)
    fig.tight_layout()
    # 下两行共用时间轴，贴紧；上图和中图之间留出标题和注释的空间
    p2, p3 = ax2.get_position(), ax3.get_position()
    ax3.set_position([p3.x0, p2.y0 - p3.height - 0.012, p3.width, p3.height])
    out = Path(out) if out else rep.with_name(f"{rep.stem}_block{block}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"图已写出：{out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    blk = int(args[args.index("--block") + 1]) if "--block" in args else 5
    reps = [a for a in args if a.endswith(".nsys-rep")]
    for r in reps or ["profiles/mem_trace_xl_seq128_fwd_bwd.nsys-rep"]:
        block_residuals(Path(r), blk)
        if "--plot" in args:
            plot_block(Path(r), blk)
