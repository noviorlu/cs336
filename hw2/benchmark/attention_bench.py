"""§4.1 / §4.2 / §4.5 attention 微基准。

被测对象是一个函数 attn(Q, K, V) -> O，Q/K/V 形状 [batch, seq, d]，没有 head 维。
§4.1 传 hw1 的 scaled_dot_product_attention，§4.2 传 torch.compile 后的版本，§4.5 传 FlashAttention。

对一个 (d, seq) 点，按作业 (iii)–(vi) 的顺序做：
    1. 造随机 Q/K/V
    2. 预热：完整跑几次 前向+反向
    3. 计时 100 次前向
    4. 跑一次前向、停在反向前，记 memory_allocated()
    5. 计时 100 次反向（每次先重新前向，只计 backward 那一段）
    每次前向/反向后 torch.cuda.synchronize()
OOM 不抛出，记在 status 里。
"""
import itertools
import json
import timeit
from dataclasses import asdict, dataclass

import pandas as pd
import torch

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


@dataclass
class AttnResult:
    d: int
    seq: int
    fwd_ms: float | None = None            # 每次前向平均 ms
    bwd_ms: float | None = None            # 每次反向平均 ms
    mem_before_bwd_gib: float | None = None  # 前向结束、反向开始前的 memory_allocated()
    status: str = "ok"                     # ok / OOM@forward / OOM@backward


def bench_attention(attn, d: int, seq: int, *, batch: int = 8, warmup: int = 5, steps: int = 100,
                    dtype=torch.float32) -> AttnResult:
    res = AttnResult(d, seq)
    torch.cuda.empty_cache()
    try:
        # 1. 随机输入，requires_grad 才有反向
        Q = torch.randn(batch, seq, d, device="cuda", dtype=dtype, requires_grad=True)
        K = torch.randn(batch, seq, d, device="cuda", dtype=dtype, requires_grad=True)
        V = torch.randn(batch, seq, d, device="cuda", dtype=dtype, requires_grad=True)

        # 2. 预热：前向和反向用的是不同的 kernel，两边都要跑到；compile/triton 版的 JIT 也在这里发生
        for _ in range(warmup):
            attn(Q, K, V).sum().backward()
        torch.cuda.synchronize()

        # 3. 前向计时。no_grad：不建图、不存 saved tensors，只量前向 kernel
        with torch.no_grad():
            t0 = timeit.default_timer()
            for _ in range(steps):
                attn(Q, K, V)
                torch.cuda.synchronize()
            res.fwd_ms = (timeit.default_timer() - t0) / steps * 1e3

        # 4. 反向前显存：带图跑一次前向，此时 S/P 等 saved tensors 都在显存里
        O = attn(Q, K, V)
        torch.cuda.synchronize()
        res.mem_before_bwd_gib = torch.cuda.memory_allocated() / 1024**3
        del O

        # 5. 反向计时。backward 需要一张新图，所以每次都要重新前向；只把 backward 那段计进去
        total = 0.0
        for _ in range(steps):
            O = attn(Q, K, V)
            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            O.sum().backward()
            torch.cuda.synchronize()
            total += timeit.default_timer() - t0
        res.bwd_ms = total / steps * 1e3

    except torch.cuda.OutOfMemoryError:
        res.status = "OOM@forward" if res.fwd_ms is None else "OOM@backward"
    finally:
        # 释放本点的张量，避免 OOM 后残留影响下一个点
        Q = K = V = O = None
        torch.cuda.empty_cache()
    return res


def attention_sweep(attn, out_path: str | None = None, *, ds=D_MODELS, seqs=SEQ_LENS, **kw) -> pd.DataFrame:
    """扫 ds × seqs 全部组合，返回 DataFrame；给 out_path（.md）时同时落 .md 和 .json。"""
    grid = list(itertools.product(ds, seqs))
    rows = []
    for i, (d, s) in enumerate(grid, 1):
        print(f"\r[{i}/{len(grid)}] d={d} seq={s}", end="", flush=True)
        rows.append(asdict(bench_attention(attn, d, s, **kw)))
    print()
    df = pd.DataFrame(rows)
    if out_path:
        with open(out_path, "w") as f:
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write(f"\n\n- {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                    f"batch {kw.get('batch', 8)}, warmup {kw.get('warmup', 5)}, steps {kw.get('steps', 100)}\n")
        with open(out_path.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=1)
    return df
