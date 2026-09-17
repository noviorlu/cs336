"""§2.2 图 2.2-1 的实测点：在每层 block 前向结束、反向结束时读 memory_allocated()，得到与 M(j) 同横轴的曲线。
small@512 / xl@128，fp32 / bf16 autocast，fwd_bwd 一步（预热一步后测第二步）。输出 peak_moment_measured.json。"""
import json, sys
import torch
from benchmark import BenchConfig
from benchmark.model_bench import build_model, build_batch
from cs336_basics.nn_utils import cross_entropy

GiB = 1024**3

def measure(size, seq, autocast):
    torch.cuda.empty_cache()
    cfg = BenchConfig(size, seq, "fwd_bwd")
    model = build_model(cfg); x, y = build_batch(cfg)
    layers = list(model.layers)
    fwd, bwd = [], []
    def rec(lst):
        torch.cuda.synchronize(); lst.append(torch.cuda.memory_allocated() / GiB)
    def fhook(m, inp, out):
        rec(fwd)
        # out 的梯度算出来的时刻 = 上一层（更靠近输出的那层）反向做完的时刻
        out.register_hook(lambda g: rec(bwd))
    for blk in layers:
        blk.register_forward_hook(fhook)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            loss = cross_entropy(model(x), y)
        loss.backward(); rec(bwd)                 # 最后一层（layer 0）反向做完
        model.zero_grad(set_to_none=True)
    step(); fwd.clear(); bwd.clear()            # 预热一步：cuBLAS 句柄等一次性分配不算进曲线
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated() / GiB
    step()
    peak = torch.cuda.max_memory_allocated() / GiB
    del model, x, y; torch.cuda.empty_cache()
    return dict(size=size, seq=seq, autocast=autocast, start=start, fwd=fwd, bwd=bwd[1:], peak=peak)

out = []
for size, seq in [("small", 512), ("xl", 128)]:
    for ac in [False, True]:
        r = measure(size, seq, ac); out.append(r)
        print(f"{size}@{seq} autocast={ac}: start {r['start']:.2f} | fwd end {r['fwd'][-1]:.2f} | bwd end {r['bwd'][-1]:.2f} | peak {r['peak']:.2f}")
json.dump(out, open("notes/assets/s2/peak_moment_measured.json", "w"), indent=1)
