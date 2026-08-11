#!/usr/bin/env python3
"""把 wandb 上所有 run 的 history 拉下来缓存成一个 json。

单独一步是因为拉 73 条 run 要几分钟，而调图样式要反复跑——缓存之后
make_plots.py 就不用再碰网络。

早期几条 run 没起名字（wandb 自动生成的 blooming-shape-1 之类），所以
除了 name 还要存 config，让绘图脚本能按超参认出它们是哪个实验。
"""
import json, os, sys
import wandb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_cache.json")
KEYS = ["train/loss", "train/lr", "train/mfu", "train/tflops",
        "train/tokens_per_sec", "train/peak_mem_gb", "eval/val_loss", "eval/train_loss"]
CFG = ["learning_rate", "batch_size", "accum", "d_model", "num_layers", "num_heads",
       "d_ff", "vocab_size", "max_iters", "seed", "norm", "ffn", "no_rope",
       "tie_embeddings", "context_length", "train_data"]

api = wandb.Api(timeout=60)
runs = api.runs("noviorlumiao/cs336-hw1")
out = {}
for i, r in enumerate(runs, 1):
    try:
        # 不能传 keys=：wandb 只返回**同时含有全部这些 key** 的行，而 train/loss
        # 和 eval/val_loss 记在不同 step 上，交集为空 → 一行都取不到。
        hist = r.history(pandas=False, samples=100000)
    except Exception as e:
        print(f"  [{i}/{len(runs)}] {r.name}: history 取不到（{e}）", file=sys.stderr)
        hist = []
    out[r.id] = {
        "name": r.name,
        "state": r.state,
        "created": str(r.created_at),
        "config": {k: r.config.get(k) for k in CFG},
        "summary": {k: r.summary.get(k) for k in ["eval/val_loss", "train/mfu", "train/peak_mem_gb", "_step"]},
        "history": [{k: v for k, v in row.items()
                     if k in KEYS + ["_step", "_runtime"] and v is not None} for row in hist],
    }
    print(f"  [{i}/{len(runs)}] {r.name:<24} {len(hist):>6} 点  {r.state}")

json.dump(out, open(OUT, "w"))
print(f"\n{len(out)} 条 → {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
