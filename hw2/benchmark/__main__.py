"""CLI：只做「命令行参数 → BenchConfig」的翻译，不含任何测量逻辑。

    uv run python -m benchmark --size medium --mode full
    uv run python -m benchmark --sweep --config mixed_precision --out notes/assets/s2/x.md
"""
import argparse

from .config import MODEL_SIZES, SWEEP_CONFIGS, BenchConfig, parse_sweep_config
from .sweep import sweep


def main():
    p = argparse.ArgumentParser(description="§2 benchmark：timeit 计时（默认）/ NVTX 探针（--nvtx）")
    p.add_argument("--model", default="basics")
    p.add_argument("--size", choices=MODEL_SIZES.keys(), default="small")
    p.add_argument("--mode", choices=["forward", "fwd_bwd", "full"], default="forward")
    p.add_argument("--inference", action="store_true", help="前向包 no_grad（只能配 --mode forward）")
    p.add_argument("--autocast", action="store_true",
                   help="前向 + loss 走 bf16 autocast；反向沿用前向的 dtype，最终 .grad 仍为 fp32")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--device", default="cuda")

    g = p.add_argument_group("NVTX（供 nsys profile 用；默认关，以免改变 §2.1 的计时基准线）")
    g.add_argument("--nvtx", action="store_true", help="插 warmup/step/forward/backward/optimizer range")
    g.add_argument("--nvtx-attn", action="store_true",
                   help="§2.2 (e)：attention 内部再插 scores/softmax/matmul 三段。"
                        "需配合 --nvtx；会把 attention 串行化，不要拿它的墙钟去对 §2.1")
    g.add_argument("--nvtx-ops", action="store_true",
                   help="§2.5 (f)：每层 block{i} range + 每个 aten 算子一个 range（emit_nvtx）。"
                        "需配合 --nvtx 和 nsys --cuda-memory-usage=true；range 数量巨大，只跑 1 步")

    g = p.add_argument_group("显存（§2.5）")
    g.add_argument("--memory-snapshot", metavar="PATH", default=None,
                   help="把测量段的显存分配历史写成 .pickle（拖进 pytorch.org/memory_viz）。"
                        "建议配 --steps 1 或 2，时间线才看得清；OOM 时也会落盘")

    g = p.add_argument_group("批量")
    g.add_argument("--sweep", action="store_true", help="按 --config 指定的 SWEEP_CONFIGS 条目批量跑")
    g.add_argument("--config", default="default", choices=SWEEP_CONFIGS.keys())
    g.add_argument("--isolate", action="store_true",
                   help="每个配置起独立子进程。测预热/首步开销必须开，否则后面的配置白捡前面的进程级预热")
    g.add_argument("--out", default=None, help="写出 .md，同名 .json 一并写出")

    a = p.parse_args()

    if a.sweep:
        cfgs = parse_sweep_config(SWEEP_CONFIGS[a.config], a.device)
    else:
        cfgs = [BenchConfig(
            model_type=a.model, size=a.size, mode=a.mode, inference=a.inference,
            warmup=a.warmup, steps=a.steps, batch_size=a.batch_size, seq_len=a.seq_len,
            vocab_size=a.vocab_size, device=a.device,
            autocast=a.autocast, nvtx=a.nvtx, nvtx_attn=a.nvtx_attn, nvtx_ops=a.nvtx_ops,
            memory_snapshot=a.memory_snapshot,
        )]
    sweep(cfgs, a.out, isolate=a.isolate)


if __name__ == "__main__":
    main()
