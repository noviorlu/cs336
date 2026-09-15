"""§2 benchmark 包。

    config.py    模型规格、sweep 定义、BenchConfig / BenchResult（纯数据，不 import torch）
    model_bench.py  整模型计时核心：run_model(cfg) → BenchResult
    nvtx.py      NVTX 探针（Probes），只服务 nsys profile；关掉时 model_bench 感知不到它
    memory.py    显存快照（record_snapshot），只服务 §2.5 memory_viz；同上
    sweep.py     批量跑、--isolate 子进程、markdown/json 落盘
    checkpointing.py  §3.2 (b) 的 checkpoint_every：每 N 层包一个 checkpoint 段
    nsys_run.py  nsys profile / nsys stats / 显存快照的 Python 包装（子进程），notebook 不碰命令行
    __main__.py  CLI：python -m benchmark

以后加的采样器照 nvtx.py / memory.py 的样子单独成文件，
在 model_bench.run_model() 的「预热 → 测量」分界处挂进去。
"""
from .config import MODEL_SIZES, SWEEP_CONFIGS, BenchConfig, BenchResult, parse_sweep_config
from .model_bench import run_model
from .sweep import run_config, sweep
from .nsys_run import nsys_profile, nsys_stats, snapshot

__all__ = ["MODEL_SIZES", "SWEEP_CONFIGS", "BenchConfig", "BenchResult", "parse_sweep_config", "run_model", "sweep", "run_config", "nsys_profile", "nsys_stats", "snapshot"]
