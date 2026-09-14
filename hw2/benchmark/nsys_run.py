"""把 nsys profile 包成 Python 调用：BenchConfig → profiles/<name>.nsys-rep。

nsys 只能从进程启动时开始采集（要 hook CUDA driver），所以底层一定是子进程；
这里把 subprocess 细节收起来，notebook / 脚本只描述「跑什么配置、存到哪」。
"""
import os
import subprocess
import sys
from pathlib import Path

from .config import BenchConfig


def nsys_profile(cfg: BenchConfig, rep: str | Path, *, memory: bool = False,
                 env: dict | None = None, quiet: bool = False) -> bool:
    """在 nsys 下跑一个 BenchConfig，报告写到 `rep`.nsys-rep，stdout/stderr 写到 `rep`.log。

    memory=True：加 --cuda-memory-usage=true 并设 PYTORCH_NO_CUDA_MEMORY_CACHING=1
                 （§2.5 (f)：关 caching allocator，nsys 才看得到单个张量的 cudaMalloc）。
    返回子进程是否成功；失败时 .log 里有 traceback。
    """
    rep = Path(rep)
    rep.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["nsys", "profile", "--trace=cuda,nvtx", "-o", str(rep), "--force-overwrite", "true"]
    run_env = {**os.environ, **(env or {})}
    if memory:
        cmd.append("--cuda-memory-usage=true")
        run_env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    cmd += ["--", sys.executable, "-m", "benchmark", *cfg.to_argv()]

    with open(rep.with_suffix(".log"), "w") as log:
        ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=run_env).returncode == 0
    if not quiet:
        print(f"{'✓' if ok else '✗ (见 .log)'} {cfg.size:<7} seq={cfg.seq_len:<5} {cfg.mode:<8} → {rep}.nsys-rep")
    return ok


def snapshot(cfg: BenchConfig, pickle_path: str | Path, quiet: bool = False) -> bool:
    """在独立子进程里跑一个配置并落 torch 显存快照（§2.5 (a)(e)）。

    独立进程是为了每次都从干净的显存池开始；OOM 也会落盘（runner 里 finally 写）。
    """
    p = Path(pickle_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = BenchConfig(**{**cfg.__dict__, "memory_snapshot": str(p)})
    with open(p.with_suffix(".log"), "w") as log:
        ok = subprocess.run([sys.executable, "-m", "benchmark", *cfg.to_argv()],
                            stdout=log, stderr=subprocess.STDOUT).returncode == 0
    if not quiet:
        print(f"{'✓' if ok else '✗ (多半是 OOM，快照照样有)'} {cfg.size:<7} seq={cfg.seq_len:<5} {cfg.mode:<8} → {p}")
    return ok


def nsys_stats(rep: str | Path, report: str = "nvtx_sum", filter_nvtx: str | None = None):
    """`nsys stats` 的 Python 版：返回 pandas.DataFrame。

    report      nvtx_sum（各 range 总耗时）| cuda_gpu_kern_sum（各 kernel GPU 时间）| ...
    filter_nvtx 只统计落在该 NVTX range 内的 kernel，如 "forward"。
    ns 列顺手换算成 ms。
    """
    import io
    import pandas as pd

    rep = Path(rep)
    if rep.suffix != ".nsys-rep":
        rep = rep.with_suffix(".nsys-rep")
    cmd = ["nsys", "stats", "--force-export=true", "--report", report, "--format", "csv", str(rep)]
    if filter_nvtx:
        cmd.insert(-1, f"--filter-nvtx={filter_nvtx}")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    # 前几行是 "Generating SQLite..." / "Processing..." 日志，表头是第一行含逗号的
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    for col in [c for c in df.columns if c.endswith("(ns)")]:
        df[col.replace("(ns)", "(ms)")] = df.pop(col) / 1e6
    return df
