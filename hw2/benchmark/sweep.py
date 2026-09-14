"""批量跑 + 落盘：一组 BenchConfig → markdown 表（带可复现脚注）+ json（含逐步原始耗时）。"""
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import date

import pandas as pd
import torch

from .config import SWEEP_CONFIGS, BenchConfig, BenchResult, parse_sweep_config
from .runner import run


def _run_isolated(cfg: BenchConfig) -> BenchResult:
    """在独立子进程里跑一个配置。

    测预热 / 首步开销这类实验必须隔离：不预热的开销大半是进程级一次性成本
    （kernel 懒加载、cuBLAS 句柄、显存池首次 cudaMalloc），同进程连跑会让后面的配置
    白捡前面的预热。
    """
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "r.md")
        proc = subprocess.run(
            [sys.executable, "-m", "benchmark", *cfg.to_argv(), "--out", out],
            capture_output=True, text=True,
        )
        jp = out.rsplit(".", 1)[0] + ".json"
        if not os.path.exists(jp):
            print(proc.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(
                f"隔离子进程无产出：size={cfg.size} mode={cfg.mode} warmup={cfg.warmup}"
            )
        with open(jp) as f:
            return BenchResult(**json.load(f)[0])


def footnote(cfgs: list[BenchConfig]) -> str:
    """每张表都要带，事后补不回来。"""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = "n/a"
    warmups = sorted({c.warmup for c in cfgs})
    steps = sorted({c.steps for c in cfgs})
    return (
        "\n\n---\n\n"
        "**测量条件**（每张表都要带，事后补不回来）\n\n"
        f"- 硬件：{gpu}｜CUDA {torch.version.cuda}\n"
        f"- 软件：Python {platform.python_version()}、torch {torch.__version__}、triton {triton_ver}\n"
        f"- 测法：warmup {warmups} 步 / measure {steps} 步，每步 `torch.cuda.synchronize()`，"
        f"计时用 `timeit.default_timer()`\n"
        f"- 采集日期：{date.today().isoformat()}\n\n"
    )


def _pretty(df: pd.DataFrame) -> str:
    """终端用的紧凑版：常量列折进标题行，avg±std 合成一列，bool 列变 ✓/·。文件里仍存完整表。"""
    from tabulate import tabulate

    df = df.copy()
    always_hide = ["model", "vocab_size"] if "vocab_size" in df.columns else ["model"]
    if len(df) > 1:
        const = {c: df[c].iloc[0] for c in df.columns if df[c].nunique() == 1 and c != "status"}
    else:
        const = {c: df[c].iloc[0] for c in always_hide}
    df = df.drop(columns=list(const))
    if {"avg_ms", "std_ms"} <= set(df.columns):
        df["ms"] = df.apply(lambda r: f"{r.avg_ms:.2f} ± {r.std_ms:.2f}" if r.avg_ms == r.avg_ms else "—", axis=1)
        df = df.drop(columns=["avg_ms", "std_ms"])
        df = df.drop(columns=[c for c in ("first_ms", "rest_avg_ms") if c in df.columns])
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].map({True: "✓", False: "·"})
    df = df.rename(columns={"peak_mem_gib": "peak GiB", "seq_len": "seq", "batch_size": "batch"})
    header = "  ".join(f"{k}={v}" for k, v in const.items())
    return f"[{header}]\n" + tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False, floatfmt=".2f")


def sweep(cfgs: list[BenchConfig], out_path: str | None = None, isolate: bool = False,
          as_frame: bool = False) -> list[BenchResult] | pd.DataFrame:
    """as_frame=True 返回 DataFrame（notebook 里 display 用），否则返回 BenchResult 列表。"""
    results = []
    for i, c in enumerate(cfgs, 1):
        tag = f"{c.size} {c.mode} seq={c.seq_len}" + (" no_grad" if c.inference else "") \
            + (" bf16" if c.autocast else "") + (f" warmup={c.warmup}" if c.warmup != 5 else "")
        # 单行原地刷新，不留历史（notebook 里跑完只剩最终表）
        print(f"\r[{i}/{len(cfgs)}] {tag:<40}", end="", flush=True)
        r = _run_isolated(c) if isolate else run(c)
        results.append(r)
        if c.is_cuda:
            torch.cuda.empty_cache()
    print("\r" + " " * 60 + "\r", end="")

    df = pd.DataFrame([{k: v for k, v in r.__dict__.items() if k != "times_ms"} for r in results])
    report = df.to_markdown(index=False) + footnote(cfgs)
    if not as_frame:
        print("\n" + _pretty(df) + "\n")

    if out_path:
        with open(out_path, "w") as f:
            f.write(report)
        json_path = out_path.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2, ensure_ascii=False)
        print(f"已写出：{out_path}\n         {json_path}（含逐步原始耗时）")
    return df if as_frame else results


def run_config(name: str, out: str, isolate: bool = False) -> pd.DataFrame:
    """SWEEP_CONFIGS[name] 一把跑完，写到 out（.md + 同名 .json），返回 DataFrame。notebook 入口。"""
    return sweep(parse_sweep_config(SWEEP_CONFIGS[name]), out, isolate=isolate, as_frame=True)
