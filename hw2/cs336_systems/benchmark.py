import argparse
import contextlib
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
import timeit
import traceback
from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any, Callable, Tuple

import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_SIZES = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

SWEEP_CONFIGS = {
    "default": {
        "model_type": ["basics"],
        "size": ["small", "medium", "large", "xl", "10B"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [5],
        "steps": [10],
        "mode_and_inference": [
            ("forward", True),
            ("forward", False),
            ("fwd_bwd", False),
            ("full", False),
        ],
    },
    "warmup_by_size": {
        "model_type": ["basics"],
        "size": ["small", "medium", "large"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [0, 1, 2, 5],
        "steps": [10],
        "mode_and_inference": [("full", False)],
    },
    "warmup_xl": {
        "model_type": ["basics"],
        "size": ["xl"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [0, 1, 2, 5],
        "steps": [10],
        "mode_and_inference": [("forward", True)],
    },
}


@dataclass
class BenchConfig:
    model_type: str
    size: str
    mode: str
    inference: bool
    warmup: int
    steps: int
    batch_size: int
    seq_len: int
    vocab_size: int
    device: str
    nvtx: bool = False          # 插 NVTX range（供 nsys profile 用），默认关

    def __post_init__(self):
        # no_grad 下不建反向图，backward 会报 "does not require grad"。
        # 早失败，别等跑到一半才崩（且那个 RuntimeError 不含 "out of memory"，
        # 会穿透 OOM 分支把整个 sweep 带走）。
        if self.inference and self.mode != "forward":
            raise ValueError(
                f"--inference 只能配 --mode forward，当前 mode={self.mode}。"
                "推理模式下不构建反向图，backward/optimizer 无从谈起。"
            )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


@dataclass
class BenchResult:
    model: str
    size: str
    seq_len: int
    batch: int
    warmup: int
    steps: int
    mode: str
    inference: bool
    avg_ms: float
    std_ms: float
    first_ms: float
    rest_avg_ms: float
    peak_mem_gb: float
    status: str
    times_ms: list[float] = field(default_factory=list)


def _sync(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.synchronize()


def _reset_peak(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.reset_peak_memory_stats()


def _range(cfg: BenchConfig, name: str, gate: list | None = None):
    """NVTX range；`--nvtx` 关闭时退化成 nullcontext，零成本。

    为什么要能关：§2.1 的计时结果已经发表，而 §2.2(a) 要拿 nsys 的前向时间
    和它对账。无条件插 NVTX 等于悄悄改了基准线，(a) 就无从比起了。

    `gate`：一个单元素列表做的开关，用来**在预热期间彻底不发 range**。
    必须如此的原因：`nsys stats --filter-nvtx` 默认只取该 range 的**第一个实例**。
    只把预热循环包进 "warmup" 是不够的——预热步内部照样会发 "forward"，
    于是 `--filter-nvtx=forward` 抓到的是被冷启动污染的那一步。
    实测症状：top kernel 的 Max 是 Med 的 9 倍（562us vs 61us）。
    """
    if cfg.nvtx and cfg.is_cuda and (gate is None or gate[0]):
        return nvtx.range(name)
    return contextlib.nullcontext()


def _peak_gb(cfg: BenchConfig) -> float:
    if not cfg.is_cuda:
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def build_model(cfg: BenchConfig) -> Any:
    config = MODEL_SIZES[cfg.size]

    if cfg.model_type == "basics":
        return BasicsTransformerLM(
            vocab_size=cfg.vocab_size,
            context_length=cfg.seq_len,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=10000.0,
            device=torch.device(cfg.device),
        )
    elif cfg.model_type == "flash":
        raise NotImplementedError("Flash Attention hasn't been implemented yet!")
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


def build_batch(cfg: BenchConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    return x, y


def make_step_fn(
    model: Any,
    opt: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    cfg: BenchConfig,
    stage_ptr: list,
    nvtx_on: list | None = None,
) -> Callable[[], None]:
    x, y = batch

    def step():
        # NVTX range 和 stage_ptr 标在同一批边界上，但用途不同、不要合并：
        # stage_ptr 管 OOM 归因，必须始终开着；NVTX 管 profiling，要能关。
        # ⚠️ 这几个 range 里没有 synchronize，所以它们的 CPU 时长是"把 kernel
        #    排完队"的时间，不是 GPU 算完的时间。§2.2(a) 要 GPU 侧耗时的话，
        #    得用 `nsys stats --report cuda_gpu_kern_sum --filter-nvtx=forward`。
        with _range(cfg, "forward", nvtx_on):
            stage_ptr[0] = "forward"
            ctx = torch.no_grad() if cfg.inference else contextlib.nullcontext()
            with ctx:
                logits = model(x)
                loss = cross_entropy(logits, y)

        if cfg.mode in ("fwd_bwd", "full"):
            with _range(cfg, "backward", nvtx_on):
                stage_ptr[0] = "backward"
                loss.backward()

        if cfg.mode == "full":
            with _range(cfg, "optimizer", nvtx_on):
                stage_ptr[0] = "optimizer"
                opt.step()

        if cfg.mode != "forward":
            with _range(cfg, "zero_grad", nvtx_on):
                model.zero_grad(set_to_none=True)

    return step


def _oom_result(cfg: BenchConfig, stage: str, peak: float, status_prefix: str) -> BenchResult:
    return BenchResult(
        model=cfg.model_type, size=cfg.size, seq_len=cfg.seq_len, batch=cfg.batch_size,
        warmup=cfg.warmup, steps=cfg.steps, mode=cfg.mode, inference=cfg.inference,
        avg_ms=float("nan"), std_ms=float("nan"),
        first_ms=float("nan"), rest_avg_ms=float("nan"),
        peak_mem_gb=round(peak, 2),
        status=f"{status_prefix} ({stage})",
    )


def benchmark(cfg: BenchConfig) -> BenchResult:
    stage_ptr = ["init"]
    model = opt = batch = step_fn = None

    try:
        _reset_peak(cfg)
        model = build_model(cfg)
        batch = build_batch(cfg)
        opt = AdamW(model.parameters(), lr=1e-4)
        nvtx_on = [False]          # 预热期间不发阶段 range，见 _range 的说明
        step_fn = make_step_fn(model, opt, batch, cfg, stage_ptr, nvtx_on)

        # 预热单独包一个 range：`--filter-nvtx` 默认只取该 range 的**第一个实例**，
        # 若预热步也叫 "step"，(b) 抓到的就是被冷启动污染的那一步。
        with _range(cfg, "warmup"):
            for _ in range(cfg.warmup):
                step_fn()
                _sync(cfg)

        _reset_peak(cfg)
        nvtx_on[0] = True          # 预热结束，从这里开始才发阶段 range

        times = []
        for _ in range(cfg.steps):
            start = timeit.default_timer()
            # sync 放在 range **内部**：这样 "step" 的时长才等于 timeit 量到的墙钟，
            # (a) 拿它和 §2.1 对账才是同一个量。
            with _range(cfg, "step"):
                step_fn()
                _sync(cfg)
            times.append((timeit.default_timer() - start) * 1000)

        peak_mem = _peak_gb(cfg)
        t = torch.tensor(times, dtype=torch.float64)
        avg = t.mean().item() if cfg.steps > 0 else float("nan")
        std = t.std().item() if cfg.steps > 1 else 0.0
        first = times[0] if times else float("nan")
        rest = t[1:].mean().item() if cfg.steps > 1 else float("nan")

        return BenchResult(
            model=cfg.model_type, size=cfg.size, seq_len=cfg.seq_len, batch=cfg.batch_size,
            warmup=cfg.warmup, steps=cfg.steps, mode=cfg.mode, inference=cfg.inference,
            avg_ms=round(avg, 2), std_ms=round(std, 2),
            first_ms=round(first, 2), rest_avg_ms=round(rest, 2),
            peak_mem_gb=round(peak_mem, 2),
            status="OK",
            times_ms=[round(v, 3) for v in times],
        )

    except RuntimeError as e:
        peak = _peak_gb(cfg)                  # 爆之前用到了多少，§2.5 要这个数
        if "out of memory" in str(e).lower():
            return _oom_result(cfg, stage_ptr[0], peak, "OOM")
        # 非 OOM 的异常：记下来但不要中断整个 sweep（一跑几十分钟，不值当全丢）
        print(f"\n!! 非 OOM 异常 @ stage={stage_ptr[0]}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return _oom_result(cfg, stage_ptr[0], peak, "ERROR")

    finally:
        # 顺序很重要：empty_cache() 只能回收"已经没人引用"的块。
        # 这些还是活着的局部变量，不 del 掉的话这句等于空转（原版就是这个 bug）。
        del step_fn, opt, batch, model
        if cfg.is_cuda:
            torch.cuda.empty_cache()
        # 注意：OOM 分支走到这里时异常还在处理中，抛出点的栈帧仍钉着张量，
        # 这一次 empty_cache() 收不干净。真正的兜底在 sweep() 里——函数返回后再收一次。


def _footnote(cfgs: list[BenchConfig]) -> str:
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
        "**读表须知**\n\n"
        "- `inference=True` 表示前向包在 `torch.no_grad()` 里（不建反向图）；"
        "`False` 则保留 autograd 图，是训练步的前向。**两者不是同一个量，别混着相减。**\n"
        "- 反向和优化器耗时是**减出来的**，不是直接测的："
        "`Backward = fwd_bwd − forward(inference=False)`，`Optimizer = full − fwd_bwd`。"
        "这隐含可加性假设；当两个被减数接近时，差值会落进噪声（可能为负）。\n"
        "- `peak_mem_gb` 只统计**测量段**（warmup 之后已 `reset_peak_memory_stats()`），"
        "且每步 `zero_grad(set_to_none=True)`——这两个设置都会影响峰值。\n"
        "- `first_ms` / `rest_avg_ms` 是第 1 步与第 2 步起的均值，用来看预热是否收敛。\n"
    )


def _cfg_to_argv(cfg: BenchConfig) -> list[str]:
    argv = [
        "--model", cfg.model_type, "--size", cfg.size, "--mode", cfg.mode,
        "--warmup", str(cfg.warmup), "--steps", str(cfg.steps),
        "--batch-size", str(cfg.batch_size), "--seq-len", str(cfg.seq_len),
        "--vocab-size", str(cfg.vocab_size), "--device", cfg.device,
    ]
    if cfg.inference:
        argv.append("--inference")
    if cfg.nvtx:
        argv.append("--nvtx")
    return argv


def _run_isolated(cfg: BenchConfig) -> BenchResult:
    """在独立子进程里跑一个配置。

    为什么需要：CUDA context 创建、kernel 懒加载、显存池首次扩张都是**进程级**的
    一次性开销。同一进程里连着扫表，第一个配置付完账，后面的全白捡——测"第一步有多贵"
    这类实验会直接得出相反结论（实测 small 的 warmup=0 惩罚：独立进程 6.85×，
    同进程排第 3 位 1.00×），而且不报任何错。
    """
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "r.md")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *_cfg_to_argv(cfg), "--out", out],
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


def sweep(cfgs: list[BenchConfig], out_path: str | None = None, isolate: bool = False):
    results = []
    for i, c in enumerate(cfgs, 1):
        print(
            f"[{i}/{len(cfgs)}] model={c.model_type:<6} size={c.size:<6} seq={c.seq_len:<4} "
            f"batch={c.batch_size:<2} warmup={c.warmup:<2} steps={c.steps:<2} "
            f"mode={c.mode:<8} inference={str(c.inference):<5}"
            f"{' [isolated]' if isolate else ''}..."
        )
        results.append(_run_isolated(c) if isolate else benchmark(c))
        # benchmark() 已经返回 → OOM 分支的异常帧此时才真正释放，这里再收一次。
        # 不收的话下一档会继承上一档的碎片，"哪一格 OOM" 就变得依赖运行顺序。
        if c.is_cuda:
            torch.cuda.empty_cache()

    df = pd.DataFrame([{k: v for k, v in r.__dict__.items() if k != "times_ms"} for r in results])
    final_output = df.to_markdown(index=False) + _footnote(cfgs)

    print("\n" + final_output + "\n")
    if out_path:
        with open(out_path, "w") as f:
            f.write(final_output)
        json_path = out_path.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2, ensure_ascii=False)
        print(f"已写出：{out_path}\n         {json_path}（含逐步原始耗时）")


def parse_sweep_config(sweep_def: dict, device: str) -> list[BenchConfig]:
    coupled = sweep_def.get("mode_and_inference", [("forward", False)])
    valid_fields = {f.name for f in fields(BenchConfig)}
    keys = [k for k in sweep_def.keys() if k in valid_fields]
    lists = [sweep_def[k] for k in keys]

    cfgs = []
    for combo in itertools.product(*lists):
        kwargs = dict(zip(keys, combo))
        kwargs["device"] = device
        for mode, inf in coupled:
            cfgs.append(BenchConfig(**kwargs, mode=mode, inference=inf))
    return cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="basics")
    parser.add_argument("--size", type=str, choices=MODEL_SIZES.keys(), default="small")
    parser.add_argument("--mode", type=str, choices=["forward", "fwd_bwd", "full"], default="forward")
    parser.add_argument("--inference", action="store_true", help="前向包 no_grad（只能配 --mode forward）")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nvtx", action="store_true",
                        help="插 NVTX range（warmup/step/forward/backward/optimizer），供 nsys profile 用。"
                             "默认关，以免改变 §2.1 的计时基准线")

    parser.add_argument("--sweep", action="store_true", help="Enable sweep mode")
    parser.add_argument("--isolate", action="store_true",
                        help="每个配置起独立子进程跑。测预热/首步开销这类实验必须开，"
                             "否则后面的配置会白捡前面的进程级预热")
    parser.add_argument("--config", type=str, default="default", help="Name of the sweep config in SWEEP_CONFIGS")
    parser.add_argument("--out", type=str, default=None, help="写出 .md，同名 .json 一并写出")

    args = parser.parse_args()

    if args.sweep:
        if args.config not in SWEEP_CONFIGS:
            raise ValueError(f"Sweep config '{args.config}' not found in SWEEP_CONFIGS dict.")
        cfgs = parse_sweep_config(SWEEP_CONFIGS[args.config], args.device)
        sweep(cfgs, args.out, isolate=args.isolate)
    else:
        cfg = BenchConfig(
            args.model, args.size, args.mode, args.inference,
            args.warmup, args.steps, args.batch_size, args.seq_len, args.vocab_size, args.device,
            args.nvtx,
        )
        sweep([cfg], args.out)


if __name__ == "__main__":
    main()
