import argparse
import contextlib
import itertools
import json
import platform
import sys
import timeit
import traceback
from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any, Callable, Tuple

import pandas as pd
import torch

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

# ==========================================
# 扫表配置区 (Sweep Configurations)
# ==========================================
SWEEP_CONFIGS = {
    "default": {
        "model_type": ["basics"],
        "size": ["small", "medium", "large", "xl", "10B"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [5],
        "steps": [10],
        # inference=True 只允许配 forward（no_grad 下 backward 无图可回）
        "mode_and_inference": [
            ("forward", True),
            ("forward", False),
            ("fwd_bwd", False),
            ("full", False),
        ],
    },
    "warmup_test": {
        "model_type": ["basics"],
        "size": ["medium"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [0, 1, 2, 5, 10],
        "steps": [10],
        "mode_and_inference": [("full", False)],
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
    first_ms: float          # 第 1 步单独拎出来：2.1(c) 的结论就藏在这里
    rest_avg_ms: float       # 第 2 步起的均值，和 first_ms 对照
    peak_mem_gb: float
    status: str
    times_ms: list[float] = field(default_factory=list)   # 逐步原始耗时，落 JSON 用


# ---------- 设备工具：CPU 上这些 CUDA 调用会直接崩，统一包一层 ----------

def _sync(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.synchronize()


def _reset_peak(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.reset_peak_memory_stats()


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
) -> Callable[[], None]:
    x, y = batch

    def step():
        stage_ptr[0] = "forward"
        ctx = torch.no_grad() if cfg.inference else contextlib.nullcontext()
        with ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)

        if cfg.mode in ("fwd_bwd", "full"):
            stage_ptr[0] = "backward"
            loss.backward()

        if cfg.mode == "full":
            stage_ptr[0] = "optimizer"
            opt.step()

        if cfg.mode != "forward":
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
        _reset_peak(cfg)                      # 从干净的水位开始
        model = build_model(cfg)
        batch = build_batch(cfg)
        opt = AdamW(model.parameters(), lr=1e-4)
        step_fn = make_step_fn(model, opt, batch, cfg, stage_ptr)

        for _ in range(cfg.warmup):
            step_fn()
            _sync(cfg)

        # ★ 峰值只统计测量段：warmup 期间分配器还在扩张，混进来的不是稳态峰值。
        #   （想看含 warmup 的峰值就把这行挪到 warmup 之前，但要同步改脚注。）
        _reset_peak(cfg)

        times = []
        for _ in range(cfg.steps):
            start = timeit.default_timer()
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


def sweep(cfgs: list[BenchConfig], out_path: str | None = None):
    results = []
    for i, c in enumerate(cfgs, 1):
        print(
            f"[{i}/{len(cfgs)}] model={c.model_type:<6} size={c.size:<6} seq={c.seq_len:<4} "
            f"batch={c.batch_size:<2} warmup={c.warmup:<2} steps={c.steps:<2} "
            f"mode={c.mode:<8} inference={str(c.inference):<5}..."
        )
        results.append(benchmark(c))
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

    parser.add_argument("--sweep", action="store_true", help="Enable sweep mode")
    parser.add_argument("--config", type=str, default="default", help="Name of the sweep config in SWEEP_CONFIGS")
    parser.add_argument("--out", type=str, default=None, help="写出 .md，同名 .json 一并写出")

    args = parser.parse_args()

    if args.sweep:
        if args.config not in SWEEP_CONFIGS:
            raise ValueError(f"Sweep config '{args.config}' not found in SWEEP_CONFIGS dict.")
        cfgs = parse_sweep_config(SWEEP_CONFIGS[args.config], args.device)
        sweep(cfgs, args.out)
    else:
        cfg = BenchConfig(
            args.model, args.size, args.mode, args.inference,
            args.warmup, args.steps, args.batch_size, args.seq_len, args.vocab_size, args.device,
        )
        sweep([cfg], args.out)


if __name__ == "__main__":
    main()
