"""timeit 计时核心：建模型 → 预热 → 逐步计时 → 汇总。一次 run_model() 对应表里的一行。

阶段划分（§2.5 的显存采样会挂在同一条线上，见 run_model() 里的注释）：

    setup  ──▶  warmup  ──▶  [reset peak / probes.live=True]  ──▶  measure  ──▶  teardown
"""
import contextlib
import sys
import timeit
import traceback
from typing import Any, Callable, Tuple

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from .config import MODEL_SIZES, BenchConfig, BenchResult
from .checkpointing import apply_checkpointing
from .memory import record_snapshot
from .nvtx import Probes


def build_model(cfg: BenchConfig) -> Any:
    spec = MODEL_SIZES[cfg.size]
    if cfg.model_type == "basics":
        return BasicsTransformerLM(
            vocab_size=cfg.vocab_size,
            context_length=cfg.seq_len,
            d_model=spec["d_model"],
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            d_ff=spec["d_ff"],
            rope_theta=10000.0,
            device=torch.device(cfg.device),
        )
    if cfg.model_type == "flash":
        raise NotImplementedError("Flash Attention hasn't been implemented yet!")
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
    probes: Probes,
    stage: list,            # 单元素 list，异常时告诉 run_model() 死在哪个阶段
) -> Callable[[], None]:
    x, y = batch

    def grad_ctx():
        return torch.no_grad() if cfg.inference else contextlib.nullcontext()

    def amp_ctx():
        # autocast 只包前向 + loss，backward 在外：反向 dtype 在前向建图时就定了，
        # 块内调 backward 不改变它，只会让块内额外算子意外走 autocast。
        # loss 也要在块内：cross_entropy 是自写函数，autocast 只认识里面的 exp/sum/log，
        # 放块内它们才会被提升到 fp32。
        if cfg.autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def step():
        with probes.emit_ops():   # §2.5 (f) 用；平时是 nullcontext
            with probes.phase("forward"):
                stage[0] = "forward"
                with grad_ctx(), amp_ctx():
                    logits = model(x)
                    loss = cross_entropy(logits, y)

            if cfg.mode in ("fwd_bwd", "full"):
                with probes.phase("backward"):
                    stage[0] = "backward"
                    loss.backward()

            if cfg.mode == "full":
                with probes.phase("optimizer"):
                    stage[0] = "optimizer"
                    opt.step()

            if cfg.mode != "forward":
                with probes.phase("zero_grad"):
                    model.zero_grad(set_to_none=True)

    return step


def _sync(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.synchronize()


def _reset_peak(cfg: BenchConfig) -> None:
    if cfg.is_cuda:
        torch.cuda.reset_peak_memory_stats()


def _peak_gib(cfg: BenchConfig) -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if cfg.is_cuda else 0.0  # GiB


def run_model(cfg: BenchConfig) -> BenchResult:
    stage = ["init"]
    probes = Probes(cfg)
    model = opt = batch = step_fn = None

    try:
        # ---- setup ----
        _reset_peak(cfg)
        model = build_model(cfg)
        if cfg.checkpoint_every:
            apply_checkpointing(model, cfg.checkpoint_every)
        batch = build_batch(cfg)
        opt = AdamW(model.parameters(), lr=1e-4)
        probes.install_attention_probes()
        probes.install_block_ranges(model)
        step_fn = make_step_fn(model, opt, batch, cfg, probes, stage)

        # §2.5 的快照从预热就开始记：xl@2048 在第一个前向就 OOM，只包测量段会什么都留不下。
        # 代价是时间线第一步带着 cuBLAS workspace 等一次性初始化，看图时看第二步。
        with record_snapshot(cfg.memory_snapshot):
            # ---- warmup ----（探针闭嘴，见 Probes.range 的注释）
            with probes.range("warmup"):
                for _ in range(cfg.warmup):
                    step_fn()
                    _sync(cfg)

            # ---- 预热 → 测量的分界 ----
            # peak 计数器清零：peak_mem_gib 只统计测量段。
            _reset_peak(cfg)
            probes.live = True

            # ---- measure ----
            times = []
            for _ in range(cfg.steps):
                start = timeit.default_timer()
                # sync 放在 range **内部**：这样 "step" 的时长才等于 timeit 量到的墙钟
                with probes.range("step"):
                    step_fn()
                    _sync(cfg)
                times.append((timeit.default_timer() - start) * 1000)

        t = torch.tensor(times, dtype=torch.float64)
        return BenchResult(
            model=cfg.model_type, size=cfg.size, seq_len=cfg.seq_len, batch=cfg.batch_size,
            warmup=cfg.warmup, steps=cfg.steps, mode=cfg.mode,
            inference=cfg.inference, autocast=cfg.autocast, checkpoint_every=cfg.checkpoint_every,
            avg_ms=round(t.mean().item(), 2) if cfg.steps > 0 else float("nan"),
            std_ms=round(t.std().item(), 2) if cfg.steps > 1 else 0.0,
            first_ms=round(times[0], 2) if times else float("nan"),
            rest_avg_ms=round(t[1:].mean().item(), 2) if cfg.steps > 1 else float("nan"),
            peak_mem_gib=round(_peak_gib(cfg), 2),
            status="OK",
            times_ms=[round(v, 3) for v in times],
        )

    except RuntimeError as e:
        peak = _peak_gib(cfg)
        if "out of memory" in str(e).lower():
            return BenchResult.failed(cfg, stage[0], peak, "OOM")
        print(f"\n!! 非 OOM 异常 @ stage={stage[0]}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return BenchResult.failed(cfg, stage[0], peak, "ERROR")

    finally:
        # ---- teardown ----
        del step_fn, opt, batch, model
        if cfg.is_cuda:
            torch.cuda.empty_cache()
