"""配置层：模型规格、sweep 定义、单次测量的输入（BenchConfig）与输出（BenchResult）。

这个文件不 import torch——纯数据，谁都能安全地 import。
"""
import itertools
from dataclasses import dataclass, field, fields

MODEL_SIZES = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

# 每个 sweep 是「字段名 → 取值列表」，parse_sweep_config 做笛卡尔积。
# mode_and_inference 例外：两者耦合（inference 只能配 forward），成对给。
SWEEP_CONFIGS = {
    # §2.1 (b)：各 size × 四种模式
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
    # §2.1 (c)：预热步数的影响。必须配 --isolate，否则后面的配置白捡前面的进程级预热
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
    # §2.4 (c)：fp32 vs bf16 autocast，只要 forward / fwd_bwd（backward 靠相减）。
    # 10B 权重就 48 GiB，什么精度都装不下，不浪费时间；xl 在 fp32 下 OOM，bf16 值得试。
    "mixed_precision": {
        "model_type": ["basics"],
        "size": ["small", "medium", "large", "xl"],
        "seq_len": [512],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [5],
        "steps": [10],
        "autocast": [False, True],
        "mode_and_inference": [("forward", False), ("fwd_bwd", False)],
    },
    # §2.5 (b)(c)：xl 在 seq {128, 2048} 下 forward / full 的峰值显存，fp32 与 bf16 各一遍。
    # 只要 peak_mem_gib 这一列，steps 给 2 就够（峰值和步数无关）。xl@2048 full 预期 OOM，
    # OOM 行的 peak_mem_gib 是炸掉前的水位，也是数据。
    "memory_xl": {
        "model_type": ["basics"],
        "size": ["xl"],
        "seq_len": [128, 2048],
        "batch_size": [4],
        "vocab_size": [10000],
        "warmup": [2],
        "steps": [2],
        "autocast": [False, True],
        "mode_and_inference": [("forward", True), ("fwd_bwd", False), ("full", False)],
    },
    # §3.2 (b)：扫 checkpoint 段长（每段几层）看 fwd_bwd 的峰值显存和 step 时间。
    # 题面 xl@2048 batch 4 在 5090 上任何段长都 OOM（参数+梯度 25.4 GiB），large@2048 batch 4 也只有
    # every ≤ 4 能跑。要让全部段长（含不 checkpoint）都出数，降到 large / batch 1 / seq 1024：
    # 一层 saved tensors ≈ 0.26 GiB，不 checkpoint 也只有 ~17 GiB。跑前重启 kernel。
    "checkpoint_large": {
        "model_type": ["basics"],
        "size": ["large"],
        "seq_len": [1024],
        "batch_size": [1],
        "vocab_size": [10000],
        "warmup": [2],
        "steps": [5],
        "checkpoint_every": [None, 1, 2, 3, 4, 6, 9, 12, 18, 36],
        "mode_and_inference": [("fwd_bwd", False)],
    },
}


@dataclass
class BenchConfig:
    """默认值与 CLI 一致，所以 BenchConfig("xl", 128, "fwd_bwd") 就能用（前三个位置参数：size, seq_len, mode）。"""
    size: str = "small"
    seq_len: int = 512
    mode: str = "forward"   # forward | fwd_bwd | full
    inference: bool = False # 前向包 no_grad（只能配 mode=forward）
    warmup: int = 5
    steps: int = 10
    batch_size: int = 4
    vocab_size: int = 10000
    model_type: str = "basics"
    device: str = "cuda"
    autocast: bool = False  # 前向 + loss 走 bf16 autocast；反向沿用前向 dtype，最终 .grad 仍为 fp32
    nvtx: bool = False      # 插 NVTX range（供 nsys profile 用），默认关，以免改变 §2.1 的计时基准线
    nvtx_attn: bool = False # 再往 attention 内部插三段 range（§2.2 (e)），需 --nvtx
    nvtx_ops: bool = False  # 每层 block range + aten 算子级 range（§2.5 (f) 显存归因），需 --nvtx
    memory_snapshot: str | None = None  # §2.5：测量段的显存快照写到这个 .pickle；None 关
    checkpoint_every: int | None = None # §3.2 (b)：每 every 层包一个 torch.utils.checkpoint 段；None 关

    def __post_init__(self):
        # 早失败，别等跑到一半才崩（且那个 RuntimeError 不含 "out of memory"，
        # 会穿透 OOM 分支把整个 sweep 带走）。
        if (self.nvtx_attn or self.nvtx_ops) and not self.nvtx:
            raise ValueError("--nvtx-attn / --nvtx-ops 需要同时开 --nvtx")
        if self.inference and self.mode != "forward":
            raise ValueError(
                f"--inference 只能配 --mode forward，当前 mode={self.mode}。"
                "推理模式下不构建反向图，backward/optimizer 无从谈起。"
            )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def to_argv(self) -> list[str]:
        """反序列化成 CLI 参数——--isolate 子进程用。加字段记得同步这里。"""
        argv = [
            "--model", self.model_type, "--size", self.size, "--mode", self.mode,
            "--warmup", str(self.warmup), "--steps", str(self.steps),
            "--batch-size", str(self.batch_size), "--seq-len", str(self.seq_len),
            "--vocab-size", str(self.vocab_size), "--device", self.device,
        ]
        for flag in ("inference", "autocast", "nvtx", "nvtx_attn", "nvtx_ops"):
            if getattr(self, flag):
                argv.append("--" + flag.replace("_", "-"))
        if self.memory_snapshot:
            argv += ["--memory-snapshot", self.memory_snapshot]
        if self.checkpoint_every:
            argv += ["--checkpoint-every", str(self.checkpoint_every)]
        return argv


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
    autocast: bool
    avg_ms: float
    std_ms: float
    first_ms: float
    rest_avg_ms: float
    peak_mem_gib: float
    status: str                                     # OK | OOM (stage) | ERROR (stage)
    checkpoint_every: int | None = None             # §3.2 (b)；旧 json 没有这列，默认 None
    times_ms: list[float] = field(default_factory=list)  # 逐步原始耗时，只进 json 不进表

    @classmethod
    def failed(cls, cfg: BenchConfig, stage: str, peak_gib: float, status: str) -> "BenchResult":
        return cls(
            model=cfg.model_type, size=cfg.size, seq_len=cfg.seq_len, batch=cfg.batch_size,
            warmup=cfg.warmup, steps=cfg.steps, mode=cfg.mode,
            inference=cfg.inference, autocast=cfg.autocast, checkpoint_every=cfg.checkpoint_every,
            avg_ms=float("nan"), std_ms=float("nan"),
            first_ms=float("nan"), rest_avg_ms=float("nan"),
            peak_mem_gib=round(peak_gib, 2),
            status=f"{status} ({stage})",
        )


def parse_sweep_config(sweep_def: dict, device: str = "cuda") -> list[BenchConfig]:
    coupled = sweep_def.get("mode_and_inference", [("forward", False)])
    valid = {f.name for f in fields(BenchConfig)}
    keys = [k for k in sweep_def if k in valid]
    cfgs = []
    for combo in itertools.product(*(sweep_def[k] for k in keys)):
        kwargs = dict(zip(keys, combo), device=device)
        for mode, inf in coupled:
            cfgs.append(BenchConfig(**kwargs, mode=mode, inference=inf))
    return cfgs
