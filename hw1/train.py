import argparse
import contextlib
import os
import re
import time

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch


# 评估用的种子偏移：和训练种子错开，保证 eval 的采样和训练的采样互不相关。
EVAL_SEED_OFFSET = 1_000_003

# 自动 checkpoint 的命名。轮转只认这个模式，所以手工留下来的文件（best.pt 之类）
# 不会被误删。
CKPT_PATTERN = re.compile(r"^ckpt_step_(\d+)\.pt$")


def load_dataset(path):
    # .npy 有 128 字节文件头，必须走 np.load；直接 np.memmap 会把文件头
    # 当成 64 个 token 读进来（值可达 3 万，撞上就是 IndexError）。
    # 裸二进制（.bin）没有头，才用 np.memmap。
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


def check_vocab_range(data, vocab_size, name, probe_tokens=1_000_000):
    """抽查 token id 是否落在词表范围内。

    动机：data/ 下同时躺着 10k(TinyStories) 和 32k(OWT) 两套 .npy，配错了会在
    Embedding 里抛 IndexError，而且是训练跑到一半随机抛。这里在启动时就拦下来。
    只查开头一段（全量 max 要读完 5.45 GB），词表配错的话前几千个 token 就会露馅。
    """
    probe = np.asarray(data[: min(len(data), probe_tokens)])
    hi = int(probe.max())
    if hi >= vocab_size:
        raise SystemExit(
            f"{name}: 抽查前 {len(probe)} 个 token，最大 id 是 {hi}，超出 --vocab_size {vocab_size}。"
            f" 多半是数据和词表配错了（是不是把 owt 的 32k 数据配了默认的 10k 词表？）"
        )


def prune_checkpoints(checkpoint_dir, keep_last_n):
    """只保留步数最大的 keep_last_n 个自动 checkpoint，返回删掉的个数。

    keep_last_n <= 0 表示全部保留。按文件名里的步数排序而不是按 mtime——
    resume 之后新写的文件 mtime 更新但步数可能更小，按 mtime 排会删错。
    """
    if keep_last_n <= 0:
        return 0

    ckpts = []
    for name in os.listdir(checkpoint_dir):
        m = CKPT_PATTERN.match(name)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))
    ckpts.sort()

    removed = 0
    for _, path in ckpts[:-keep_last_n]:
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            # 删不掉不是训练该崩的理由，报一声继续跑
            print(f"  (跳过) 无法删除旧 checkpoint {path}: {e}")
    return removed


def evaluate_loss(model, dataset, eval_iters, batch_size, context_length, device, seed, ctx):
    model.eval()
    # 每次调用都从同一个种子重开，所以每个 eval point 看到的是完全相同的样本，
    # 曲线上的起伏才是模型在变，而不是采样噪声。
    rng = np.random.default_rng(seed)
    total = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(dataset, batch_size, context_length, device, rng=rng)
            with ctx:
                logits = model(x)
                loss = cross_entropy(logits, y)
            total += loss.item()
    model.train()
    return total / eval_iters


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    # Data params
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npy or raw .bin)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (.npy or raw .bin)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="只保留最近 N 个 ckpt_step_*.pt，0 或负数表示全部保留；"
                             "手工改名的文件（如 best.pt）不受影响")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)

    # Logging and Evaluation
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                        help="矩阵乘的自动混合精度，仅在 CUDA 上生效")
    parser.add_argument("--compile", action="store_true", help="用 torch.compile 加速，仅在 CUDA 上生效")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    # bf16 autocast：bf16 的动态范围和 fp32 一样宽，不需要 GradScaler；
    # norm / softmax / 残差这些逐元素算子由 autocast 自己留在 fp32。
    use_bf16 = args.dtype == "bfloat16" and is_cuda and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    # 1. Memory-mapped data loading
    print(f"Loading data from {args.train_data} ...")
    train_data = load_dataset(args.train_data)
    val_data = load_dataset(args.val_data)
    print(f"Train data: {len(train_data)} tokens | val: {len(val_data)} tokens")

    check_vocab_range(train_data, args.vocab_size, "train_data")
    check_vocab_range(val_data, args.vocab_size, "val_data")

    # 2. Initialize Model —— device 在构造时就对齐，不要建完再 .to()
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f} M params | device {device} | bf16 autocast: {use_bf16}")

    # raw_model 始终指向未编译的模块。torch.compile 返回的 OptimizedModule 的
    # state_dict 每个 key 都带 `_orig_mod.` 前缀，直接存下来就装不回普通模型了。
    raw_model = model
    if args.compile:
        if is_cuda:
            model = torch.compile(model)
        else:
            print("--compile 只在 CUDA 上启用，已跳过")

    # 3. Initialize Optimizer
    #    weight decay 只打给 2D 及以上的权重矩阵。RMSNorm 的 gain 是 1D、初值为 1，
    #    把它往 0 拉没有正则意义，只是在削弱归一化本身。
    decay_params, no_decay_params = [], []
    for p in raw_model.parameters():
        if p.requires_grad:
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)
    print(f"weight decay: {sum(p.numel() for p in decay_params)/1e6:.2f} M params | "
          f"no decay: {sum(p.numel() for p in no_decay_params)/1e6:.2f} M params")

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    start_iter = 1
    if args.resume:
        start_iter = load_checkpoint(args.resume, raw_model, optimizer) + 1
        print(f"Resumed from {args.resume}, continuing at iter {start_iter}")

    print(f"Starting training on {device} ...")

    tokens_seen = 0
    t0 = time.perf_counter()

    # 4. Training Loop
    #    评估和存盘放在每轮的**末尾**（见 --- E ---）。放在开头的话，标着
    #    `ckpt_step_N` 的文件其实是第 N-1 步之后的状态，从它 resume 会静默跳掉一步，
    #    打印的 "Step N: val_loss" 也比实际少一次更新。
    for it in range(start_iter, args.max_iters + 1):
        # --- A. Learning Rate Scheduling ---
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- B. Forward Pass ---
        # batch 由 (seed, it) 唯一决定：同一个种子跑两次数据流完全一致，
        # 从 checkpoint 续训也能接着原来的数据往下走，不需要存 RNG 状态。
        x, y = get_batch(train_data, args.batch_size, args.context_length, device,
                         rng=np.random.default_rng([args.seed, it]))
        with ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)

        # --- C. Backward Pass & Optimizer Step ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 用 raw_model：compile 后的 wrapper 和它共享同一批 Parameter 对象，
        # 但从未编译的那个取更直白，也不会被类型检查器当成普通函数。
        gradient_clipping(raw_model.parameters(), args.max_grad_norm)

        optimizer.step()

        tokens_seen += args.batch_size * args.context_length

        # --- D. Logging ---
        if it % args.log_interval == 0:
            loss_val = loss.item()          # 这一步本身就是一次同步，放在计时之前
            dt = time.perf_counter() - t0
            print(f"step {it} | loss {loss_val:.4f} | lr {lr:.4e} | {tokens_seen/dt:.0f} tok/s")
            tokens_seen = 0
            t0 = time.perf_counter()

        # --- E. Evaluation and Checkpointing（在这一步的更新之后）---
        if it % args.eval_interval == 0 or it == args.max_iters:
            train_loss = evaluate_loss(model, train_data, args.eval_iters, args.batch_size,
                                       args.context_length, device, args.seed + EVAL_SEED_OFFSET, ctx)
            val_loss = evaluate_loss(model, val_data, args.eval_iters, args.batch_size,
                                     args.context_length, device, args.seed + EVAL_SEED_OFFSET + 1, ctx)
            print(f"Step {it}: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")

            # 先写临时文件再 rename：中途被打断也不会留下半截的 checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_step_{it}.pt")
            tmp_path = ckpt_path + ".tmp"
            save_checkpoint(raw_model, optimizer, it, tmp_path)
            os.replace(tmp_path, ckpt_path)
            # 轮转放在 replace 之后：新的那份已经完整落盘，才轮到删旧的
            prune_checkpoints(args.checkpoint_dir, args.keep_last_n)

            # 评估和存盘不计入训练吞吐
            tokens_seen = 0
            t0 = time.perf_counter()


if __name__ == "__main__":
    main()
