import argparse
import contextlib
import os
import re
import sys
import time
import yaml

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM, build_attention_mask
from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.bpe_tokenizer import lookup_token_id


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


def model_flops_per_token(args):
    """一步训练（前向+反向）每个 token 的模型 FLOPs，用来算 MFU。

    不用常见的 6P 近似，因为它在小模型上偏得很厉害，而且偏的方向不止一个：
      - 6P 把 embedding 当成算力。查表是索引，0 FLOP。本作业 V=10000、d=512，
        embedding 占 P 的 22.6%，等于凭空多出 30.7 MFLOPs/token
      - 6P 又漏掉注意力里那两个 T² 项（QKᵀ 和 scores@V）
    这里按 CS336 §5.4 的口径逐个算子加，前向 F 再乘 3（反向约等于前向的 2 倍）。
    参数在 config 里改了（d_ff、层数……）这里自动跟着变。
    """
    T, L, d, dff, V = (args.context_length, args.num_layers,
                       args.d_model, args.d_ff, args.vocab_size)
    n_ff = 3 if args.ffn == "swiglu" else 2      # SwiGLU 三个矩阵，无门控的 SiLU 两个
    # MHA：d_q = d_kv = d_model。RMSNorm、SiLU、残差这些逐元素算子的 FLOPs 相比
    # 矩阵乘可以忽略（数量级差 d 倍），消融开关不影响这个函数除 n_ff 外的部分。
    f_layer = 4 * 2 * T * d * d + 2 * (2 * T * T * d) + n_ff * 2 * T * d * dff
    forward = L * f_layer + 2 * T * d * V          # 末尾是 lm_head
    return 3.0 * forward / T


def evaluate_loss(model, dataset, eval_iters, batch_size, context_length, device, seed, ctx, doc_sep_id=None):
    model.eval()
    # 每次调用都从同一个种子重开，所以每个 eval point 看到的是完全相同的样本，
    # 曲线上的起伏才是模型在变，而不是采样噪声。
    rng = np.random.default_rng(seed)
    total = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(dataset, batch_size, context_length, device, rng=rng)
            with ctx:
                seq_len = x.shape[-1]
                mask = build_attention_mask(seq_len, x.device, x=x, doc_sep_id=doc_sep_id)
                
                logits = model(x, mask=mask)
                loss = cross_entropy(logits, y)
            total += loss.item()
    model.train()
    return total / eval_iters


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    parser.add_argument("--config", type=str, default=None, help="传入 JSON 或 YAML 配置文件路径，直接加载所有超参数")
    
    # Data params
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data (.npy or raw .bin)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data (.npy or raw .bin)")
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
    parser.add_argument("--doc_mask_vocab", type=str, default=None,
                        help="词表 json 的路径（如 data/tinystories_vocab.json）。给了就启用 "
                             "document mask：从词表里解析出 <|endoftext|> 的 id，挡住注意力跨越"
                             "文档边界。代价是每步多一个 [B, 1, m, m] 的 bool 张量。"
                             "不手填 id 是因为它随词表变（10k 是 9999、32k 是 31999），"
                             "填错不会崩，只会静默生成一份错误的 mask")

    # 架构消融（handout §7.3）。默认值就是基线模型，不传任何一个 = 原来的行为。
    parser.add_argument("--norm", type=str, default="pre", choices=["pre", "post", "none"],
                        help="RMSNorm 放哪。pre=基线（式 25/26）；post=post_norm_ablation（式 27/28，"
                             "同时去掉 ln_final）；none=layer_norm_ablation（连 ln_final 一起拆光）")
    parser.add_argument("--ffn", type=str, default="swiglu", choices=["swiglu", "silu"],
                        help="前馈网络类型。swiglu=基线（3 个矩阵）；silu=swiglu_ablation 的对照组"
                             "（2 个矩阵，无门控）。选 silu 且没手动给 --d_ff 时，d_ff 自动取 "
                             "4·d_model 来对齐参数量")
    parser.add_argument("--no_rope", action="store_true",
                        help="no_pos_emb：完全去掉位置编码（NoPE），和基线的 RoPE 对比")

    # Logging and Evaluation
    parser.add_argument("--eval_interval", type=int, default=0, help="0表示自动设为总步数的 1/10")
    parser.add_argument("--eval_iters", type=int, default=20, help="每次评测使用的 batch 数量")
    parser.add_argument("--log_interval", type=int, default=0, help="0表示自动设为总步数的 1/100")
    
    # MFU 相关配置
    parser.add_argument("--peak_flops", type=float, default=209.5e12,
                        help="显卡理论峰值 FLOPs，算 MFU 的分母。默认 209.5e12 = RTX 5090 的 "
                             "bf16 tensor core（FP32 累加，PyTorch 走的就是这条）。"
                             "要填对应 dtype 的**稠密**峰值：5090 那个 419 是带 sparsity 的，"
                             "104.8 是 FP32 shader；A100 bf16 是 312e12，H100 SXM 是 990e12")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                        help="矩阵乘的自动混合精度，仅在 CUDA 上生效")
    parser.add_argument("--compile", action="store_true", help="用 torch.compile 加速，仅在 CUDA 上生效")
    parser.add_argument("--wandb", type=str, default=None, help="Weights & Biases project name. If provided, enables wandb logging.")

    # 第一遍解析：抓取 --config 和 --resume
    args_config, remaining = parser.parse_known_args()
    config_path = args_config.config

    # 如果没传 --config 但传了 --resume，自动去 checkpoint 同级目录下找配置
    if not config_path and args_config.resume:
        ckpt_dir = os.path.dirname(args_config.resume)
        if os.path.exists(os.path.join(ckpt_dir, "config.yaml")):
            config_path = os.path.join(ckpt_dir, "config.yaml")
        elif os.path.exists(os.path.join(ckpt_dir, "config.json")):
            config_path = os.path.join(ckpt_dir, "config.json")

    if config_path:
        if config_path.endswith((".yaml", ".yml")):
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        # 用文件里的参数去覆盖 argparse 的默认值
        parser.set_defaults(**config_dict)

    # 第二遍解析：用命令行传进来的参数去覆盖（如果有的话）
    args = parser.parse_args()

    if not args.train_data or not args.val_data:
        parser.error("必须提供 --train_data 和 --val_data（可通过命令行传，或写在 config 文件里）")

    # SiLU 前馈只有两个矩阵，d_ff 要取 4·d_model 才和 SwiGLU 的 8/3·d_model×3 对齐
    # 参数量（handout 明确要求）。只在没显式指定 --d_ff 时自动改，命令行给了就听命令行。
    if args.ffn == "silu" and not any(a.split("=")[0] == "--d_ff" for a in sys.argv[1:]):
        args.d_ff = 4 * args.d_model
        print(f"[ffn=silu] d_ff 自动改为 4·d_model = {args.d_ff}（对齐 SwiGLU 参数量）")

    # 自动换算 interval (如果未手动设置或设为0)
    if args.log_interval <= 0:
        args.log_interval = max(1, args.max_iters // 100)
    if args.eval_interval <= 0:
        args.eval_interval = max(1, args.max_iters // 10)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 将所有的超参数保存到 checkpoint 目录下，防止以后忘了 d_model、num_layers 等无状态参数
    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    doc_sep_id = None
    if args.doc_mask_vocab is not None:
        doc_sep_id = lookup_token_id(args.doc_mask_vocab)
        if doc_sep_id >= args.vocab_size:
            raise SystemExit(
                f"{args.doc_mask_vocab} 解析出的分隔符 id 是 {doc_sep_id}，"
                f"超出 --vocab_size {args.vocab_size}——词表和数据配错了")
        print(f"document mask 已启用，<|endoftext|> id = {doc_sep_id}")

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
        rope_theta=None if args.no_rope else args.rope_theta,
        norm=args.norm,
        ffn=args.ffn,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = model_flops_per_token(args)
    arch = f"norm={args.norm} ffn={args.ffn} d_ff={args.d_ff} pos={'NoPE' if args.no_rope else 'RoPE'}"
    print(f"Model: {n_params/1e6:.2f} M params | {arch}")
    print(f"       device {device} | bf16 autocast: {use_bf16}")
    print(f"MFU 口径: {flops_per_token/1e6:.2f} MFLOPs/token (精确) "
          f"vs {6*n_params/1e6:.2f} (6P 近似，高估 {6*n_params/flops_per_token-1:+.1%}) "
          f"| 峰值 {args.peak_flops/1e12:.1f} TFLOPS")

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

    # Record step 0 baseline before training begins
    if start_iter == 1:
        train_loss = evaluate_loss(model, train_data, args.eval_iters, args.batch_size,
                                   args.context_length, device, args.seed + EVAL_SEED_OFFSET, ctx, doc_sep_id)
        val_loss = evaluate_loss(model, val_data, args.eval_iters, args.batch_size,
                                 args.context_length, device, args.seed + EVAL_SEED_OFFSET + 1, ctx, doc_sep_id)
        print(f"Step 0: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")
        if args.wandb:
            wandb.log({
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
                "step": 0,
            }, step=0)

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
            seq_len = x.shape[-1]
            mask = build_attention_mask(seq_len, x.device, x=x, doc_sep_id=doc_sep_id)
            
            logits = model(x, mask=mask)
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
            tok_per_sec = tokens_seen / dt
            
            # 计算 MFU (Model FLOPs Utilization)
            flops_per_sec = tok_per_sec * flops_per_token
            mfu = (flops_per_sec / args.peak_flops) * 100

            # 峰值显存：batch_size_experiment 要拿它画"batch 撞到显存墙"那条线。
            # max_memory_allocated 是进程启动以来的高水位，不随 log 归零——正好，
            # 峰值出现在反向的中段，每步都一样，取高水位就够。
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0

            print(f"step {it} | loss {loss_val:.4f} | lr {lr:.4e} | {tok_per_sec:.0f} tok/s "
                  f"| {flops_per_sec/1e12:.1f} TFLOPS | mfu {mfu:.2f}% | peak {peak_gb:.2f} GB")
            if args.wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tflops": flops_per_sec / 1e12,
                    "train/mfu": mfu,
                    "train/peak_mem_gb": peak_gb,
                    "step": it,
                }, step=it)
            tokens_seen = 0
            t0 = time.perf_counter()

        # --- E. Evaluation and Checkpointing（在这一步的更新之后）---
        if it % args.eval_interval == 0 or it == args.max_iters:
            train_loss = evaluate_loss(model, train_data, args.eval_iters, args.batch_size,
                                       args.context_length, device, args.seed + EVAL_SEED_OFFSET, ctx, doc_sep_id)
            val_loss = evaluate_loss(model, val_data, args.eval_iters, args.batch_size,
                                     args.context_length, device, args.seed + EVAL_SEED_OFFSET + 1, ctx, doc_sep_id)
            print(f"Step {it}: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")
            if args.wandb:
                wandb.log({
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "step": it,
                }, step=it)

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
