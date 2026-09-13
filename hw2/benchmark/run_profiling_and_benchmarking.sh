#!/usr/bin/env bash
# §2 Profiling and Benchmarking —— 全部实验一键复现
#
# 用法：bash benchmark/run_profiling_and_benchmarking.sh [2.1|2.2|2.3|2.4|2.5|all]   （默认 all）
#
# 产出全部落到 notes/assets/s2/（表 .md + 逐步原始耗时 .json），
# nsys 报告落到 profiles/。文件名带关键配置，事后能对上是哪张表。
#
# 总耗时估计（RTX 5090）：2.1 约 5 min，2.2 约 10 min，2.3 无（只 cat 结果），2.4 约 2 min，2.5 约 3 min。
set -euo pipefail
cd "$(dirname "$0")/.."   # 回到 hw2/ 根目录，所有路径以此为准
HERE=benchmark

SECTION="${1:-all}"
OUT=notes/assets/s2
PROF=profiles
mkdir -p "$OUT" "$PROF"

BENCH="uv run python -m benchmark"

want() { [[ "$SECTION" == all || "$SECTION" == "$1" ]]; }
title() { printf '\n\033[1m━━ %s ━━\033[0m\n' "$*"; }

# ─────────────────────────────────────────────────────────────────────────────
if want 2.1; then
title "§2.1 (b) 各阶段耗时：5 size × {forward/no_grad, forward, fwd_bwd, full}"
# 测什么：每个 size 四种模式的墙钟均值 ± 标准差；backward / optimizer 靠相减得到。
# 预期：xl 在 forward（保留计算图）就 OOM，10B 建模型即 OOM——这两行是 §4 FlashAttention
#       和 §3 激活检查点的伏笔，OOM 本身就是数据，别删。
$BENCH --sweep --config default --out "$OUT/stages_b4_seq512.md"

title "§2.1 (c) 不预热会怎样：small/medium/large × warmup {0,1,2,5}"
# 测什么：warmup=0 时首步比稳态慢多少、对均值和标准差污染多大；warmup=1 之后还有没有残余。
# --isolate 必须开：不预热的开销大半是进程级一次性成本（kernel 懒加载、cuBLAS 句柄、
# 显存池首次 cudaMalloc），同进程连跑后面的配置会白捡前面的预热。
$BENCH --sweep --config warmup_by_size --isolate --out "$OUT/warmup_by_size_full.md"
$BENCH --sweep --config warmup_xl      --isolate --out "$OUT/warmup_xl_forward_nograd.md"
fi

# ─────────────────────────────────────────────────────────────────────────────
if want 2.2; then
# 一个 nsys 采集函数，(a)–(e) 共用。$1=size $2=seq $3=文件名后缀，其余参数透传给 benchmark。
# 每份报告旁边留 .log，里面有 benchmark 自己那行 timeit 结果——(a) 要拿它和 nsys 的数对账。
nsys_run() {
  local size=$1 seq=$2 tag=$3; shift 3
  local rep="$PROF/${size}_seq${seq}_${tag}"
  printf '▶ %-7s seq=%-5s [%s] ... ' "$size" "$seq" "$tag"
  if uv run nsys profile --trace=cuda,nvtx -o "$rep" --force-overwrite true -- \
        python -m benchmark --size "$size" --seq-len "$seq" --nvtx --warmup 5 --steps 5 "$@" \
        > "$rep.log" 2>&1; then
    printf '✓  timeit %s\n' "$(grep -m1 '^| basics' "$rep.log" | awk -F'|' '{printf "%s ms", $11}' | xargs)"
  else
    printf '✗  失败（见 %s.log）\n' "$rep"
  fi
}

title "§2.2 (a)–(d) nsys：small/medium × seq {256,512,1024}，每档采一份 full"
# 测什么：(a) NVTX forward range 的宽度 vs timeit（同机同天对账，profiler 系统性偏高 2–8%）
#         (b) --filter-nvtx=forward 下 GPU 时间最长的 kernel；加上 backward 后是否还是它
#         (c) 非 matmul kernel（elementwise / reduce）占前向多少，随 seq 怎么变
#         (d) 同一份里 forward range vs step range 的 matmul 占比
# 选型：PDF 要"两个 size × 三个 >128 的 2 的幂，最大档取显存装得下的最长"。large 只跑得到
#       512 凑不齐三档，所以选 small+medium，1024 正是两者各自的上限（2048 都 OOM）。
# 六档一份 full 就够，不用另采 --inference：实测 medium@1024 no_grad 前向与训练步中的前向
#       kernel 次数完全相同（1244）、matmul 占比差 <0.3pp——no_grad 省的是 CPU 侧建图记账，
#       不产生额外 GPU kernel。
# --trace=cuda,nvtx 是轻量档。要 aten 算子级细节另加 --pytorch=functions-trace,autograd-shapes-nvtx，
#       但观测开销明显变大，(a) 的对账仍用轻量档。
for size in small medium; do
  for seq in 256 512 1024; do
    nsys_run "$size" "$seq" full --mode full
  done
done

title "§2.2 (e) attention 内 scores / softmax / matmul 三段占比"
# 测什么：把 scaled_dot_product_attention 换成带三段 phase 的等价实现，每段结尾 sync，
#         range 宽度 = GPU 耗时。只看段间比，不拿墙钟对 §2.1（串行化后 forward 慢约 7%）。
#         反向不会触发这些 range（autograd 在工作线程跑 grad_fn，不进 Python 函数体）。
for size in small medium; do
  for seq in 256 512 1024; do
    nsys_run "$size" "$seq" attn --mode forward --nvtx-attn
  done
done

cat <<TIP
取数：
  uv run nsys stats --force-export=true --report nvtx_sum            $PROF/<size>_seq<N>_full.nsys-rep
  uv run nsys stats --force-export=true --report cuda_gpu_kern_sum --filter-nvtx="forward" \
                                                                     $PROF/<size>_seq<N>_full.nsys-rep
  uv run nsys stats --force-export=true --report nvtx_sum            $PROF/<size>_seq<N>_attn.nsys-rep
TIP
fi

# ─────────────────────────────────────────────────────────────────────────────
if want 2.3; then
title "§2.3 累加 1000 次 0.01"
# 一次性实验，脚本已删；结果在 notes/assets/s2/accumulation_1000x0.01.txt。
# 结论：精度由累加器 dtype 决定——fp16 累加漂到 9.95，bf16 累加卡死在 4.0，fp32 累加无论加数是什么都在 10.00x。
cat "$OUT/accumulation_1000x0.01.txt"
fi

# ─────────────────────────────────────────────────────────────────────────────
if want 2.4; then
title "§2.4 (a)(b) ToyModel 在 fp16 / bf16 autocast 下各组件的 dtype"
# 一次性实验，脚本已删；结果在 notes/assets/s2/autocast_dtypes.txt。
cat "$OUT/autocast_dtypes.txt"

title "§2.4 (c) fp32 vs bf16 autocast：small/medium/large/xl × {forward, fwd_bwd}"
# 测什么：两种精度的前向/反向耗时与峰值显存，加速比随模型大小的趋势。
# 注意：fp32 基准是 allow_tf32=False（SIMT GEMM），bf16 走 Tensor Core，
#       加速比 = 位宽减半 + 硬件路径切换两个效应叠加。
$BENCH --sweep --config mixed_precision --out "$OUT/mixed_precision_b4_seq512.md"
fi

# ─────────────────────────────────────────────────────────────────────────────
if want 2.5; then
title "§2.5 (a)(e) 显存时间线：xl × seq {128, 2048} × {forward, full} → .pickle"
# 测什么：(a) 从 Active Memory Timeline 的峰形能否认出 forward / backward / optimizer 三个阶段
#         (e) 把 Detail 拉低只看最大的几笔分配，它们多大、调用栈指向哪里
# 快照从预热开始记，--warmup 1 --steps 1 共两步：第一步带初始化噪声，看第二步的三个峰。
# xl@2048 full 预期在第一个前向就 OOM：快照仍会落盘，能看到炸掉前的时间线。
# 看图：把 profiles/mem_*.pickle 拖进 https://pytorch.org/memory_viz
#      （已画好的 png 和最大分配清单在 notes/assets/s2/mem_xl_*.png、memory_top_allocs.txt）
for seq in 128 2048; do
  for mode in "forward --inference" "full"; do
    tag=$(echo "$mode" | cut -d' ' -f1)
    printf '▶ xl seq=%-5s %-8s ... ' "$seq" "$tag"
    # shellcheck disable=SC2086
    $BENCH --size xl --seq-len "$seq" --mode $mode --warmup 1 --steps 1 \
      --memory-snapshot "$PROF/mem_xl_seq${seq}_${tag}.pickle" > "$PROF/mem_xl_seq${seq}_${tag}.log" 2>&1 \
      && echo ✓ || echo "✗（多半是 OOM，见 $PROF/mem_xl_seq${seq}_${tag}.log；快照照样有）"
  done
done

title "§2.5 (f) nsys 显存 trace：单层 TransformerBlock 为反向存了多少"
# 测什么：block{i} 前向 range 内分配、且到 range 结束还没释放的字节 = 这层的 residuals；
#         按包着它们的 aten range 归组 → top 5 算子；反向该层的净变化 Δ → 梯度 = Δ + residuals。
# PYTORCH_NO_CUDA_MEMORY_CACHING=1 必须开：否则 caching allocator 从池里复用，nsys 只看到池增长，
#         看不到每个张量。会慢很多，所以 warmup 0 / steps 1，xl@128 fwd_bwd（full 会 OOM）。
# 看图：nsys-ui profiles/mem_trace_xl_seq128_fwd_bwd.nsys-rep，NVTX 行找 block5，下面 "CUDA memory usage" 行是阶梯。
rep="$PROF/mem_trace_xl_seq128_fwd_bwd"
printf '▶ xl seq=128 fwd_bwd [memory trace] ... '
PYTORCH_NO_CUDA_MEMORY_CACHING=1 uv run nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    -o "$rep" --force-overwrite true -- \
    python -m benchmark --size xl --seq-len 128 --mode fwd_bwd --nvtx --nvtx-ops --warmup 0 --steps 1 \
    > "$rep.log" 2>&1 && echo ✓ || echo "✗（见 $rep.log）"
# 归因：block5 前向 range 内分配、活过 range 结束的 = residual，按最内层 aten range 归组；
#       反向用 seq 编号对回该层，梯度 ≈ 净变化 + 释放的 residual。预期 ≈ 每层参数 × 4 B = 400 MiB。
uv run python -m benchmark.memory "$rep.nsys-rep" --block 5 --plot 2>/dev/null | tee "$OUT/memory_block5_residuals.txt"
cp "${rep}_block5.png" "$OUT/nsys_block5_memory.png"

title "§2.5 (b)(c) 峰值显存：xl × seq {128, 2048} × {forward, full} × {fp32, bf16}"
# 测什么：(b) 两种 seq 各自的 forward / full 峰值   (c) bf16 autocast 省了多少
# 用的是 torch.cuda.max_memory_allocated()（预热后清零，只统计测量段）。
$BENCH --sweep --config memory_xl --out "$OUT/memory_xl_peak.md"
fi

title "完成。产出："
ls -1 "$OUT"
