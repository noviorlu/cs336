#!/usr/bin/env bash
# §2.2 nsys_profile 批量采集
#
# 用法：cd hw2 && bash scripts/profile_s22.sh [输出目录]
#
# 选型依据（见 notes/todo.md §2.2）：PDF 要"两个 size × 三个 >128 的 2 的幂，
# 最大那档取显存装得下的最长"。large 只跑得到 512 凑不齐三档，所以选 small+medium，
# 三档 256/512/1024——1024 正是两者各自的上限（2048 都 OOM）。
#
# 每个配置采两份：
#   full   —— 完整训练步，(a)(b)(c) 从它的 forward range 取数，(d) 的训练侧也是它
#   infer  —— --mode forward --inference，(d) 的推理基准
# PDF 的 (d) 原文是 "compared to doing inference (forward pass only)"，
# 所以基准必须是 no_grad 的推理，不是训练步里的前向。
set -euo pipefail

OUT="${1:-profiles}"
SIZES=(small medium)
SEQS=(256 512 1024)
WARMUP=5          # 和 §2.1 一致，(a) 对账才是同一口径
STEPS=5           # --filter-nvtx 只取第一个实例，5 步够用且报告小
TRACE="cuda,nvtx" # 轻量档。要 aten 级细节另跑 --pytorch=functions-trace，见文末

mkdir -p "$OUT"

run() {
  local size=$1 seq=$2 tag=$3; shift 3
  local rep="$OUT/${size}_seq${seq}_${tag}"
  printf '▶ %-7s seq=%-5s [%s] ... ' "$size" "$seq" "$tag"
  if uv run nsys profile --trace="$TRACE" -o "$rep" --force-overwrite true -- \
        python cs336_systems/benchmark.py \
          --size "$size" --seq-len "$seq" --nvtx \
          --warmup "$WARMUP" --steps "$STEPS" "$@" > "$rep.log" 2>&1; then
    # benchmark.py 自己那行 timeit 结果也留着——(a) 要拿它和 nsys 的数对账
    printf '✓  %s\n' "$(grep -m1 '^| basics' "$rep.log" | awk -F'|' '{printf "%s ms", $10}' | xargs)"
  else
    printf '✗  失败（见 %s.log）\n' "$rep"
  fi
}

for size in "${SIZES[@]}"; do
  for seq in "${SEQS[@]}"; do
    run "$size" "$seq" full  --mode full
    run "$size" "$seq" infer --mode forward --inference
  done
done

echo
echo "产出："
ls -1sh "$OUT"/*.nsys-rep 2>/dev/null || echo "  （没有产出，检查上面的 .log）"
cat <<'TIP'

取数示例：
  uv run nsys stats --force-export=true --report nvtx_sum           PROF.nsys-rep
  uv run nsys stats --force-export=true --report cuda_gpu_kern_sum \
      --filter-nvtx="forward"                                       PROF.nsys-rep

要 aten 算子级的细节（(e) 若走"按形状认"那条路会用到），重跑一份：
  加 --pytorch=functions-trace,autograd-shapes-nvtx
  ⚠️ 观测开销明显变大，(a) 的对账仍应使用本脚本的轻量档
TIP
