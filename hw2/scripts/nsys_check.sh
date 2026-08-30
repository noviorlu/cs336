#!/usr/bin/env bash
# nsys 工具链验收。用法：cd hw2 && bash scripts/nsys_check.sh
#
# 为什么需要它：`nsys --version` 通过不代表能用。Ubuntu 源里的 2022.4.2 能跑、
# 能认出 5090，但收尾报 "Importer error status"，只吐 .qdstrm、生不成 .nsys-rep，
# nsys stats 全废——而且**脚本正常退出**，失败信号极弱。
#
# 更阴的是：新版 nsys 是软链在 .venv/bin/ 里的，而 .venv/bin 排在 PATH 最前。
# 软链一旦消失（uv 重建 venv、conda 的 diff 环境被删），nsys 会**静默回退**到
# /usr/bin/nsys 那个坏版本。所以每次 profile 前都该先过这道断言。
set -euo pipefail
OUT="${1:-/tmp/nsys_check}"

VER=$(uv run nsys --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9]+' | head -1)
MAJOR=${VER%%.*}
echo "[1/4] nsys 版本 = ${VER:-<未知>}  ($(uv run which nsys 2>/dev/null | tail -1))"
if [ -z "$VER" ] || [ "$MAJOR" -lt 2024 ]; then
  echo "  ✗ 版本过旧或识别失败。5090 (Blackwell/sm_120) 需要 2024+。"
  echo "    修：ln -sf /home/yc/miniconda3/envs/diff/nsight-compute-2025.3.1/host/target-linux-x64/nsys .venv/bin/nsys"
  exit 1
fi

echo "[2/4] profile ..."
uv run nsys profile -o "$OUT" --force-overwrite true -t cuda,nvtx \
    python scripts/nsys_check.py 2>&1 | grep -E '^\[smoke\]|Generated:' || true
[ -f "$OUT.nsys-rep" ] || { echo "  ✗ 没有生成 .nsys-rep（这正是旧版的症状）"; exit 1; }

echo "[3/4] NVTX range 汇总（§2.2 要靠它过滤掉 warmup）"
uv run nsys stats --force-export=true --report nvtx_sum "$OUT.nsys-rep" 2>/dev/null \
    | grep -E 'warmup|forward|backward|optimizer|attention' | head -6 || echo "  ✗ NVTX 汇总为空"

echo "[4/4] 按 NVTX 过滤 kernel（§2.2(b)(e) 的核心能力）"
uv run nsys stats --force-export=true --report cuda_gpu_kern_sum --filter-nvtx="scaled dot product attention" \
    "$OUT.nsys-rep" 2>/dev/null | sed -n '/Time (%)/,$p' | cut -c1-116 | head -6 || echo "  ✗ NVTX 过滤为空"

echo "✓ nsys 工具链可用"
