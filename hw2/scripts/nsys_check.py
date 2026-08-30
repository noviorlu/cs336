"""nsys 冒烟测试：一次验完 §2.2 需要的全部能力。
不是 §2.2 的交付物，只是工具链验收。跑法见文末。"""
import torch, torch.cuda.nvtx as nvtx
import cs336_basics.model as M
from cs336_basics.nn_utils import cross_entropy

# --- §2.2 要求的猴补丁：给 attention 内部打 NVTX ---
_orig = M.scaled_dot_product_attention
CALLS = {"n": 0}

def annotated_sdpa(Q, K, V, mask=None):
    CALLS["n"] += 1
    with nvtx.range("scaled dot product attention"):
        return _orig(Q, K, V, mask)

M.scaled_dot_product_attention = annotated_sdpa

cfg = dict(vocab_size=10000, context_length=256, d_model=768, num_layers=4,
           num_heads=12, d_ff=3072, rope_theta=10000.0)
dev = "cuda"
model = M.TransformerLM(**cfg, device=torch.device(dev))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randint(0, cfg["vocab_size"], (4, cfg["context_length"]), device=dev)
y = torch.randint(0, cfg["vocab_size"], (4, cfg["context_length"]), device=dev)

with nvtx.range("warmup"):                       # PDF: 用 NVTX 把 warmup 圈出来好过滤掉
    for _ in range(2):
        cross_entropy(model(x), y).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

for step in range(3):
    with nvtx.range(f"step{step}"):
        with nvtx.range("forward"):
            loss = cross_entropy(model(x), y)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer"):
            opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

print(f"[smoke] annotated attention 被调用 {CALLS['n']} 次 "
      f"(期望 {cfg['num_layers']}*(2+3)={cfg['num_layers']*5})")
print(f"[smoke] device = {torch.cuda.get_device_name(0)}, torch = {torch.__version__}")
