"""§3.2 (b)：把 model.layers 每 `every` 层包成一个 torch.utils.checkpoint 段。

只服务「扫段长测峰值」这一个实验；关掉（every=None）时 model_bench 感知不到它。
段是一个 nn.Module，签名与单层相同（x, mask=, token_positions=），所以 TransformerLM.forward
里 `for layer in self.layers` 那行不用改——它看到的还是一串「层」，只是每个内部跑 `every` 层。
"""
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointSegment(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def _run(self, x, mask, token_positions):
        for layer in self.layers:
            x = layer(x, mask=mask, token_positions=token_positions)
        return x

    def forward(self, x, mask=None, token_positions=None):
        # use_reentrant=False：前向只留 (x, mask, token_positions) 这几个输入（entry），段内 saved tensors 全丢
        return checkpoint(self._run, x, mask, token_positions, use_reentrant=False)


def apply_checkpointing(model: nn.Module, every: int) -> nn.Module:
    """就地把 model.layers 换成 ceil(L/every) 个 CheckpointSegment。every=1 即每层一个 checkpoint。"""
    layers = list(model.layers)
    model.layers = nn.ModuleList(
        CheckpointSegment(layers[i:i + every]) for i in range(0, len(layers), every)
    )
    return model
