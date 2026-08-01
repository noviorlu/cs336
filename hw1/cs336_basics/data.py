import torch
import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, Int
from torch import Tensor

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[
    Int[Tensor, "batch context"],
    Int[Tensor, "batch context"],
]:
    """
    Sample language modeling input sequences and their corresponding labels from the dataset.

    返回 (x, y)：
      x, y      形状 [batch_size, context_length]，dtype int32。
                y 是 x 整体右移一格：位置 j 的输入 x[:, j] 要预测的就是 y[:, j]。
    """
    max_start_idx = len(dataset) - context_length - 1

    if rng is None:
        ix = np.random.randint(0, max_start_idx + 1, size=batch_size)
    else:
        ix = rng.integers(0, max_start_idx + 1, size=batch_size)

    # idx[b, j] = ix[b] + j，形状 [batch_size, context_length]
    idx = np.add.outer(ix, np.arange(context_length))
    x = dataset[idx]
    y = dataset[idx + 1]

    # uint16 走完 PCIe，到显存里再转，传输量是 2 字节/token 而不是 4。
    # 只能转成 int32 或 int64：PyTorch 的索引 kernel 只按这两种类型特化，
    # int8/int16/uint* 一律 IndexError。取 int32 省一半。
    x_tensor = torch.from_numpy(x).to(device).to(torch.int32)
    y_tensor = torch.from_numpy(y).to(device).to(torch.int32)

    return x_tensor, y_tensor
