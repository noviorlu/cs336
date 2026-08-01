import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and their corresponding labels from the dataset.

    rng: 可选的 numpy Generator。不传就用全局 np.random（作业测试走的是这条路）；
         传了就用它，于是 batch 变成 (种子, 步数) 的确定性函数——训练可精确复现、
         中断后可从 checkpoint 精确续上，评估每次也能看到完全相同的样本。
    """
    # 由于我们需要同时切出 x (长度为 context_length) 和 y (向后错一位，长度也为 context_length)
    # 所以必须保证切片的终点 i + 1 + context_length <= len(dataset)
    # 因此最大的起始索引应该是 len(dataset) - context_length - 1
    max_start_idx = len(dataset) - context_length - 1

    # 随机抽取 batch_size 个合法的起始索引
    # randint / integers 的区间都是 [low, high)，所以右边界要 +1
    if rng is None:
        ix = np.random.randint(0, max_start_idx + 1, size=batch_size)
    else:
        ix = rng.integers(0, max_start_idx + 1, size=batch_size)

    # 按照索引切片并堆叠 (stack) 成 (batch_size, context_length) 大小的矩阵
    x = np.stack([dataset[i : i + context_length] for i in ix])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in ix])
    
    # 转成 PyTorch Tensor，确保是 int64(long) 类型，并立刻送到指定设备 (device)
    x_tensor = torch.from_numpy(x.astype(np.int64)).to(device)
    y_tensor = torch.from_numpy(y.astype(np.int64)).to(device)
    
    return x_tensor, y_tensor
