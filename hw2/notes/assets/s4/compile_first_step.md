| model   | size   |   seq_len |   batch |   warmup |   steps | mode   | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status   | checkpoint_every   | compile   |
|:--------|:-------|----------:|--------:|---------:|--------:|:-------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------|:-------------------|:----------|
| basics  | small  |       512 |       4 |        0 |       5 | full   | False       | False      |   121.83 |   143.01 |     377.65 |         57.87 |           5.04 | OK       |                    | False     |
| basics  | small  |       512 |       4 |        0 |       5 | full   | False       | False      |  2932.28 |  6453.86 |   14477.3  |         46.03 |           4.33 | OK       |                    | True      |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [0] 步 / measure [5] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-20

