| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status   |   checkpoint_every |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------|-------------------:|
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   235.87 |     0.53 |     235.82 |        235.88 |          15.02 | OK       |                nan |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   302.1  |     1.13 |     301.29 |        302.3  |           7.82 | OK       |                  1 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   305.58 |     0.48 |     305.52 |        305.6  |           8.03 | OK       |                  2 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   310.69 |     1.15 |     311.02 |        310.61 |           8.23 | OK       |                  3 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   310.54 |     0.74 |     310.38 |        310.58 |           8.44 | OK       |                  4 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   312.21 |     0.86 |     310.94 |        312.53 |           8.85 | OK       |                  6 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   309.34 |     1.04 |     308.95 |        309.43 |           9.47 | OK       |                  9 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   312.34 |     1.08 |     311.85 |        312.47 |          10.09 | OK       |                 12 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   313.1  |     1.38 |     312.9  |        313.15 |          11.32 | OK       |                 18 |
| basics  | large  |      1024 |       1 |        2 |       5 | fwd_bwd | False       | False      |   313.34 |     0.44 |     313.17 |        313.38 |          15.03 | OK       |                 36 |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [2] 步 / measure [5] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

