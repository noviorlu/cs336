| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status         |   checkpoint_every |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------------|-------------------:|
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          28.67 | OOM (forward)  |                nan |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |  3623.13 |    24.35 |    3605.91 |       3640.34 |          15.36 | OK             |                  1 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |  3755.84 |     0.61 |    3755.42 |       3756.27 |          18.94 | OK             |                  2 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |  3756.93 |     1.05 |    3757.67 |       3756.18 |          22.52 | OK             |                  3 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |  3734.84 |     2.56 |    3736.65 |       3733.02 |          26.1  | OK             |                  4 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          28.09 | OOM (backward) |                  6 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          28.01 | OOM (backward) |                  9 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          27.98 | OOM (backward) |                 12 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          27.94 | OOM (backward) |                 18 |
| basics  | large  |      2048 |       4 |        1 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          27.9  | OOM (backward) |                 36 |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [1] 步 / measure [2] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

