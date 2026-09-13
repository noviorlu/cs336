| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gb | status        |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|--------------:|:--------------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    16.88 |     0.2  |      16.75 |         16.89 |          3.98 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    50.66 |     0.09 |      50.6  |         50.67 |          4.08 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | True       |     8.9  |     0.43 |       8.73 |          8.92 |          3.16 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |    28.34 |     0.51 |      28.3  |         28.34 |          3.18 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    49.25 |     1.1  |      48.73 |         49.31 |         10.49 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   151.63 |     0.79 |     151.34 |        151.66 |         10.58 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | True       |    25.36 |     0.65 |      25.15 |         25.38 |          8.34 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |    82.61 |     1.09 |      81.77 |         82.7  |          8.36 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   115.78 |     1.21 |     115.7  |        115.79 |         20.19 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   337.43 |     2.11 |     343.34 |        336.78 |         20.28 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | True       |    49.94 |     0.42 |      50.83 |         49.84 |         16.53 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |   168.62 |     4.21 |     166.74 |        168.83 |         16.61 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.18 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.18 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | True       |   nan    |   nan    |     nan    |        nan    |         29.29 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |   nan    |   nan    |     nan    |        nan    |         29.29 | OOM (forward) |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-12

