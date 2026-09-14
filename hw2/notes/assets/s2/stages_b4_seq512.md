| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status        |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:--------------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | True        | False      |    17.65 |     0.12 |      17.42 |         17.67 |           0.72 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    17.88 |     0.15 |      17.85 |         17.89 |           3.98 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    53.72 |     0.21 |      53.4  |         53.75 |           4.08 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    57.43 |     0.27 |      57.24 |         57.45 |           5.04 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | True        | False      |    50.95 |     0.22 |      50.83 |         50.97 |           1.9  | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    51.37 |     0.28 |      51.16 |         51.39 |          10.49 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   160.87 |     0.62 |     161.31 |        160.82 |          10.58 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   172.02 |     1.25 |     172.49 |        171.97 |          13.74 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | True        | False      |   121.08 |     0.43 |     121.23 |        121.07 |           4.11 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   120.27 |     0.42 |     120.74 |        120.22 |          20.19 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   359.98 |     3.03 |     355.84 |        360.44 |          20.28 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   387.54 |     0.59 |     388.09 |        387.48 |          27.51 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | True        | False      |   357.23 |     2.64 |     357.02 |        357.25 |          13.47 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.06 | OOM (forward) |
| basics  | 10B    |       512 |       4 |        5 |      10 | forward | True        | False      |   nan    |   nan    |     nan    |        nan    |          29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.31 | OOM (init)    |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

