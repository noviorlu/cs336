| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status        |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:--------------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    17.37 |     0.31 |      17.88 |         17.31 |           3.99 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    52.4  |     0.39 |      52.71 |         52.37 |           4.08 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | True       |     9.04 |     0.36 |       8.84 |          9.06 |           3.16 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |    29.18 |     0.42 |      29.39 |         29.15 |           3.18 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    50.54 |     0.46 |      50.4  |         50.55 |          10.49 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   156.47 |     0.65 |     156.54 |        156.46 |          10.58 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | True       |    25.36 |     0.4  |      25.99 |         25.29 |           8.34 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |    84.91 |     0.75 |      84.67 |         84.94 |           8.36 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   119.66 |     1.23 |     118.92 |        119.74 |          20.19 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   353.28 |     1.67 |     351.72 |        353.45 |          20.28 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | True       |    52.62 |     0.24 |      53.22 |         52.55 |          16.54 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |   175.45 |     0.85 |     176.42 |        175.35 |          16.61 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | True       |   nan    |   nan    |     nan    |        nan    |          29.12 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | True       |   nan    |   nan    |     nan    |        nan    |          29.12 | OOM (forward) |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

