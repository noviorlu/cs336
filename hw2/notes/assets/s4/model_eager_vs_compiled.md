| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status         | checkpoint_every   | compile   |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------------|:-------------------|:----------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    16.53 |     0.05 |      16.42 |         16.54 |           3.98 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    14.07 |     0.24 |      14    |         14.08 |           3.26 | OK             |                    | True      |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    49.82 |     0.59 |      49.62 |         49.85 |           4.08 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    40.72 |     1.1  |      39.35 |         40.87 |           3.37 | OK             |                    | True      |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    53.8  |     1.34 |      53.7  |         53.82 |           5.04 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    43.85 |     1.3  |      42.68 |         43.98 |           4.33 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    48.23 |     0.97 |      47.36 |         48.33 |          10.48 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    42.02 |     1.33 |      42.2  |         42    |           8.59 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   154.37 |     5.9  |     147.71 |        155.11 |          10.58 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   123.01 |     1.11 |     122.47 |        123.06 |           8.69 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   165.23 |     1.11 |     166.41 |        165.1  |          13.74 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   131.74 |     1.09 |     130.75 |        131.84 |          11.85 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   117.16 |     1.08 |     118.22 |        117.04 |          20.18 | OK             |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |    95.29 |     0.35 |      95.03 |         95.32 |          16.65 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   374.88 |    20.41 |     407.32 |        371.28 |          20.28 | OK             |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   271.01 |     2.11 |     272.46 |        270.84 |          16.75 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          27.41 | OOM (backward) |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   293    |     3.53 |     291.66 |        293.15 |          23.97 | OK             |                    | True      |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-10-04

