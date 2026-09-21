| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status         | checkpoint_every   | compile   |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------------|:-------------------|:----------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    17.86 |     0.28 |      17.91 |         17.86 |           3.98 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    15.17 |     0.3  |      15.63 |         15.12 |           3.26 | OK             |                    | True      |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    53.02 |     0.76 |      52.67 |         53.05 |           4.08 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    43.18 |     0.74 |      42.87 |         43.21 |           3.37 | OK             |                    | True      |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    57.2  |     0.33 |      57.38 |         57.18 |           5.04 | OK             |                    | False     |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    46.44 |     0.38 |      46.08 |         46.48 |           4.33 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    51.11 |     0.75 |      50.44 |         51.18 |          10.48 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    42.67 |     0.52 |      42.41 |         42.7  |           8.59 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   158.85 |     0.94 |     159.62 |        158.76 |          10.58 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   126.75 |     1.09 |     127.16 |        126.7  |           8.69 | OK             |                    | True      |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   170.6  |     0.46 |     170.5  |        170.61 |          13.74 | OK             |                    | False     |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   136.29 |     0.56 |     135.15 |        136.41 |          11.85 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   122.76 |     1.43 |     121.67 |        122.88 |          20.18 | OK             |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |    98.09 |     0.4  |      97.34 |         98.17 |          16.65 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   360.84 |     0.51 |     360.96 |        360.83 |          20.28 | OK             |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   276.54 |     0.43 |     276.65 |        276.53 |          16.75 | OK             |                    | True      |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          27.51 | OOM (backward) |                    | False     |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   301.22 |     0.49 |     300.54 |        301.3  |          23.97 | OK             |                    | True      |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-20

