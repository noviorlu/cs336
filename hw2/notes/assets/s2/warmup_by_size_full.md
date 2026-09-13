| model   | size   |   seq_len |   batch |   warmup |   steps | mode   | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gb | status   |
|:--------|:-------|----------:|--------:|---------:|--------:|:-------|:------------|:-----------|---------:|---------:|-----------:|--------------:|--------------:|:---------|
| basics  | small  |       512 |       4 |        0 |      10 | full   | False       | False      |    86.53 |   102.29 |     377.65 |         54.18 |          5.04 | OK       |
| basics  | small  |       512 |       4 |        1 |      10 | full   | False       | False      |    53.98 |     0.08 |      54.01 |         53.98 |          5.04 | OK       |
| basics  | small  |       512 |       4 |        2 |      10 | full   | False       | False      |    54.82 |     1.07 |      54.8  |         54.82 |          5.04 | OK       |
| basics  | small  |       512 |       4 |        5 |      10 | full   | False       | False      |    54.78 |     0.96 |      55.04 |         54.75 |          5.04 | OK       |
| basics  | medium |       512 |       4 |        0 |      10 | full   | False       | False      |   192.39 |    95.2  |     463.34 |        162.29 |         13.74 | OK       |
| basics  | medium |       512 |       4 |        1 |      10 | full   | False       | False      |   164.43 |     1.71 |     163.88 |        164.49 |         13.74 | OK       |
| basics  | medium |       512 |       4 |        2 |      10 | full   | False       | False      |   167.51 |     6.78 |     177.4  |        166.41 |         13.74 | OK       |
| basics  | medium |       512 |       4 |        5 |      10 | full   | False       | False      |   171.06 |     8.75 |     161.3  |        172.14 |         13.74 | OK       |
| basics  | large  |       512 |       4 |        0 |      10 | full   | False       | False      |   401.13 |    90.96 |     659.05 |        372.47 |         27.51 | OK       |
| basics  | large  |       512 |       4 |        1 |      10 | full   | False       | False      |   366.67 |     3.09 |     373.77 |        365.88 |         27.51 | OK       |
| basics  | large  |       512 |       4 |        2 |      10 | full   | False       | False      |   381.87 |    10.95 |     372.88 |        382.86 |         27.51 | OK       |
| basics  | large  |       512 |       4 |        5 |      10 | full   | False       | False      |   368.79 |     3.25 |     367.08 |        368.98 |         27.51 | OK       |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [0, 1, 2, 5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-12

