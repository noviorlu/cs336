| model   | size   |   seq_len |   batch |   warmup |   steps | mode   | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status   |
|:--------|:-------|----------:|--------:|---------:|--------:|:-------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:---------|
| basics  | small  |       512 |       4 |        0 |      10 | full   | False       | False      |    89.98 |   104.81 |     388.28 |         56.84 |           5.04 | OK       |
| basics  | small  |       512 |       4 |        1 |      10 | full   | False       | False      |    56.81 |     0.41 |      56.88 |         56.8  |           5.04 | OK       |
| basics  | small  |       512 |       4 |        2 |      10 | full   | False       | False      |    57.08 |     0.32 |      56.83 |         57.1  |           5.04 | OK       |
| basics  | small  |       512 |       4 |        5 |      10 | full   | False       | False      |    57.1  |     0.48 |      57.76 |         57.03 |           5.04 | OK       |
| basics  | medium |       512 |       4 |        0 |      10 | full   | False       | False      |   200.76 |    99.5  |     483.91 |        169.29 |          13.74 | OK       |
| basics  | medium |       512 |       4 |        1 |      10 | full   | False       | False      |   169.91 |     1.33 |     169.26 |        169.98 |          13.74 | OK       |
| basics  | medium |       512 |       4 |        2 |      10 | full   | False       | False      |   171.46 |     2.1  |     172.26 |        171.37 |          13.74 | OK       |
| basics  | medium |       512 |       4 |        5 |      10 | full   | False       | False      |   170.8  |     1.56 |     168.76 |        171.03 |          13.74 | OK       |
| basics  | large  |       512 |       4 |        0 |      10 | full   | False       | False      |   412.29 |    86.84 |     659.42 |        384.83 |          27.51 | OK       |
| basics  | large  |       512 |       4 |        1 |      10 | full   | False       | False      |   382    |     2.19 |     378.66 |        382.37 |          27.51 | OK       |
| basics  | large  |       512 |       4 |        2 |      10 | full   | False       | False      |   380.25 |     0.98 |     380.84 |        380.19 |          27.51 | OK       |
| basics  | large  |       512 |       4 |        5 |      10 | full   | False       | False      |   383.79 |     0.27 |     384.12 |        383.75 |          27.51 | OK       |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [0, 1, 2, 5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

