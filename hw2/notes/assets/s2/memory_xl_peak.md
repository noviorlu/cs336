| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status          |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:----------------|
| basics  | xl     |       128 |       4 |        2 |       2 | forward | True        | False      |    80.28 |     2.52 |      78.5  |         82.07 |          12.91 | OK              |
| basics  | xl     |       128 |       4 |        2 |       2 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          29.25 | OOM (optimizer) |
| basics  | xl     |       128 |       4 |        2 |       2 | forward | True        | True       |    35.55 |     0.21 |      35.4  |         35.7  |          19.19 | OK              |
| basics  | xl     |       128 |       4 |        2 |       2 | full    | False       | True       |   nan    |   nan    |     nan    |        nan    |          29.24 | OOM (optimizer) |
| basics  | xl     |      2048 |       4 |        2 |       2 | forward | True        | False      |  2084.46 |    33.38 |    2060.85 |       2108.06 |          21.39 | OK              |
| basics  | xl     |      2048 |       4 |        2 |       2 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          25.97 | OOM (forward)   |
| basics  | xl     |      2048 |       4 |        2 |       2 | forward | True        | True       |  1074.68 |     3.8  |    1077.37 |       1072    |          25.28 | OK              |
| basics  | xl     |      2048 |       4 |        2 |       2 | full    | False       | True       |   nan    |   nan    |     nan    |        nan    |          26.79 | OOM (forward)   |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [2] 步 / measure [2] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

