| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gb | status   |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|--------------:|:---------|
| basics  | xl     |       512 |       4 |        0 |      10 | forward | True        | False      |   353.85 |    42.68 |     475.09 |        340.38 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        1 |      10 | forward | True        | False      |   341.8  |     1.58 |     340.3  |        341.96 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        2 |      10 | forward | True        | False      |   351.82 |     8.01 |     346.55 |        352.41 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | True        | False      |   346.44 |     3.48 |     340.93 |        347.05 |         13.46 | OK       |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [0, 1, 2, 5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-12

