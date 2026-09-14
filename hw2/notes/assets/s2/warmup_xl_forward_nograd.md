| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status   |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|--------------:|:---------|
| basics  | xl     |       512 |       4 |        0 |      10 | forward | True        | False      |   368.76 |    43.64 |     492.96 |        354.96 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        1 |      10 | forward | True        | False      |   354.52 |     1.12 |     353.3  |        354.65 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        2 |      10 | forward | True        | False      |   352.37 |     1.03 |     351.38 |        352.48 |         13.46 | OK       |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | True        | False      |   352.75 |     0.25 |     353.03 |        352.71 |         13.46 | OK       |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [0, 1, 2, 5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-13

