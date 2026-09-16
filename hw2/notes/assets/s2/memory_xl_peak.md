| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gib | status          | checkpoint_every   |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|---------------:|:----------------|:-------------------|
| basics  | xl     |       128 |       4 |        2 |       2 | forward | True        | False      |    79.78 |     0.05 |      79.74 |         79.82 |          12.9  | OK              |                    |
| basics  | xl     |       128 |       4 |        2 |       2 | fwd_bwd | False       | False      |   237.65 |     0.79 |     238.2  |        237.09 |          25.56 | OK              |                    |
| basics  | xl     |       128 |       4 |        2 |       2 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          28.38 | OOM (optimizer) |                    |
| basics  | xl     |       128 |       4 |        2 |       2 | forward | True        | True       |    34.99 |     0.14 |      35.09 |         34.89 |          19.18 | OK              |                    |
| basics  | xl     |       128 |       4 |        2 |       2 | fwd_bwd | False       | True       |   132.32 |     4.54 |     129.11 |        135.53 |          25.55 | OK              |                    |
| basics  | xl     |       128 |       4 |        2 |       2 | full    | False       | True       |   nan    |   nan    |     nan    |        nan    |          28.46 | OOM (optimizer) |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | forward | True        | False      |  2051.48 |     4.12 |    2054.4  |       2048.57 |          21.38 | OK              |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |          25.96 | OOM (forward)   |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |          25.96 | OOM (forward)   |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | forward | True        | True       |  1056.4  |     0.06 |    1056.45 |       1056.36 |          25.27 | OK              |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | fwd_bwd | False       | True       |   nan    |   nan    |     nan    |        nan    |          26.78 | OOM (forward)   |                    |
| basics  | xl     |      2048 |       4 |        2 |       2 | full    | False       | True       |   nan    |   nan    |     nan    |        nan    |          26.78 | OOM (forward)   |                    |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [2] 步 / measure [2] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-16

