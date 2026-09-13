| model   | size   |   seq_len |   batch |   warmup |   steps | mode    | inference   | autocast   |   avg_ms |   std_ms |   first_ms |   rest_avg_ms |   peak_mem_gb | status        |
|:--------|:-------|----------:|--------:|---------:|--------:|:--------|:------------|:-----------|---------:|---------:|-----------:|--------------:|--------------:|:--------------|
| basics  | small  |       512 |       4 |        5 |      10 | forward | True        | False      |    16.41 |     0.03 |      16.42 |         16.41 |          0.72 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | forward | False       | False      |    16.53 |     0.02 |      16.53 |         16.52 |          3.98 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |    49.43 |     0.03 |      49.44 |         49.42 |          4.08 | OK            |
| basics  | small  |       512 |       4 |        5 |      10 | full    | False       | False      |    52.84 |     0.08 |      52.82 |         52.84 |          5.04 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | True        | False      |    47.36 |     0.12 |      47.12 |         47.38 |          1.9  | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | forward | False       | False      |    48.03 |     0.07 |      47.84 |         48.05 |         10.49 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   148.48 |     0.09 |     148.36 |        148.49 |         10.58 | OK            |
| basics  | medium |       512 |       4 |        5 |      10 | full    | False       | False      |   162.71 |     5.04 |     158.12 |        163.22 |         13.74 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | True        | False      |   116.25 |     3    |     112.93 |        116.61 |          4.11 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | forward | False       | False      |   114.91 |     2.6  |     119.09 |        114.44 |         20.19 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   346.47 |     8.41 |     353.04 |        345.74 |         20.28 | OK            |
| basics  | large  |       512 |       4 |        5 |      10 | full    | False       | False      |   371.52 |     9.91 |     357.66 |        373.06 |         27.51 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | True        | False      |   345.3  |     5.27 |     345.24 |        345.3  |         13.47 | OK            |
| basics  | xl     |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.06 | OOM (forward) |
| basics  | xl     |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.06 | OOM (forward) |
| basics  | 10B    |       512 |       4 |        5 |      10 | forward | True        | False      |   nan    |   nan    |     nan    |        nan    |         29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | forward | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | fwd_bwd | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.31 | OOM (init)    |
| basics  | 10B    |       512 |       4 |        5 |      10 | full    | False       | False      |   nan    |   nan    |     nan    |        nan    |         29.31 | OOM (init)    |

---

**测量条件**（每张表都要带，事后补不回来）

- 硬件：NVIDIA GeForce RTX 5090｜CUDA 13.0
- 软件：Python 3.13.9、torch 2.11.0+cu130、triton 3.6.0
- 测法：warmup [5] 步 / measure [10] 步，每步 `torch.cuda.synchronize()`，计时用 `timeit.default_timer()`
- 采集日期：2026-09-12

