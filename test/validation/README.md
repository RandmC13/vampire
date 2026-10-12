# VAMPIRE validation

Checks that a vampire binary (serial, CUDA DP/SP, multi-GPU, ...) produces the correct results by comparing against
analytic solutions and committed reference outputs from the serial CPU code.

```
python3 test/validation/vtest.py run --exe ./vampire-cuda --label gpu-dp [--precision sp] [--jobs N]
python3 test/validation/vtest.py compare --test gpu-dp --ref cpu
```

- `run` copies each `cases/<case>/` to `results/<label>/<case>/<run>/`, appends `--extra` lines (and
  `sim:integrator-random-seed` for seeded cases) to the input, and runs the binary there.
- `compare` exits non-zero on any failure. Labels are looked up in `results/` first, then `reference/`.
- `./tests.sh -v` runs `vampire-serial` and compares against `reference/cpu`.

## Case format (`cases/<case>/case.json`)

All case inputs use `output:column-headers` and `output:precision=16`

| kind | runs | pass criterion |
| ------ | ------ | ---------------- |
| `analytic` / `precession` | 1 | max abs error vs analytic damped precession <= `tolerance[precision].atol` |
| `analytic` / `thermal` | `seeds` | seed-mean of time-averaged <mz> (after `skip_rows`) within `k`*stderr + `atol` of the Boltzmann average |
| `deterministic` (T=0) | 1 | every row: abs(test-ref) <= atol + rtol*abs(ref) (`tolerance[precision]`); reports "identical" when bit-equal |
| `statistical` (T>0) | `seeds` | abs(mean_test - mean_ref) <= `k` * sqrt(se_test^2 + se_ref^2) over seed time-averages |

Precision (`dp`/`sp`) is taken from the test run's `meta.json` (set with `run --precision`).
Statistical checks fail spuriously ~0.1 % of the time per quantity: re-run with `--seed-offset 100` before treating a failure as real.
