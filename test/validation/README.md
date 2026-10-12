# VAMPIRE validation

Checks that a vampire binary (serial, CUDA DP/SP, multi-GPU, ...) produces the correct results by comparing against
analytic solutions and committed reference outputs from the serial CPU code.

```
python3 test/validation/vtest.py run --exe ./vampire-cuda --label gpu-dp [--precision sp] [--jobs N]
python3 test/validation/vtest.py compare --test gpu-dp --ref cpu
```

- `run` copies each `cases/<case>/` to `results/<label>/<case>/<run>/`, appends `--extra` lines (and
  `sim:integrator-random-seed` for seeded cases) to the input, and runs the binary there.
- `compare` exits non-zero on any failure; passing cases are summarised in one line (`--verbose` shows every quantity). Labels are looked up in `results/` first, then `reference/`.
- `./tests.sh -v` runs `vampire-serial` and compares against `reference/cpu`.

## GPU validation

`test/validation/run_gpu_suite.sh [label-prefix]` builds the CUDA version in double and single precision, runs the
suite with each build (and the `hamr-perf` benchmark), checks that the CUDA binary with `gpu:acceleration=off`
reproduces the serial references exactly, and packs results, build logs and system information into
`test/validation/results/gpu-suite-<host>-<date>.tar.gz`. It runs `make clean` before each build and at the end.

## Case format (`cases/<case>/case.json`)

All case inputs use `output:column-headers` and `output:precision=16`

| kind | runs | pass criterion |
| ------ | ------ | ---------------- |
| `analytic` / `precession` | 1 | max abs error vs analytic damped precession <= `tolerance[precision].atol` |
| `analytic` / `thermal` | `seeds` | seed-mean of time-averaged <mz> (after `skip_rows`) within `k`*stderr + `atol` of the Boltzmann average |
| `deterministic` (T=0) | 1 | every row: abs(test-ref) <= atol*max(abs(ref)) + rtol*abs(ref) (`tolerance[precision]`); reports "identical" when bit-equal |
| `statistical` (T>0) | `seeds` | abs(mean_test - mean_ref) <= `k` * sqrt(se_test^2 + se_ref^2) over seed time-averages (after `skip_rows`); with `"per_row": true` every output row is compared separately (e.g. one row per temperature) |
| `benchmark` | 1 | not compared, reports simulation run time; only run when named with `--cases` (e.g. `hamr-perf` for profiling) |

Precision (`dp`/`sp`) is taken from the test run's `meta.json` (set with `run --precision`).
Statistical checks fail spuriously ~0.1 % of the time per quantity: re-run with `--seed-offset 100` before treating a failure as real.

## Cases

- `analytic-*`: physics vs exact results (single-spin precession; Boltzmann average for non-interacting spins).
- `t0-*`: deterministic T=0 trajectories. `t0-anis-<term>` isolates one anisotropy term each; `biaxial`, `triaxial`
  and `cubic-rotated` are not implemented in the CUDA module, so they are expected to fail on GPU until fixed.
- `timeseries-T300`, `curie-llg`, `curie-mc`: thermal statistics (LLG and Monte Carlo are each compared only with
  the same integrator).
- `hamr-small` (statistical), `hamr-t0` (deterministic), `hamr-perf` (benchmark, ~168k atoms).
