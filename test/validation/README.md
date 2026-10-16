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

`test/validation/profile_gpu.sh [case] [dp|sp]` profiles one case (default `hamr-perf`) with Nsight Systems.

`test/validation/ab_compare.sh [ref-commit] [dp|sp] [seeds]` compares the CUDA build of a reference commit
(default `HEAD`) with the working tree, for changes expected to give statistically equivalent rather than identical
results (for example a different random number sequence): it runs every finite temperature LLG case with both builds
using more seeds (default 32), compares them with each other and with the CPU references, and times the
`hamr-perf` and `thermal-perf` benchmarks with both builds.

`test/validation/identity_compare.sh [ref-commit]` (default `develop`) is for changes that must not change any
result: it builds the CUDA version of a reference commit and of the working tree in both precisions, runs every
case with both (once as committed and once with extra statistics, configuration and checkpoint output) and compares
every output file byte for byte, ignoring only comment lines and timings.

## Multi-GPU validation

`run --ranks N` starts each run with `mpirun -np N` (the `MPIRUN` environment variable replaces `mpirun`, e.g.
`MPIRUN="mpirun --oversubscribe"`). `vtest.py configs --test A --ref B` compares the final spin configurations of two
result sets atom by atom (atoms are paired by their coordinates, since MPI runs list atoms in rank order) and reports
whether they are bit for bit identical; runs need `--extra config:atoms=end --extra config:output-format=binary`.

**Bit for bit identical results on different numbers of GPUs require `gpu:decomposition-independent-order`**
(multi-GPU version only, off by default). It sums the exchange contributions of every atom in an order given by the
atom's position in the crystal (its unit cell and its number within the unit cell) instead of its local atom number,
which depends on the decomposition. Without it, runs on different numbers of GPUs agree only to floating point
round-off. It costs a sort of the neighbour list at start-up and has no measurable effect on the time per step.

`test/validation/mgpu_check.sh [ranks ...]` (default `1 2 4`) builds the CPU MPI version and the multi-GPU version
(`make cuda-multigpu`) in both precisions and runs the zero temperature cases on each number of ranks, which may share
one GPU. One-rank runs are compared with the serial references and multi-rank runs with the CPU MPI version on the
same number of ranks (most cases start from random spins drawn per rank). `t0-spin-spiral` starts from a state that
does not depend on the decomposition: it is compared with the serial references on every number of ranks, and its
final spins must be bit for bit identical on every number of ranks (the script enables
`gpu:decomposition-independent-order` for the GPU runs). `t0-exchange-tensor` also starts from that state, with
tensorial exchange whose order of summation depends on the decomposition, so it is only compared with the references. The other LLG cases whose initial state does not depend on
the decomposition (finite temperature, HAMR and program cases) are run with one seed on every number of ranks and
their final spins must also be bit for bit identical, since the multi-GPU version draws the thermal noise of each atom
from its key and the time step; with all their seeds on one rank they are compared statistically with the serial
references. Finally `timeseries-T300` and `hamr-small` are rerun on every number of ranks saving a single checkpoint part way
(`sim:save-checkpoint=continuous` with a rate of 60 % of the time steps) and continued from it
(`sim:load-checkpoint=continue`); the final spins of the continuation must be bit for bit those of the uninterrupted
run.

`run --seeds N` overrides the number of seeds of statistical cases. `compare --summary` prints the signed z-scores
of all statistical comparisons pooled together: when test and reference sample the same distribution they are
approximately standard normal (mean ~0, rms ~1, ~4.6 % beyond 2), which is a sharper test than the per-quantity
pass/fail. Rows of a per-row case are correlated, so judge the pooled values rather than a single case.

## Case format (`cases/<case>/case.json`)

All case inputs use `output:column-headers` and `output:precision=16`

| kind | runs | pass criterion |
| ------ | ------ | ---------------- |
| `analytic` / `precession` | 1 | max abs error vs analytic damped precession <= `tolerance[precision].atol` |
| `analytic` / `thermal` | `seeds` | seed-mean of time-averaged <mz> (after `skip_rows`) within `k`*stderr + `atol` of the Boltzmann average |
| `deterministic` (T=0) | 1 | every row: abs(test-ref) <= abs + atol*max(abs(ref)) + rtol*abs(ref) (`tolerance[precision]`, `abs` optional, default 0, for quantities close to zero in every compared row); an optional `rows` entry limits the comparison to the first rows, used in single precision for random initial states where round-off is amplified chaotically; reports "identical" when bit-equal |
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
- `hamr-small` (statistical), `hamr-t0` (deterministic), `hamr-perf` (benchmark, ~168k atoms); `thermal-perf` (benchmark, the same system as a 300 K time series).
