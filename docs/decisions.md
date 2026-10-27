# Design decisions

Why hamrvis is built the way it is. Newest decisions at the bottom of each section; dates are when they were made.

## Scope and history

- Grew out of one-off pages built per run (explorer + analysis HTML with data embedded, 2026-10-14/15, for 0.3-1.6M
  atom write-only runs on a desktop GPU). Those pages liked by the user: grain and atom views, scrollable animation,
  laser and write-field overlays, charts and hover data everywhere. hamrvis keeps all of that.
- 2026-10-27, user brief: one tool for every future HAMR run of a PhD, covering the full chain (relax, write, read);
  collaborative design; lives on its own orphan branch `hamr-vis` (worktree `.claude/worktrees/hamr-vis` of the
  user's VAMPIRE fork), not merged into VAMPIRE until the user or supervisor decide it is ready. Code and data
  formats must be documented for the user and for future agents.
- First real data: the Viking H3 chain (200 x 60 x 20 nm, 8,724,701 atoms incl. a 10 nm non-magnetic sensor layer,
  416 grains, 33 bits at 50 m/s, 4.29 ns write on 6 A40 GPUs, 430 binary file-per-node snapshots = ~90 GB).

## Architecture (2026-10-27)

- **Reduce where the data is, view anywhere.** Moving 90 GB to a laptop is the bottleneck, and a browser cannot hold
  it either, so a Python reducer runs on the cluster next to the output and writes a bundle; only the bundle is
  copied. Alternatives rejected: copying raw data (90 GB per run), a per-run generated web page (every old page
  freezes at the version that built it, and a 16 MB page cannot hold atom data of a large run), a desktop GUI
  (dependencies, no sharing).
- **One general viewer + a local server** (`hamrvis view`, Python stdlib `http.server`) instead of a page per run:
  one codebase, every bundle opens in the same, latest viewer. Snapshot frames are separate files the viewer
  fetches on demand, so a multi-GB bundle costs the browser one frame at a time.
- **`hamrvis export`** inlines viewer and a thinned bundle into one HTML file for sharing / claude.ai artifacts
  (16 MB limit): grain frames thinned with `--grain-step`.
- **Dependencies**: Python 3.8+ and numpy only (Viking's `SciPy-bundle/2023.07-gfbf-2023a` module has them); the
  viewer is vanilla JS + canvas/SVG, no build step, one `app.js`, so export can inline it. Fonts from Google Fonts
  with system fallbacks (works offline with the fallback).

## Reduction (2026-10-27)

- Snapshot binary layout (`src/config`): coordinates `u64 n, i32 type[n], i32 category[n], f64 xyz[3n],
  i32 grain[n]`; spins `u64 n, f64 xyz[3n]`; text variants line-based. The `.meta` file lists the data files, so
  file-per-node, file-per-process and single-file outputs are handled alike.
- Only magnetic atoms (materials without `non-magnetic` in the .mat file) are binned: ~45 % of an H3 snapshot is
  the sensor layer and fill.
- **Top-down columns at the atomic pitch** (2.5 A: 800 x 240 for H3) and a **per-layer volume at 2x the pitch**
  (400 x 120 x 40 layers); int8 components (resolution 1/127). Measured on the final H3 snapshot: top 0.35 MB,
  volume 4.6 MB per snapshot gzipped, so ~150 MB + ~2.0 GB for all 430 snapshots. The user's standing preference is
  never to lower output fidelity to save disk, so the volume is on by default (`--no-volume` drops it).
- The volume serves the one-layer view, the x-z cross-section at any y, and (later) any other slice.
- `grain.txt` per-grain block only (int16); the per-material block (H3: 5 materials x 416 grains x 4) is skipped.
- Parallel over snapshots with `multiprocessing` (fork): ~2 s per H3 snapshot per core here.
- Read stage: `dipole-field.txt` cells gridded by ID (see bundle-format.md). The per-atom
  `atomistic_dipole_field.txt` is not reduced yet (marked `shortcut:` in reduce.py).

## Accuracy (2026-10-27, after the user's review of the first mockup)

- **Grain shapes come from the atoms.** The first mockup drew `grain_shapes.txt` polygons. Checked against the H3
  atoms, only 85.6 % of magnetic columns lay inside their own grain's polygon (5.1 % in another grain's, 9.4 % in
  none): the polygons are the Voronoi cells before `voronoi-elliptical-rounding`, they miss grains that wrap across
  the periodic boundaries, and some are matched to the wrong grain. The reducer now builds a top-down grain-ID map
  (one value per 2.5 A column, from each atom's grain ID in the coordinate files; H3: 0 columns hold two grains)
  and the viewer colours those pixels. Polygons are used only when a bundle has no coordinates, and are labelled
  approximate.
- **Grain centres and atom counts come from the atoms** (periodic circular means). `grain-coordinates.txt` counts
  non-magnetic atoms assigned to a grain: in H3 all `fill` atoms belong to grain 0 (48,483 atoms listed, ~19k
  magnetic), which moved its listed centre 15 nm. Its magnetisation in `grain.txt` is not affected.
- **Built-in checks**, stored in the manifest and shown under Run details: (1) every stored column value against
  the exact float64 mean (int8 rounding, <= 1/254); (2) per-grain mean spin of every snapshot against `grain.txt`.
  VAMPIRE writes snapshot k at step `k x rate + 1`, one step after the grain row of the same output, so for a
  finite-temperature run the comparison is one step apart (differences = one step of thermal motion; ~2e-2 for the
  290-atom grains of the test fixture). At 0 K and at a shared step the means agree to 0 (tests/fixture-t0).
- **What reduction keeps and drops** (90 GB -> ~2.2 GB for H3): all grain-level time series at full time resolution
  (int16, 3e-5); every snapshot (none dropped by default); every magnetic column at full atomic in-plane
  resolution (through-thickness mean); every atomic layer at 2 x 2 sites (0.5 nm); all as int8 per component
  (resolution 0.008). Dropped: the non-magnetic atoms, the per-atom detail inside a 2 x 2 x 1 cell, and precision
  beyond 1/127. Analysis numbers (outcomes, freezing temperature) are computed from the full-precision files.
- **The random start of H3 is real**: the supervisor's material file has `initial-spin-direction = random`; the MC
  relax orders each grain up or down at random (47 % up), and all 416 grains start the write with the sign they
  ended the relax with. The earlier 3070 runs started saturated (`0,0,1`), which made writing easy to see. For
  random media the "m_z vs intended" colour mode shows agreement with the bit each grain sits in, and greys out
  grains the laser never melts (H3: only 148 of 416 grains, a ~24 nm band, melt in a 58 nm track).

## Generality (2026-10-27, user: everything must work for future runs)

- Head model now copies VAMPIRE's track logic: `max_tracks = floor((Ly - 1 A - 2 padding) / track_size)`,
  `bits_per_track = floor((Lx - 1 A) / bit_size)`, tracks reduced to `round(number_of_bits / bits_per_track)` only
  if that is fewer; sequences trimmed or padded with field-free 0 bits to fill the tracks (user-defined sequences
  read from `hamr:bit-sequence`). Checked against the 2-track overnight run's `B_applied` (1e-16 T) and head
  positions in its log.
- `grain.txt` leading columns are taken from the input's `grain:` keywords in order (the overnight run has
  `time-steps`, H3 `real-time`); unknown-width keywords before `grain:magnetisation` stop the reducer with a
  message naming the table to extend.
- Grids adapt (TOP_MAX, LAYER_MAX in reduce.py); `--layer-factor 1` gives per-atom layers.
- Coordinate files without grain IDs: Voronoi outlines, labelled approximate; snapshot checks against grain.txt
  need the IDs and are skipped.
- Truncated snapshots (copies caught mid-transfer) are skipped with a warning and listed in the manifest.
- Space-time rows shrink below 1 px for large runs (canvas height capped), track separators for several tracks;
  time ticks spaced by the run length.

## Viewer changes from the second review (2026-10-27)

- Theme toggle moved next to the run name (it pushed the stage buttons on narrow screens).
- Start / end buttons (Home / End keys) to compare the first and last frames.
- Correct-grain counts follow the timeline (bits lane, inspector: "at t" and "at end"); the analysis section
  keeps the final numbers.
- *Bit cells* overlay: bit regions on every track, and the band hotter than the freezing temperature.
- Atom views print the snapshot time on the image when it is not the timeline's time. Grains view = grain.txt at
  every grain frame (1 ps in H3); atom views = the nearest spin snapshot (10 ps in H3), so with few snapshots the
  two differ by the time between them.

## Physics choices

- The head field is rebuilt from the input with VAMPIRE's own trapezoid (`docs/analysis.md`), not read from
  `output`: H3 did not write `B_applied`. **Finding: the H3 write (inputs from the supervisor's template) ran at a
  2 T peak field, not 1 T**: with only `sim:applied-field-strength` set, Hmax/Hmin keep their defaults (+1/-1 T)
  and the ramp climbs Hmax - Hmin. Reported to the user 2026-10-27; it is a property of the inputs, not of hamrvis.
- Correctness uses the signed z field (`bit polarity x unit-vector z`): H3 writes with (0, 0, -1).
- Freezing temperature measured from the data (median centre temperature at the decision moments), 588 K for H3,
  instead of requiring a Curie temperature on the command line.

## Visual design (2026-10-27, chosen by the user from two layouts and three colour options)

- **Aligned lanes**: every view of the write shares the down-track axis (film x), stacked like oscilloscope
  channels, with one crosshair through all lanes; inspector column on the right; transport bar under the lanes.
  Down-track position is the coordinate everything in recording is about (bits, transitions, laser, read-back).
- Colour is reserved for data. m_z: the first explorer's map (user preference, 2026-10-27): blue #2a78d6 (down),
  grey #f0efec, red #e34948 (up), saturating at |m_z| = 0.85; dark theme #3987e5 / #383835 / #e66767. Fields use a
  separate green-purple map. Correct = green, wrong = ochre (red and blue already mean up and down). Laser
  isotherms in ink; the field box takes the colour of the field direction.
- User feedback 2026-10-27: transport bar directly under the film (the main view); slow-motion speeds (0.1x-0.5x);
  space-time as in the first explorer (one row per melted grain sorted by x, time left to right, laser pass dashed,
  decision ticks), not a lane on the film axis; labelled data instead of prose everywhere: a scientific tool.
- Chrome: cool aluminium / slate neutrals (the platter), one typeface (Instrument Sans, condensed bold for titles,
  tabular figures for numbers), sentence-case labels.

## Testing

- `tests/fixture-run`: a 2,624-atom granular HAMR write made with VAMPIRE (regenerate: `tests/make_fixture.sh`).
- `tests/test_hamrvis.py`: field model vs VAMPIRE's B_applied, default 2 T peak, column averages vs a direct numpy
  computation, analysis output, export, and the viewer's JavaScript model vs the Python one (node).
- Visual checks: screenshots with Playwright + the `@sparticuz/chromium` npm package (the Playwright browser
  download is blocked by the sandbox firewall; Ubuntu's chromium is a snap stub).

## Not done yet / ideas

- Relax stage: user is happy with it for now (2026-10-27). Ideas to return to: grain |m| and m_z histograms over MC
  steps; time to equilibrate (|m| reaching its plateau) per grain and overall; fraction of grains up against MC
  step; grain-size against |m| (small grains are superparamagnetic); comparison of the relaxed state with the write's
  first frame (checkpoint hand-over check, done by hand on 2026-10-27: 416 of 416 grains keep their sign).
- Run comparison (e.g. the supervisor's grain-rounding sweep) was not requested for the first version.
- Movie export (canvas MediaRecorder to WebM, or ffmpeg from the reducer) and SVG export.
- Per-atom read-back field; peak temperature measured from snapshots rather than the laser model.
