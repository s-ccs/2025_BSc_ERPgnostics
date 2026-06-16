# Simulation: E4 64x64 ResNet18 Parameter Search

This directory is self-contained. It searches simulator parameters with
three strategies so that a ResNet18 trained purely on **simulated** `sigmoid`
versus `no_class` ERP images transfers as well as possible to the **real**
`fixations_dataset`.

The goal is to reproduce realistic `sigmoid` / `no_class` instances: candidate
quality is the balanced accuracy of the trained network on the real fixation
validation set.

## Search strategies

- **broad random** — domain randomisation; each parameter drawn uniformly and
  independently.
- **Monte Carlo** — the same uniform draw kept as a separate baseline.
- **Latin hypercube** — a stratified design that spreads candidates more evenly
  across the 48 parameter dimensions.

A hand-crafted baseline configuration (the "starting parameters") is evaluated
unchanged for reference.

## Layout

```
src/simulation/
├── run_search.jl            # Entry point: builds a RunConfig and calls run_search
├── SimulationPipeline.jl    # Module: loads dependencies and includes the pipeline
├── Project.toml             # Self-contained package environment
├── erpgen.jl, erpgen/       # VENDORED simulator (see "Vendored simulator" below)
├── pipeline/
│   ├── config.jl            # RunConfig, seed derivation, device helpers
│   ├── image_pipeline.jl    # Shared sort → z-score → Gaussian → resize pipeline
│   ├── parameter_space.jl   # 48-dim parameter space; cfg ⇄ parameter vector
│   ├── fixations_data.jl    # Loads the real validation set + dataset dimensions
│   ├── search_strategies.jl # Candidate proposal per strategy
│   ├── resnet18.jl          # Single-channel ResNet18 (random / pretrained)
│   ├── training.jl          # Simulate, train, evaluate, sanity gate, score
│   ├── preview.jl           # Sim-vs-real preview figure
│   ├── reporting.jl         # Evaluation loop, aggregation, all CSV exports
│   └── run.jl               # Orchestration (run_search)
└── outputs/                 # Results are written here (see "Outputs")
```

## Running

From the repository root:

```bash
julia --project=src/simulation -e 'import Pkg; Pkg.instantiate()'        # once
julia -t auto --project=src/simulation src/simulation/run_search.jl
```

`run_search.jl` activates this project, so the `--project` flag is optional once
the environment is instantiated.

**Devices and threads.** When a CUDA device is functional it is used for the
ResNet18 training and inference (with `allowscalar(false)` and device 0,
mirroring `src/real_data_training`); otherwise the run falls back to CPU. The
ERPGen simulator always runs on the CPU and parallelises the per-image
simulation across `Threads.nthreads()` — start Julia with `-t auto` to use all
CPU cores. BLAS is pinned to a single thread so it does not oversubscribe the
cores during simulation.

**Seeds.** There is no global root seed: whenever a fresh seed is needed the
pipeline simply uses `time_ns()` (`new_seed()`), so each run is independently
random. The simulated training images are drawn fresh on every call.

**Identical to the real data.** The simulated trials are kept 1:1 with the real
fixation recording — no trial dropout and no cropping — and go through the exact
same image pipeline, so simulated and real images match in dimensions and
processing.

### Test vs. full run

`run_search.jl` defaults to a **test** budget of **three candidates per
strategy**. To reproduce the thesis budget, set in `build_test_config`:

```julia
strategy_budgets = Dict(:broad_random => 12, :latin_hypercube => 48, :monte_carlo => 48)
```

Other knobs (repeats, images per pattern, training hyper-parameters, thresholds)
live in `RunConfig` in `pipeline/config.jl`.

## Outputs

All outputs go to `outputs/strategy_64x64_resnet18/`:

- `best_run.csv` — the overall winning candidate and its 48 parameters.
- `preview_default.png`, `preview_broad_random.png`, `preview_latin_hypercube.png`,
  `preview_monte_carlo.png` — three simulated (top) vs. three real (bottom)
  sigmoid images in the square week_24 style, for the baseline parameters and for
  each strategy's best candidate. These files are overwritten on every run.
- `posthoc_exports/` — per-candidate summaries, rankings, per-method summary,
  baseline summary, sanity results, and run metadata.
- `extra_exports/` — strategy efficiency curve, score distributions, repeat
  stability, prediction balance, long-format parameters, parameter-performance
  correlations, and the runtime budget.

## Vendored simulator

`erpgen.jl` and the `erpgen/` folder are copied **verbatim** from
`notebooks/data_generation/` and are intentionally left unchanged so the
simulator behaves identically to the original experiment. The pipeline makes the
simulator reproducible under explicit seeds by replacing its time-based RNG
helpers at load time (`install_deterministic_erpgen_rng!` in `pipeline/run.jl`),
without modifying the vendored files.

Everything outside `erpgen/` is a clean, fully documented reimplementation that
follows the Julia style guide and uses docstrings throughout.
