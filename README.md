# **BSc-Thesis:** Automated Pattern Detec­tion in ERP Images Using Convolutional Neural Net­works (CNN)
**Author:** *Benjamin Borchert*

**Supervisor(s):** *Vladimir Mikheev*

**Year:** *2026*

## Project Description

ERP images keep the single-trial structure of EEG that grand-averaging hides: trials are sorted by an experimental variable and stacked into a 2D image, where recurring shapes (e.g. a sigmoid band) can point to cognitive effects. Inspecting them by hand does not scale, and labelled data are scarce.

This thesis tests two things:

1. **Sim-to-real transfer** — whether a CNN trained only on *simulated* ERP images (built with [UnfoldSim.jl](https://github.com/unfoldtoolbox/UnfoldSim.jl)) can detect the `sigmoid` pattern in *real* ERP images, and how large the remaining simulation-to-real gap is.
2. **Manual labelling** — whether a (pretrained) ResNet18, trained on a pool of manually labelled real ERP images with a pattern-preserving augmentation, can detect these patterns reliably.

Main result: training on real labels with the augmentation reaches a balanced accuracy of ~0.92 at a 64×64 input size; the sim-to-real gap and simulation runtime remain the main open problems. The full write-up is in [`report/thesis/thesis.typ`](report/thesis/thesis.typ).

## Zotero Library Path

The bibliography is kept as a BibTeX file in the report folder: [`report/thesis/refs.bib`](report/thesis/refs.bib).

## Instruction for a new student

The code is **Julia** (tested with 1.12). Each subfolder under [`src/`](src/) is **self-contained**: it ships its own `Project.toml`/`Manifest.toml` and is run by activating that folder. Python is only used by a few Label Studio helper scripts in `notebooks/`.

### 0. Data format

Real ERP data lives in [`datasets/`](datasets/), one JLD2 bundle per dataset/participant (`events.jld2`, optional `labels.jld2`, and `signals/<channel>.jld2`). The bundles store **preprocessed, event-locked trials**, not raw EEG. The exact schema is documented in [`datasets/DATA_FORMAT.md`](datasets/DATA_FORMAT.md).

ERP-image steps (sort → per-timepoint z-score → Gaussian smoothing → resize) are applied **after** loading by the shared helpers in [`scripts/erp_pipeline/`](scripts/erp_pipeline/).

### 1. Plot one ERP image end-to-end

```bash
julia --project=src/examples -e 'import Pkg; Pkg.instantiate()'   # once
julia --project=src/examples src/examples/explore_erp.jl          # load → process → augment → plot
```

### 2. Simulation parameter search (RQ 1)

Searches simulator parameters (broad-random, Monte Carlo, Latin hypercube) so a ResNet18 trained on simulated `sigmoid` vs `no_class` images transfers to the real `fixations_dataset`.

```bash
julia --project=src/simulation -e 'import Pkg; Pkg.instantiate()'   # once
julia -t auto --project=src/simulation src/simulation/run_search.jl  # -t auto for CPU simulation; CUDA used if available
```

`run_search.jl` defaults to a small **test** budget (3 candidates/strategy). For the thesis budget set `strategy_budgets = Dict(:broad_random => 12, :latin_hypercube => 48, :monte_carlo => 48)` in `build_test_config`. Outputs go to `src/simulation/outputs/strategy_64x64_resnet18/`. See [`src/simulation/README.md`](src/simulation/README.md).

### 3. Real-data training (RQ 2)

Trains/scores a pretrained ResNet18 on manually labelled real ERP images using 200-trial slicing + the four sort/polarity augmentations.

```bash
julia --project=src/real_data_training -e 'import Pkg; Pkg.instantiate()'   # once
julia src/real_data_training/run_pipeline.jl                                # config.jl activates the folder
```

Outputs (in `src/real_data_training/`): `lean_parent_scores.csv` (one score per dataset/sort-variable/channel), `lean_augmentation_scores.csv`, and `final_model.jld2`. See [`src/real_data_training/README.md`](src/real_data_training/README.md).

### Original exploratory workflow

The cleaned, reproducible pipelines above are derived from the week-by-week exploration in [`notebooks/`](notebooks/) (data-source screening, Label Studio export/import, dataset building, figure generation). These are kept for provenance but are not needed to reproduce the main results.

## Overview of Folder Structure 

```
│projectdir          <- Project's main folder. It is initialized as a Git
│                       repository with a reasonable .gitignore file.
│
├── report           <- **Immutable and add-only!**
│   ├── proposal     <- Proposal PDF
│   ├── thesis       <- Final Thesis PDF
│   ├── talks        <- PDFs (and optionally pptx etc) of the Intro,
|   |                   Midterm & Final-Talk
|
├── _research        <- WIP scripts, code, notes, comments,
│   |                   to-dos and anything in an alpha state.
│
├── datasets         <- Preprocessed, event-locked ERP data as JLD2 bundles,
│   |                   one folder per dataset/participant (see DATA_FORMAT.md).
│
├── results          <- Exported result tables (e.g. cross-validation CSVs).
│
├── plots            <- All exported plots go here, best in date folders.
|   |                   Note that to ensure reproducibility it is required that all plots can be
|   |                   recreated using the plotting scripts in the scripts folder.
|
├── notebooks        <- Pluto, Jupyter, Weave or any other mixed media notebooks.*
│
├── scripts          <- Various scripts, e.g. simulations, plotting, analysis,
│   │                   The scripts use the `src` folder for their base code.
│
├── src              <- Source code for use in this project. Contains functions,
│                       structures and modules that are used throughout
│                       the project and in multiple scripts.
│
├── test             <- Folder containing tests for `src`.
│   └── runtests.jl  <- Main test file
│   └── setup.jl     <- Setup test environment
│
├── README.md        <- Top-level README. A fellow student needs to be able to
|   |                   continue your project. Think about her!!
|
├── .gitignore       <- focused on Julia, but some Matlab things as well
│
├── (Manifest.toml)  <- Contains full list of exact package versions used currently.
|── (Project.toml)   <- Main project file, allows activation and installation.
└── (Requirements.txt)<- in case of python project - can also be an anaconda file, MakeFile etc.
                        
```