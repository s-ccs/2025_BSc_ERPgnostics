# `src/`

Source code for this project, grouped by purpose. Each subfolder is **self
contained**: it carries its own flat `Project.toml` (and `Manifest.toml`) and is
run by activating that folder, so there is no global/shared environment.

```
src/
├── simulation/              <- Simulated ERP-image generation + ResNet18 parameter search
│   ├── SimulationPipeline.jl    main module (ties the includes together)
│   ├── run_search.jl            entry point (activates this folder, runs a search)
│   ├── erpgen.jl, erpgen/       simulated-ERP generator
│   ├── pipeline/                image pipeline, parameter space, training, reporting
│   ├── outputs/                 search results and preview plots
│   ├── Project.toml             self-contained environment
│   ├── Manifest.toml
│   └── README.md
│
├── real_data_training/      <- Score real ERP images with a pretrained ResNet18
│   ├── run_pipeline.jl          entry point: julia src/real_data_training/run_pipeline.jl
│   ├── config.jl                activates this folder, loads the engine, constants
│   ├── model_engine.jl          vendored, self-contained model engine (no notebooks/ dep)
│   ├── data_loading.jl, augmentation.jl, model.jl
│   ├── train_cv.jl, train_final.jl, predict_unlabeled.jl, aggregate_scores.jl
│   ├── erpgnostics_topoplot_explorer.jl   GLMakie score explorer
│   ├── reference/               bundled reference data (e.g. positions_128.jld2)
│   ├── final_model.jld2, lean_*_scores.csv   run outputs (overwritten each run)
│   ├── Project.toml             self-contained environment (only the packages used)
│   ├── Manifest.toml
│   └── README.md
│
├── examples/                <- Small, runnable usage examples
│   ├── explore_erp.jl           load → process → augment → plot one ERP image
│   ├── explore_erp.ipynb        notebook version of the same walkthrough
│   ├── Project.toml             self-contained environment (only the packages used)
│   ├── Manifest.toml
│   └── README.md
│
└── README.md                <- this file
```

## Where are the ERP helper functions?

The reusable ERP loading / processing / augmentation / plotting helpers live in
[`../scripts/erp_pipeline/`](../scripts/erp_pipeline/). The `examples/` here are
their only caller and `include` them from there.

## Running a folder

Activate the folder you want and run its entry point, e.g.:

```bash
julia src/real_data_training/run_pipeline.jl     # config.jl activates the folder itself
julia --project=src/examples src/examples/explore_erp.jl
julia --project=src/simulation src/simulation/run_search.jl
```
