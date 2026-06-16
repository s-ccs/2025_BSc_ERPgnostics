# `src/`

Reusable project code should live here.

The ERP loading, processing, augmentation, and plotting helpers live in
[`erp_pipeline/`](erp_pipeline/).

Before refactoring the notebooks into final source code, use this data contract:

- [DATA_FORMAT.md](DATA_FORMAT.md) describes the required bundle format for real
  data.
- [DATA_FORMAT_OVERVIEW.md](DATA_FORMAT_OVERVIEW.md) shows the structure and
  relationships graphically.

To Start training

`julia src/real_data_training/run_pipeline.jl`

To Start test topoplot

`julia --project=notebooks/model_test src/real_data_training/erpgnostics_topoplot_explorer.jl`