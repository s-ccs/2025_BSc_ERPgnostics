# `examples/`

Small, runnable walkthroughs of the ERP image pipeline. They show how to load a
dataset, build and process an ERP image, create the model-ready augmentations
and plot the result — using the reusable helpers in
[`../../scripts/erp_pipeline/`](../../scripts/erp_pipeline/).

| file | what it does |
|------|--------------|
| `explore_erp.jl` | script version (cell-delimited with `# %%`) of the full walkthrough |
| `explore_erp.ipynb` | notebook version of the same steps |

## Environment

This folder is self-contained: it carries its own flat `Project.toml` /
`Manifest.toml` and activates itself, so it does not depend on any other
environment. The dependencies are exactly those the `erp_pipeline` helpers need.

## Run

```bash
# script (activates src/examples and includes the helpers itself)
julia src/examples/explore_erp.jl

# notebook: open src/examples/explore_erp.ipynb and run the setup cell first
```

Both read the bundled datasets from the repository's `datasets/` folder.
