#!/usr/bin/env python3
"""Generate `notebooks/week_15/try_new_data.ipynb`."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "week_15" / "try_new_data.ipynb"


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


cells = [
    markdown_cell(
        """# Public ERP Comparison Datasets

This notebook uses **already downloaded** public ERP datasets and visualizes them with the same base pipeline as the other Week 15 notebooks:

- baseline correction on the prestimulus window `-0.2s..0s`, then post-stimulus segment only
- sorting by the selected sort variable
- `zscore_timepoints(...)` per time index across trials
- Gaussian low-pass
- resize to `64x64`
- asymmetric zero-anchored colorbar from `ERPImageUtils`

The registered sources combine:

- five cleaned ERP CORE components already prepared in this repo
- the previously audited public shortlist sources that were still missing locally and have now been materialized as comparison bundles
- additional EYE-EEG and PhysioNet ERPBCI bundles prepared from their official public example releases

The notebook also shows a few reference images from the existing fixation dataset so the visual structure can be compared directly.

The datasets are expected to already exist under `notebooks/datasets`. This notebook only imports them.
"""
    ),
    code_cell(
        """import Pkg

week15_dir = if isfile(joinpath(pwd(), "try_new_data.ipynb"))
    pwd()
else
    joinpath(pwd(), "notebooks", "week_15")
end

Pkg.activate(joinpath(week15_dir, "..", "model_test"))

using CairoMakie
using DataFrames

include(joinpath(week15_dir, "try_new_data_helpers.jl"))
using .Week15TryNewData

println("Week15 dir: ", week15_dir)
println("Comparison dataset keys: ", COMPARISON_DATASET_KEYS)
println("Target image size: ", REAL_TARGET_SIZE)"""
    ),
    markdown_cell(
        """## Load Pre-Downloaded Data

All comparison bundles have already been downloaded and converted into a Julia-friendly `HDF5 + events.csv` layout under `notebooks/datasets`.
"""
    ),
    code_cell(
        """bundles = [load_clean_dataset_bundle(key) for key in COMPARISON_DATASET_KEYS]

external_dataset_summary_df(bundles)"""
    ),
    markdown_cell(
        """## Axis Audit

The table below checks that each source is interpreted as `(channel, time, trial)` inside Julia, that the trial axis matches the number of event rows for each subject, and that the stored epoch time axis still includes the prestimulus segment.
"""
    ),
    code_cell(
        """dataset_axis_audit_df(bundles)"""
    ),
    markdown_cell(
        """## Available Sort Columns

The table below lists every available non-constant sort column in `events.csv`.
`preview_default = true` marks the columns that are plotted automatically.
Pure bookkeeping columns such as `sample_index`, `epoch_index`, and `source_file`
remain visible in the table, but are not auto-plotted because they mostly reflect
acquisition order rather than a meaningful experimental sort.
"""
    ),
    code_cell(
        """available_sort_columns_df(bundles)"""
    ),
    markdown_cell(
        """## Source Overview

The following tables show the official sources and the loader notes stored with each prepared bundle.
"""
    ),
    code_cell(
        """dataset_source_overview_df(bundles)"""
    ),
    code_cell(
        """for bundle in bundles
    println("\\n=== ", bundle.dataset_key, " ===")
    display(dataset_source_example_df(bundle))
end"""
    ),
    markdown_cell(
        """## Fixation Reference

This section shows a few reference images from the fixation dataset already present in the repo, processed with the same image pipeline.
"""
    ),
    code_cell(
        """fixation_summary_df()"""
    ),
    code_cell(
        """fixation_cache = load_fixation_reference_cache(per_sort_var = 2)
fig_fix = plot_fixation_reference_grid(fixation_cache)
fig_fix"""
    ),
    markdown_cell(
        """## ERP Previews By Sort Variable

For each dataset and each preview-default sort variable, the notebook plots a compact set of subjects and preferred channels so it is easy to see which patterns look visually similar to the fixation dataset.
"""
    ),
    code_cell(
        """for bundle in bundles
    println("\\n=== PREVIEWS FOR ", bundle.dataset_key, " ===")
    for spec in recommended_preview_specs(bundle)
        preview = build_dataset_sort_preview(bundle;
            sort_col = spec.sort_col,
            filters = spec.filters,
        )
        fig = plot_dataset_sort_preview(preview)
        display(fig)
    end
end"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Julia 1.11.6",
            "language": "julia",
            "name": "julia-1.11",
        },
        "language_info": {
            "file_extension": ".jl",
            "mimetype": "application/julia",
            "name": "julia",
            "version": "1.11.6",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
