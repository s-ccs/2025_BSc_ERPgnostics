# Post-submit ResNet18 ERP scoring pipeline

A lean, notebook-free Julia pipeline that scores ERP images with a pretrained
[Metalhead.jl ResNet18](https://fluxml.ai/Metalhead.jl/stable/api/resnet/) and
merges old and newly computed scores into duplicate-free CSVs.

## Run

```bash
julia notebooks/post_submit/run_pipeline.jl
```

The pipeline reuses the shared `notebooks/model_test` package environment and the
Week-20 ResNet/Metalhead engine (`build_resnet_single_channel_pretrained`,
training, prediction, Gaussian-reference image pipeline). It does **not** depend
on any notebook.

## What it does

1. **Materialise labelled samples** — every labelled `(dataset, sort_variable,
   channel)` row from `datasets/` is split into fixed-trial chunks (200 trials)
   and each chunk gets the four sort/polarity augmentations.
2. **5-fold cross validation** — a fresh pretrained ResNet18 per fold; the union
   of validation predictions gives an **out-of-fold** score per labelled
   augmentation. A parent's score is the mean of its augmentation scores
   (`source = cv_labeled`).
3. **Final model** — one fresh pretrained ResNet18 trained on *all* labelled
   augmented samples.
4. **Score unlabelled combinations** — the scoring universe is every `(dataset,
   sort_variable, channel)` where the sort variable appears in that dataset's
   labels (any label, including `no_class`), across all channels. A combination
   is **unlabelled** when it is in this universe but has no label row of its own.
   Every unlabelled combination is **trial-sliced into 200-trial chunks exactly
   like the training data** (whole-parent fallback only when it has fewer than
   200 trials), each slice × augmentation is scored with the final model, and the
   per-combination mean is taken (`source = new_unlabeled`). This keeps the model
   on the trial dimension it was trained on.
5. **Combine** — `cv_labeled` (labelled) and `new_unlabeled` (unlabelled) are
   disjoint by construction, so every `(dataset, sorting_variable, channel)` gets
   **exactly one** model score. The old CSVs are intentionally not used as a
   source: labelled → CV, unlabelled → final model, one fresh score per
   combination.

The same augmentation (`augmentation.jl`) is used in every step, so labelled and
unlabelled scores are comparable.

> `unlabelled` is **not** a class — it only means no human looked at that
> combination. The model still classifies it (the score is P(pattern)).

## Outputs

Final lean CSVs (written next to this README):

| file | one row per | key columns |
|------|-------------|-------------|
| `lean_parent_scores.csv` | **combination** | `dataset, sorting_variable, channel, parent, manual_label, has_manual_label, score` |
| `lean_augmentation_scores.csv` | augmentation | `dataset, sorting_variable, channel, parent, augmentation_id, augmentation_name, score, source` |

`manual_label` is the human label or `unlabelled`.
`lean_parent_scores.csv` is the main deliverable: one row, one score per
`(dataset, sorting_variable, channel)`. The score always equals the mean of that
combination's augmentation scores.

## Interactive explorer

The old ERPgnostics-style clickable topoplot explorer has a lean post-submit
version that reads `lean_parent_scores.csv` and the built `datasets/` directly:

```bash
julia --project=notebooks/model_test notebooks/post_submit/erpgnostics_topoplot_explorer.jl
```

Useful overrides:

```bash
POST_SUBMIT_START_DATASET=02_new_roamm_reading \
POST_SUBMIT_START_SORT=fixation_duration \
julia --project=notebooks/model_test notebooks/post_submit/erpgnostics_topoplot_explorer.jl
```

Set `POST_SUBMIT_PARENT_SCORES` or `POST_SUBMIT_DATASETS_ROOT` to point at a
different score CSV or dataset root. Set `POST_SUBMIT_EXPLORER_SMOKE=true` to
validate loading without opening the GLMakie window.

The detail ERP image is rendered at the dataset's native trial/time resolution:
trials are sorted, each timepoint is z-scored across trials, then the same
Gaussian smoothing settings as the score pipeline are applied. No resize is
applied in the explorer detail view.

Every run recomputes everything from scratch and overwrites the two lean CSVs.
Per-fold and final training metrics are written to `outputs/` as diagnostics.

## Modules

| file | responsibility |
|------|----------------|
| `config.jl` | environment, paths, constants, seeds, logging |
| `data_loading.jl` | dataset/label/channel discovery and loading |
| `augmentation.jl` | shared sort/polarity/z-score/chunking + image preprocessing |
| `model.jl` | ResNet18 build/train/predict wrappers + artifact save/load |
| `train_cv.jl` | stratified fold assignment + 5-fold CV |
| `train_final.jl` | final model on all labelled augmented samples |
| `predict_unlabeled.jl` | score missing target combinations |
| `aggregate_scores.jl` | merge/dedup → lean CSVs |
| `erpgnostics_topoplot_explorer.jl` | interactive GLMakie topoplot + ERP-image explorer for lean scores |
| `run_pipeline.jl` | orchestration (recompute + overwrite) |

## Environment overrides

`POST_SUBMIT_EPOCHS`, `POST_SUBMIT_LR`, `POST_SUBMIT_FOLDS`,
`POST_SUBMIT_TARGET_TRIALS`,
`POST_SUBMIT_BATCHSIZE_GPU` (default 64, thesis value),
`POST_SUBMIT_BATCHSIZE_CPU` (default 8),
`POST_SUBMIT_LABEL_SMOOTHING` (default 0.02, thesis value).
