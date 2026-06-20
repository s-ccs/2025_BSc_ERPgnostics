# Real-data ResNet18 ERP training

Scores ERP images with a pretrained ResNet18 and writes CSV and model as outputs.

## Run

```bash
julia src/real_data_training/run_pipeline.jl
```

Every run recomputes everything and overwrites the output files in this folder.

## Pipeline

1. Load manual labels from `datasets/`.
2. Build 200-trial ERP image chunks and the four sort/polarity augmentations.
3. Score labeled combinations with 5-fold out-of-fold predictions.
4. Train one final model on all labeled samples.
5. Score unlabeled combinations with the final model.
6. Merge everything into one score per `(dataset, sorting_variable, channel)`.

`labeled` means the combination occurs in the dataset label files. The label is
one of the six ERP image classes or `no_class`.

`unlabeled` means the `(dataset, event/sort variable, channel)` combination is
part of the scoring universe but does not occur in the label files.

The final score is the mean of the underlying augmentation scores. Unlabeled
combinations are sliced like training data; whole-parent scoring is only the
fallback for combinations with fewer than 200 valid trials.

## Outputs

| file | content |
|------|---------|
| `lean_parent_scores.csv` | main output: one row and one score per combination |
| `lean_augmentation_scores.csv` | per-slice/per-augmentation scores behind each final score |
| `final_model.jld2` | final ResNet18 state trained on all labeled data |

## Explorer

```bash
julia --project=src/real_data_training src/real_data_training/erpgnostics_topoplot_explorer.jl
```

## Key Scripts

| file | role |
|------|------|
| `run_pipeline.jl` | orchestration |
| `augmentation.jl` | chunking and four augmentations |
| `train_cv.jl` | labeled out-of-fold scores |
| `train_final.jl` | final model |
| `predict_unlabeled.jl` | unlabeled scores |
| `aggregate_scores.jl` | merge and write lean CSVs |
