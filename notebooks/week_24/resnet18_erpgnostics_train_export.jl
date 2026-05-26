# %% [markdown]
# # ResNet18 ERPgnostics training export
#
# This notebook trains the binary ERP-pattern ResNet18 on the real JLD2 datasets
# under `datasets/` and writes a reusable model artifact. It intentionally does
# not build the ERPgnostics explorer; use
# `resnet18_erpgnostics_import_explorer.ipynb` for visualization.

# %%
# Optional quick-run overrides. Uncomment before running this first cell.
# ENV["WEEK24_RUN_CV"] = "false"
# ENV["WEEK24_RESNET18_EPOCHS"] = "1"

include(joinpath(@__DIR__, "resnet18_erpgnostics_common.jl"))

TRAIN_EXPORT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_train_export")
TRAIN_MODEL_ARTIFACT = model_artifact_path(TRAIN_EXPORT_DIR)
EXPORT_PARENT_SCORES = lowercase(get(ENV, "WEEK24_EXPORT_PARENT_SCORES", "true")) in ("true", "1", "yes")

println("Training output directory: ", TRAIN_EXPORT_DIR)
println("Model artifact path: ", TRAIN_MODEL_ARTIFACT)
println("TARGET_TRIALS = ", TARGET_TRIALS)
println("EXPORT_PARENT_SCORES = ", EXPORT_PARENT_SCORES)

# %% [markdown]
# ## Train
#
# By default this runs the same 5-fold CV plus final all-data model as the
# combined notebook. For a quick smoke run, set these before executing:
#
# ```julia
# ENV["WEEK24_RUN_CV"] = "false"
# ENV["WEEK24_RESNET18_EPOCHS"] = "1"
# ```

# %%
training_run = run_training_pipeline(
    output_dir = TRAIN_EXPORT_DIR,
    nepochs = TRAIN_EPOCHS,
    lr = TRAIN_LR,
    k_folds = K_FOLDS,
    seed = GLOBAL_SEED,
    run_cv = RUN_CV,
)

# %% [markdown]
# ## Export Model
#
# The artifact stores the full CPU `Flux.state(model)` plus metadata. This is
# important for ResNet18 because BatchNorm running statistics are model state,
# not trainable parameters. Saving only `Flux.trainables(model)` makes the
# imported model produce nearly constant scores.

# %%
artifact_metadata = Dict{String, Any}(
    "source_notebook" => "notebooks/week_24/resnet18_erpgnostics_train_export.ipynb",
    "output_dir" => TRAIN_EXPORT_DIR,
    "run_config_path" => joinpath(TRAIN_EXPORT_DIR, "run_config.json"),
    "final_train_metrics_path" => joinpath(TRAIN_EXPORT_DIR, "final_train_metrics.csv"),
    "fold_metrics_path" => joinpath(TRAIN_EXPORT_DIR, "fold_metrics.csv"),
    "trained_on" => "real JLD2 datasets from datasets/, excluding datasets/simulated",
    "all_parent_scores_path" => parent_scores_path(TRAIN_EXPORT_DIR),
    "all_augmentation_scores_path" => augmentation_scores_path(TRAIN_EXPORT_DIR),
)

save_resnet18_model_artifact(TRAIN_MODEL_ARTIFACT, training_run.final_model; metadata = artifact_metadata)
println("Saved model artifact: ", TRAIN_MODEL_ARTIFACT)

training_run.final.metrics_df

# %% [markdown]
# ## Export Model Scores
#
# This writes one parent probability per dataset/sort-variable/channel ERP image
# for all real datasets under `datasets/`, excluding `datasets/simulated`.
# Sort variables are included only when the dataset has at least one manual
# pattern label for that sort variable.

# %%
if EXPORT_PARENT_SCORES
    println("Scoring all labelled-sort parent ERP images with the final ResNet18.")
    parent_score_run = score_dataset_parent_images(
        training_run.final_model,
        training_run.final.device;
        labels_df = training_run.labels_df,
    )
    saved_score_paths = save_parent_score_outputs(TRAIN_EXPORT_DIR, parent_score_run)
    println("Saved parent scores: ", saved_score_paths.parent_scores_path)
    println("Saved augmentation scores: ", saved_score_paths.augmentation_scores_path)
    if nrow(parent_score_run.skipped_df) > 0
        @warn "Some dataset/sort/channel combinations could not be scored." skipped_rows = nrow(parent_score_run.skipped_df)
    end
    first(parent_score_run.score_df, min(12, nrow(parent_score_run.score_df)))
else
    println("Skipping parent-score export because WEEK24_EXPORT_PARENT_SCORES=false.")
    parent_score_run = nothing
end

# %% [markdown]
# ## Training Summary

# %%
if nrow(training_run.cv.metrics_df) > 0
    summarize_metrics(training_run.cv.metrics_df)
else
    training_run.final.metrics_df
end
