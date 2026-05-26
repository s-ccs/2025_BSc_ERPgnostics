# %% [markdown]
# # ResNet18 ERPgnostics explorer from saved model
#
# This notebook does not train. It loads the exported ResNet18 artifact from
# `resnet18_erpgnostics_train_export.ipynb` only when saved scores are missing.
# The preferred path is to load the train-export CSV scores and browse all real
# datasets with at least one manual pattern label for a sort variable.

# %%
include(joinpath(@__DIR__, "resnet18_erpgnostics_common.jl"))

DEFAULT_TRAIN_EXPORT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_train_export")
DEFAULT_MODEL_ARTIFACT = model_artifact_path(DEFAULT_TRAIN_EXPORT_DIR)
MODEL_ARTIFACT_PATH = get(ENV, "WEEK24_MODEL_ARTIFACT", DEFAULT_MODEL_ARTIFACT)
SAVED_SCORE_DIR = get(ENV, "WEEK24_SCORE_OUTPUT_DIR", DEFAULT_TRAIN_EXPORT_DIR)
USE_SAVED_MODEL_SCORES = lowercase(get(ENV, "WEEK24_USE_SAVED_MODEL_SCORES", "true")) in ("true", "1", "yes")

EXPLORER_OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_import_explorer")
mkpath(EXPLORER_OUTPUT_DIR)

println("Model artifact path: ", MODEL_ARTIFACT_PATH)
println("Saved score directory: ", SAVED_SCORE_DIR)
println("Explorer output directory: ", EXPLORER_OUTPUT_DIR)

# %% [markdown]
# ## System Diagnostics
#
# The static plots below work with CairoMakie. The clickable explorer needs a
# frontend that renders WGLMakie/Bonito HTML with a live Julia connection. In
# practice that is more reliable in a browser-backed IJulia/Jupyter session than
# in VS Code's static plot preview.

# %%
function package_version_by_name(name::AbstractString)
    for (_, dep) in Pkg.dependencies()
        if dep.name == String(name)
            return dep.version === nothing ? "not installed" : string(dep.version)
        end
    end
    return "not installed"
end

system_diagnostics = DataFrame(
    item = [
        "Julia",
        "WGLMakie",
        "Bonito",
        "CairoMakie",
        "IJulia",
        "jupyter executable",
        "VS Code PID env",
        "model artifact exists",
        "saved parent scores exist",
        "saved augmentation scores exist",
    ],
    value = [
        string(VERSION),
        package_version_by_name("WGLMakie"),
        package_version_by_name("Bonito"),
        package_version_by_name("CairoMakie"),
        package_version_by_name("IJulia"),
        something(Sys.which("jupyter"), "not found on PATH"),
        get(ENV, "VSCODE_PID", "not set"),
        string(isfile(MODEL_ARTIFACT_PATH)),
        string(isfile(parent_scores_path(SAVED_SCORE_DIR))),
        string(isfile(augmentation_scores_path(SAVED_SCORE_DIR))),
    ],
)

system_diagnostics

# %% [markdown]
# ## Load Labels and Scores

# %%
labels_df = load_all_real_labels()
positive_sort_summary = positive_dataset_sort_variable_summary(labels_df)
dataset_sort_variables = dataset_sort_variable_map(labels_df; require_pattern = true)
available_dataset_keys = sort(collect(keys(dataset_sort_variables)))

positive_sort_summary

# %% [markdown]
# ## Load or Compute Model Output
#
# The score shown in the topoplots is the parent-image probability: the mean
# `prob_class` over the four sort/polarity augmentations for that channel and
# sort variable. The train-export notebook writes these CSV files so this
# explorer can run without recomputing model inference.

# %%
loaded_model = nothing
model = nothing
device = cpu
use_cuda = false

using_saved_scores = USE_SAVED_MODEL_SCORES && saved_parent_scores_available(SAVED_SCORE_DIR)
if using_saved_scores
    println("Loading saved model scores from $(SAVED_SCORE_DIR).")
    score_run = load_parent_score_outputs(SAVED_SCORE_DIR)
else
    println("Saved model scores are missing or disabled; loading ResNet18 and scoring parent images now.")
    device, use_cuda = Generalization.setup_device()
    loaded_model = load_resnet18_model_artifact(MODEL_ARTIFACT_PATH; device = device)
    model = loaded_model.model
    score_run = score_dataset_parent_images(
        model,
        device;
        labels_df = labels_df,
        dataset_sort_variables = dataset_sort_variables,
    )
    save_parent_score_outputs(EXPLORER_OUTPUT_DIR, score_run)
end

score_df = score_run.score_df
augmentation_score_df = score_run.augmentation_df

CSV.write(parent_scores_path(EXPLORER_OUTPUT_DIR), score_df)
augmentation_export_df = :processed_img in propertynames(augmentation_score_df) ?
    select(augmentation_score_df, Not(:processed_img)) :
    augmentation_score_df
CSV.write(augmentation_scores_path(EXPLORER_OUTPUT_DIR), augmentation_export_df)

INITIAL_DATASET_KEY = initial_dataset_key(score_df)
INITIAL_SORT_VARIABLE = initial_sort_variable(score_df, INITIAL_DATASET_KEY)
INITIAL_CHANNEL_NAME = initial_channel_name(score_df, INITIAL_DATASET_KEY, INITIAL_SORT_VARIABLE)

DataFrame(
    item = [
        "score_source",
        "n_scored_datasets",
        "n_parent_images",
        "n_augmentation_images",
        "initial_dataset",
        "initial_sort_variable",
        "initial_channel",
    ],
    value = [
        using_saved_scores ? "saved_csv" : "computed_from_model",
        string(length(score_dataset_keys(score_df))),
        string(nrow(score_df)),
        string(nrow(augmentation_score_df)),
        INITIAL_DATASET_KEY,
        INITIAL_SORT_VARIABLE,
        INITIAL_CHANNEL_NAME,
    ],
)

# %%
first(score_df, 12)

# %%
score_diagnostics = combine(
    groupby(score_df, [:dataset_key, :sort_variable]),
    :score_class => minimum => :min_score,
    :score_class => maximum => :max_score,
    :score_class => mean => :mean_score,
    :score_class => std => :std_score,
    nrow => :n_channels,
)

if all(coalesce.(score_diagnostics.std_score, 0.0) .< 1e-4)
    @warn "All parent scores are nearly constant. This usually means the model artifact is stale/incomplete or the final model underfit. Rerun the training-export notebook and make sure it writes resnet18_final_state.jld2 and score CSVs."
end

score_diagnostics

# %% [markdown]
# ## Visible Static ERPgnostics Overview
#
# This cell always uses CairoMakie, so it should display reliably in VS Code and
# Jupyter. The following interactive cell is optional.

# %%
CairoMakie.activate!(type = "svg")

score_filename_part(s) = replace(String(s), r"[^A-Za-z0-9_.-]" => "_")

topoplot_fig = plot_dataset_score_topoplots(
    score_df,
    INITIAL_DATASET_KEY;
    sort_variables = score_sort_variables(score_df, INITIAL_DATASET_KEY),
)
display(topoplot_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "parent_score_topoplots_$(score_filename_part(INITIAL_DATASET_KEY)).svg"), topoplot_fig)

for dataset_key in score_dataset_keys(score_df)
    overview_fig = plot_dataset_score_topoplots(
        score_df,
        dataset_key;
        sort_variables = score_sort_variables(score_df, dataset_key),
    )
    CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "parent_score_topoplots_$(score_filename_part(dataset_key)).svg"), overview_fig)
end

detail_fig = plot_dataset_parent_image(score_df, INITIAL_DATASET_KEY, INITIAL_SORT_VARIABLE, INITIAL_CHANNEL_NAME)
display(detail_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "detail_$(score_filename_part(INITIAL_DATASET_KEY))_$(score_filename_part(INITIAL_SORT_VARIABLE))_$(score_filename_part(INITIAL_CHANNEL_NAME)).svg"), detail_fig)

augmented_inputs_fig = plot_augmented_model_inputs(augmentation_score_df, INITIAL_DATASET_KEY, INITIAL_SORT_VARIABLE, INITIAL_CHANNEL_NAME)
display(augmented_inputs_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "model_inputs_$(score_filename_part(INITIAL_DATASET_KEY))_$(score_filename_part(INITIAL_SORT_VARIABLE))_$(score_filename_part(INITIAL_CHANNEL_NAME)).svg"), augmented_inputs_fig)

println("Saved static explorer exports in $(EXPLORER_OUTPUT_DIR).")

# %% [markdown]
# ## Manual Detail Selector
#
# If WGLMakie is not rendered by the notebook frontend, use this small selector:
# change `SELECT_DATASET_KEY`, `SELECT_SORT_VARIABLE`, and
# `SELECT_CHANNEL_NAME`, then rerun the cell. Use only sort variables returned
# by `score_sort_variables(score_df, SELECT_DATASET_KEY)`.

# %%
SELECT_DATASET_KEY = INITIAL_DATASET_KEY
SELECT_SORT_VARIABLE = INITIAL_SORT_VARIABLE
SELECT_CHANNEL_NAME = INITIAL_CHANNEL_NAME

manual_detail_fig = plot_dataset_parent_image(score_df, SELECT_DATASET_KEY, SELECT_SORT_VARIABLE, SELECT_CHANNEL_NAME)
display(manual_detail_fig)

# %% [markdown]
# ## Browser-Based Clickable Explorer
#
# VS Code often shows Makie output in a static plot preview. This cell avoids
# that renderer: it starts a local Bonito/WGLMakie server and opens the explorer
# in your browser. Keep the Julia kernel running while using the page.
#
# The browser app uses full-resolution ERP detail images and recalculates the
# image-specific colorbar for every selected channel.

# %%
browser_explorer = start_browser_dataset_explorer(
    score_df,
    augmentation_score_df;
    dataset_keys = score_dataset_keys(score_df),
    initial_dataset_key = INITIAL_DATASET_KEY,
    initial_sort_variable = INITIAL_SORT_VARIABLE,
    initial_channel = INITIAL_CHANNEL_NAME,
    use_original_erp_images = true,
    port = 9384,
    open_browser = true,
)

browser_explorer.message

# %% [markdown]
# ## Optional Inline WGLMakie Explorer
#
# This may work in browser-backed Jupyter/IJulia, but the browser-server cell
# above is the more reliable option for this setup.

# %%
interactive_fig = interactive_dataset_explorer(
    score_df,
    augmentation_score_df;
    dataset_keys = score_dataset_keys(score_df),
    initial_dataset_key = INITIAL_DATASET_KEY,
    initial_sort_variable = INITIAL_SORT_VARIABLE,
    initial_channel = INITIAL_CHANNEL_NAME,
    use_original_erp_images = true,
)

interactive_fig
