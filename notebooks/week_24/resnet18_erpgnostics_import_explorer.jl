# %% [markdown]
# # ResNet18 ERPgnostics explorer from saved model
#
# This notebook does not train. It loads the exported ResNet18 artifact from
# `resnet18_erpgnostics_train_export.ipynb`, scores the fixation-reference
# parent ERP images, and builds ERPgnostics-style visualizations.

# %%
include(joinpath(@__DIR__, "resnet18_erpgnostics_common.jl"))

DEFAULT_TRAIN_EXPORT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_train_export")
DEFAULT_MODEL_ARTIFACT = model_artifact_path(DEFAULT_TRAIN_EXPORT_DIR)
MODEL_ARTIFACT_PATH = get(ENV, "WEEK24_MODEL_ARTIFACT", DEFAULT_MODEL_ARTIFACT)

EXPLORER_OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_import_explorer")
mkpath(EXPLORER_OUTPUT_DIR)

println("Model artifact path: ", MODEL_ARTIFACT_PATH)
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
    ],
)

system_diagnostics

# %% [markdown]
# ## Load Model

# %%
device, use_cuda = Generalization.setup_device()
loaded_model = load_resnet18_model_artifact(MODEL_ARTIFACT_PATH; device = device)
model = loaded_model.model

loaded_model.metadata

# %% [markdown]
# ## Score Reference ERP Parents
#
# The score shown in the topoplots is the parent-image probability: the mean
# `prob_class` over the four sort/polarity augmentations for that channel and
# sort variable.

# %%
labels_df = load_all_real_labels()
reference_sort_variables, reference_sort_summary = choose_reference_sort_variables(labels_df)
reference_sort_summary

# %%
reference_scores = score_reference_parent_images(
    model,
    device;
    labels_df = labels_df,
    sort_variables = reference_sort_variables,
)

score_df = reference_scores.score_df
augmentation_score_df = reference_scores.augmentation_df

CSV.write(joinpath(EXPLORER_OUTPUT_DIR, "reference_parent_scores.csv"), score_df)
CSV.write(joinpath(EXPLORER_OUTPUT_DIR, "reference_augmentation_scores.csv"), select(augmentation_score_df, Not(:processed_img)))

first(score_df, 12)

# %%
score_diagnostics = combine(
    groupby(score_df, :sort_variable),
    :score_class => minimum => :min_score,
    :score_class => maximum => :max_score,
    :score_class => mean => :mean_score,
    :score_class => std => :std_score,
    nrow => :n_channels,
)

if all(coalesce.(score_diagnostics.std_score, 0.0) .< 1e-4)
    @warn "All reference scores are nearly constant. This usually means the model artifact is stale/incomplete or the final model underfit. Rerun the training-export notebook and make sure it writes resnet18_final_state.jld2."
end

score_diagnostics

# %% [markdown]
# ## Visible Static ERPgnostics Overview
#
# This cell always uses CairoMakie, so it should display reliably in VS Code and
# Jupyter. The following interactive cell is optional.

# %%
CairoMakie.activate!(type = "svg")

topoplot_fig = plot_reference_score_topoplots(score_df; sort_variables = reference_sort_variables)
display(topoplot_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "reference_parent_score_topoplots.svg"), topoplot_fig)

detail_fig = plot_reference_parent_image(score_df, first(reference_sort_variables), REFERENCE_INITIAL_CHANNEL)
display(detail_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "reference_detail_$(first(reference_sort_variables))_$(REFERENCE_INITIAL_CHANNEL).svg"), detail_fig)

augmented_inputs_fig = plot_augmented_model_inputs(augmentation_score_df, first(reference_sort_variables), REFERENCE_INITIAL_CHANNEL)
display(augmented_inputs_fig)
CairoMakie.save(joinpath(EXPLORER_OUTPUT_DIR, "reference_model_inputs_$(first(reference_sort_variables))_$(REFERENCE_INITIAL_CHANNEL).svg"), augmented_inputs_fig)

println("Saved static explorer exports in $(EXPLORER_OUTPUT_DIR).")

# %% [markdown]
# ## Manual Detail Selector
#
# If WGLMakie is not rendered by the notebook frontend, use this small selector:
# change `SELECT_SORT_VARIABLE` and `SELECT_CHANNEL_NAME`, then rerun the cell.

# %%
SELECT_SORT_VARIABLE = first(reference_sort_variables)
SELECT_CHANNEL_NAME = REFERENCE_INITIAL_CHANNEL

manual_detail_fig = plot_reference_parent_image(score_df, SELECT_SORT_VARIABLE, SELECT_CHANNEL_NAME)
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
browser_explorer = start_browser_reference_explorer(
    score_df,
    augmentation_score_df;
    sort_variables = reference_sort_variables,
    initial_sort_variable = first(reference_sort_variables),
    initial_channel = REFERENCE_INITIAL_CHANNEL,
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
interactive_fig = interactive_reference_explorer(
    score_df,
    augmentation_score_df;
    sort_variables = reference_sort_variables,
    initial_sort_variable = first(reference_sort_variables),
    initial_channel = REFERENCE_INITIAL_CHANNEL,
    use_original_erp_images = true,
)

interactive_fig
