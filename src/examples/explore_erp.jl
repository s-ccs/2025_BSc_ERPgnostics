# %%
import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(joinpath(REPO_ROOT, "scripts"))

include(joinpath(REPO_ROOT, "src", "erp_data.jl"))
include(joinpath(REPO_ROOT, "src", "erp_processing.jl"))
include(joinpath(REPO_ROOT, "src", "erp_augmentation.jl"))
include(joinpath(REPO_ROOT, "src", "erp_plot.jl"))

# %%
datasets = list_datasets()
isempty(datasets) && error("No datasets found.")
println("Available datasets:")
println(join(datasets, "\n"))

# %%
dataset_key = "fixations_dataset" in datasets ? "fixations_dataset" : first(datasets)
channels = list_channels(dataset_key)
sort_variables = list_sort_variables(dataset_key)

channel_name = "ch042" in channels ? "ch042" : first(channels)
preferred_sort_variables = ["duration", "rt_ms", "sac_amplitude"]
preferred_available = [name for name in preferred_sort_variables if name in sort_variables]
sort_variable = isempty(preferred_available) ? first(sort_variables) : first(preferred_available)

println("Selected dataset: ", dataset_key)
println("Selected channel: ", channel_name)
println("Selected sort variable: ", sort_variable)

# %%
events_bundle = load_events(dataset_key)
labels = load_labels(dataset_key)
signal_bundle = load_signal(dataset_key, channel_name)

println("Events rows: ", nrow(events_bundle.events))
println("Labels rows: ", nrow(labels))
println("Signal size: ", size(signal_bundle.data_time_trials))

# %%
trial_order = trial_sort_order(events_bundle.events, sort_variable)
sorted_trials = sort_trials(signal_bundle.data_time_trials, trial_order)
zscored_trials = zscore_timepoints(sorted_trials)
erp_image = trials_time_image(zscored_trials)
smoothed_image = smooth_image(erp_image)
resized_image = resize_image(smoothed_image)

println("ERP image size: ", size(erp_image))
println("Smoothed image size: ", size(smoothed_image))
println("Resized image size: ", size(resized_image))

# %%
fig = plot_erp_image(dataset_key, channel_name, sort_variable)
display(fig)

# %%
target_trials = min(200, size(signal_bundle.data_time_trials, 2))
augmented = prepare_augmented_images(
    dataset_key,
    channel_name,
    sort_variable;
    target_trials = target_trials,
)

println("Augmented images: ", length(augmented.images))
println(first(augmented.metadata, min(8, nrow(augmented.metadata))))
