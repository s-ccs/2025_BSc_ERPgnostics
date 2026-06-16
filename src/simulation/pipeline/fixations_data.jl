# =============================================================================
# Real fixation data
#
# Loads the labelled `fixations_dataset` bundle, keeps the `sigmoid` and
# `no_class` channels, and turns each into a 64x64 image with the shared
# pipeline. These images form the real validation set that scores every
# simulator candidate.
# =============================================================================

"""
    RealValidationData

Bundle of the real validation set and the dataset dimensions the simulator must
match.

# Fields
- `events::DataFrame`: the per-trial event table of the fixation recording.
- `eval_df::DataFrame`: one row per labelled channel, including `processed_img`.
- `tensor::Array{Float32,4}`: validation images as a `h x w x 1 x n` tensor.
- `labels::Vector{Int}`: binary labels (1 = sigmoid, 0 = no_class).
- `n_trials::Int`: number of trials per channel.
- `n_timepoints::Int`: post-onset timepoints per trial.
- `sampling_rate::Float64`: sampling rate in Hz.
- `epoch_duration_s::Float64`: trial duration in seconds derived from the above.
"""
struct RealValidationData
    events::DataFrame
    eval_df::DataFrame
    tensor::Array{Float32, 4}
    labels::Vector{Int}
    n_trials::Int
    n_timepoints::Int
    sampling_rate::Float64
    epoch_duration_s::Float64
end

"""
    dataset_dir(config::RunConfig) -> String

Return the directory of the fixation dataset bundle.
"""
dataset_dir(config::RunConfig) = joinpath(config.datasets_root, config.fixations_dataset_key)

"""
    signal_path(config::RunConfig, channel_name) -> String

Return the JLD2 path of one channel signal file inside the dataset bundle.

# Arguments
- `config::RunConfig`: locates the dataset.
- `channel_name`: channel file name without extension (e.g. `"ch096"`).

# Returns
- `String`: absolute path to `signals/<channel_name>.jld2`.
"""
signal_path(config::RunConfig, channel_name) = joinpath(dataset_dir(config), "signals", string(channel_name, ".jld2"))

"""
    load_fixations_tables(config::RunConfig)

Load the events table, labels table, and dataset metadata of the fixation
bundle.

# Arguments
- `config::RunConfig`: locates the dataset bundle.

# Returns
- `Tuple{DataFrame,DataFrame,Dict}`: `(events, labels, metadata)`.
"""
function load_fixations_tables(config::RunConfig)
    dir = dataset_dir(config)
    events_path = joinpath(dir, "events.jld2")
    labels_path = joinpath(dir, "labels.jld2")
    isfile(events_path) || error("Missing events file: $(events_path)")
    isfile(labels_path) || error("Missing labels file: $(labels_path)")

    events = JLD2.load(events_path, "events")
    labels = JLD2.load(labels_path, "labels")
    metadata = JLD2.load(events_path, "metadata")
    return events, labels, metadata
end

"""
    load_channel_signal(config, channel_name, cache) -> Matrix{Float32}

Load a channel signal matrix, caching it so a channel used by several labels is
read from disk only once.

# Arguments
- `config::RunConfig`: locates the signal files.
- `channel_name`: channel to load.
- `cache::Dict{String,Matrix{Float32}}`: channel cache, updated in place.

# Returns
- `Matrix{Float32}`: the channel data in timepoints-by-trials layout.
"""
function load_channel_signal(config::RunConfig, channel_name, cache::Dict{String, Matrix{Float32}})
    key = String(channel_name)
    return get!(cache, key) do
        path = signal_path(config, key)
        isfile(path) || error("Missing signal file for $(key): $(path)")
        Matrix{Float32}(JLD2.load(path, "data_time_trials"))
    end
end

"""
    build_real_eval_dataframe(config, events, labels)

Build the real validation table: keep `sigmoid`/`no_class` labels, attach the
binary label, and render each channel into a `config.target_size` image.

# Arguments
- `config::RunConfig`: image pipeline settings and dataset location.
- `events::DataFrame`: per-trial event table.
- `labels::DataFrame`: label table with `channel_name`, `sort_variable`, `erp_class`.

# Returns
- `DataFrame`: one row per labelled channel with a `processed_img` column.
"""
function build_real_eval_dataframe(config::RunConfig, events::DataFrame, labels::DataFrame)
    keep = [String(row.erp_class) in (SIGMOID_CLASS, NO_CLASS) for row in eachrow(labels)]
    selected = copy(labels[keep, :])
    selected.binary_label = Int.(String.(selected.erp_class) .== SIGMOID_CLASS)
    sort!(selected, [:erp_class, :sort_variable, :channel_name])

    rows = NamedTuple[]
    images = Matrix{Float32}[]
    signal_cache = Dict{String, Matrix{Float32}}()

    for (label_index, row) in enumerate(eachrow(selected))
        data_time_trials = load_channel_signal(config, row.channel_name, signal_cache)
        size(data_time_trials, 2) == nrow(events) || error("Trial count mismatch for $(row.channel_name).")
        sort_col = Symbol(String(row.sort_variable))
        img = shared_preprocess(data_time_trials, events, sort_col, config)
        push!(rows, (
            sample_id = length(rows) + 1,
            label_index = label_index,
            dataset_key = config.fixations_dataset_key,
            channel_name = String(row.channel_name),
            sort_variable = String(row.sort_variable),
            erp_class = String(row.erp_class),
            binary_label = Int(row.binary_label),
            n_trials = nrow(events),
            n_timepoints = size(data_time_trials, 1),
        ))
        push!(images, img)
    end

    out = DataFrame(rows)
    out.processed_img = images
    return out
end

"""
    load_real_validation_data(config::RunConfig) -> RealValidationData

Load and preprocess the full real validation set and the dataset dimensions the
simulator must reproduce.

# Arguments
- `config::RunConfig`: dataset location and image pipeline settings.

# Returns
- `RealValidationData`: the validation images, labels, and recording dimensions.
"""
function load_real_validation_data(config::RunConfig)
    events, labels, metadata = load_fixations_tables(config)
    eval_df = build_real_eval_dataframe(config, events, labels)

    tensor = images_to_tensor(eval_df.processed_img)
    y = Int.(eval_df.binary_label)

    n_trials = nrow(events)
    n_timepoints = Int(metadata["n_timepoints_post"])
    sampling_rate = Float64(metadata["sampling_rate_hz"])
    epoch_duration_s = Float64(n_timepoints - 1) / sampling_rate

    # Guard against a validation set that cannot measure two-class separation.
    count(==(1), y) > 0 || error("No sigmoid rows in real validation set.")
    count(==(0), y) > 0 || error("No no_class rows in real validation set.")
    all(size.(eval_df.processed_img) .== Ref(config.target_size)) ||
        error("Some real images do not match the target size $(config.target_size).")

    return RealValidationData(events, eval_df, tensor, y, n_trials, n_timepoints, sampling_rate, epoch_duration_s)
end
