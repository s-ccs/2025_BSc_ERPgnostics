# data_loading.jl
#
# Dataset discovery and loading for the post-submit pipeline. Reads the built
# JLD2 datasets under `datasets/<dataset>/`:
#   * events.jld2   -> per-trial event table + metadata
#   * labels.jld2   -> manual ERP-pattern labels (channel x sort_variable)
#   * signals/<channel>.jld2 -> time x trials signal matrix + metadata
#
# Signals and events are cached so repeated channel/sort combinations do not
# reload the same matrices.

dataset_dir(dataset_key::AbstractString) = joinpath(DATASETS_ROOT, dataset_key)
events_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "events.jld2")
labels_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "labels.jld2")
signals_dir(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "signals")
signal_path(dataset_key::AbstractString, channel_name::AbstractString) =
    joinpath(signals_dir(dataset_key), string(channel_name, ".jld2"))

"""
    discover_real_dataset_keys() -> Vector{String}

Find every usable dataset under `datasets/`.

# Returns
- `Vector{String}`: sorted dataset keys that have all three required pieces
  (`events.jld2`, `labels.jld2`, `signals/`). `simulated` is intentionally
  excluded.

# Behavior
Throws an error if no usable dataset is found.
"""
function discover_real_dataset_keys()
    keys = String[]
    for dataset_key in sort(readdir(DATASETS_ROOT))
        dataset_key == "simulated" && continue
        dir = dataset_dir(dataset_key)
        if isdir(dir) && isfile(events_path(dataset_key)) && isfile(labels_path(dataset_key)) && isdir(signals_dir(dataset_key))
            push!(keys, dataset_key)
        end
    end
    isempty(keys) && error("No real built datasets found in $(DATASETS_ROOT).")
    return keys
end

"Parse the trailing integer of a `chNNN` channel name, or `0` if it does not match."
function channel_index_from_name(channel_name::AbstractString)
    m = match(r"^ch(\d+)$", String(channel_name))
    m === nothing && return 0
    return parse(Int, m.captures[1])
end

# --------------------------------------------------------------------------- #
# Caching context
# --------------------------------------------------------------------------- #
"Load `events.jld2` for `dataset_key`, returning `(events, metadata)`."
function load_events_file(dataset_key::AbstractString)
    path = events_path(dataset_key)
    isfile(path) || error("Missing events file: $(path)")
    return (
        events = JLD2.load(path, "events"),
        metadata = JLD2.load(path, "metadata"),
    )
end

"Load one channel signal, returning `(data_time_trials, metadata, channel_idx)`."
function load_signal_file(dataset_key::AbstractString, channel_name::AbstractString)
    path = signal_path(dataset_key, channel_name)
    isfile(path) || error("Missing signal file: $(path)")
    data = Matrix{Float32}(JLD2.load(path, "data_time_trials"))
    metadata = JLD2.load(path, "metadata")
    return (
        data_time_trials = data,
        metadata = metadata,
        channel_idx = Int(get(metadata, "channel_idx", channel_index_from_name(channel_name))),
    )
end

"""
    build_data_context()

Create an empty cache holding one events table per dataset and one signal matrix
per (dataset, channel), so repeated lookups do not reload the same files.
"""
build_data_context() = (
    events_cache = Dict{String, Any}(),
    signal_cache = Dict{Tuple{String, String}, Any}(),
)

"Events table for `dataset_key`, loaded on first use and cached in `ctx`."
events_for_dataset(ctx, dataset_key::AbstractString) =
    get!(() -> load_events_file(dataset_key), ctx.events_cache, String(dataset_key))

"Signal bundle for `(dataset_key, channel_name)`, loaded on first use and cached in `ctx`."
function signal_for_channel(ctx, dataset_key::AbstractString, channel_name::AbstractString)
    key = (String(dataset_key), String(channel_name))
    return get!(() -> load_signal_file(dataset_key, channel_name), ctx.signal_cache, key)
end

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
"""
    load_dataset_labels(dataset_key) -> DataFrame

Read the manual ERP-pattern labels for one dataset.

# Arguments
- `dataset_key::AbstractString`: dataset folder name under `datasets/`.

# Returns
- `DataFrame`: one row per valid label, with `dataset_key`, `dataset_label`,
  `channel_name`, `channel_idx`, `sort_variable`, `erp_class`, `erp_class_id`,
  and `binary_label` (0 for `no_class`, 1 otherwise). A label is kept only when
  its class is known, its sort column exists in the events, and the channel
  signal file is present.
"""
function load_dataset_labels(dataset_key::AbstractString)
    raw = JLD2.load(labels_path(dataset_key), "labels")
    events_bundle = load_events_file(dataset_key)
    dataset_label = String(get(events_bundle.metadata, "dataset_label", dataset_key))

    rows = NamedTuple[]
    for row in eachrow(raw)
        channel_name = cellstr(row.channel_name)
        sort_variable = cellstr(row.sort_variable)
        erp_class = cellstr(row.erp_class)
        # Keep a label only if its class is known, its sort column exists in the
        # events, and the channel's signal file is on disk.
        haskey(CLASS_ID, erp_class) || continue
        Symbol(sort_variable) in propertynames(events_bundle.events) || continue
        isfile(signal_path(dataset_key, channel_name)) || continue

        sig_meta = JLD2.load(signal_path(dataset_key, channel_name), "metadata")
        push!(rows, (
            dataset_key = String(dataset_key),
            dataset_label = dataset_label,
            channel_name = channel_name,
            channel_idx = Int(get(sig_meta, "channel_idx", channel_index_from_name(channel_name))),
            sort_variable = sort_variable,
            erp_class = erp_class,
            erp_class_id = CLASS_ID[erp_class],
            binary_label = erp_class == "no_class" ? 0 : 1,
        ))
    end
    return DataFrame(rows)
end

"""
    load_all_real_labels() -> DataFrame

Concatenate the valid labels of every dataset into the project-wide label pool.

# Returns
- `DataFrame`: all rows from [`load_dataset_labels`](@ref) across every dataset,
  sorted by `(dataset_key, sort_variable, channel_name)` and tagged with a stable
  1-based `source_row_id`.

# Behavior
Throws an error if no labeled rows are found anywhere.
"""
function load_all_real_labels()
    parts = DataFrame[]
    for dataset_key in discover_real_dataset_keys()
        labels = load_dataset_labels(dataset_key)
        isempty(labels) || push!(parts, labels)
    end
    labels = isempty(parts) ? DataFrame() : vcat(parts...; cols = :union)
    isempty(labels) && error("No labeled real ERP rows were found in $(DATASETS_ROOT).")
    sort!(labels, [:dataset_key, :sort_variable, :channel_name])
    labels.source_row_id = collect(1:nrow(labels))
    return labels
end

"A copy of `sample_df` without the heavy `processed_img` column (for writing/printing)."
labels_without_images(sample_df::DataFrame) = select(sample_df, Not(:processed_img))

# --------------------------------------------------------------------------- #
# Channels and sort-variable maps
# --------------------------------------------------------------------------- #
"""
    dataset_channels(dataset_key) -> DataFrame

List the channels available on disk for one dataset.

# Returns
- `DataFrame` with `dataset_key`, `channel_name`, `channel_idx`, one row per
  signal file, sorted by `(channel_idx, channel_name)`.
"""
function dataset_channels(dataset_key::AbstractString)
    files = filter(path -> endswith(path, ".jld2"), readdir(signals_dir(dataset_key); join = false))
    rows = NamedTuple[]
    for file in files
        channel_name = replace(file, ".jld2" => "")
        metadata = JLD2.load(signal_path(dataset_key, channel_name), "metadata")
        push!(rows, (
            dataset_key = String(dataset_key),
            channel_name = channel_name,
            channel_idx = Int(get(metadata, "channel_idx", channel_index_from_name(channel_name))),
        ))
    end
    channels = DataFrame(rows)
    sort!(channels, [:channel_idx, :channel_name])
    return channels
end

"Channel names on disk for `dataset_key` (the `channel_name` column of [`dataset_channels`](@ref))."
dataset_channel_names(dataset_key::AbstractString) = String.(dataset_channels(dataset_key).channel_name)

"""
    dataset_sort_variable_summary(labels_df) -> DataFrame

Count manual labels per (dataset, sort_variable).

# Arguments
- `labels_df::DataFrame`: a label pool such as [`load_all_real_labels`](@ref).

# Returns
- `DataFrame` with `dataset_key`, `dataset_label`, `sort_variable`, `n_pattern`
  (count of positive/pattern labels) and `n_labels` (total labels). Empty input
  yields an empty frame with the same columns.
"""
function dataset_sort_variable_summary(labels_df::DataFrame)
    isempty(labels_df) && return DataFrame(
        dataset_key = String[], dataset_label = String[], sort_variable = String[],
        n_pattern = Int[], n_labels = Int[],
    )
    summary = combine(
        groupby(labels_df, [:dataset_key, :dataset_label, :sort_variable]),
        :binary_label => sum => :n_pattern,
        nrow => :n_labels,
    )
    sort!(summary, [:dataset_key, :n_pattern, :sort_variable], rev = [false, true, false])
    return summary
end

"""
    sort_variables_for_dataset(labels_df, dataset_key; require_pattern=true) -> Vector{String}

Sort variables that carry manual labels for one dataset.

# Arguments
- `labels_df::DataFrame`: the label pool.
- `dataset_key::AbstractString`: dataset to filter on.
- `require_pattern::Bool=true`: if true, keep only sort variables with at least
  one positive pattern label.

# Returns
- `Vector{String}`: matching sort variables, most-pattern-rich first.
"""
function sort_variables_for_dataset(labels_df::DataFrame, dataset_key::AbstractString; require_pattern::Bool = true)
    summary = dataset_sort_variable_summary(labels_df)
    sub = summary[summary.dataset_key .== String(dataset_key), :]
    require_pattern && (sub = sub[Int.(sub.n_pattern) .> 0, :])
    sort!(sub, [:n_pattern, :sort_variable], rev = [true, false])
    return String.(sub.sort_variable)
end

"""
    configured_extra_sort_variables(dataset_key) -> Vector{String}

Sort variables configured in [`EXTRA_SORT_VARIABLES_BY_DATASET`](@ref) that should
be scored for `dataset_key` even without a manual label. Errors if a configured
name is not an actual event column.
"""
function configured_extra_sort_variables(dataset_key::AbstractString)
    extras = get(EXTRA_SORT_VARIABLES_BY_DATASET, String(dataset_key), String[])
    isempty(extras) && return String[]

    # Fail loudly if a configured extra sort variable is not an actual event column.
    events = load_events_file(dataset_key).events
    missing = [sort_variable for sort_variable in extras if !(Symbol(sort_variable) in propertynames(events))]
    isempty(missing) || error(
        "Configured sort variables missing from $(dataset_key): $(join(missing, ", "))."
    )
    return String.(extras)
end

"""
    dataset_sort_variable_map(labels_df; require_pattern=true) -> Dict{String, Vector{String}}

Build the scoring universe's sort-variable axis: each dataset mapped to the sort
variables that should be scored for it (labeled ones per `require_pattern`, plus
any configured extras). Datasets with no sort variables are omitted.
"""
function dataset_sort_variable_map(labels_df::DataFrame; require_pattern::Bool = true)
    out = Dict{String, Vector{String}}()
    for dataset_key in discover_real_dataset_keys()
        sort_variables = sort_variables_for_dataset(labels_df, dataset_key; require_pattern = require_pattern)
        append!(sort_variables, configured_extra_sort_variables(dataset_key))
        sort_variables = unique(sort_variables)
        isempty(sort_variables) || (out[String(dataset_key)] = sort_variables)
    end
    return out
end

"""
    combined_label_lookup(labels_df) -> NamedTuple

Index the manual ground truth by `(dataset, sort_variable, channel)`.

# Returns
A `NamedTuple` of four `Dict`s keyed by the `(dataset, sort_variable, channel)`
tuple: `erp_class` (the pattern, or `"no_class"`), `binary_label` (0/1),
`n_manual_labels`, and `n_manual_pattern_labels`. Used to annotate scored rows
with the manual label when one exists.
"""
function combined_label_lookup(labels_df::DataFrame)
    erp_lookup = Dict{Tuple{String, String, String}, String}()
    binary_lookup = Dict{Tuple{String, String, String}, Int}()
    count_lookup = Dict{Tuple{String, String, String}, Int}()
    pattern_count_lookup = Dict{Tuple{String, String, String}, Int}()

    isempty(labels_df) && return (
        erp_class = erp_lookup, binary_label = binary_lookup,
        n_manual_labels = count_lookup, n_manual_pattern_labels = pattern_count_lookup,
    )

    for group in groupby(labels_df, [:dataset_key, :sort_variable, :channel_name])
        first_row = group[1, :]
        key = (cellstr(first_row.dataset_key), cellstr(first_row.sort_variable), cellstr(first_row.channel_name))
        positive_idxs = findall(Int.(group.binary_label) .== 1)
        count_lookup[key] = nrow(group)
        pattern_count_lookup[key] = length(positive_idxs)
        if isempty(positive_idxs)
            erp_lookup[key] = "no_class"      # only no_class labels -> negative combo
            binary_lookup[key] = 0
        else
            erp_lookup[key] = cellstr(group.erp_class[first(positive_idxs)])  # first pattern wins
            binary_lookup[key] = 1
        end
    end

    return (
        erp_class = erp_lookup, binary_label = binary_lookup,
        n_manual_labels = count_lookup, n_manual_pattern_labels = pattern_count_lookup,
    )
end
