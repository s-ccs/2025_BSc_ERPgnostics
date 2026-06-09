using DataFrames
using JLD2

const EXPECTED_LABEL_COLUMNS = (
    channel_name = String[],
    sort_variable = String[],
    erp_class = String[],
)

"""
    default_data_root()

Return the default repository data directory used by the ERP source helpers.

# Arguments
- None.

# Returns
- `String`: Normalized path to `datasets/` at the repository root.
"""
function default_data_root()
    return normpath(joinpath(@__DIR__, "..", "datasets"))
end

"""
    dataset_path(dataset_key; data_root=default_data_root())

Build the path to one dataset folder.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: Path to `data_root/dataset_key`.
"""
function dataset_path(dataset_key; data_root = default_data_root())
    return joinpath(data_root, String(dataset_key))
end

"""
    events_path(dataset_key; data_root=default_data_root())

Build the path to the dataset event table file.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: Path to `events.jld2` for the dataset.
"""
function events_path(dataset_key; data_root = default_data_root())
    return joinpath(dataset_path(dataset_key; data_root = data_root), "events.jld2")
end

"""
    labels_path(dataset_key; data_root=default_data_root())

Build the path to the optional dataset label table file.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: Path to `labels.jld2` for the dataset.
"""
function labels_path(dataset_key; data_root = default_data_root())
    return joinpath(dataset_path(dataset_key; data_root = data_root), "labels.jld2")
end

"""
    signals_path(dataset_key; data_root=default_data_root())

Build the path to the folder containing per-channel signal files.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: Path to the dataset `signals/` folder.
"""
function signals_path(dataset_key; data_root = default_data_root())
    return joinpath(dataset_path(dataset_key; data_root = data_root), "signals")
end

"""
    signal_path(dataset_key, channel_name; data_root=default_data_root())

Build the path to one channel signal file.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel file name without `.jld2`.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: Path to the channel `.jld2` file.
"""
function signal_path(dataset_key, channel_name; data_root = default_data_root())
    return joinpath(signals_path(dataset_key; data_root = data_root), string(channel_name, ".jld2"))
end

"""
    require_file(path)

Validate that a required file exists before loading it.

# Arguments
- `path`: File path to check.

# Returns
- The original `path` if it exists.

# Throws
- `ErrorException`: If the file does not exist.
"""
function require_file(path)
    isfile(path) || error("Required file not found: $(path)")
    return path
end

"""
    require_directory(path)

Validate that a required directory exists before listing it.

# Arguments
- `path`: Directory path to check.

# Returns
- The original `path` if it exists.

# Throws
- `ErrorException`: If the directory does not exist.
"""
function require_directory(path)
    isdir(path) || error("Required directory not found: $(path)")
    return path
end

"""
    empty_labels()

Create the standard empty label table used when `labels.jld2` is missing.

# Arguments
- None.

# Returns
- `DataFrame`: Empty table with `channel_name`, `sort_variable`, and `erp_class`.
"""
function empty_labels()
    return DataFrame(EXPECTED_LABEL_COLUMNS)
end

"""
    cellstring(value)

Convert table cell values to strings for label matching.

# Arguments
- `value`: Any scalar value from a DataFrame cell.

# Returns
- `String`: Empty string for `missing` or `nothing`, otherwise `String(value)`.
"""
function cellstring(value)
    (ismissing(value) || value === nothing) && return ""
    return String(value)
end

"""
    list_datasets(data_root=default_data_root())

List dataset folders that contain an event file and a signal directory.

# Arguments
- `data_root`: Root folder containing dataset folders.

# Returns
- `Vector{String}`: Sorted dataset keys.
"""
function list_datasets(data_root = default_data_root())
    isdir(data_root) || return String[]

    dataset_keys = String[]
    for name in readdir(data_root)
        path = joinpath(data_root, name)
        if isdir(path) && isfile(joinpath(path, "events.jld2")) && isdir(joinpath(path, "signals"))
            push!(dataset_keys, String(name))
        end
    end
    sort!(dataset_keys)
    return dataset_keys
end

"""
    load_events(dataset_key; data_root=default_data_root())

Load the event table and dataset metadata from `events.jld2`.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `NamedTuple`: `(events, metadata)` where `events` is a `DataFrame`.
"""
function load_events(dataset_key; data_root = default_data_root())
    path = require_file(events_path(dataset_key; data_root = data_root))
    return (
        events = DataFrame(JLD2.load(path, "events")),
        metadata = JLD2.load(path, "metadata"),
    )
end

"""
    load_labels(dataset_key; data_root=default_data_root())

Load label rows from `labels.jld2`, or return the standard empty label table.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `DataFrame`: Label table with `channel_name`, `sort_variable`, and `erp_class`.
"""
function load_labels(dataset_key; data_root = default_data_root())
    path = labels_path(dataset_key; data_root = data_root)
    isfile(path) || return empty_labels()

    labels = DataFrame(JLD2.load(path, "labels"))
    for column in propertynames(empty_labels())
        column in propertynames(labels) || error("labels.jld2 is missing required column $(column).")
    end
    return labels
end

"""
    list_channels(dataset_key; data_root=default_data_root())

List channel names available in a dataset's `signals/` folder.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `Vector{String}`: Sorted channel names without `.jld2`.
"""
function list_channels(dataset_key; data_root = default_data_root())
    path = require_directory(signals_path(dataset_key; data_root = data_root))
    channels = String[]
    for name in readdir(path)
        endswith(name, ".jld2") || continue
        push!(channels, first(splitext(name)))
    end
    sort!(channels)
    return channels
end

"""
    load_signal(dataset_key, channel_name; data_root=default_data_root())

Load one channel signal matrix and its metadata.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel file name without `.jld2`.
- `data_root`: Root folder containing dataset folders.

# Returns
- `NamedTuple`: `(data_time_trials, metadata)` where the matrix is `Float32`
  in `timepoints x trials` layout.
"""
function load_signal(dataset_key, channel_name; data_root = default_data_root())
    path = require_file(signal_path(dataset_key, channel_name; data_root = data_root))
    data_time_trials = Matrix{Float32}(JLD2.load(path, "data_time_trials"))
    return (
        data_time_trials = data_time_trials,
        metadata = JLD2.load(path, "metadata"),
    )
end

"""
    list_sort_variables(dataset_key; data_root=default_data_root())

List event columns that can be used as sort variables.

# Arguments
- `dataset_key`: Dataset folder name.
- `data_root`: Root folder containing dataset folders.

# Returns
- `Vector{String}`: Event column names.
"""
function list_sort_variables(dataset_key; data_root = default_data_root())
    events = load_events(dataset_key; data_root = data_root).events
    return String.(propertynames(events))
end

"""
    labels_for(dataset_key, channel_name, sort_variable; data_root=default_data_root())

Return label rows for one dataset, channel, and sort variable combination.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel name used by the signal file.
- `sort_variable`: Event column used for trial sorting.
- `data_root`: Root folder containing dataset folders.

# Returns
- `DataFrame`: Matching label rows, or an empty DataFrame if the combination is
  unlabeled or no label file exists.
"""
function labels_for(dataset_key, channel_name, sort_variable; data_root = default_data_root())
    labels = load_labels(dataset_key; data_root = data_root)
    isempty(labels) && return labels

    channel_matches = [cellstring(value) == String(channel_name) for value in labels[!, :channel_name]]
    sort_matches = [cellstring(value) == String(sort_variable) for value in labels[!, :sort_variable]]
    return labels[channel_matches .& sort_matches, :]
end
