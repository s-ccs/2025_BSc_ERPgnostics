module ERPDataIO

using DataFrames
using JLD2

export ERP_DATA_DIR
export list_datasets
export list_sort_variables
export list_channels
export list_labeled_channels
export list_labeled_triples
export list_all_labeled_triples
export load_dataset_metadata
export trial_sort_order
export sort_trials_by_order
export load_erp

const ERP_DATA_DIR = joinpath(@__DIR__, "..", "datasets") |> normpath

const SORT_TIEBREAKER_COLUMNS = [
    :source_part_index,
    :source_epoch_index,
    :subject_label,
    :epoch_index,
    :sample_index,
    :event_rank_within_type,
    :flash_index_within_run,
    :flash_index_within_trial,
    :onset_s,
    :stimulus_onset_s,
]

const LOCAL_WITHIN_SOURCE_SORT_COLUMNS = Set([
    :sample_index,
    :source_epoch_index,
    :trial_block_index,
    :flash_index_within_run,
    :flash_index_within_trial,
])

const SOURCE_PART_SORT_COLUMN_CANDIDATES = [
    :source_part_index,
    :run_label,
    :source_file,
]

function dataset_dir(dataset_key::AbstractString)
    return joinpath(ERP_DATA_DIR, String(dataset_key))
end

function events_path(dataset_key::AbstractString)
    return joinpath(dataset_dir(dataset_key), "events.jld2")
end

function labels_path(dataset_key::AbstractString)
    return joinpath(dataset_dir(dataset_key), "labels.jld2")
end

function signal_path(dataset_key::AbstractString, channel_name::AbstractString)
    return joinpath(dataset_dir(dataset_key), "signals", string(channel_name, ".jld2"))
end

function require_file(path::AbstractString)
    isfile(path) || error("Required ERP dataset file not found: $(path).")
    return path
end

function list_datasets()
    isdir(ERP_DATA_DIR) || return String[]
    keys = String[]
    for name in readdir(ERP_DATA_DIR)
        dir = joinpath(ERP_DATA_DIR, name)
        isdir(dir) && isfile(joinpath(dir, "events.jld2")) && push!(keys, String(name))
    end
    sort!(keys)
    return keys
end

function read_labels(dataset_key::AbstractString)
    return JLD2.load(require_file(labels_path(dataset_key)), "labels")
end

function list_sort_variables(dataset_key::AbstractString)
    labels = read_labels(dataset_key)
    values = unique(String.(labels.sort_variable))
    sort!(values)
    return values
end

function list_channels(dataset_key::AbstractString)
    labels = read_labels(dataset_key)
    values = unique(String.(labels.channel_name))
    sort!(values)
    return values
end

function list_labeled_channels(dataset_key::AbstractString, sort_variable::AbstractString)
    labels = read_labels(dataset_key)
    mask = String.(labels.sort_variable) .== String(sort_variable)
    values = unique(String.(labels.channel_name[mask]))
    sort!(values)
    return values
end

function list_labeled_triples(dataset_key::AbstractString)
    labels = read_labels(dataset_key)
    out = select(labels, [:channel_name, :sort_variable, :erp_class])
    out.channel_name = String.(out.channel_name)
    out.sort_variable = String.(out.sort_variable)
    out.erp_class = String.(out.erp_class)
    return out
end

function list_all_labeled_triples()
    frames = DataFrame[]
    for dataset_key in list_datasets()
        labels = list_labeled_triples(dataset_key)
        insertcols!(labels, 1, :dataset_key => fill(String(dataset_key), nrow(labels)))
        push!(frames, labels)
    end
    isempty(frames) && return DataFrame(
        dataset_key = String[],
        channel_name = String[],
        sort_variable = String[],
        erp_class = String[],
    )
    return vcat(frames...; cols = :union)
end

function load_dataset_metadata(dataset_key::AbstractString)
    return JLD2.load(require_file(events_path(dataset_key)), "metadata")
end

function find_labeled_row(labels::DataFrame, channel_name::AbstractString, sort_variable::AbstractString)
    matches = findall(
        (String.(labels.channel_name) .== String(channel_name)) .&
        (String.(labels.sort_variable) .== String(sort_variable))
    )
    isempty(matches) && error("Combination not labeled — not supported: channel=$(channel_name), sort_variable=$(sort_variable).")
    length(matches) == 1 || error("Duplicate labelled combination found for channel=$(channel_name), sort_variable=$(sort_variable).")
    return labels[matches[1], :]
end

function unique_nonmissing_count(values)
    return length(unique(collect(skipmissing(values))))
end

function source_part_sort_column(df::DataFrame)
    for col in SOURCE_PART_SORT_COLUMN_CANDIDATES
        col in propertynames(df) || continue
        unique_nonmissing_count(df[!, col]) > 1 || continue
        return col
    end
    return nothing
end

function effective_sort_columns(df::DataFrame, sort_col::Symbol)
    sort_cols = Symbol[sort_col]
    if sort_col in LOCAL_WITHIN_SOURCE_SORT_COLUMNS
        subject_col = :subject_label in propertynames(df) && unique_nonmissing_count(df[!, :subject_label]) > 1 ?
            :subject_label : nothing
        source_col = source_part_sort_column(df)
        prefix_cols = Symbol[]
        if subject_col !== nothing && subject_col != sort_col
            push!(prefix_cols, subject_col)
        end
        if source_col !== nothing && source_col != sort_col && !(source_col in prefix_cols)
            push!(prefix_cols, source_col)
        end
        if !isempty(prefix_cols)
            sort_cols = vcat(prefix_cols, [sort_col])
        end
    end
    for col in SORT_TIEBREAKER_COLUMNS
        col == sort_col && continue
        col in propertynames(df) || continue
        col in sort_cols && continue
        push!(sort_cols, col)
    end
    return sort_cols
end

function fallback_trial_sort_order(df::DataFrame, sort_col::Symbol)
    sort_col in propertynames(df) || error("Sort variable $(sort_col) is not present in events.")
    row_col = :__row_idx__
    sort_cols = effective_sort_columns(df, sort_col)
    order_df = DataFrame()
    order_df[!, row_col] = collect(1:nrow(df))
    for col in sort_cols
        order_df[!, col] = copy(df[!, col])
    end
    sort!(order_df, vcat(sort_cols, [row_col]))
    return Int.(order_df[!, row_col])
end

function trial_sort_order(df::DataFrame, sort_col::Symbol)
    return fallback_trial_sort_order(df, sort_col)
end

function sort_trials_by_order(data_time_trials::AbstractMatrix, trial_order)
    order = Int.(collect(trial_order))
    length(order) == size(data_time_trials, 2) ||
        error("Trial order length $(length(order)) does not match trial count $(size(data_time_trials, 2)).")
    return Matrix{Float32}(data_time_trials[:, order])
end

function load_erp(dataset_key::AbstractString, channel_name::AbstractString, sort_variable::AbstractString)
    labels = read_labels(dataset_key)
    labeled_row = find_labeled_row(labels, channel_name, sort_variable)
    events_file = require_file(events_path(dataset_key))
    signal_file = require_file(signal_path(dataset_key, channel_name))

    events = JLD2.load(events_file, "events")
    dataset_metadata = JLD2.load(events_file, "metadata")
    data_time_trials = Matrix{Float32}(JLD2.load(signal_file, "data_time_trials"))
    signal_metadata = JLD2.load(signal_file, "metadata")

    size(data_time_trials, 2) == nrow(events) ||
        error("Signal trial count does not match events for $(dataset_key), channel $(channel_name).")

    sort_col = Symbol(sort_variable)
    sort_col in propertynames(events) || error("Sort variable $(sort_variable) is not present in events.")
    order = trial_sort_order(events, sort_col)
    data_sorted = sort_trials_by_order(data_time_trials, order)
    sort_values = collect(events[!, sort_col])[order]
    merged_metadata = Dict{String, Any}(dataset_metadata)
    for (key, value) in signal_metadata
        merged_metadata[String(key)] = value
    end

    return (
        data_time_trials = data_sorted,
        sort_values = sort_values,
        trial_order = Int.(order),
        erp_class = String(labeled_row.erp_class),
        events = events[order, :],
        metadata = merged_metadata,
    )
end

end

using .ERPDataIO: ERP_DATA_DIR
using .ERPDataIO: list_datasets
using .ERPDataIO: list_sort_variables
using .ERPDataIO: list_channels
using .ERPDataIO: list_labeled_channels
using .ERPDataIO: list_labeled_triples
using .ERPDataIO: list_all_labeled_triples
using .ERPDataIO: load_dataset_metadata
using .ERPDataIO: trial_sort_order
using .ERPDataIO: sort_trials_by_order
using .ERPDataIO: load_erp
