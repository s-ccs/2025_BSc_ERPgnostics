module Week15TryNewData

using CSV
using CairoMakie
using DataFrames
using HDF5
using Images: imresize
using ImageFiltering: imfilter
using JSON3
using Printf: @sprintf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: gaussian_kernel, zscore_timepoints, clipped_color_stats_quantile_zero_ticks

export ERP_CORE_DATASET_KEYS
export SHORTLIST_PUBLIC_DATASET_KEYS
export ADDITIONAL_PUBLIC_DATASET_KEYS
export NEW_PUBLIC_DATASET_KEYS
export SIGMOID_PUBLIC_DATASET_KEYS
export COMPARISON_DATASET_KEYS
export ERP_CORE_SUBJECT_IDS
export REAL_TARGET_SIZE
export ensure_erp_core_clean_datasets!
export load_clean_dataset_bundle
export external_dataset_summary_df
export dataset_axis_audit_df
export dataset_source_overview_df
export dataset_source_example_df
export available_sort_columns
export available_sort_columns_df
export recommended_preview_specs
export sort_order_audit_df
export build_dataset_sort_preview
export plot_dataset_sort_preview
export plot_all_dataset_sort_previews
export fixation_summary_df
export fixation_sort_order_audit_df
export load_fixation_reference_cache
export plot_fixation_reference_grid

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATASETS_ROOT = joinpath(REPO_ROOT, "notebooks", "datasets")
const PREPARE_SCRIPT = joinpath(REPO_ROOT, "scripts", "prepare_erp_core_clean_datasets.py")
const PREPARE_PYTHON = let venv_python = joinpath(REPO_ROOT, ".venv_8bit", "bin", "python")
    isfile(venv_python) ? venv_python : "python"
end

const ERP_CORE_DATASET_KEYS = [
    "erp_core_p3_clean",
    "erp_core_n170_clean",
    "erp_core_n400_clean",
    "erp_core_n2pc_clean",
    "erp_core_mmn_clean",
    "erp_core_lrp_clean",
]
const SHORTLIST_PUBLIC_DATASET_KEYS = [
    "bi2013a_public",
    "bi2014a_public",
    "bi2014b_public",
    "bi2015a_public",
    "bi2015b_public",
    "cattan2019_vr_public",
    "bnci_008_2014_public",
    "bigp3bci_studya_public",
]
const ADDITIONAL_PUBLIC_DATASET_KEYS = [
    "eye_eeg_reading_fixations",
    "eye_eeg_freeviewing_fixations",
    "eye_eeg_sceneviewing_tobii_fixations",
    "erpbci_public",
]
const NEW_PUBLIC_DATASET_KEYS = [
    "nod_eeg_public",
    # "zuco2_nr_public",  # Large OSF source; use the dedicated source notebook/import gate.
]
const SIGMOID_PUBLIC_DATASET_KEYS = [
    "eye_eeg_reading_fixations",
    "eye_eeg_freeviewing_fixations",
    "eye_eeg_sceneviewing_tobii_fixations",
    "zuco2_nr_public",
    "roamm_reading_fixations",
    "eegeyenet_saccades",
    "saccade_onset_face_vr",
    "erp_core_lrp_clean",
    "nencki_symfonia_srt",
    "openneuro_gonogo_ds002680",
    "confidence_perceptual_decisions",
    "kilo_word_erp",
    "02_new_eegeyenet_saccades",
    "02_new_zuco2_reading_fixations",
    "02_new_raccoons_reading",
    "02_new_roamm_reading",
    "02_new_unfold_facefreeview",
]
const COMPARISON_DATASET_KEYS = vcat(
    ERP_CORE_DATASET_KEYS,
    SHORTLIST_PUBLIC_DATASET_KEYS,
    ADDITIONAL_PUBLIC_DATASET_KEYS,
    NEW_PUBLIC_DATASET_KEYS,
)
const ERP_CORE_SUBJECT_IDS = [1, 2, 3, 4]
const DATASET_COMPONENT_KEYS = Dict(
    "erp_core_p3_clean" => "p3",
    "erp_core_n170_clean" => "n170",
    "erp_core_n400_clean" => "n400",
    "erp_core_n2pc_clean" => "n2pc",
    "erp_core_mmn_clean" => "mmn",
    "erp_core_lrp_clean" => "lrp",
)

const REAL_TARGET_SIZE = (64, 64)
const REAL_PREVIEW_TIME_WINDOW_S = (0f0, 1f0)
# Gaussian smoothing factor passed through notebooks/utils/erp_image_utils.jl.
const LOWPASS_SIGMA_FACTOR = 75f0
const FILTER_BORDER = "reflect"
const REAL_BASELINE_WINDOW_S = (-0.2f0, 0f0)

const FIXATIONS_DATASET_DIR = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", "fixations_dataset")
const FIXATION_H5_PATH = joinpath(FIXATIONS_DATASET_DIR, "data_fixations.hdf5")
const FIXATION_EVENTS_CSV_PATH = joinpath(FIXATIONS_DATASET_DIR, "events.csv")
const FIXATION_PRE_STIM_S = 0.5
const FIXATION_SAMPLING_RATE = 512
const FIXATION_TIME_ZERO_IDX = Int(round(FIXATION_PRE_STIM_S * FIXATION_SAMPLING_RATE)) + 1
const FIXATION_REFERENCE_SORT_COLUMNS = [
    :duration,
    :sac_amplitude,
    :fix_avgpos_x,
    :fix_avgpupilsize,
    :fix_type,
    :latency,
]
const FIXATION_NON_SORT_COLS = Set([:id, :picID, :trialnum, :stim_set, :stim_file])
const FIXATION_REFERENCE_RNG_SEED = 20260319

const SORT_COLUMN_EXCLUDE = Set([
    :dataset_key,
    :component,
    :subject_id,
    :subject_label,
    :response_code,
    :stimulus_onset_s,
    :response_onset_s,
    :source_event_item,
    :source_set_relpath,
    :source_eventlist_relpath,
])
const PREVIEW_SORT_COLUMN_EXCLUDE = Set([
    :sample_index,
    :epoch_index,
    :source_file,
])
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

dataset_dir(dataset_key::AbstractString) = joinpath(DATASETS_ROOT, dataset_key)
h5_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "epochs.hdf5")
events_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "events.csv")
metadata_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "metadata.json")

function ensure_erp_core_clean_datasets!(;
        dataset_keys::Vector{String} = collect(ERP_CORE_DATASET_KEYS),
        subject_ids::Vector{Int} = collect(ERP_CORE_SUBJECT_IDS))
    missing = String[]
    for dataset_key in dataset_keys
        if !(isfile(h5_path(dataset_key)) && isfile(events_path(dataset_key)) && isfile(metadata_path(dataset_key)))
            push!(missing, dataset_key)
        end
    end
    isempty(missing) && return nothing

    @assert isfile(PREPARE_SCRIPT) "Preparation script not found: $PREPARE_SCRIPT"
    components = [DATASET_COMPONENT_KEYS[key] for key in missing]
    cmd = Cmd(vcat(
        [PREPARE_PYTHON, PREPARE_SCRIPT, "--output-root", DATASETS_ROOT, "--components"],
        components,
        ["--subjects"],
        string.(subject_ids),
    ))
    run(cmd)
    return nothing
end

function infer_sampling_rate(times_s::AbstractVector{<:Real})
    length(times_s) < 2 && return 0
    dt = Float64(times_s[2] - times_s[1])
    dt <= 0 && return 0
    return round(Int, 1 / dt)
end

function read_subject_metadata(path::AbstractString, subject_label::AbstractString)
    return h5open(path, "r") do f
        g = f["subjects"][subject_label]
        times_s = Float32.(read(g["times_s"]))
        channel_names = String.(read(g["channel_names"]))
        n_channels = Int(read(HDF5.attributes(g)["n_channels"]))
        n_timepoints = Int(read(HDF5.attributes(g)["n_timepoints"]))
        n_trials = Int(read(HDF5.attributes(g)["n_trials"]))
        @assert length(times_s) == n_timepoints "Time axis metadata mismatch for $(subject_label)"
        @assert length(channel_names) == n_channels "Channel axis metadata mismatch for $(subject_label)"
        (
            subject_label = subject_label,
            times_s = times_s,
            channel_names = channel_names,
            sfreq_hz = Float64(read(HDF5.attributes(g)["sfreq_hz"])),
            n_channels = n_channels,
            n_timepoints = n_timepoints,
            n_trials = n_trials,
            source_set_relpath = String(read(HDF5.attributes(g)["source_set_relpath"])),
            source_eventlist_relpath = String(read(HDF5.attributes(g)["source_eventlist_relpath"])),
        )
    end
end

function ensure_channel_time_trial(epochs::AbstractArray, n_channels::Int, n_timepoints::Int, n_trials::Int)
    expected = (n_channels, n_timepoints, n_trials)
    size(epochs) == expected && return epochs

    perm = if size(epochs) == (n_trials, n_timepoints, n_channels)
        (3, 2, 1)
    elseif size(epochs) == (n_timepoints, n_channels, n_trials)
        (2, 1, 3)
    elseif size(epochs) == (n_trials, n_channels, n_timepoints)
        (2, 3, 1)
    elseif size(epochs) == (n_timepoints, n_trials, n_channels)
        (3, 1, 2)
    elseif size(epochs) == (n_channels, n_trials, n_timepoints)
        (1, 3, 2)
    else
        nothing
    end

    perm === nothing && error(
        "Unexpected epoch tensor layout $(size(epochs)); expected axes compatible with " *
        "(channel, time, trial) = $(expected)"
    )
    normalized = permutedims(epochs, perm)
    @assert size(normalized) == expected "Failed to normalize epoch tensor axes."
    return normalized
end

function load_subject_data(path::AbstractString, subject_label::AbstractString)
    return h5open(path, "r") do f
        g = f["subjects"][subject_label]
        times_s = Float32.(read(g["times_s"]))
        channel_names = String.(read(g["channel_names"]))
        n_channels = Int(read(HDF5.attributes(g)["n_channels"]))
        n_timepoints = Int(read(HDF5.attributes(g)["n_timepoints"]))
        n_trials = Int(read(HDF5.attributes(g)["n_trials"]))
        raw_epochs = read(g["epochs"])
        epochs = ensure_channel_time_trial(raw_epochs, n_channels, n_timepoints, n_trials)
        @assert length(times_s) == n_timepoints "Time axis metadata mismatch for $(subject_label)"
        @assert length(channel_names) == n_channels "Channel axis metadata mismatch for $(subject_label)"
        (
            subject_label = subject_label,
            epochs = epochs,
            times_s = times_s,
            channel_names = channel_names,
            sfreq_hz = Float64(read(HDF5.attributes(g)["sfreq_hz"])),
            n_channels = n_channels,
            n_timepoints = n_timepoints,
            n_trials = n_trials,
            source_set_relpath = String(read(HDF5.attributes(g)["source_set_relpath"])),
            source_eventlist_relpath = String(read(HDF5.attributes(g)["source_eventlist_relpath"])),
        )
    end
end

function metadata_string_list(x)
    return [String(v) for v in x]
end

function load_clean_dataset_bundle(dataset_key::AbstractString)
    for path in [h5_path(dataset_key), events_path(dataset_key), metadata_path(dataset_key)]
        @assert isfile(path) "Pre-downloaded dataset file not found: $path"
    end
    meta = JSON3.read(read(metadata_path(dataset_key), String))
    events = CSV.read(events_path(dataset_key), DataFrame)
    selected_subjects = metadata_string_list(meta.selected_subjects)
    sample_subject = first(selected_subjects)
    subject_meta = read_subject_metadata(h5_path(dataset_key), sample_subject)
    return (
        dataset_key = String(dataset_key),
        dataset_dir = dataset_dir(dataset_key),
        h5_path = h5_path(dataset_key),
        events_path = events_path(dataset_key),
        metadata_path = metadata_path(dataset_key),
        metadata = meta,
        events = events,
        subject_labels = selected_subjects,
        channel_names = subject_meta.channel_names,
        times_s = subject_meta.times_s,
        sampling_rate = infer_sampling_rate(subject_meta.times_s),
        n_channels = subject_meta.n_channels,
        n_timepoints = subject_meta.n_timepoints,
    )
end

function unique_nonmissing_count(values)
    return length(unique(collect(skipmissing(values))))
end

function available_sort_columns(bundle; require_variation::Bool = true)
    present = Symbol[]
    for col in propertynames(bundle.events)
        col in SORT_COLUMN_EXCLUDE && continue
        values = bundle.events[!, col]
        require_variation && unique_nonmissing_count(values) <= 1 && continue
        push!(present, col)
    end

    preferred = Symbol.(metadata_string_list(bundle.metadata.recommended_sort_columns))
    ordered = Symbol[]
    for col in preferred
        col in present && push!(ordered, col)
    end
    for col in present
        col in ordered || push!(ordered, col)
    end
    return ordered
end

function available_sort_columns_df(bundles)
    rows = NamedTuple[]
    for bundle in bundles
        for col in available_sort_columns(bundle)
            values = bundle.events[!, col]
            push!(rows, (
                dataset_key = bundle.dataset_key,
                component = String(bundle.metadata.component),
                sort_col = String(col),
                unique_values = unique_nonmissing_count(values),
                value_type = string(eltype(values)),
                preview_default = !(col in PREVIEW_SORT_COLUMN_EXCLUDE),
            ))
        end
    end
    return DataFrame(rows)
end

function external_dataset_summary_df(bundles)
    rows = NamedTuple[]
    for bundle in bundles
        push!(rows, (
            dataset_key = bundle.dataset_key,
            component = String(bundle.metadata.component),
            subjects = length(bundle.subject_labels),
            channels = bundle.n_channels,
            timepoints = bundle.n_timepoints,
            tmin_s = first(bundle.times_s),
            tmax_s = last(bundle.times_s),
            sampling_rate_hz = bundle.sampling_rate,
            trials_total = nrow(bundle.events),
            sort_columns = join(string.(available_sort_columns(bundle)), ", "),
        ))
    end
    return DataFrame(rows)
end

function dataset_axis_audit_df(bundles)
    rows = NamedTuple[]
    for bundle in bundles
        for subject_label in bundle.subject_labels
            subj = load_subject_data(bundle.h5_path, subject_label)
            expected = (subj.n_channels, subj.n_timepoints, subj.n_trials)
            observed = size(subj.epochs)
            event_rows = nrow(bundle.events[bundle.events.subject_label .== subject_label, :])
            push!(rows, (
                dataset_key = bundle.dataset_key,
                component = String(bundle.metadata.component),
                subject_label = String(subject_label),
                observed_axes = string(observed),
                expected_axes = string(expected),
                tmin_s = first(subj.times_s),
                tmax_s = last(subj.times_s),
                event_rows = event_rows,
                trial_axis = size(subj.epochs, 3),
                status = observed == expected && event_rows == size(subj.epochs, 3) ? "ok" : "mismatch",
            ))
        end
    end
    return DataFrame(rows)
end

function dataset_source_overview_df(bundles)
    rows = NamedTuple[]
    for bundle in bundles
        push!(rows, (
            dataset_key = bundle.dataset_key,
            component = String(bundle.metadata.component),
            source_component = String(bundle.metadata.source_component),
            source_scripts = String(bundle.metadata.source_processing_scripts),
            reader_docs = String(bundle.metadata.reader_docs),
            selected_subjects = join(bundle.subject_labels, ", "),
        ))
    end
    return DataFrame(rows)
end

function dataset_source_example_df(bundle)
    rows = NamedTuple[]
    examples = bundle.metadata.official_source_examples
    for key in propertynames(examples)
        push!(rows, (
            source_file = String(key),
            excerpt = String(getproperty(examples, key)),
        ))
    end
    return DataFrame(rows)
end

function preview_sort_columns(bundle)
    cols = [col for col in available_sort_columns(bundle) if !(col in PREVIEW_SORT_COLUMN_EXCLUDE)]
    isempty(cols) && return available_sort_columns(bundle)
    return cols
end

function recommended_preview_specs(bundle)
    return [(sort_col = col, filters = Pair{Symbol, Any}[]) for col in preview_sort_columns(bundle)]
end

function post_stim_indices(times_s::AbstractVector{<:Real};
        time_window_s::Tuple{<:Real, <:Real} = REAL_PREVIEW_TIME_WINDOW_S)
    tmin = Float32(time_window_s[1])
    tmax = Float32(time_window_s[2])
    @assert tmin <= tmax "Invalid preview time window: $(time_window_s)"
    idx = findall(t -> tmin <= t <= tmax, Float32.(times_s))
    isempty(idx) && error("No timepoints found in preview window $(time_window_s).")
    return idx
end

function sortvalues_from(df::DataFrame, col::Symbol)
    values = df[!, col]
    if eltype(values) <: Number
        return Float64.(values)
    end
    return string.(values)
end

function source_part_sort_column(df::DataFrame)
    present = propertynames(df)
    for col in SOURCE_PART_SORT_COLUMN_CANDIDATES
        col in present || continue
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

function trial_sort_order(df::DataFrame, sort_col::Symbol)
    row_col = :__row_idx__
    sort_cols = effective_sort_columns(df, sort_col)
    sort_cols_with_row = vcat(sort_cols, [row_col])

    order_df = DataFrame()
    order_df[!, row_col] = collect(1:nrow(df))
    for col in sort_cols
        order_df[!, col] = copy(df[!, col])
    end

    sort!(order_df, sort_cols_with_row)
    return Int.(order_df[!, row_col])
end

function merged_events_trials(bundle; filters = Pair{Symbol, Any}[])
    frames = DataFrame[]
    for subject_label in bundle.subject_labels
        push!(frames, select_subject_events(bundle, subject_label; filters = filters))
    end
    isempty(frames) && error("No subject events available for $(bundle.dataset_key).")
    return vcat(frames...; cols = :union)
end

function audit_sort_order_rows!(rows::Vector{NamedTuple}, bundle, subject_label::AbstractString,
        events_trials::DataFrame, sort_columns)
    for sort_col in sort_columns
        sort_col in propertynames(events_trials) || continue
        sort_cols = effective_sort_columns(events_trials, sort_col)
        sort_cols_with_row = vcat(sort_cols, [:__row_idx__])
        order = trial_sort_order(events_trials, sort_col)

        order_df = DataFrame()
        order_df[!, :__row_idx__] = collect(1:nrow(events_trials))
        for col in sort_cols
            order_df[!, col] = copy(events_trials[!, col])
        end
        sort!(order_df, sort_cols_with_row)
        expected_order = Int.(order_df[!, :__row_idx__])

        permutation_ok = sort(order) == collect(1:nrow(events_trials))
        matches_expected = order == expected_order
        source_guard = length(sort_cols) > 1 && first(sort_cols) != sort_col
        status = permutation_ok && matches_expected ? "ok" : "mismatch"

        push!(rows, (
            dataset_key = bundle.dataset_key,
            subject_label = String(subject_label),
            sort_col = String(sort_col),
            effective_sort_columns = join(string.(sort_cols_with_row), ", "),
            n_trials = nrow(events_trials),
            unique_values = unique_nonmissing_count(events_trials[!, sort_col]),
            source_guard = source_guard,
            status = status,
        ))
    end
    return rows
end

function sort_order_audit_df(bundle; sort_columns = available_sort_columns(bundle),
        include_merged::Bool = false)
    rows = NamedTuple[]
    for subject_label in bundle.subject_labels
        events_trials = select_subject_events(bundle, subject_label)
        audit_sort_order_rows!(rows, bundle, subject_label, events_trials, sort_columns)
    end
    if include_merged && length(bundle.subject_labels) > 1
        audit_sort_order_rows!(rows, bundle, "merged_experiment",
            merged_events_trials(bundle), sort_columns)
    end
    return DataFrame(rows)
end

function build_base_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    @assert size(data_time_trials, 2) == nrow(events_trials) "Trial count mismatch between matrix and events."
    @assert sort_col in propertynames(events_trials) "Sort column not found: $sort_col"

    data_z = zscore_timepoints(data_time_trials)
    order = trial_sort_order(events_trials, sort_col)
    data_sorted = Float32.(data_z[:, order])
    return Float32.(permutedims(data_sorted, (2, 1)))
end

function baseline_correct_time_trials(
    data_time_trials::AbstractMatrix,
    times_s::AbstractVector{<:Real};
    baseline_window::Tuple{Float32, Float32} = REAL_BASELINE_WINDOW_S,
)
    baseline_idx = findall(t -> baseline_window[1] <= t <= baseline_window[2], Float32.(times_s))
    isempty(baseline_idx) && return Float32.(data_time_trials)

    x = Float32.(data_time_trials)
    baseline = mean(x[baseline_idx, :]; dims = 1)
    return Float32.(x .- baseline)
end

function process_erp_image(img_trials_time::AbstractMatrix,
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing;
        lowpass::Bool = true,
        sigma_factor::Float32 = LOWPASS_SIGMA_FACTOR)
    filtered = Float32.(img_trials_time)
    if lowpass && min(size(filtered)...) > 1
        smooth_size = target_size === nothing ? size(img_trials_time) : target_size
        kernel = gaussian_kernel(sigma_factor, size(img_trials_time), smooth_size)
        filtered = Float32.(imfilter(filtered, kernel, FILTER_BORDER))
    end
    target_size === nothing && return filtered
    return size(filtered) == target_size ? filtered : Float32.(imresize(filtered, target_size))
end

function value_matches(x, target)
    ismissing(x) && return false
    if target isa Function
        return Bool(target(x))
    elseif target isa AbstractVector && !(target isa AbstractString)
        return any(value_matches(x, item) for item in target)
    elseif x isa AbstractString || target isa AbstractString
        return lowercase(string(x)) == lowercase(string(target))
    end
    return x == target
end

function apply_filters(df::DataFrame, filters)
    isempty(filters) && return copy(df)
    mask = trues(nrow(df))
    for (col, target) in filters
        col in propertynames(df) || error("Filter column not found: $col")
        mask .&= [value_matches(v, target) for v in df[!, col]]
    end
    return copy(df[mask, :])
end

function filters_label(filters)
    isempty(filters) && return "none"
    parts = String[]
    for (col, target) in filters
        if target isa AbstractVector && !(target isa AbstractString)
            push!(parts, "$(col)=" * join(string.(target), "|"))
        else
            push!(parts, "$(col)=$(target)")
        end
    end
    return join(parts, ", ")
end

function preview_title(component::AbstractString, sort_col::Symbol, filters)
    base = "$(component) | sort=$(sort_col)"
    isempty(filters) && return base
    return base * " | filters: " * filters_label(filters)
end

function select_subject_events(bundle, subject_label::AbstractString; filters = Pair{Symbol, Any}[])
    rows = bundle.events[bundle.events.subject_label .== subject_label, :]
    rows = apply_filters(rows, filters)
    sort!(rows, :epoch_index)
    @assert !isempty(rows) "No events left for $(subject_label) after filtering: $(filters_label(filters))"
    return rows
end

function select_preview_subjects(bundle; filters = Pair{Symbol, Any}[], n_subjects::Int = 2)
    rows = apply_filters(bundle.events, filters)
    counts = combine(groupby(rows, :subject_label), nrow => :n_trials)
    sort!(counts, [:n_trials, :subject_label], rev = [true, false])
    return String.(counts.subject_label[1:min(n_subjects, nrow(counts))])
end

function select_preview_channels(bundle; n_channels::Int = 2)
    preferred = Set(metadata_string_list(bundle.metadata.preferred_channels))
    ordered = String[]
    for channel in bundle.channel_names
        channel in preferred && push!(ordered, channel)
    end
    for channel in bundle.channel_names
        channel in ordered || push!(ordered, channel)
    end
    return ordered[1:min(n_channels, length(ordered))]
end

function first_n_with_repeats(items, n::Int)
    n <= 0 && return items[1:0]
    isempty(items) && error("Cannot select $(n) preview items from an empty collection.")
    return [items[mod1(idx, length(items))] for idx in 1:n]
end

function numeric_range_or_nothing(values)
    if eltype(values) <: Number
        return (minimum(values), maximum(values))
    end
    return nothing
end

function build_dataset_image(bundle;
        subject_label::AbstractString,
        channel_name::AbstractString,
        sort_col::Symbol,
        filters = Pair{Symbol, Any}[],
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        baseline_correct::Bool = true,
        time_window_s::Tuple{<:Real, <:Real} = REAL_PREVIEW_TIME_WINDOW_S)
    subj = load_subject_data(bundle.h5_path, subject_label)
    @assert size(subj.epochs) == (subj.n_channels, subj.n_timepoints, subj.n_trials) "Unexpected tensor axes for $(subject_label)"
    channel_idx = findfirst(==(channel_name), subj.channel_names)
    channel_idx === nothing && error("Channel $(channel_name) not found in $(subject_label)")

    events_subset = select_subject_events(bundle, subject_label; filters = filters)
    epoch_indices = Int.(events_subset.epoch_index)
    post_idx = post_stim_indices(subj.times_s; time_window_s = time_window_s)
    post_times_s = subj.times_s[post_idx]
    @assert minimum(epoch_indices) >= 1 "Epoch indices must be 1-based."
    @assert maximum(epoch_indices) <= size(subj.epochs, 3) "Epoch index exceeds trial axis."

    data_full_time_trials = reshape(
        Float32.(subj.epochs[channel_idx, :, epoch_indices]),
        subj.n_timepoints,
        length(epoch_indices),
    )
    @assert size(data_full_time_trials) == (subj.n_timepoints, length(epoch_indices)) "Expected full (time, trial) slice."
    if baseline_correct
        data_full_time_trials = baseline_correct_time_trials(data_full_time_trials, subj.times_s)
    end

    data_time_trials = reshape(
        Float32.(data_full_time_trials[post_idx, :]),
        length(post_idx),
        length(epoch_indices),
    )
    @assert size(data_time_trials) == (length(post_idx), length(epoch_indices)) "Expected (time, trial) slice."
    img_base = build_base_image(data_time_trials, events_subset, sort_col)
    @assert size(img_base) == (length(epoch_indices), length(post_idx)) "Expected (trial, time) ERP image."
    img_processed = process_erp_image(img_base, target_size; lowpass = lowpass)
    sort_range = numeric_range_or_nothing(events_subset[!, sort_col])

    return (
        dataset_key = bundle.dataset_key,
        component = String(bundle.metadata.component),
        subject_label = String(subject_label),
        channel_name = String(channel_name),
        channel_idx = Int(channel_idx),
        sort_col = sort_col,
        filters = filters,
        filters_label = filters_label(filters),
        n_trials = nrow(events_subset),
        n_timepoints_post = length(post_idx),
        time_start_s = Float32(first(post_times_s)),
        time_end_s = Float32(last(post_times_s)),
        sampling_rate_hz = subj.sfreq_hz,
        source_set_relpath = subj.source_set_relpath,
        source_eventlist_relpath = subj.source_eventlist_relpath,
        base_img = img_base,
        image = img_processed,
        original_size = size(img_base),
        resized_size = size(img_processed),
        sort_range = sort_range,
    )
end

function build_dataset_merged_image(bundle;
        channel_name::AbstractString,
        sort_col::Symbol,
        filters = Pair{Symbol, Any}[],
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        baseline_correct::Bool = true,
        time_window_s::Tuple{<:Real, <:Real} = REAL_PREVIEW_TIME_WINDOW_S)
    data_parts = Matrix{Float32}[]
    event_parts = DataFrame[]
    subject_labels = String[]
    channel_indices = Int[]
    post_len = nothing
    sfreq_hz = nothing
    time_start_s = nothing
    time_end_s = nothing

    for subject_label in bundle.subject_labels
        subj = load_subject_data(bundle.h5_path, subject_label)
        @assert size(subj.epochs) == (subj.n_channels, subj.n_timepoints, subj.n_trials) "Unexpected tensor axes for $(subject_label)"
        channel_idx = findfirst(==(channel_name), subj.channel_names)
        channel_idx === nothing && continue

        events_subset = select_subject_events(bundle, subject_label; filters = filters)
        epoch_indices = Int.(events_subset.epoch_index)
        post_idx = post_stim_indices(subj.times_s; time_window_s = time_window_s)
        post_times_s = subj.times_s[post_idx]
        post_len === nothing || post_len == length(post_idx) ||
            error("Cannot merge $(bundle.dataset_key): post-stimulus length differs across subjects.")
        sfreq_hz === nothing || sfreq_hz == subj.sfreq_hz ||
            error("Cannot merge $(bundle.dataset_key): sampling rate differs across subjects.")
        time_start_s === nothing || time_start_s == first(post_times_s) ||
            error("Cannot merge $(bundle.dataset_key): preview start time differs across subjects.")
        time_end_s === nothing || time_end_s == last(post_times_s) ||
            error("Cannot merge $(bundle.dataset_key): preview end time differs across subjects.")
        post_len = length(post_idx)
        sfreq_hz = subj.sfreq_hz
        time_start_s = first(post_times_s)
        time_end_s = last(post_times_s)

        data_full_time_trials = reshape(
            Float32.(subj.epochs[channel_idx, :, epoch_indices]),
            subj.n_timepoints,
            length(epoch_indices),
        )
        if baseline_correct
            data_full_time_trials = baseline_correct_time_trials(data_full_time_trials, subj.times_s)
        end
        data_time_trials = reshape(
            Float32.(data_full_time_trials[post_idx, :]),
            length(post_idx),
            length(epoch_indices),
        )

        push!(data_parts, data_time_trials)
        push!(event_parts, events_subset)
        push!(subject_labels, String(subject_label))
        push!(channel_indices, Int(channel_idx))
    end

    isempty(data_parts) && error("Channel $(channel_name) not found in any selected subject for $(bundle.dataset_key).")
    data_time_trials = hcat(data_parts...)
    events_merged = vcat(event_parts...; cols = :union)
    @assert size(data_time_trials, 2) == nrow(events_merged) "Trial count mismatch after merging subjects."

    img_base = build_base_image(data_time_trials, events_merged, sort_col)
    @assert size(img_base) == (nrow(events_merged), post_len) "Expected merged (trial, time) ERP image."
    img_processed = process_erp_image(img_base, target_size; lowpass = lowpass)
    sort_range = numeric_range_or_nothing(events_merged[!, sort_col])
    subject_label = length(subject_labels) == 1 ? first(subject_labels) : "merged_experiment"

    return (
        dataset_key = bundle.dataset_key,
        component = String(bundle.metadata.component),
        subject_label = subject_label,
        channel_name = String(channel_name),
        channel_idx = first(channel_indices),
        sort_col = sort_col,
        filters = filters,
        filters_label = filters_label(filters),
        n_trials = nrow(events_merged),
        n_timepoints_post = post_len,
        time_start_s = Float32(time_start_s),
        time_end_s = Float32(time_end_s),
        sampling_rate_hz = sfreq_hz,
        source_set_relpath = "merged",
        source_eventlist_relpath = "merged",
        base_img = img_base,
        image = img_processed,
        original_size = size(img_base),
        resized_size = size(img_processed),
        sort_range = sort_range,
    )
end

function build_dataset_sort_preview(bundle;
        sort_col::Symbol,
        filters = Pair{Symbol, Any}[],
        n_samples::Int = 16,
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        rng_seed::Int = 42,
        merge_subjects::Bool = false,
        baseline_correct::Bool = true,
        prefer_preferred_channels::Bool = false,
        time_window_s::Tuple{<:Real, <:Real} = REAL_PREVIEW_TIME_WINDOW_S)
    all_subjects = select_preview_subjects(bundle; filters = filters,
        n_subjects = length(bundle.subject_labels))
    all_channels = String.(bundle.channel_names)
    rng = MersenneTwister(rng_seed + hash(sort_col))

    function order_with_preferred(channels)
        prefer_preferred_channels || return channels[randperm(rng, length(channels))]
        preferred = Set(metadata_string_list(bundle.metadata.preferred_channels))
        pref_present = [c for c in channels if c in preferred]
        rest = [c for c in channels if !(c in preferred)]
        return vcat(pref_present[randperm(rng, length(pref_present))],
                    rest[randperm(rng, length(rest))])
    end

    if merge_subjects
        ordered = order_with_preferred(all_channels)
        selected_channels = first_n_with_repeats(ordered, n_samples)
        images = Matrix{Float32}[]
        metadata = NamedTuple[]
        for channel_name in selected_channels
            sample = build_dataset_merged_image(bundle;
                channel_name = channel_name,
                sort_col = sort_col,
                filters = filters,
                target_size = target_size,
                lowpass = lowpass,
                baseline_correct = baseline_correct,
                time_window_s = time_window_s,
            )
            push!(images, sample.image)
            push!(metadata, sample)
        end

        return (
            dataset_key = bundle.dataset_key,
            component = String(bundle.metadata.component),
            sort_col = sort_col,
            filters = filters,
            filters_label = filters_label(filters),
            subjects = length(bundle.subject_labels) == 1 ? String.(bundle.subject_labels) : ["merged_experiment"],
            channels = selected_channels,
            images = images,
            metadata = metadata,
        )
    end

    ordered_channels = order_with_preferred(all_channels)
    pairs = [(s, c) for s in all_subjects for c in ordered_channels]
    ordered_pairs = if prefer_preferred_channels
        pairs
    else
        pairs[randperm(rng, length(pairs))]
    end
    selected = first_n_with_repeats(ordered_pairs, n_samples)

    images = Matrix{Float32}[]
    metadata = NamedTuple[]
    for (subject_label, channel_name) in selected
        sample = build_dataset_image(bundle;
            subject_label = subject_label,
            channel_name = channel_name,
            sort_col = sort_col,
            filters = filters,
            target_size = target_size,
            lowpass = lowpass,
            baseline_correct = baseline_correct,
            time_window_s = time_window_s,
        )
        push!(images, sample.image)
        push!(metadata, sample)
    end

    subjects_used = unique([s for (s, _) in selected])
    channels_used = unique([c for (_, c) in selected])

    return (
        dataset_key = bundle.dataset_key,
        component = String(bundle.metadata.component),
        sort_col = sort_col,
        filters = filters,
        filters_label = filters_label(filters),
        subjects = subjects_used,
        channels = channels_used,
        images = images,
        metadata = metadata,
    )
end

function shared_color_stats(images)
    vals = Float32[]
    sizehint!(vals, sum(length, images))
    for img in images
        append!(vals, vec(Float32.(img)))
    end
    _, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(reshape(vals, :, 1))
    return (
        colorrange = colorrange,
        tick_vals = tick_vals,
        tick_labels = tick_labels,
        cmap = cmap,
    )
end

function image_color_stats(img::AbstractMatrix)
    clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(Float32.(img))
    return (
        clipped = clipped,
        colorrange = colorrange,
        tick_vals = tick_vals,
        tick_labels = tick_labels,
        cmap = cmap,
    )
end

function format_elapsed_time_label(seconds_after_zero::Real)
    ms = round(Int, Float64(seconds_after_zero) * 1000)
    if abs(ms) >= 1000
        return @sprintf("%.2f s", Float64(seconds_after_zero))
    end
    return "$(ms) ms"
end

function axis_tick_positions(display_len::Int)
    return unique([1, Int(round((display_len + 1) / 2)), display_len])
end

function scaled_axis_ticks(display_len::Int, original_len::Int)
    tick_vals = axis_tick_positions(display_len)
    if display_len <= 1 || original_len <= 1
        tick_labels = string.(fill(original_len, length(tick_vals)))
    else
        tick_labels = [
            string(Int(round(1 + (pos - 1) * (original_len - 1) / (display_len - 1))))
            for pos in tick_vals
        ]
    end
    return (tick_vals, tick_labels)
end

function time_axis_ticks(time_start_s::Real, time_end_s::Real)
    start_s = Float64(time_start_s)
    stop_s = Float64(time_end_s)
    tick_times = unique([start_s, (start_s + stop_s) / 2, stop_s])
    return (tick_times, format_elapsed_time_label.(tick_times))
end

function trial_rank_ticks(n_trials::Integer)
    tick_vals = unique([1, Int(round((n_trials + 1) / 2)), n_trials])
    return (tick_vals, string.(tick_vals))
end

function axis_ticks(sample)
    xticks = time_axis_ticks(sample.time_start_s, sample.time_end_s)
    yticks = trial_rank_ticks(sample.n_trials)
    return (
        xticks = xticks,
        yticks = yticks,
    )
end

function plot_dataset_sort_preview(preview; n_cols::Int = 4)
    n_total = length(preview.images)
    n_cols = min(n_cols, n_total)
    n_rows = cld(n_total, n_cols)

    fig = Figure(size = (380 * n_cols + 80, 295 * n_rows + 140), figure_padding = 20)
    title = preview_title(preview.component, preview.sort_col, preview.filters)
    Label(fig[0, 1:n_cols], title; fontsize = 22, tellwidth = false)

    for idx in 1:n_total
        row = cld(idx, n_cols)
        col = mod1(idx, n_cols)
        stats = image_color_stats(preview.images[idx])
        img = stats.clipped
        meta = preview.metadata[idx]
        ticks = axis_ticks(meta)
        sort_text = meta.sort_range === nothing ? "categorical sort asc" :
            @sprintf("sort asc %.1f..%.1f", meta.sort_range[1], meta.sort_range[2])

        cell = GridLayout(fig[row, col])
        ax = Axis(cell[1, 1];
            title = "$(meta.subject_label) | $(meta.channel_name)",
            xlabel = row == n_rows ? "time after onset" : "",
            ylabel = col == 1 ? "trial rank (ascending)" : "",
            yreversed = false,
            titlesize = 13,
            xlabelsize = 12,
            ylabelsize = 12,
            xticklabelsize = 9,
            yticklabelsize = 9,
        )
        ax.xticks = ticks.xticks
        ax.yticks = ticks.yticks

        hm = heatmap!(
            ax,
            range(Float64(meta.time_start_s), Float64(meta.time_end_s); length = size(img, 2)),
            range(1, Float64(meta.n_trials); length = size(img, 1)),
            permutedims(img, (2, 1));
            colormap = stats.cmap,
            colorrange = stats.colorrange,
        )

        info = @sprintf(
            "n=%d | %s | %.0f Hz",
            meta.n_trials,
            sort_text,
            meta.sampling_rate_hz,
        )
        text!(ax, 0.02, 0.02, text = info, space = :relative, align = (:left, :bottom), fontsize = 8)

        Colorbar(
            cell[1, 2],
            hm;
            ticks = (stats.tick_vals, stats.tick_labels),
            ticklabelsize = 8,
            width = 12,
        )
    end
    return fig
end

function plot_all_dataset_sort_previews(bundle;
        sort_columns = available_sort_columns(bundle),
        n_samples::Int = 16,
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        rng_seed::Int = 42,
        n_cols::Int = 4,
        merge_subjects::Bool = true,
        baseline_correct::Bool = true,
        prefer_preferred_channels::Bool = false,
        time_window_s::Tuple{<:Real, <:Real} = REAL_PREVIEW_TIME_WINDOW_S)
    rows = NamedTuple[]
    for (idx, sort_col) in enumerate(sort_columns)
        preview = build_dataset_sort_preview(bundle;
            sort_col = sort_col,
            n_samples = n_samples,
            target_size = target_size,
            lowpass = lowpass,
            rng_seed = rng_seed + 1009 * idx,
            merge_subjects = merge_subjects,
            baseline_correct = baseline_correct,
            prefer_preferred_channels = prefer_preferred_channels,
            time_window_s = time_window_s,
        )
        display(plot_dataset_sort_preview(preview; n_cols = n_cols))
        push!(rows, (
            sort_col = String(sort_col),
            n_images = length(preview.images),
            subjects = join(preview.subjects, ", "),
            channels = join(preview.channels, ", "),
        ))
    end
    return DataFrame(rows)
end

function find_erps_dataset(file)
    candidates = ["epochs", "/epochs", "erps", "/erps", "data", "/data/data_fixations.hdf5", "data/data_fixations.hdf5"]
    for key in candidates
        if haskey(file, key)
            obj = file[key]
            if obj isa HDF5.Dataset
                return obj
            end
        end
    end

    function first_dataset(group)
        for key in keys(group)
            obj = group[key]
            if obj isa HDF5.Dataset
                return obj
            elseif obj isa HDF5.Group
                nested = first_dataset(obj)
                nested === nothing || return nested
            end
        end
        return nothing
    end

    dataset = first_dataset(file)
    dataset === nothing && error("No dataset found in HDF5 file.")
    return dataset
end

function with_erps_dataset(func::Function, path::AbstractString)
    return h5open(path, "r") do file
        dataset = find_erps_dataset(file)
        return func(dataset)
    end
end

function extract_fixation_channel_trials(erps, events::DataFrame, channel::Int; post_stim_only::Bool = true)
    @assert 1 <= channel <= size(erps, 1) "Channel out of range: $channel"
    start_idx = post_stim_only ? FIXATION_TIME_ZERO_IDX : 1
    data = Float32.(erps[channel, start_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], copy(events[1:n, :])
end

function fixation_axis_ticks(meta)
    xticks = time_axis_ticks(0f0, Float32((meta.n_timepoints_post - 1) / FIXATION_SAMPLING_RATE))
    yticks = trial_rank_ticks(meta.n_trials)
    return (xticks = xticks, yticks = yticks)
end

function fixation_sort_columns(events::DataFrame; preferred_only::Bool = false)
    present = Symbol[]
    for col in propertynames(events)
        col in FIXATION_NON_SORT_COLS && continue
        unique_nonmissing_count(events[!, col]) <= 1 && continue
        push!(present, col)
    end

    if preferred_only
        return [col for col in FIXATION_REFERENCE_SORT_COLUMNS if col in present]
    end

    ordered = Symbol[]
    for col in FIXATION_REFERENCE_SORT_COLUMNS
        col in present && push!(ordered, col)
    end
    for col in present
        col in ordered || push!(ordered, col)
    end
    return ordered
end

function fixation_summary_df()
    for path in [FIXATION_H5_PATH, FIXATION_EVENTS_CSV_PATH]
        @assert isfile(path) "File not found: $path"
    end
    events = CSV.read(FIXATION_EVENTS_CSV_PATH, DataFrame)
    n_channels = with_erps_dataset(FIXATION_H5_PATH) do erps
        size(erps, 1)
    end

    rows = NamedTuple[]
    for sort_var in fixation_sort_columns(events)
        values = events[!, sort_var]
        push!(rows, (
            sort_var = String(sort_var),
            unique_values = unique_nonmissing_count(values),
            value_type = string(eltype(values)),
            channels_available = n_channels,
        ))
    end
    return DataFrame(rows)
end

function fixation_sort_order_audit_df(; preferred_only::Bool = false)
    for path in [FIXATION_H5_PATH, FIXATION_EVENTS_CSV_PATH]
        @assert isfile(path) "File not found: $path"
    end

    events = CSV.read(FIXATION_EVENTS_CSV_PATH, DataFrame)
    rows = NamedTuple[]
    for sort_var in fixation_sort_columns(events; preferred_only = preferred_only)
        sort_cols = effective_sort_columns(events, sort_var)
        sort_cols_with_row = vcat(sort_cols, [:__row_idx__])
        order = trial_sort_order(events, sort_var)

        order_df = DataFrame()
        order_df[!, :__row_idx__] = collect(1:nrow(events))
        for col in sort_cols
            order_df[!, col] = copy(events[!, col])
        end
        sort!(order_df, sort_cols_with_row)
        expected_order = Int.(order_df[!, :__row_idx__])

        push!(rows, (
            sort_col = String(sort_var),
            effective_sort_columns = join(string.(sort_cols_with_row), ", "),
            n_trials = nrow(events),
            unique_values = unique_nonmissing_count(events[!, sort_var]),
            source_guard = length(sort_cols) > 1 && first(sort_cols) != sort_var,
            status = sort(order) == collect(1:nrow(events)) && order == expected_order ? "ok" : "mismatch",
        ))
    end
    return DataFrame(rows)
end

function load_fixation_reference_cache(;
        per_sort_var::Int = 16,
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        rng_seed::Int = FIXATION_REFERENCE_RNG_SEED,
        preferred_only::Bool = false)
    for path in [FIXATION_H5_PATH, FIXATION_EVENTS_CSV_PATH]
        @assert isfile(path) "File not found: $path"
    end

    events = CSV.read(FIXATION_EVENTS_CSV_PATH, DataFrame)
    sort_vars = fixation_sort_columns(events; preferred_only = preferred_only)
    rng = MersenneTwister(rng_seed)

    rows = NamedTuple[]
    images = Matrix{Float32}[]
    with_erps_dataset(FIXATION_H5_PATH) do erps
        n_channels = size(erps, 1)
        for sort_var in sort_vars
            picks = first_n_with_repeats(randperm(rng, n_channels), per_sort_var)
            for (pick_rank, channel) in enumerate(picks)
                data_full, events_full = extract_fixation_channel_trials(erps, events, channel)
                base_img = build_base_image(data_full, events_full, sort_var)
                proc_img = process_erp_image(base_img, target_size; lowpass = lowpass)
                push!(rows, (
                    channel = channel,
                    sort_var = String(sort_var),
                    selection_rank = pick_rank,
                    selection_mode = "seeded_random",
                    original_size = size(base_img),
                    resized_size = size(proc_img),
                    n_trials = size(base_img, 1),
                    n_timepoints_post = size(base_img, 2),
                ))
                push!(images, proc_img)
            end
        end
    end

    return (
        images = images,
        meta = DataFrame(rows),
    )
end

function plot_fixation_reference_grid(cache; n_cols::Int = 4, sort_var = nothing)
    plot_cache = cache
    if sort_var !== nothing
        sort_var_label = String(sort_var)
        idxs = findall(==(sort_var_label), String.(cache.meta.sort_var))
        @assert !isempty(idxs) "No fixation reference images available for sort_var=$(sort_var_label)."
        plot_cache = (
            images = cache.images[idxs],
            meta = cache.meta[idxs, :],
        )
    end

    @assert !isempty(plot_cache.images) "No fixation reference images available."
    n = length(plot_cache.images)
    n_cols = min(max(n_cols, 1), n)
    n_rows = cld(n, n_cols)

    fig = Figure(size = (510 * n_cols, 320 * n_rows + 90), figure_padding = 18)

    for idx in 1:n
        row = cld(idx, n_cols)
        col = mod1(idx, n_cols)
        stats = image_color_stats(plot_cache.images[idx])
        img = stats.clipped
        meta = plot_cache.meta[idx, :]
        ticks = fixation_axis_ticks(meta)

        cell = GridLayout(fig[row, col])
        ax = Axis(cell[1, 1];
            title = "$(meta.sort_var) | ch$(meta.channel)",
            xlabel = row == n_rows ? "time after onset" : "",
            ylabel = col == 1 ? "trial rank (ascending)" : "",
            yreversed = false,
            xticks = ticks.xticks,
            yticks = ticks.yticks,
            titlesize = 14,
            xlabelsize = 12,
            ylabelsize = 12,
            xticklabelsize = 10,
            yticklabelsize = 10,
        )
        hm = heatmap!(
            ax,
            range(0, Float64((meta.n_timepoints_post - 1) / FIXATION_SAMPLING_RATE); length = size(img, 2)),
            range(1, Float64(meta.n_trials); length = size(img, 1)),
            permutedims(img, (2, 1));
            colormap = stats.cmap,
            colorrange = stats.colorrange,
        )
        Colorbar(
            cell[1, 2],
            hm;
            ticks = (stats.tick_vals, stats.tick_labels),
            ticklabelsize = 9,
            width = 14,
        )
    end
    resized = plot_cache.meta[1, :resized_size]
    title_sort = sort_var === nothing ? "all sort variables" : "sort=$(String(sort_var))"
    Label(fig[0, 1:n_cols], "Fixation reference images | $(title_sort) | low-pass + $(resized)";
        fontsize = 20, tellwidth = false)
    return fig
end

end
