module Week23ERPPatternExamples

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
    ]))
    for candidate in candidates
        if isdir(joinpath(candidate, "notebooks")) && isdir(joinpath(candidate, "scripts"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir).")
end

const REPO_ROOT = find_repo_root()
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
Pkg.activate(MODEL_ENV_DIR)

using CairoMakie
using CSV
using DataFrames
using HDF5
using Printf: @sprintf
using Random

include(joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_erp_export_helpers.jl"))
using .Week21LabelStudioERPExport

const Export = Week21LabelStudioERPExport
const Week15 = Export.Week15TryNewData

const NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_23")
const ANNOTATIONS_CSV = joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_annotations_all.csv")
const THESIS_EXPORT_DIR = joinpath(NOTEBOOK_DIR, "erp_pattern_thesis_exports")
const RNG_SEED = 20260516
const EXAMPLES_PER_CLASS = 20

const PATTERN_CLASSES = [
    "sigmoid",
    "tilted_bar",
    "one_sided_fan",
    "two_sided_fan",
    "diverging_bar",
    "hourglass",
]

const THESIS_EXPORT_SPECS = [
    (erp_class = "sigmoid", dataset_key = "fixations_dataset", channel_name = "ch096", sort_variable = "duration"),
    (erp_class = "sigmoid", dataset_key = "eye_eeg_freeviewing_fixations", channel_name = "Oz", sort_variable = "fixation_duration_ms"),
    (erp_class = "tilted_bar", dataset_key = "02_new_roamm_reading", channel_name = "F8", sort_variable = "gaze_x"),
    (erp_class = "tilted_bar", dataset_key = "erp_core_n170_clean", channel_name = "C6", sort_variable = "reaction_time_ms"),
    (erp_class = "one_sided_fan", dataset_key = "02_new_eegeyenet_saccades", channel_name = "E61", sort_variable = "saccade_duration_ms"),
    (erp_class = "one_sided_fan", dataset_key = "02_new_eegeyenet_saccades", channel_name = "E83", sort_variable = "saccade_duration"),
    (erp_class = "two_sided_fan", dataset_key = "fixations_dataset", channel_name = "ch125", sort_variable = "sac_amplitude"),
    (erp_class = "two_sided_fan", dataset_key = "fixations_dataset", channel_name = "ch058", sort_variable = "sac_amplitude"),
    (erp_class = "diverging_bar", dataset_key = "02_new_unfold_facefreeview", channel_name = "P3", sort_variable = "saccade_amplitude"),
    (erp_class = "diverging_bar", dataset_key = "02_new_unfold_facefreeview", channel_name = "O1", sort_variable = "saccade_amplitude"),
    (erp_class = "hourglass", dataset_key = "kilo_word_erp", channel_name = "FC5", sort_variable = "number_of_letters"),
    (erp_class = "hourglass", dataset_key = "02_new_eegeyenet_saccades", channel_name = "E21", sort_variable = "saccade_duration_ms"),
]

const EXCLUDED_DATASET_KEYS = Set([
    # This accidental raw import is already excluded from the Week 23 training
    # dataset build. It currently has no positive pattern labels, but keeping the
    # exclusion here prevents it from silently entering future figure selections.
    "02_new_eeget_rsod",
])

const BASELINE_CORRECT_BY_DATASET = Dict{String, Bool}(
    "02_new_eegeyenet_saccades" => false,
    "02_new_roamm_reading" => false,
    "02_new_unfold_facefreeview" => false,
    "eegeyenet_saccades" => false,
    "erp_core_n170_clean" => true,
    "erp_core_n2pc_clean" => true,
    "eye_eeg_freeviewing_fixations" => false,
    "eye_eeg_reading_fixations" => false,
    "eye_eeg_sceneviewing_tobii_fixations" => false,
    "fixations_dataset" => false,
    "kilo_word_erp" => false,
    "nod_eeg_public" => true,
)

const REFERENCE_DATASET_KEY = "fixations_dataset"
const REFERENCE_DATA_DIR = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY)
const REFERENCE_H5_PATH = joinpath(REFERENCE_DATA_DIR, "data_fixations.hdf5")
const REFERENCE_EVENTS_PATH = joinpath(REFERENCE_DATA_DIR, "events.csv")
const REFERENCE_SAMPLING_RATE = 512.0
const REFERENCE_PRE_STIM_S = 0.5f0
const REFERENCE_TIME_ZERO_IDX = Int(round(Float64(REFERENCE_PRE_STIM_S) * REFERENCE_SAMPLING_RATE)) + 1

export ANNOTATIONS_CSV
export THESIS_EXPORT_DIR
export THESIS_EXPORT_SPECS
export RNG_SEED
export EXAMPLES_PER_CLASS
export PATTERN_CLASSES
export load_labelled_annotations
export select_labelled_examples
export select_examples_by_specs
export reconstruct_full_resolution_examples
export available_examples_by_class
export plot_single_erp_image
export display_class_examples
export export_individual_erp_svgs
export selected_examples_table
export run_pattern_example_pipeline

cellstr(x) = ismissing(x) || x === nothing ? "" : string(x)

function parse_int_cell(x; default::Int = 0)
    s = strip(cellstr(x))
    isempty(s) && return default
    parsed = tryparse(Int, s)
    return parsed === nothing ? default : parsed
end

function humanize_class(cls::AbstractString)
    words = split(replace(String(cls), "_" => " "), " ")
    title_words = String[]
    for word in words
        isempty(word) && continue
        rest = firstindex(word) == lastindex(word) ? "" : String(word[nextind(word, firstindex(word)):end])
        push!(title_words, string(uppercase(first(word))) * rest)
    end
    return join(title_words, " ")
end

function display_dataset_key(dataset_key::AbstractString)
    pretty = Dict(
        "02_new_eegeyenet_saccades" => "EEGEyeNet saccades",
        "02_new_roamm_reading" => "ROAMM reading",
        "02_new_unfold_facefreeview" => "Unfold face freeview",
        "erp_core_n170_clean" => "ERP CORE N170",
        "erp_core_n2pc_clean" => "ERP CORE N2pc",
        "eye_eeg_freeviewing_fixations" => "EYE-EEG freeviewing",
        "eye_eeg_sceneviewing_tobii_fixations" => "EYE-EEG sceneviewing",
        "fixations_dataset" => "Fixations dataset",
        "kilo_word_erp" => "Kilo-word ERP",
    )
    haskey(pretty, String(dataset_key)) && return pretty[String(dataset_key)]

    cleaned = replace(String(dataset_key), r"^\d+_+" => "")
    cleaned = replace(cleaned, r"^new_+" => "")
    cleaned = replace(cleaned, r"_new_" => "_")
    return replace(cleaned, "_" => " ")
end

function display_sort_variable(sort_variable::AbstractString)
    return replace(String(sort_variable), "_" => " ")
end

function stable_slug(x)
    y = lowercase(String(x))
    y = replace(y, r"[^a-z0-9]+" => "_")
    y = replace(y, r"(^_+|_+$)" => "")
    return isempty(y) ? "item" : y
end

function normalize_reference_sort_variables!(df::DataFrame)
    for row in eachrow(df)
        if cellstr(row.dataset_key) == REFERENCE_DATASET_KEY && cellstr(row.sort_variable) == "latency"
            row.sort_variable = "rt_ms"
        end
    end
    return df
end

function load_labelled_annotations(; annotations_csv::AbstractString = ANNOTATIONS_CSV)
    isfile(annotations_csv) || error("Missing annotations CSV: $(annotations_csv). Run update_labelstudio_annotation_tracking.py first.")
    raw = CSV.read(annotations_csv, DataFrame)

    required = [
        :dataset_key,
        :dataset_label,
        :channel_name,
        :channel_idx,
        :sort_variable,
        :label_status,
        :erp_class,
        :annotation_updated_at,
        :annotation_id,
    ]
    missing_cols = [col for col in required if !(col in propertynames(raw))]
    isempty(missing_cols) || error("Annotation CSV is missing columns: $(join(string.(missing_cols), ", "))")

    keep = [
        cellstr(row.label_status) == "classified" &&
        cellstr(row.erp_class) in PATTERN_CLASSES &&
        !(cellstr(row.dataset_key) in EXCLUDED_DATASET_KEYS)
        for row in eachrow(raw)
    ]
    labels = copy(raw[keep, :])

    for col in [:dataset_key, :dataset_label, :channel_name, :sort_variable, :erp_class]
        labels[!, col] = cellstr.(labels[!, col])
    end
    labels.channel_idx_int = [parse_int_cell(v) for v in labels.channel_idx]
    normalize_reference_sort_variables!(labels)

    # Keep the latest manual annotation for a channel/sort origin. The tracking
    # table is append-like across Label Studio batches, so this guards against
    # stale duplicates without changing any source files.
    sort!(
        labels,
        [:dataset_key, :channel_name, :sort_variable, :annotation_updated_at, :annotation_id],
        rev = [false, false, false, true, true],
    )
    deduplicated = labels[.!nonunique(labels, [:dataset_key, :channel_name, :sort_variable]), :]
    sort!(deduplicated, [:erp_class, :dataset_key, :sort_variable, :channel_name])
    deduplicated.source_row_id = collect(1:nrow(deduplicated))
    return deduplicated
end

function shuffled_copy(items::AbstractVector, rng::AbstractRNG)
    values = collect(items)
    length(values) <= 1 && return values
    return values[randperm(rng, length(values))]
end

function select_indices_round_robin_by_dataset(labels::DataFrame, class_indices::Vector{Int};
        class_rank::Int,
        examples_per_class::Int,
        seed::Int)
    rng = MersenneTwister(seed + 1009 * class_rank)
    sorted_indices = sort(class_indices; by = idx -> (
        cellstr(labels.dataset_key[idx]),
        cellstr(labels.sort_variable[idx]),
        cellstr(labels.channel_name[idx]),
        parse_int_cell(labels.annotation_id[idx]),
    ))

    datasets = sort(unique(cellstr.(labels.dataset_key[sorted_indices])))
    datasets = shuffled_copy(datasets, rng)
    by_dataset = Dict{String, Vector{Int}}()
    for dataset_key in datasets
        idxs = [idx for idx in sorted_indices if cellstr(labels.dataset_key[idx]) == dataset_key]
        by_dataset[dataset_key] = shuffled_copy(idxs, rng)
    end

    selected = Int[]
    while length(selected) < examples_per_class
        progressed = false
        for dataset_key in datasets
            bucket = by_dataset[dataset_key]
            isempty(bucket) && continue
            push!(selected, popfirst!(bucket))
            progressed = true
            length(selected) == examples_per_class && break
        end
        progressed || break
    end
    return selected
end

function select_labelled_examples(labels::DataFrame;
        examples_per_class::Int = EXAMPLES_PER_CLASS,
        seed::Int = RNG_SEED)
    rows = Int[]
    for (class_rank, cls) in enumerate(PATTERN_CLASSES)
        class_indices = findall(==(cls), cellstr.(labels.erp_class))
        isempty(class_indices) && error("No manually labelled examples found for class $(cls).")
        selected = select_indices_round_robin_by_dataset(
            labels,
            class_indices;
            class_rank = class_rank,
            examples_per_class = min(examples_per_class, length(class_indices)),
            seed = seed,
        )
        append!(rows, selected)
    end

    selected_df = copy(labels[rows, :])
    selected_df.figure_row = zeros(Int, nrow(selected_df))
    selected_df.figure_col = zeros(Int, nrow(selected_df))
    for (class_rank, cls) in enumerate(PATTERN_CLASSES)
        idxs = findall(==(cls), cellstr.(selected_df.erp_class))
        for (col, idx) in enumerate(idxs)
            selected_df.figure_row[idx] = class_rank
            selected_df.figure_col[idx] = col
        end
    end
    sort!(selected_df, [:figure_row, :figure_col])
    return selected_df
end

function available_examples_by_class(labels::DataFrame)
    rows = NamedTuple[]
    for cls in PATTERN_CLASSES
        idxs = findall(==(cls), cellstr.(labels.erp_class))
        push!(rows, (
            pattern_class = cls,
            available_manual_labels = length(idxs),
            selected_by_default = min(EXAMPLES_PER_CLASS, length(idxs)),
        ))
    end
    return DataFrame(rows)
end

function select_examples_by_specs(labels::DataFrame, specs = THESIS_EXPORT_SPECS)
    rows = Int[]
    for spec in specs
        idx = findfirst(eachindex(labels.erp_class)) do i
            cellstr(labels.erp_class[i]) == String(spec.erp_class) &&
            cellstr(labels.dataset_key[i]) == String(spec.dataset_key) &&
            cellstr(labels.channel_name[i]) == String(spec.channel_name) &&
            cellstr(labels.sort_variable[i]) == String(spec.sort_variable)
        end
        idx === nothing && error(
            "Missing labelled example: class=$(spec.erp_class), dataset=$(spec.dataset_key), " *
            "channel=$(spec.channel_name), sort=$(spec.sort_variable).",
        )
        push!(rows, idx)
    end

    selected_df = copy(labels[rows, :])
    selected_df.figure_row = collect(1:nrow(selected_df))
    selected_df.figure_col = ones(Int, nrow(selected_df))
    return selected_df
end

function baseline_correct_for_dataset(dataset_key::AbstractString)
    return get(BASELINE_CORRECT_BY_DATASET, String(dataset_key), false)
end

function build_reconstruction_context()
    return (
        bundle_cache = Dict{String, Any}(),
        subject_cache = Dict{String, Any}(),
        origin_cache = Dict{Tuple{String, String}, Any}(),
        reference_origin_cache = Dict{String, Any}(),
    )
end

function reference_channel_idx(channel_name::AbstractString, fallback_idx::Integer)
    m = match(r"^ch(\d+)$", String(channel_name))
    if m !== nothing
        return parse(Int, m.captures[1])
    end
    fallback_idx > 0 || error("Cannot infer reference channel index from $(channel_name).")
    return Int(fallback_idx)
end

function load_reference_events()
    isfile(REFERENCE_EVENTS_PATH) || error("Missing reference events file: $(REFERENCE_EVENTS_PATH)")
    events = CSV.read(REFERENCE_EVENTS_PATH, DataFrame)
    if !(:subject_label in propertynames(events))
        events.subject_label = fill("reference_fixations", nrow(events))
    end
    if !(:epoch_index in propertynames(events))
        events.epoch_index = collect(1:nrow(events))
    end
    if :latency in propertynames(events) && !(:rt_ms in propertynames(events))
        events.rt_ms = copy(events.latency)
    end
    return events
end

function reference_origin(row, ctx)
    channel_name = cellstr(row.channel_name)
    return get!(ctx.reference_origin_cache, channel_name) do
        isfile(REFERENCE_H5_PATH) || error("Missing reference HDF5 file: $(REFERENCE_H5_PATH)")
        events = load_reference_events()
        channel_idx = reference_channel_idx(channel_name, Int(row.channel_idx_int))
        data_time_trials = h5open(REFERENCE_H5_PATH, "r") do fid
            dataset = fid["data"]["data_fixations.hdf5"]
            1 <= channel_idx <= size(dataset, 1) || error("Reference channel index out of range: $(channel_idx).")
            n = min(size(dataset, 3), nrow(events))
            Float32.(dataset[channel_idx, REFERENCE_TIME_ZERO_IDX:size(dataset, 2), 1:n])
        end
        n = min(size(data_time_trials, 2), nrow(events))
        return (
            data_time_trials = Matrix{Float32}(data_time_trials[:, 1:n]),
            events = events[1:n, :],
            subject_label = "reference_fixations",
            channel_idx = channel_idx,
            n_trials = n,
            n_timepoints_post = size(data_time_trials, 1),
            time_start_s = 0.0f0,
            time_end_s = Float32((size(data_time_trials, 1) - 1) / REFERENCE_SAMPLING_RATE),
            sampling_rate_hz = Float64(REFERENCE_SAMPLING_RATE),
            baseline_correct = false,
            source_h5 = REFERENCE_H5_PATH,
            source_events = REFERENCE_EVENTS_PATH,
        )
    end
end

function standard_origin(row, ctx)
    dataset_key = cellstr(row.dataset_key)
    channel_name = cellstr(row.channel_name)
    key = (dataset_key, channel_name)
    return get!(ctx.origin_cache, key) do
        bundle = get!(ctx.bundle_cache, dataset_key) do
            Week15.load_clean_dataset_bundle(dataset_key)
        end
        subject_caches = get!(ctx.subject_cache, dataset_key) do
            Export.load_subject_cache(bundle)
        end
        origin = Export.merged_channel_trials_from_cache(
            bundle,
            subject_caches,
            channel_name;
            baseline_correct = baseline_correct_for_dataset(dataset_key),
        )
        return merge(origin, (
            baseline_correct = baseline_correct_for_dataset(dataset_key),
            source_h5 = String(bundle.h5_path),
            source_events = String(bundle.events_path),
        ))
    end
end

function origin_for_label(row, ctx)
    dataset_key = cellstr(row.dataset_key)
    dataset_key == REFERENCE_DATASET_KEY && return reference_origin(row, ctx)
    return standard_origin(row, ctx)
end

function reconstruct_full_resolution_examples(selected_labels::DataFrame)
    ctx = build_reconstruction_context()
    samples = NamedTuple[]
    for row in eachrow(selected_labels)
        origin = origin_for_label(row, ctx)
        sort_col = Symbol(cellstr(row.sort_variable))
        sort_col in propertynames(origin.events) || error(
            "Sort column $(sort_col) missing for $(row.dataset_key) $(row.channel_name)."
        )
        img = Export.smooth_erp_image(origin.data_time_trials, origin.events, sort_col)
        push!(samples, (
            erp_class = cellstr(row.erp_class),
            dataset_key = cellstr(row.dataset_key),
            dataset_label = cellstr(row.dataset_label),
            channel_name = cellstr(row.channel_name),
            channel_idx = Int(origin.channel_idx),
            sort_variable = String(sort_col),
            figure_row = Int(row.figure_row),
            figure_col = Int(row.figure_col),
            label_studio_task_id = hasproperty(row, :label_studio_task_id) ? cellstr(row.label_studio_task_id) : "",
            annotation_id = hasproperty(row, :annotation_id) ? cellstr(row.annotation_id) : "",
            annotation_updated_at = hasproperty(row, :annotation_updated_at) ? cellstr(row.annotation_updated_at) : "",
            tracking_key = hasproperty(row, :tracking_key) ? cellstr(row.tracking_key) : "",
            n_trials = Int(size(img, 1)),
            n_timepoints = Int(size(img, 2)),
            time_start_s = Float32(origin.time_start_s),
            time_end_s = Float32(origin.time_end_s),
            sampling_rate_hz = Float64(origin.sampling_rate_hz),
            baseline_correct = Bool(origin.baseline_correct),
            source_h5 = String(origin.source_h5),
            source_events = String(origin.source_events),
            image = Matrix{Float32}(img),
        ))
    end
    return samples
end

function format_time_tick(t::Real)
    ms = round(Int, 1000 * Float64(t))
    abs(ms) >= 1000 && return @sprintf("%.2f", Float64(t))
    return @sprintf("%.1f", Float64(t))
end

function time_axis_ticks(start_s::Real, end_s::Real)
    values = unique(Float64[start_s, (Float64(start_s) + Float64(end_s)) / 2, end_s])
    return (values, format_time_tick.(values))
end

function time_axis_ticks_with_timepoints(sample)
    times = unique(Float64[sample.time_start_s, (Float64(sample.time_start_s) + Float64(sample.time_end_s)) / 2, sample.time_end_s])
    start_s = Float64(sample.time_start_s)
    end_s = Float64(sample.time_end_s)
    denom = max(end_s - start_s, eps(Float64))
    labels = [
        begin
            timepoint = clamp(round(Int, 1 + (t - start_s) / denom * (sample.n_timepoints - 1)), 1, sample.n_timepoints)
            "$(format_time_tick(t))\n$(timepoint)"
        end
        for t in times
    ]
    return (times, labels)
end

function trial_axis_ticks(n_trials::Integer)
    values = unique([1, Int(round((Int(n_trials) + 1) / 2)), Int(n_trials)])
    return (values, string.(values))
end

function subplot_title(sample)
    return @sprintf(
        "%s\n%s\nch=%s | sort=%s",
        humanize_class(sample.erp_class),
        display_dataset_key(sample.dataset_key),
        sample.channel_name,
        display_sort_variable(sample.sort_variable),
    )
end

function plot_single_erp_image(sample; figure_size = (900, 650))
    stats = Export.clipped_color_stats_quantile_zero_ticks(Float32.(sample.image))
    img = stats[1]
    colorrange = stats[2]
    tick_vals = stats[3]
    tick_labels = stats[4]
    cmap = stats[5]

    fig = Figure(size = figure_size, figure_padding = 24)
    ax = Axis(fig[1, 1];
        title = subplot_title(sample),
        titlesize = 34,
        xlabel = "time after onset (s)\ntimepoint",
        ylabel = "sorted trials",
        xlabelsize = 30,
        ylabelsize = 30,
        xticklabelsize = 26,
        yticklabelsize = 26,
        xticks = time_axis_ticks_with_timepoints(sample),
        yticks = trial_axis_ticks(sample.n_trials),
    )
    hm = heatmap!(
        ax,
        range(Float64(sample.time_start_s), Float64(sample.time_end_s); length = size(img, 2)),
        range(1, Float64(sample.n_trials); length = size(img, 1)),
        permutedims(img, (2, 1));
        colormap = cmap,
        colorrange = colorrange,
        rasterize = true,
    )
    Colorbar(
        fig[1, 2],
        hm;
        ticks = (tick_vals, tick_labels),
        ticklabelsize = 24,
        width = 26,
    )
    colgap!(fig.layout, 14)
    resize_to_layout!(fig)
    return fig
end

function export_filename(sample, idx::Integer, ext::AbstractString)
    parts = [
        @sprintf("%02d", idx),
        sample.erp_class,
        display_dataset_key(sample.dataset_key),
        sample.channel_name,
        sample.sort_variable,
    ]
    return join(stable_slug.(parts), "__") * "." * String(ext)
end

function export_individual_erp_svgs(samples;
        output_dir::AbstractString = THESIS_EXPORT_DIR,
        formats = ("svg",))
    mkpath(output_dir)
    table = selected_examples_table(samples)
    exported = String[]

    for (idx, sample) in enumerate(samples)
        fig = plot_single_erp_image(sample)
        for ext in formats
            path = joinpath(output_dir, export_filename(sample, idx, String(ext)))
            CairoMakie.save(path, fig)
            push!(exported, path)
        end
    end

    CSV.write(joinpath(output_dir, "exported_examples.csv"), table)
    return exported
end

function display_class_examples(samples, pattern_class::AbstractString)
    class_samples = [sample for sample in samples if sample.erp_class == pattern_class]
    isempty(class_samples) && error("No reconstructed examples for class $(pattern_class).")
    for (idx, sample) in enumerate(class_samples)
        display(plot_single_erp_image(sample))
    end
    return selected_examples_table(class_samples)
end

function selected_examples_table(samples)
    rows = NamedTuple[]
    for sample in samples
        push!(rows, (
            pattern_class = sample.erp_class,
            dataset_key = sample.dataset_key,
            channel = sample.channel_name,
            sort_variable = sample.sort_variable,
            n_trials = sample.n_trials,
            n_timepoints = sample.n_timepoints,
            time_start_s = sample.time_start_s,
            time_end_s = sample.time_end_s,
            baseline_correct = sample.baseline_correct,
            label_studio_task_id = sample.label_studio_task_id,
            annotation_id = sample.annotation_id,
        ))
    end
    table = DataFrame(rows)
    class_order = Dict(cls => idx for (idx, cls) in enumerate(PATTERN_CLASSES))
    table.__class_order = [class_order[row.pattern_class] for row in eachrow(table)]
    sort!(table, [:__class_order, :dataset_key, :channel, :sort_variable])
    select!(table, Not(:__class_order))
    return table
end

function run_pattern_example_pipeline(;
        annotations_csv::AbstractString = ANNOTATIONS_CSV,
        examples_per_class::Int = EXAMPLES_PER_CLASS,
        seed::Int = RNG_SEED)
    labels = load_labelled_annotations(annotations_csv = annotations_csv)
    selected = select_labelled_examples(labels; examples_per_class = examples_per_class, seed = seed)
    samples = reconstruct_full_resolution_examples(selected)
    table = selected_examples_table(samples)
    return (
        labels = labels,
        selected_labels = selected,
        samples = samples,
        selected_examples = table,
    )
end

function main()
    result = run_pattern_example_pipeline()
    show(result.selected_examples; allrows = true, allcols = true)
    println()
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .Week23ERPPatternExamples
    Week23ERPPatternExamples.main()
end
