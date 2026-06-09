module Week23ERPDataBuild

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

using CSV
using DataFrames
using HDF5
using JLD2

include(joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_erp_export_helpers.jl"))
using .Week21LabelStudioERPExport

const Week21 = Week21LabelStudioERPExport
const Week15 = Week21.Week15TryNewData

const ANNOTATIONS_CSV = joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_annotations_all.csv")
const ERP_DATASETS_ROOT = joinpath(REPO_ROOT, "datasets")
const REFERENCE_DATASET_KEY = "fixations_dataset"
const REFERENCE_DATA_DIR = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY)
const REFERENCE_H5_PATH = joinpath(REFERENCE_DATA_DIR, "data_fixations.hdf5")
const REFERENCE_EVENTS_PATH = joinpath(REFERENCE_DATA_DIR, "events.csv")
const REFERENCE_SAMPLING_RATE = 512.0
const REFERENCE_PRE_STIM_S = 0.5f0
const REFERENCE_TIME_ZERO_IDX = Int(round(Float64(REFERENCE_PRE_STIM_S) * REFERENCE_SAMPLING_RATE)) + 1

const EXPECTED_DATASET_KEYS = [
    "02_new_eegeyenet_saccades",
    "02_new_roamm_reading",
    "02_new_unfold_facefreeview",
    "eegeyenet_saccades",
    "erp_core_n170_clean",
    "erp_core_n2pc_clean",
    "eye_eeg_freeviewing_fixations",
    "eye_eeg_reading_fixations",
    "eye_eeg_sceneviewing_tobii_fixations",
    "fixations_dataset",
    "kilo_word_erp",
    "nod_eeg_public",
]

const EXCLUDED_DATASET_KEYS = Set([
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

function erp_core_primary_citation()
    return Dict{String, Any}(
        "authors" => "Kappenman, Farrens, Zhang, Stewart, Luck",
        "year" => 2021,
        "title" => "ERP CORE: An open resource for human event-related potential research",
        "venue" => "NeuroImage 225, 117465",
        "doi" => "10.1016/j.neuroimage.2020.117465",
    )
end

const CITATIONS = Dict{String, Any}(
    "fixations_dataset" => Dict{String, Any}(
        "status" => "TODO",
        "note" => "Reference Fixation Dataset - citation pending",
    ),
    "erp_core_n170_clean" => Dict{String, Any}(
        "primary" => erp_core_primary_citation(),
        "data" => Dict{String, Any}(
            "authors" => "Kappenman, Farrens, Zhang, Stewart, Luck",
            "year" => 2020,
            "title" => "ERP CORE N170: Data Files and Analysis Scripts",
            "url" => "https://osf.io/pfde9/",
        ),
    ),
    "erp_core_n2pc_clean" => Dict{String, Any}(
        "primary" => erp_core_primary_citation(),
        "data" => Dict{String, Any}(
            "authors" => "Kappenman, Farrens, Zhang, Stewart, Luck",
            "year" => 2020,
            "title" => "ERP CORE N2pc: Data Files and Analysis Scripts",
            "url" => "https://osf.io/yefrq/",
        ),
    ),
    "eye_eeg_reading_fixations" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Dimigen, Sommer, Hohlfeld, Jacobs, Kliegl",
            "year" => 2011,
            "title" => "Coregistration of eye movements and EEG in natural reading: Analyses and review",
            "venue" => "J. Exp. Psychol.: General 140(4), 552-572",
            "doi" => "10.1037/a0023885",
        ),
        "data" => Dict{String, Any}(
            "authors" => "Dimigen",
            "year" => 2021,
            "title" => "EYE-EEG: Download test datasets",
            "url" => "https://www.eyetracking-eeg.org/testdata.html",
        ),
    ),
    "eye_eeg_freeviewing_fixations" => Dict{String, Any}(
        "data" => Dict{String, Any}(
            "authors" => "Dimigen",
            "year" => 2021,
            "title" => "EYE-EEG: Download test datasets",
            "url" => "https://www.eyetracking-eeg.org/testdata.html",
        ),
    ),
    "eye_eeg_sceneviewing_tobii_fixations" => Dict{String, Any}(
        "data" => Dict{String, Any}(
            "authors" => "Dimigen",
            "year" => 2021,
            "title" => "EYE-EEG: Download test datasets",
            "url" => "https://www.eyetracking-eeg.org/testdata.html",
        ),
    ),
    "eegeyenet_saccades" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Kastrati, Płomecka, Pascual, Wolf, Gillioz, Wattenhofer, Langer",
            "year" => 2021,
            "title" => "EEGEyeNet: A Simultaneous EEG and Eye-Tracking Dataset and Benchmark for Eye Movement Prediction",
            "venue" => "NeurIPS Datasets & Benchmarks",
            "doi" => "10.48550/arXiv.2111.05100",
        ),
        "data_osf" => Dict{String, Any}(
            "url" => "https://osf.io/ktv7m/",
        ),
        "data_openneuro" => Dict{String, Any}(
            "authors" => "Płomecka, Kastrati, Langer",
            "year" => 2025,
            "title" => "EEGEyeNet Dataset",
            "doi" => "10.18112/openneuro.ds005872.v1.0.0",
        ),
    ),
    "02_new_eegeyenet_saccades" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Kastrati, Płomecka, Pascual, Wolf, Gillioz, Wattenhofer, Langer",
            "year" => 2021,
            "title" => "EEGEyeNet: A Simultaneous EEG and Eye-Tracking Dataset and Benchmark for Eye Movement Prediction",
            "venue" => "NeurIPS Datasets & Benchmarks",
            "doi" => "10.48550/arXiv.2111.05100",
        ),
        "data_osf" => Dict{String, Any}(
            "url" => "https://osf.io/ktv7m/",
        ),
        "data_openneuro" => Dict{String, Any}(
            "authors" => "Płomecka, Kastrati, Langer",
            "year" => 2025,
            "title" => "EEGEyeNet Dataset",
            "doi" => "10.18112/openneuro.ds005872.v1.0.0",
        ),
    ),
    "02_new_roamm_reading" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Data on the Brain and Mind Tutorial Track",
            "year" => 2025,
            "title" => "Reading Observed At Mindless Moments (ROAMM)",
            "url" => "https://data-brain-mind.github.io/tutorials/reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/",
        ),
        "data" => Dict{String, Any}(
            "url" => "https://osf.io/kmvgb/",
        ),
    ),
    "02_new_unfold_facefreeview" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Ehinger, Dimigen",
            "year" => 2019,
            "title" => "Unfold: An integrated toolbox for overlap correction, non-linear modeling, and regression-based EEG analysis",
            "venue" => "PeerJ 7, e7838",
            "doi" => "10.7717/peerj.7838",
        ),
        "data" => Dict{String, Any}(
            "authors" => "Ehinger, Dimigen",
            "year" => 2018,
            "url" => "https://osf.io/wbz7x/",
        ),
    ),
    "kilo_word_erp" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Dufau, Grainger, Midgley, Holcomb",
            "year" => 2015,
            "title" => "A Thousand Words Are Worth a Picture: Snapshots of Printed-Word Processing in an Event-Related Potential Megastudy",
            "venue" => "Psychological Science 26(12), 1887-1897",
            "doi" => "10.1177/0956797615603934",
        ),
        "data" => Dict{String, Any}(
            "authors" => "Dufau et al.",
            "year" => 2015,
            "title" => "The Kilo-Word ERP Database: Lexical Decision",
            "url" => "https://osf.io/72b89/",
        ),
    ),
    "nod_eeg_public" => Dict{String, Any}(
        "primary" => Dict{String, Any}(
            "authors" => "Zhang, Zhou, Zhen, Tang, Li, Zhen",
            "year" => 2025,
            "title" => "A Large-Scale MEG and EEG Dataset for Object Recognition in Naturalistic Scenes",
            "venue" => "Scientific Data 12(1), 857",
            "doi" => "10.1038/s41597-025-05174-7",
        ),
        "data" => Dict{String, Any}(
            "authors" => "Zhang et al.",
            "year" => 2025,
            "title" => "NOD-EEG",
            "doi" => "10.18112/openneuro.ds005811.v1.0.9",
        ),
    ),
)

function cell_string(x)
    return ismissing(x) ? "" : String(x)
end

function normalize_reference_sort_variables!(df::DataFrame)
    has_reference_rt = any(
        (cell_string(row.dataset_key) == REFERENCE_DATASET_KEY) &&
        (cell_string(row.sort_variable) == "rt_ms")
        for row in eachrow(df)
    )
    has_reference_rt && return df

    for row in eachrow(df)
        if cell_string(row.dataset_key) == REFERENCE_DATASET_KEY && cell_string(row.sort_variable) == "latency"
            row.sort_variable = "rt_ms"
        end
    end
    return df
end

function load_latest_labels(; annotations_csv::AbstractString = ANNOTATIONS_CSV)
    isfile(annotations_csv) || error("Missing annotations CSV: $(annotations_csv).")
    raw = CSV.read(annotations_csv, DataFrame)
    keep = [
        cell_string(row.label_status) == "classified" &&
        !(cell_string(row.dataset_key) in EXCLUDED_DATASET_KEYS)
        for row in eachrow(raw)
    ]
    labels = copy(raw[keep, :])
    labels.dataset_key = cell_string.(labels.dataset_key)
    labels.dataset_label = cell_string.(labels.dataset_label)
    labels.channel_name = cell_string.(labels.channel_name)
    labels.sort_variable = cell_string.(labels.sort_variable)
    labels.erp_class = cell_string.(labels.erp_class)
    normalize_reference_sort_variables!(labels)

    sort!(labels,
        [:dataset_key, :channel_name, :sort_variable, :annotation_updated_at, :annotation_id],
        rev = [false, false, false, true, true],
    )
    deduplicated = labels[.!nonunique(labels, [:dataset_key, :channel_name, :sort_variable]), :]
    out = select(deduplicated, [:dataset_key, :dataset_label, :channel_name, :channel_idx, :sort_variable, :erp_class])
    sort!(out, [:dataset_key, :sort_variable, :channel_name])

    observed = sort(unique(String.(out.dataset_key)))
    expected = sort(EXPECTED_DATASET_KEYS)
    observed == expected || error(
        "Unexpected labelled dataset keys. Expected $(join(expected, ", ")), got $(join(observed, ", "))."
    )
    return out
end

function labels_for_storage(labels::DataFrame)
    out = select(labels, [:channel_name, :sort_variable, :erp_class])
    out.channel_name = String.(out.channel_name)
    out.sort_variable = String.(out.sort_variable)
    out.erp_class = String.(out.erp_class)
    return out
end

function dataset_dir(dataset_key::AbstractString; output_root::AbstractString = ERP_DATASETS_ROOT)
    return joinpath(output_root, dataset_key)
end

function signal_path(dataset_key::AbstractString, channel_name::AbstractString;
        output_root::AbstractString = ERP_DATASETS_ROOT)
    return joinpath(dataset_dir(dataset_key; output_root = output_root), "signals", string(channel_name, ".jld2"))
end

function prepare_output_dirs!(dataset_key::AbstractString; output_root::AbstractString = ERP_DATASETS_ROOT)
    dir = dataset_dir(dataset_key; output_root = output_root)
    signals_dir = joinpath(dir, "signals")
    mkpath(signals_dir)
    for path in readdir(signals_dir; join = true)
        isfile(path) && endswith(path, ".jld2") && rm(path; force = true)
    end
    return dir, signals_dir
end

function canonical_trial_order(events::DataFrame)
    for col in [:subject_label, :epoch_index]
        col in propertynames(events) || error("Events table is missing canonical sort column $(col).")
    end
    order = collect(1:nrow(events))
    sort!(order; by = idx -> (events.subject_label[idx], events.epoch_index[idx], idx))
    return order
end

function canonical_events_from_caches(subject_caches)
    frames = DataFrame[cache.events for cache in subject_caches]
    events = vcat(frames...; cols = :union)
    order = canonical_trial_order(events)
    return events[order, :]
end

function assert_trial_alignment(saved_events::DataFrame, channel_events::DataFrame)
    nrow(saved_events) == nrow(channel_events) ||
        error("Trial count mismatch between events.jld2 and signal source events.")
    for col in [:subject_label, :epoch_index]
        saved_events[!, col] == channel_events[!, col] ||
            error("Trial order mismatch for canonical column $(col).")
    end
    return nothing
end

function source_pre_stim_s(times_s::AbstractVector{<:Real})
    return Float32(max(0.0, -Float64(first(times_s))))
end

function string_sort_columns(cols)
    return String.(string.(cols))
end

function dataset_label(labels::DataFrame, dataset_key::AbstractString)
    values = unique(String.(labels.dataset_label))
    values = filter(!isempty, values)
    isempty(values) && return String(dataset_key)
    return first(values)
end

function dataset_citation(dataset_key::AbstractString)
    haskey(CITATIONS, dataset_key) || error("Missing citation metadata for $(dataset_key).")
    return CITATIONS[dataset_key]
end

function write_events_file(path::AbstractString, events::DataFrame, metadata::Dict{String, Any})
    JLD2.jldsave(path; events = events, metadata = metadata)
    return path
end

function write_labels_file(path::AbstractString, labels::DataFrame, dataset_key::AbstractString)
    metadata = Dict{String, Any}("dataset_key" => String(dataset_key))
    JLD2.jldsave(path; labels = labels_for_storage(labels), metadata = metadata)
    return path
end

function write_signal_file(path::AbstractString, data_time_trials::Matrix{Float32},
        dataset_key::AbstractString, channel_name::AbstractString, channel_idx::Integer)
    metadata = Dict{String, Any}(
        "dataset_key" => String(dataset_key),
        "channel_name" => String(channel_name),
        "channel_idx" => Int(channel_idx),
    )
    JLD2.jldsave(path; data_time_trials = data_time_trials, metadata = metadata)
    return path
end

function standard_events_metadata(dataset_key::AbstractString, label::AbstractString,
        bundle, subject_caches, events::DataFrame)
    cache = first(subject_caches)
    post_times = Float32.(cache.times_s[cache.post_idx])
    return Dict{String, Any}(
        "dataset_key" => String(dataset_key),
        "dataset_label" => String(label),
        "citation" => dataset_citation(dataset_key),
        "sampling_rate_hz" => Float64(cache.sfreq_hz),
        "time_start_s" => Float32(first(post_times)),
        "time_end_s" => Float32(last(post_times)),
        "n_trials" => Int(nrow(events)),
        "n_timepoints_post" => Int(length(cache.post_idx)),
    )
end

function build_standard_dataset!(dataset_key::AbstractString, labels::DataFrame;
        output_root::AbstractString = ERP_DATASETS_ROOT)
    dir, _ = prepare_output_dirs!(dataset_key; output_root = output_root)
    bundle = Week15.load_clean_dataset_bundle(dataset_key)
    subject_caches = Week21.load_subject_cache(bundle)
    events = canonical_events_from_caches(subject_caches)
    label = dataset_label(labels, dataset_key)
    metadata = standard_events_metadata(dataset_key, label, bundle, subject_caches, events)

    write_events_file(joinpath(dir, "events.jld2"), events, metadata)
    write_labels_file(joinpath(dir, "labels.jld2"), labels, dataset_key)

    channel_names = sort(unique(String.(labels.channel_name)))
    for channel_name in channel_names
        origin = Week21.merged_channel_trials_from_cache(
            bundle,
            subject_caches,
            channel_name;
            baseline_correct = Bool(BASELINE_CORRECT_BY_DATASET[dataset_key]),
        )
        order = canonical_trial_order(origin.events)
        channel_events = origin.events[order, :]
        assert_trial_alignment(events, channel_events)
        data_time_trials = Matrix{Float32}(origin.data_time_trials[:, order])
        write_signal_file(
            signal_path(dataset_key, channel_name; output_root = output_root),
            data_time_trials,
            dataset_key,
            channel_name,
            Int(origin.channel_idx),
        )
    end

    return (
        dataset_key = String(dataset_key),
        n_trials = nrow(events),
        n_timepoints_post = Int(metadata["n_timepoints_post"]),
        n_channels = length(channel_names),
        n_labels = nrow(labels),
    )
end

function reference_channel_idx(channel_name::AbstractString)
    m = match(r"^ch(\d+)$", String(channel_name))
    m === nothing && error("Reference channel name must look like ch001, got $(channel_name).")
    return parse(Int, m.captures[1])
end

function load_reference_events()
    events = CSV.read(REFERENCE_EVENTS_PATH, DataFrame)
    events.subject_label = fill("reference_fixations", nrow(events))
    events.epoch_index = collect(1:nrow(events))
    if :latency in propertynames(events) && !(:rt_ms in propertynames(events))
        events.rt_ms = copy(events.latency)
    end
    order = canonical_trial_order(events)
    return events[order, :]
end

function load_reference_epochs()
    return h5open(REFERENCE_H5_PATH, "r") do fid
        Float32.(read(fid["data"]["data_fixations.hdf5"]))
    end
end

function reference_events_metadata(dataset_key::AbstractString, label::AbstractString,
        events::DataFrame, n_timepoints_post::Int)
    return Dict{String, Any}(
        "dataset_key" => String(dataset_key),
        "dataset_label" => String(label),
        "citation" => dataset_citation(dataset_key),
        "sampling_rate_hz" => Float64(REFERENCE_SAMPLING_RATE),
        "time_start_s" => 0.0f0,
        "time_end_s" => Float32((n_timepoints_post - 1) / REFERENCE_SAMPLING_RATE),
        "n_trials" => Int(nrow(events)),
        "n_timepoints_post" => Int(n_timepoints_post),
    )
end

function build_reference_dataset!(labels::DataFrame; output_root::AbstractString = ERP_DATASETS_ROOT)
    dataset_key = REFERENCE_DATASET_KEY
    dir, _ = prepare_output_dirs!(dataset_key; output_root = output_root)
    events = load_reference_events()
    epochs = load_reference_epochs()
    n_timepoints_post = size(epochs, 2) - REFERENCE_TIME_ZERO_IDX + 1
    label = dataset_label(labels, dataset_key)
    metadata = reference_events_metadata(dataset_key, label, events, n_timepoints_post)

    write_events_file(joinpath(dir, "events.jld2"), events, metadata)
    write_labels_file(joinpath(dir, "labels.jld2"), labels, dataset_key)

    channel_names = sort(unique(String.(labels.channel_name)))
    for channel_name in channel_names
        channel_idx = reference_channel_idx(channel_name)
        1 <= channel_idx <= size(epochs, 1) || error("Reference channel index out of range: $(channel_idx).")
        data_time_trials = Matrix{Float32}(epochs[channel_idx, REFERENCE_TIME_ZERO_IDX:end, :])
        size(data_time_trials, 2) == nrow(events) ||
            error("Reference trial count mismatch for $(channel_name).")
        write_signal_file(
            signal_path(dataset_key, channel_name; output_root = output_root),
            data_time_trials,
            dataset_key,
            channel_name,
            channel_idx,
        )
    end

    return (
        dataset_key = String(dataset_key),
        n_trials = nrow(events),
        n_timepoints_post = n_timepoints_post,
        n_channels = length(channel_names),
        n_labels = nrow(labels),
    )
end

function build_dataset!(dataset_key::AbstractString, labels::DataFrame;
        output_root::AbstractString = ERP_DATASETS_ROOT)
    if dataset_key == REFERENCE_DATASET_KEY
        return build_reference_dataset!(labels; output_root = output_root)
    end
    return build_standard_dataset!(dataset_key, labels; output_root = output_root)
end

function build_erp_datasets!(; annotations_csv::AbstractString = ANNOTATIONS_CSV,
        output_root::AbstractString = ERP_DATASETS_ROOT)
    labels = load_latest_labels(annotations_csv = annotations_csv)
    mkpath(output_root)
    rows = NamedTuple[]
    grouped = groupby(labels, :dataset_key)
    for dataset_key in EXPECTED_DATASET_KEYS
        group_index = findfirst(key -> String(key.dataset_key) == dataset_key, keys(grouped))
        group_index === nothing && error("No labels found for expected dataset $(dataset_key).")
        dataset_labels = DataFrame(grouped[group_index])
        println("Building $(dataset_key): $(nrow(dataset_labels)) labelled triples.")
        push!(rows, build_dataset!(dataset_key, dataset_labels; output_root = output_root))
    end
    summary = DataFrame(rows)
    sort!(summary, :dataset_key)
    return summary
end

function dataset_summary(; output_root::AbstractString = ERP_DATASETS_ROOT)
    rows = NamedTuple[]
    isdir(output_root) || return DataFrame(rows)
    for dataset_key in sort(readdir(output_root))
        dir = joinpath(output_root, dataset_key)
        events_path = joinpath(dir, "events.jld2")
        labels_path = joinpath(dir, "labels.jld2")
        signals_dir = joinpath(dir, "signals")
        isdir(dir) && isfile(events_path) && isfile(labels_path) && isdir(signals_dir) || continue
        events_metadata = JLD2.load(events_path, "metadata")
        labels = JLD2.load(labels_path, "labels")
        signal_count = count(path -> isfile(path) && endswith(path, ".jld2"), readdir(signals_dir; join = true))
        push!(rows, (
            dataset_key = String(dataset_key),
            dataset_label = String(events_metadata["dataset_label"]),
            n_trials = Int(events_metadata["n_trials"]),
            n_timepoints_post = Int(events_metadata["n_timepoints_post"]),
            n_labels = nrow(labels),
            n_signal_files = signal_count,
        ))
    end
    return DataFrame(rows)
end

function main()
    summary = build_erp_datasets!()
    println("Build complete.")
    show(summary; allrows = true, allcols = true)
    println()
    return summary
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .Week23ERPDataBuild
    Week23ERPDataBuild.main()
end
