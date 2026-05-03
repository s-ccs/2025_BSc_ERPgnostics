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
using CUDA
using DataFrames
using Dates
using Flux
using HDF5
using JSON3
using Printf: @sprintf
using Random
using Statistics

include(joinpath(REPO_ROOT, "notebooks", "week_20", "resnet_fixation_generalization_experiment.jl"))
using .Week20ResNetFixationGeneralization

include(joinpath(REPO_ROOT, "notebooks", "week_20", "resnet18_data_source_screening.jl"))
using .Week20ResNet18DataSourceScreening

include(joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_erp_export_helpers.jl"))
using .Week21LabelStudioERPExport

const Generalization = Week20ResNetFixationGeneralization
const Screening = Week20ResNet18DataSourceScreening
const Export = Week21LabelStudioERPExport
const CNNUtils = Generalization.ERPCNNExperimentUtils
const Week15 = Export.Week15TryNewData

const NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_21")
const ANNOTATIONS_CSV = joinpath(NOTEBOOK_DIR, "labelstudio_annotations_all.csv")
const OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_labeled_erp_cv")
const MODEL_NAME = "resnet18_pretrained_labeled_erp_binary"
const K_FOLDS = parse(Int, get(ENV, "WEEK21_RESNET18_FOLDS", "5"))
const TRAIN_EPOCHS = parse(Int, get(ENV, "WEEK21_RESNET18_EPOCHS", string(Generalization.TRAIN_EPOCHS)))
const TRAIN_LR = parse(Float32, get(ENV, "WEEK21_RESNET18_LR", string(Generalization.TRAIN_LR)))
const GLOBAL_SEED = parse(Int, get(ENV, "WEEK21_RESNET18_SEED", "20260501"))
const TARGET_TRIALS = parse(Int, get(ENV, "WEEK21_TARGET_TRIALS", "150"))
const TARGET_SIZE = Generalization.TARGET_SIZE
const NO_CLASS_CHUNKS_PER_ORIGIN = parse(Int, get(ENV, "WEEK21_NO_CLASS_CHUNKS_PER_ORIGIN", "1"))

const CLASS_ID = Dict(
    "no_class" => 0,
    "sigmoid" => 1,
    "one_sided_fan" => 2,
    "two_sided_fan" => 3,
    "diverging_bar" => 4,
    "hourglass" => 5,
    "tilted_bar" => 6,
)
const PATTERN_CLASSES = Set([k for k in keys(CLASS_ID) if k != "no_class"])
const EXCLUDED_DATASET_KEYS = Set([
    # This source was an accidental raw-data test import and should not affect
    # the model trained on the curated labeling batches.
    "02_new_eeget_rsod",
])

const REFERENCE_DATASET_KEY = "fixations_dataset"
const REFERENCE_LABEL = "Reference Fixation Dataset"
const REFERENCE_DATA_DIR = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY)
const REFERENCE_H5_PATH = joinpath(REFERENCE_DATA_DIR, "data_fixations.hdf5")
const REFERENCE_EVENTS_PATH = joinpath(REFERENCE_DATA_DIR, "events.csv")
const REFERENCE_SAMPLING_RATE = 512.0
const REFERENCE_PRE_STIM_S = 0.5
const REFERENCE_TIME_ZERO_IDX = Int(round(REFERENCE_PRE_STIM_S * REFERENCE_SAMPLING_RATE)) + 1

function cellstr(x)
    return ismissing(x) ? "" : string(x)
end

function truthy(x)
    lowercase(strip(cellstr(x))) in ("true", "1", "yes")
end

function stable_slug(x)
    return Export.stable_slug(cellstr(x))
end

function write_json(path::AbstractString, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

function clean_output_dir!(output_dir::AbstractString)
    mkpath(output_dir)
    for child in readdir(output_dir; join = true)
        isfile(child) && rm(child; force = true)
    end
    return output_dir
end

function parse_int_cell(x; default::Int = 0)
    s = strip(cellstr(x))
    isempty(s) && return default
    parsed = tryparse(Int, s)
    parsed === nothing && return default
    return parsed
end

function load_labeled_annotations(; annotations_csv::AbstractString = ANNOTATIONS_CSV)
    isfile(annotations_csv) || error("Missing annotations CSV: $(annotations_csv). Run update_labelstudio_annotation_tracking.py first.")
    df = CSV.read(annotations_csv, DataFrame)

    keep = [
        cellstr(row.label_status) == "classified" &&
        haskey(CLASS_ID, cellstr(row.erp_class)) &&
        !(cellstr(row.dataset_key) in EXCLUDED_DATASET_KEYS)
        for row in eachrow(df)
    ]
    labels = copy(df[keep, :])
    labels.source_row_id = collect(1:nrow(labels))
    labels.erp_class = cellstr.(labels.erp_class)
    labels.dataset_key = cellstr.(labels.dataset_key)
    labels.dataset_label = cellstr.(labels.dataset_label)
    labels.channel_name = cellstr.(labels.channel_name)
    labels.channel_idx_int = [parse_int_cell(v) for v in labels.channel_idx]
    labels.sort_variable = cellstr.(labels.sort_variable)
    labels.tracking_key = cellstr.(labels.tracking_key)
    labels.erp_class_id = [CLASS_ID[cellstr(v)] for v in labels.erp_class]
    labels.binary_label = Int.(labels.erp_class .!= "no_class")
    labels.is_pattern_class = labels.binary_label .== 1

    sort!(labels, [:dataset_key, :sort_variable, :channel_name, :label_studio_task_id])
    labels.source_row_id = collect(1:nrow(labels))
    return labels
end

function source_status_row(source_status_df::DataFrame, dataset_key::AbstractString)
    idx = findfirst(k -> !ismissing(k) && String(k) == String(dataset_key), source_status_df.dataset_key)
    idx === nothing && error("Dataset $(dataset_key) is not present in Week-19 source status.")
    return source_status_df[idx, :]
end

function reference_channel_name(channel_idx::Integer)
    return @sprintf("ch%03d", Int(channel_idx))
end

function load_reference_erps()
    h5open(REFERENCE_H5_PATH, "r") do fid
        return read(fid["data"]["data_fixations.hdf5"])
    end
end

function build_data_context(source_status_df::DataFrame)
    return (
        source_status_df = source_status_df,
        bundle_cache = Dict{String, Any}(),
        subject_cache = Dict{String, Any}(),
        origin_cache = Dict{Tuple{String, String}, Any}(),
        reference_erps = load_reference_erps(),
        reference_events = CSV.read(REFERENCE_EVENTS_PATH, DataFrame),
    )
end

function standard_origin(row, ctx)
    dataset_key = cellstr(row.dataset_key)
    channel_name = cellstr(row.channel_name)
    key = (dataset_key, channel_name)
    return get!(ctx.origin_cache, key) do
        status = source_status_row(ctx.source_status_df, dataset_key)
        bundle = get!(ctx.bundle_cache, dataset_key) do
            Week15.load_clean_dataset_bundle(dataset_key)
        end
        subject_caches = get!(ctx.subject_cache, dataset_key) do
            Export.load_subject_cache(bundle)
        end
        Export.merged_channel_trials_from_cache(
            bundle,
            subject_caches,
            channel_name;
            baseline_correct = Bool(status.baseline_correct),
        )
    end
end

function reference_origin(row, ctx)
    channel_idx = Int(row.channel_idx_int)
    @assert 1 <= channel_idx <= size(ctx.reference_erps, 1) "Reference channel out of range: $(channel_idx)"
    n = min(size(ctx.reference_erps, 3), nrow(ctx.reference_events))
    data_time_trials = Float32.(ctx.reference_erps[channel_idx, REFERENCE_TIME_ZERO_IDX:end, 1:n])
    events = ctx.reference_events[1:n, :]
    return (
        data_time_trials = data_time_trials,
        events = events,
        subject_label = "reference_fixations",
        channel_idx = channel_idx,
        n_trials = nrow(events),
        n_timepoints_post = size(data_time_trials, 1),
        time_start_s = 0.0f0,
        time_end_s = Float32((size(data_time_trials, 1) - 1) / REFERENCE_SAMPLING_RATE),
        sampling_rate_hz = REFERENCE_SAMPLING_RATE,
    )
end

function origin_for_label(row, ctx)
    dataset_key = cellstr(row.dataset_key)
    if dataset_key == REFERENCE_DATASET_KEY
        return reference_origin(row, ctx)
    end
    return standard_origin(row, ctx)
end

function annotate_origin_trials!(labels::DataFrame, ctx)
    n_trials = Int[]
    n_timepoints = Int[]
    for row in eachrow(labels)
        origin = origin_for_label(row, ctx)
        sort_col = Symbol(cellstr(row.sort_variable))
        sort_col in propertynames(origin.events) || error("Sort column $(sort_col) missing for $(row.tracking_key)")
        push!(n_trials, Int(origin.n_trials))
        push!(n_timepoints, Int(origin.n_timepoints_post))
    end
    labels.origin_n_trials = n_trials
    labels.origin_n_timepoints = n_timepoints
    return labels
end

function sorted_order(events::DataFrame, sort_col::Symbol)
    return Week15.trial_sort_order(events, sort_col)
end

function stable_seed(parts...)
    h = UInt64(0xcbf29ce484222325)
    prime = UInt64(0x100000001b3)
    for part in parts
        for b in codeunits(string(part))
            h = (h ⊻ UInt64(b)) * prime
        end
        h = (h ⊻ UInt64(0xff)) * prime
    end
    return Int(mod(h, UInt64(typemax(Int) - 1))) + 1
end

function fill_remainder_indices(order::Vector{Int}, remainder_idxs::Vector{Int}, target_trials::Int,
        source_row_id::Int, sort_variable::AbstractString)

    needed = target_trials - length(remainder_idxs)
    needed <= 0 && return Int[]

    used = Set(remainder_idxs)
    candidates = [idx for idx in order if !(idx in used)]
    if length(candidates) < needed
        candidates = copy(order)
    end

    seed = stable_seed(GLOBAL_SEED, source_row_id, sort_variable, target_trials, length(order))
    rng = MersenneTwister(seed)
    return shuffle(rng, candidates)[1:needed]
end

function target_trial_mod_chunks(order::Vector{Int}, target_trials::Int;
        source_row_id::Int,
        sort_variable::AbstractString)

    n = length(order)
    target_trials <= 0 && error("target_trials must be positive")
    n < target_trials && error("Cannot make fixed-size chunks of $(target_trials) from only $(n) trials")
    full_chunk_count = div(n, target_trials)
    remainder_count = rem(n, target_trials)
    full_chunk_count >= 1 || error("Cannot build mod chunks for n=$(n), target_trials=$(target_trials).")

    full_bins = [Int[] for _ in 1:full_chunk_count]
    remainder = Int[]
    rank = 1

    while rank <= n
        progressed = false
        for bin in full_bins
            if length(bin) < target_trials && rank <= n
                push!(bin, order[rank])
                rank += 1
                progressed = true
            end
        end
        if remainder_count > 0 && length(remainder) < remainder_count && rank <= n
            push!(remainder, order[rank])
            rank += 1
            progressed = true
        end
        progressed || break
    end

    @assert all(length(bin) == target_trials for bin in full_bins) "Full mod chunks did not reach target_trials."
    @assert length(remainder) == remainder_count "Remainder chunk size mismatch."

    chunks = NamedTuple[]
    for (chunk_index, idxs) in enumerate(full_bins)
        push!(chunks, (
            chunk_index = Int(chunk_index),
            chunk_count = 0,
            chunk_role = "full_mod_split",
            full_mod_split_k = Int(full_chunk_count),
            remainder_trials = Int(remainder_count),
            unique_trial_count_before_fill = Int(length(unique(idxs))),
            reused_fill_count = 0,
            filler_indices = Int[],
            trial_indices = copy(idxs),
        ))
    end

    if remainder_count > 0
        filler = fill_remainder_indices(order, remainder, target_trials, source_row_id, sort_variable)
        idxs = vcat(remainder, filler)
        @assert length(idxs) == target_trials "Filled remainder chunk did not reach target_trials."
        push!(chunks, (
            chunk_index = Int(length(chunks) + 1),
            chunk_count = 0,
            chunk_role = "distributed_remainder_filled",
            full_mod_split_k = Int(full_chunk_count),
            remainder_trials = Int(remainder_count),
            unique_trial_count_before_fill = Int(length(unique(remainder))),
            reused_fill_count = Int(length(filler)),
            filler_indices = filler,
            trial_indices = idxs,
        ))
    end

    chunk_count = length(chunks)
    return [merge(c, (chunk_count = chunk_count,)) for c in chunks]
end

function no_class_chunk_indices(chunks, source_row_id::Int)
    isempty(chunks) && return Int[]
    keep_n = min(NO_CLASS_CHUNKS_PER_ORIGIN, length(chunks))
    start = mod(source_row_id - 1, length(chunks)) + 1
    return [mod(start + j - 2, length(chunks)) + 1 for j in 1:keep_n]
end

function preprocess_fixed_trial_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    return CNNUtils.preprocess_pipeline_from_trials(
        data_time_trials,
        events_trials,
        sort_col;
        pipeline_name = :gaussian_reference,
        target_size = TARGET_SIZE,
        low_pass_sigma = Generalization.LOWPASS_SIGMA,
        lowpass_kernel_size = Generalization.LOWPASS_KERNEL_SIZE,
        filter_border = Generalization.FILTER_BORDER,
    )
end

function materialize_augmented_samples(labels::DataFrame, ctx; target_trials::Int)
    rows = NamedTuple[]
    images = Matrix{Float32}[]

    for row in eachrow(labels)
        origin = origin_for_label(row, ctx)
        sort_col = Symbol(cellstr(row.sort_variable))
        order = sorted_order(origin.events, sort_col)
        chunks = target_trial_mod_chunks(
            order,
            target_trials;
            source_row_id = Int(row.source_row_id),
            sort_variable = cellstr(row.sort_variable),
        )
        chunk_idxs = row.binary_label == 1 ? collect(eachindex(chunks)) : no_class_chunk_indices(chunks, Int(row.source_row_id))

        for chunk_idx in chunk_idxs
            chunk = chunks[chunk_idx]
            idxs = chunk.trial_indices
            events_part = origin.events[idxs, :]
            data_part = origin.data_time_trials[:, idxs]
            img = preprocess_fixed_trial_image(data_part, events_part, sort_col)
            variant = chunk.reused_fill_count == 0 ?
                @sprintf("modtarget_%04d_part%03d", target_trials, chunk.chunk_index) :
                @sprintf("modtarget_%04d_remainder%03d_fill%03d", target_trials, chunk.chunk_index, chunk.reused_fill_count)

            push!(rows, (
                sample_id = length(rows) + 1,
                source_row_id = Int(row.source_row_id),
                tracking_key = cellstr(row.tracking_key),
                dataset_key = cellstr(row.dataset_key),
                dataset_label = cellstr(row.dataset_label),
                channel_name = cellstr(row.channel_name),
                channel_idx = Int(row.channel_idx_int),
                sort_variable = cellstr(row.sort_variable),
                erp_class = cellstr(row.erp_class),
                erp_class_id = Int(row.erp_class_id),
                binary_label = Int(row.binary_label),
                label_studio_project_id = cellstr(row.label_studio_project_id),
                label_studio_task_id = cellstr(row.label_studio_task_id),
                annotation_id = cellstr(row.annotation_id),
                image_file = cellstr(row.image_file),
                origin_subject_label = cellstr(origin.subject_label),
                origin_n_trials = Int(origin.n_trials),
                origin_n_timepoints = Int(origin.n_timepoints_post),
                target_trials = Int(target_trials),
                variant = variant,
                chunk_index = Int(chunk.chunk_index),
                chunk_count = Int(chunk.chunk_count),
                chunk_role = String(chunk.chunk_role),
                full_mod_split_k = Int(chunk.full_mod_split_k),
                remainder_trials = Int(chunk.remainder_trials),
                unique_trial_count_before_fill = Int(chunk.unique_trial_count_before_fill),
                reused_fill_count = Int(chunk.reused_fill_count),
                filler_indices = join(string.(chunk.filler_indices), " "),
                n_trials = Int(length(idxs)),
                trial_indices = join(string.(idxs), " "),
            ))
            push!(images, img)
        end
    end

    sample_df = DataFrame(rows)
    sample_df.processed_img = images
    return sample_df
end

function assign_stratified_folds!(sample_df::DataFrame; k::Int = K_FOLDS, seed::Int = GLOBAL_SEED)
    rng = MersenneTwister(seed)
    folds = zeros(Int, nrow(sample_df))
    total_counts = zeros(Int, k)
    class_counts = [Dict{String, Int}() for _ in 1:k]

    function assign!(idx::Int, fold::Int)
        cls = cellstr(sample_df.erp_class[idx])
        folds[idx] = fold
        total_counts[fold] += 1
        class_counts[fold][cls] = get(class_counts[fold], cls, 0) + 1
    end

    for cls in sort(unique(cellstr.(sample_df.erp_class)))
        cls_indices = findall(==(cls), cellstr.(sample_df.erp_class))
        source_ids = unique(Int.(sample_df.source_row_id[cls_indices]))
        source_ids = shuffle(rng, source_ids)

        for source_id in source_ids
            idxs = findall(i -> cellstr(sample_df.erp_class[i]) == cls && Int(sample_df.source_row_id[i]) == source_id, 1:nrow(sample_df))
            idxs = shuffle(rng, idxs)
            used_folds = Set{Int}()
            for idx in idxs
                candidates = [fold for fold in 1:k if !(fold in used_folds)]
                isempty(candidates) && (candidates = collect(1:k))
                best_fold = candidates[1]
                best_key = (typemax(Int), typemax(Int), typemax(Int))
                for fold in candidates
                    key = (
                        get(class_counts[fold], cls, 0),
                        total_counts[fold],
                        fold,
                    )
                    if key < best_key
                        best_key = key
                        best_fold = fold
                    end
                end
                assign!(idx, best_fold)
                push!(used_folds, best_fold)
            end
        end
    end

    sample_df.fold = folds
    @assert all(folds .>= 1) "Some samples were not assigned to a fold."
    return sample_df
end

function fold_distribution_tables(sample_df::DataFrame)
    fold_binary_df = combine(groupby(sample_df, [:fold, :binary_label]), nrow => :count)
    sort!(fold_binary_df, [:fold, :binary_label])

    fold_class_df = combine(groupby(sample_df, [:fold, :erp_class]), nrow => :count)
    sort!(fold_class_df, [:erp_class, :fold])

    fold_dataset_df = combine(groupby(sample_df, [:fold, :dataset_key, :binary_label]), nrow => :count)
    sort!(fold_dataset_df, [:fold, :dataset_key, :binary_label])

    return fold_binary_df, fold_class_df, fold_dataset_df
end

function images_to_tensor(sample_df::DataFrame)
    return CNNUtils.images_to_tensor(sample_df.processed_img)
end

function metrics_for_indices(model, X::Array{Float32, 4}, y::Vector{Int}, idxs::Vector{Int};
        batchsize::Int, device::Function)
    logits, probs = Generalization.predict_logits_probs(model, X[:, :, :, idxs]; batchsize = batchsize, device = device)
    y_true = y[idxs]
    y_pred = [probs[2, i] >= probs[1, i] ? 1 : 0 for i in axes(probs, 2)]
    metrics = CNNUtils.compute_metrics(y_pred, y_true)
    return metrics, logits, probs, y_true, y_pred
end

function prediction_rows(sample_df::DataFrame, idxs::Vector{Int}, fold::Int,
        logits::AbstractMatrix, probs::AbstractMatrix, y_true::Vector{Int}, y_pred::Vector{Int})
    rows = NamedTuple[]
    for (j, idx) in enumerate(idxs)
        r = sample_df[idx, :]
        prob_no = Float32(probs[1, j])
        prob_class = Float32(probs[2, j])
        push!(rows, (
            model_name = MODEL_NAME,
            fold = fold,
            sample_id = Int(r.sample_id),
            source_row_id = Int(r.source_row_id),
            tracking_key = cellstr(r.tracking_key),
            dataset_key = cellstr(r.dataset_key),
            dataset_label = cellstr(r.dataset_label),
            channel_name = cellstr(r.channel_name),
            channel_idx = Int(r.channel_idx),
            sort_variable = cellstr(r.sort_variable),
            erp_class = cellstr(r.erp_class),
            erp_class_id = Int(r.erp_class_id),
            true_binary_label = Int(y_true[j]),
            predicted_binary_label = Int(y_pred[j]),
            predicted_class_binary = y_pred[j] == 1 ? "class" : "no_class",
            correct = Int(y_true[j] == y_pred[j]),
            logit_no_class = Float32(logits[1, j]),
            logit_class = Float32(logits[2, j]),
            prob_no_class = prob_no,
            prob_class = prob_class,
            confidence = Float32(max(prob_no, prob_class)),
            class_margin = Float32(prob_class - prob_no),
            variant = cellstr(r.variant),
            chunk_index = Int(r.chunk_index),
            chunk_count = Int(r.chunk_count),
            n_trials = Int(r.n_trials),
            origin_n_trials = Int(r.origin_n_trials),
            reused_fill_count = Int(r.reused_fill_count),
            image_file = cellstr(r.image_file),
            label_studio_project_id = cellstr(r.label_studio_project_id),
            label_studio_task_id = cellstr(r.label_studio_task_id),
            annotation_id = cellstr(r.annotation_id),
        ))
    end
    return rows
end

function run_resnet18_cv(sample_df::DataFrame;
        nepochs::Int = TRAIN_EPOCHS,
        lr::Float32 = TRAIN_LR,
        seed::Int = GLOBAL_SEED)

    X = images_to_tensor(sample_df)
    y = Int.(sample_df.binary_label)
    device, use_cuda = Generalization.setup_device()
    batchsize = use_cuda ? Generalization.TRAIN_BATCHSIZE_GPU : Generalization.TRAIN_BATCHSIZE_CPU

    metric_rows = NamedTuple[]
    history_parts = DataFrame[]
    pred_rows = NamedTuple[]

    for fold in 1:K_FOLDS
        train_idx = findall(!=(fold), Int.(sample_df.fold))
        val_idx = findall(==(fold), Int.(sample_df.fold))

        Random.seed!(seed + fold)
        model, pretrained_params_loaded = Generalization.build_resnet_single_channel_pretrained(18)
        model = device(model)

        println("$(MODEL_NAME) | fold $(fold)/$(K_FOLDS) | train=$(length(train_idx)) | val=$(length(val_idx))")
        model, history_df, train_time_s = Generalization.train_full_model!(
            model,
            X[:, :, :, train_idx],
            y[train_idx];
            model_name = "$(MODEL_NAME)_fold$(fold)",
            nepochs = nepochs,
            lr = lr,
            batchsize = batchsize,
            seed = seed + fold,
            device = device,
        )
        history_df.fold = fill(fold, nrow(history_df))
        push!(history_parts, history_df)

        train_metrics, _, _, _, _ = metrics_for_indices(
            model, X, y, train_idx;
            batchsize = Generalization.PREDICT_BATCHSIZE,
            device = device,
        )
        val_metrics, val_logits, val_probs, y_true, y_pred = metrics_for_indices(
            model, X, y, val_idx;
            batchsize = Generalization.PREDICT_BATCHSIZE,
            device = device,
        )

        append!(pred_rows, prediction_rows(sample_df, val_idx, fold, val_logits, val_probs, y_true, y_pred))

        push!(metric_rows, (
            model_name = MODEL_NAME,
            fold = fold,
            n_train = length(train_idx),
            n_val = length(val_idx),
            train_accuracy = Float64(train_metrics.accuracy),
            train_balanced_accuracy = Float64(train_metrics.balanced_accuracy),
            train_macro_f1 = Float64(train_metrics.macro_f1),
            train_precision = Float64(train_metrics.precision),
            train_recall = Float64(train_metrics.recall),
            val_accuracy = Float64(val_metrics.accuracy),
            val_balanced_accuracy = Float64(val_metrics.balanced_accuracy),
            val_macro_f1 = Float64(val_metrics.macro_f1),
            val_precision = Float64(val_metrics.precision),
            val_recall = Float64(val_metrics.recall),
            train_time_s = Float64(train_time_s),
            pretrained_params_loaded = pretrained_params_loaded,
            batchsize = batchsize,
            use_cuda = use_cuda,
        ))

        model = nothing
        CUDA.functional() && CUDA.reclaim()
        GC.gc(true)
    end

    return (
        metrics_df = DataFrame(metric_rows),
        history_df = isempty(history_parts) ? DataFrame() : vcat(history_parts...; cols = :union),
        predictions_df = DataFrame(pred_rows),
    )
end

function summarize_metrics(metrics_df::DataFrame)
    summary = combine(
        groupby(metrics_df, :model_name),
        :val_accuracy => mean => :val_accuracy_mean,
        :val_accuracy => std => :val_accuracy_std,
        :val_balanced_accuracy => mean => :val_balanced_accuracy_mean,
        :val_balanced_accuracy => std => :val_balanced_accuracy_std,
        :val_macro_f1 => mean => :val_macro_f1_mean,
        :val_macro_f1 => std => :val_macro_f1_std,
        :val_precision => mean => :val_precision_mean,
        :val_recall => mean => :val_recall_mean,
        :train_time_s => mean => :train_time_mean_s,
        :pretrained_params_loaded => maximum => :pretrained_params_loaded,
    )
    return summary
end

function sample_plan_for_csv(sample_df::DataFrame)
    out = select(sample_df, Not(:processed_img))
    return out
end

function run_experiment(;
        output_dir::AbstractString = OUTPUT_DIR,
        nepochs::Int = TRAIN_EPOCHS,
        lr::Float32 = TRAIN_LR,
        k_folds::Int = K_FOLDS,
        seed::Int = GLOBAL_SEED)

    k_folds == K_FOLDS || error("K_FOLDS is constant in this script; set WEEK21_RESNET18_FOLDS before running.")
    clean_output_dir!(output_dir)

    println("Loading Label Studio annotations.")
    labels = load_labeled_annotations()
    source_status_df = Screening.discover_week19_data_sources()
    ctx = build_data_context(source_status_df)

    println("Resolving source origins and trial counts for ", nrow(labels), " labels.")
    annotate_origin_trials!(labels, ctx)
    minimum_origin_trials = minimum(Int.(labels.origin_n_trials))
    target_trials = TARGET_TRIALS
    target_trials <= minimum_origin_trials || error(
        "Configured target_trials=$(target_trials) exceeds minimum origin trial count $(minimum_origin_trials).",
    )
    println(
        "Configured fixed trial count per augmented ERP image: ",
        target_trials,
        " | minimum origin trial count: ",
        minimum_origin_trials,
    )

    CSV.write(joinpath(output_dir, "labels_used.csv"), labels)

    label_summary_df = combine(groupby(labels, [:dataset_key, :erp_class, :binary_label]), nrow => :count)
    sort!(label_summary_df, [:dataset_key, :erp_class])
    CSV.write(joinpath(output_dir, "label_summary.csv"), label_summary_df)

    println("Materializing fixed-trial augmented ERP images.")
    sample_df = materialize_augmented_samples(labels, ctx; target_trials = target_trials)
    assign_stratified_folds!(sample_df; k = k_folds, seed = seed)
    CSV.write(joinpath(output_dir, "sample_plan.csv"), sample_plan_for_csv(sample_df))

    fold_binary_df, fold_class_df, fold_dataset_df = fold_distribution_tables(sample_df)
    CSV.write(joinpath(output_dir, "fold_distribution_binary.csv"), fold_binary_df)
    CSV.write(joinpath(output_dir, "fold_distribution_pattern_class.csv"), fold_class_df)
    CSV.write(joinpath(output_dir, "fold_distribution_dataset.csv"), fold_dataset_df)

    augmented_summary_df = combine(groupby(sample_df, [:erp_class, :binary_label]), nrow => :count)
    sort!(augmented_summary_df, [:binary_label, :erp_class])
    CSV.write(joinpath(output_dir, "augmented_label_summary.csv"), augmented_summary_df)

    println("Training and validating ResNet18 with ", k_folds, "-fold CV.")
    cv = run_resnet18_cv(sample_df; nepochs = nepochs, lr = lr, seed = seed)
    CSV.write(joinpath(output_dir, "fold_metrics.csv"), cv.metrics_df)
    CSV.write(joinpath(output_dir, "train_history.csv"), cv.history_df)
    CSV.write(joinpath(output_dir, "validation_predictions.csv"), cv.predictions_df)

    metrics_summary_df = summarize_metrics(cv.metrics_df)
    CSV.write(joinpath(output_dir, "metrics_summary.csv"), metrics_summary_df)

    write_json(joinpath(output_dir, "run_config.json"), Dict(
        "created_at" => string(now()),
        "annotations_csv" => ANNOTATIONS_CSV,
        "output_dir" => output_dir,
        "model_name" => MODEL_NAME,
        "architecture" => "ResNet18 single-channel, ImageNet-pretrained first convolution projected to 1 channel",
        "task" => "binary ERP pattern class vs no_class",
        "k_folds" => k_folds,
        "fold_policy" => "stratified by the 7 manual labels; augmented variants from each source row are spread over folds when possible",
        "augmentation_policy" => "sort trials, build floor(n / target_trials) round-robin mod-split chunks, distribute remainder trials one per round into a separate remainder chunk, fill that remainder chunk to target_trials with deterministic random unique trials from the same origin; positive rows keep all chunks, no_class rows keep one deterministic chunk by default",
        "target_trials" => target_trials,
        "target_trials_source" => "WEEK21_TARGET_TRIALS environment variable, default 150",
        "minimum_origin_trials" => minimum_origin_trials,
        "target_size" => TARGET_SIZE,
        "preprocessing" => "sort -> zscore_timepoints -> Gaussian smoothing -> resize to 64x64",
        "lowpass_sigma" => Generalization.LOWPASS_SIGMA,
        "lowpass_kernel_size" => Generalization.LOWPASS_KERNEL_SIZE,
        "filter_border" => Generalization.FILTER_BORDER,
        "nepochs" => nepochs,
        "lr" => lr,
        "seed" => seed,
        "excluded_dataset_keys" => collect(EXCLUDED_DATASET_KEYS),
        "no_class_chunks_per_origin" => NO_CLASS_CHUNKS_PER_ORIGIN,
        "n_labeled_rows_used" => nrow(labels),
        "n_augmented_images" => nrow(sample_df),
    ))

    println("Output dir: ", output_dir)
    println(metrics_summary_df)
    return (
        labels_df = labels,
        sample_df = sample_df,
        fold_binary_df = fold_binary_df,
        fold_class_df = fold_class_df,
        metrics_df = cv.metrics_df,
        metrics_summary_df = metrics_summary_df,
        predictions_df = cv.predictions_df,
        output_dir = output_dir,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment()
end
