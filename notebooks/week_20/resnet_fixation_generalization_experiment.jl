module Week20ResNetFixationGeneralization

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
const NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_20")
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
const DATASETS_ROOT = joinpath(REPO_ROOT, "notebooks", "datasets")
const OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet_fixation_generalization")

Pkg.activate(MODEL_ENV_DIR)

using CSV
using CairoMakie
using CUDA
using DataFrames
using Dates
using Flux
using HDF5
using ImageFiltering: imfilter
using Images: imresize
using JSON3
using MLUtils: DataLoader
using Metalhead
using Random
using Statistics

include(joinpath(REPO_ROOT, "notebooks", "utils", "erp_cnn_experiment_utils.jl"))
using .ERPCNNExperimentUtils

include(joinpath(REPO_ROOT, "notebooks", "week_15", "try_new_data_helpers.jl"))
using .Week15TryNewData

export run_experiment

const SAMPLING_RATE = 512
const PRE_STIM_S = 0.5
const TIME_ZERO_IDX = Int(round(PRE_STIM_S * SAMPLING_RATE)) + 1
const TARGET_SIZE = (64, 64)
const LOWPASS_SIGMA = 75.0f0
const LOWPASS_KERNEL_SIZE = (21, 21)
const FILTER_BORDER = "reflect"
const POSITIVE_SPLIT_K = 4
const NO_CLASS_SPLIT_K = 4
const GLOBAL_SEED = 20260420
const NO_CLASS_PICK_SEED = GLOBAL_SEED + 1
const TRAINING_SEED_BASE = GLOBAL_SEED + 10_000
const TRAIN_EPOCHS = 8
const TRAIN_BATCHSIZE_GPU = 32
const TRAIN_BATCHSIZE_CPU = 8
const TRAIN_LR = 3f-4
const PREDICT_BATCHSIZE = 64

const TARGET_DATASET_SPECS = [
    (
        dataset_key = "erp_core_n2pc_clean",
        label = "ERP CORE N2PC",
        sort_col = :reaction_time_ms,
        mod_split_k = 2,
        baseline_correct = true,
    ),
    (
        dataset_key = "erp_core_n170_clean",
        label = "ERP CORE N170",
        sort_col = :reaction_time_ms,
        mod_split_k = 2,
        baseline_correct = true,
    ),
    (
        dataset_key = "eye_eeg_freeviewing_fixations",
        label = "EYE-EEG FREEVIEWING FIXATIONS",
        sort_col = :fixation_duration_ms,
        mod_split_k = 1,
        baseline_correct = false,
    ),
    (
        dataset_key = "eye_eeg_reading_fixations",
        label = "EYE-EEG READING FIXATIONS",
        sort_col = :fixation_duration_ms,
        mod_split_k = 1,
        baseline_correct = false,
    ),
    (
        dataset_key = "eye_eeg_sceneviewing_tobii_fixations",
        label = "EYE-EEG SCENEVIEWING TOBII FIXATIONS",
        sort_col = :fixation_duration_ms,
        mod_split_k = 1,
        baseline_correct = false,
    ),
    (
        dataset_key = "roamm_reading_fixations",
        label = "ROAMM READING FIXATIONS",
        sort_col = :fixation_duration_ms,
        mod_split_k = 1,
        baseline_correct = false,
    ),
]

class_name(label::Integer) = label == 1 ? "class" : "no_class"

function stable_slug(x::AbstractString)
    y = lowercase(x)
    y = replace(y, r"[^a-z0-9]+" => "_")
    y = replace(y, r"(^_+|_+$)" => "")
    return isempty(y) ? "item" : y
end

function clean_outputs_dir!(output_dir::AbstractString)
    mkpath(output_dir)
    for child in readdir(output_dir; join = true)
        if isfile(child) && (endswith(child, ".csv") || endswith(child, ".png") || endswith(child, ".txt"))
            rm(child; force = true)
        end
    end
    return output_dir
end

function setup_device()
    if CUDA.functional()
        CUDA.allowscalar(false)
        CUDA.device!(0)
        println("CUDA device: ", CUDA.name(CUDA.device()))
        return gpu, true
    end
    println("CUDA is not functional; running on CPU with a smaller batch size.")
    return cpu, false
end

function collect_arrays_recursive_local(x, acc = Vector{Any}())
    if x isa AbstractArray
        push!(acc, x)
    elseif x isa NamedTuple
        for k in keys(x)
            collect_arrays_recursive_local(getfield(x, k), acc)
        end
    elseif x isa Tuple
        for xi in x
            collect_arrays_recursive_local(xi, acc)
        end
    end
    return acc
end

function project_first_conv_weights(src_weight::AbstractArray, dst_inchannels::Int)
    @assert ndims(src_weight) == 4 "Expected a 4D convolution kernel."
    src_inchannels = size(src_weight, 3)
    src_inchannels == dst_inchannels && return copy(src_weight)
    projected = mean(src_weight; dims = 3)
    return repeat(projected, 1, 1, dst_inchannels, 1)
end

function load_resnet_pretrained_project_firstconv!(model, weight_key::AbstractString)
    src_state = Metalhead.loadweights(weight_key)
    dst_arrays = Flux.trainables(model)
    src_arrays = collect_arrays_recursive_local(src_state)

    @assert !isempty(dst_arrays) "Destination model has no trainable arrays."
    @assert !isempty(src_arrays) "Source pretrained state has no arrays."

    matched = 0
    first_dst = dst_arrays[1]
    first_src = src_arrays[1]

    if ndims(first_dst) == 4 && ndims(first_src) == 4 &&
       size(first_dst, 1) == size(first_src, 1) &&
       size(first_dst, 2) == size(first_src, 2) &&
       size(first_dst, 4) == size(first_src, 4)

        projected = project_first_conv_weights(first_src, size(first_dst, 3))
        @assert size(projected) == size(first_dst) "Projected first convolution has wrong size."
        copyto!(first_dst, projected)
        matched += 1
        dst_start = 2
        src_start = 2
    else
        dst_start = 1
        src_start = 1
    end

    j = src_start
    for i in dst_start:length(dst_arrays)
        d = dst_arrays[i]
        while j <= length(src_arrays) && size(src_arrays[j]) != size(d)
            j += 1
        end
        j <= length(src_arrays) || error("Failed to map pretrained weights for destination size $(size(d)).")
        copyto!(d, src_arrays[j])
        matched += 1
        j += 1
    end

    return matched
end

resnet_backbone(model) = isdefined(Metalhead, :backbone) ? getfield(Metalhead, :backbone)(model) : model.layers.layers[1]
resnet_classifier(model) = isdefined(Metalhead, :classifier) ? getfield(Metalhead, :classifier)(model) : model.layers.layers[2]

function build_resnet_single_channel_pretrained(depth::Int; n_classes::Int = 2, in_channels::Int = 1)
    weight_key = "resnet$(depth)-IMAGENET1K_V1"
    base = Metalhead.ResNet(depth; pretrain = false, inchannels = in_channels, nclasses = 1000)
    matched = load_resnet_pretrained_project_firstconv!(base, weight_key)

    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(
        old_head.layers[1],
        old_head.layers[2],
        Dense(in_dim => n_classes),
    )

    return Chain(features, new_head), matched
end

function model_specs()
    return [
        (
            name = "resnet18_pretrained",
            depth = 18,
            builder = () -> build_resnet_single_channel_pretrained(18),
        ),
        (
            name = "resnet34_pretrained",
            depth = 34,
            builder = () -> build_resnet_single_channel_pretrained(34),
        ),
    ]
end

loss_fn(model, x, y) = Flux.Losses.logitcrossentropy(model(x), y)

function train_full_model!(model, X::Array{Float32, 4}, y_binary::Vector{Int};
    model_name::String,
    nepochs::Int,
    lr::Float32,
    batchsize::Int,
    seed::Int,
    device::Function)

    Random.seed!(seed)
    y_oh = Flux.onehotbatch(y_binary, 0:1) |> Array{Float32}
    train_loader = DataLoader((X, y_oh); batchsize = batchsize, shuffle = true)
    opt_state = Flux.setup(Flux.Adam(lr), model)
    history_rows = NamedTuple[]

    Flux.trainmode!(model)
    total_time_s = @elapsed begin
        for epoch in 1:nepochs
            running_loss = 0f0
            n_batches = 0
            epoch_time_s = @elapsed begin
                for (xb_cpu, yb_cpu) in train_loader
                    xb = device(xb_cpu)
                    yb = device(yb_cpu)
                    loss_val, grads = Flux.withgradient(model) do m
                        loss_fn(m, xb, yb)
                    end
                    opt_state, model = Flux.update!(opt_state, model, grads[1])
                    running_loss += Float32(loss_val)
                    n_batches += 1
                end
            end
            avg_loss = Float64(running_loss / max(1, n_batches))
            push!(history_rows, (
                model_name = model_name,
                epoch = epoch,
                avg_loss = avg_loss,
                epoch_time_s = epoch_time_s,
                n_batches = n_batches,
            ))
            println("$(model_name) | epoch $(epoch)/$(nepochs) | loss=$(round(avg_loss; digits = 5))")
        end
    end

    return model, DataFrame(history_rows), total_time_s
end

function predict_logits_probs(model, X::Array{Float32, 4}; batchsize::Int, device::Function)
    Flux.testmode!(model, true)
    n = size(X, 4)
    logits_all = Array{Float32}(undef, 2, n)
    probs_all = Array{Float32}(undef, 2, n)
    for start_idx in 1:batchsize:n
        idx = start_idx:min(start_idx + batchsize - 1, n)
        logits = Array(cpu(model(device(X[:, :, :, idx]))))
        probs = Flux.softmax(Float32.(logits); dims = 1)
        logits_all[:, idx] .= Float32.(logits)
        probs_all[:, idx] .= Float32.(probs)
    end
    return logits_all, probs_all
end

function evaluate_training_fit(model, X::Array{Float32, 4}, y::Vector{Int}; batchsize::Int, device::Function)
    logits, probs = predict_logits_probs(model, X; batchsize = batchsize, device = device)
    y_pred = [probs[2, i] >= probs[1, i] ? 1 : 0 for i in axes(probs, 2)]
    metrics = compute_metrics(y_pred, y)
    return (
        accuracy = Float64(metrics.accuracy),
        balanced_accuracy = Float64(metrics.balanced_accuracy),
        macro_f1 = Float64(metrics.macro_f1),
        precision = Float64(metrics.precision),
        recall = Float64(metrics.recall),
        prob_class_mean = mean(Float64.(probs[2, :])),
        confidence_mean = mean(Float64.(max.(probs[1, :], probs[2, :]))),
    )
end

function find_last_dense(model)
    found = Dense[]
    function visit(x)
        if x isa Dense
            push!(found, x)
        elseif hasproperty(x, :layers)
            for layer in getproperty(x, :layers)
                visit(layer)
            end
        elseif x isa Tuple
            for xi in x
                visit(xi)
            end
        end
        return nothing
    end
    visit(model)
    isempty(found) && error("No Dense layer found in model.")
    return last(found)
end

function output_head_weights_df(model, model_name::AbstractString)
    dense = find_last_dense(model)
    rows = NamedTuple[]
    weight = Array(cpu(dense.weight))
    bias = Array(cpu(dense.bias))
    for output_idx in axes(weight, 1)
        label = output_idx == 2 ? 1 : 0
        for feature_idx in axes(weight, 2)
            push!(rows, (
                model_name = String(model_name),
                output_index = output_idx,
                output_label = label,
                output_class = class_name(label),
                feature_index = feature_idx,
                weight = Float32(weight[output_idx, feature_idx]),
                bias = Float32(bias[output_idx]),
            ))
        end
    end
    return DataFrame(rows)
end

function split_indices_sorted_modulo(events_trials::DataFrame, sort_col::Symbol, k::Int)
    n = nrow(events_trials)
    n == 0 && return [Int[]]
    k_eff = min(max(k, 1), n)
    order = Week15TryNewData.trial_sort_order(events_trials, sort_col)
    groups = [Int[] for _ in 1:k_eff]
    for (rank, idx) in enumerate(order)
        push!(groups[((rank - 1) % k_eff) + 1], idx)
    end
    return groups
end

function post_stim_indices_from_times(times_s::AbstractVector{<:Real})
    return Week15TryNewData.post_stim_indices(
        times_s;
        time_window_s = Week15TryNewData.REAL_PREVIEW_TIME_WINDOW_S,
    )
end

function merged_channel_trials(bundle, channel_name::AbstractString; baseline_correct::Bool)
    data_parts = Matrix{Float32}[]
    event_parts = DataFrame[]
    subject_labels = String[]
    channel_indices = Int[]
    post_len = nothing
    sfreq_hz = nothing
    time_start_s = nothing
    time_end_s = nothing

    for subject_label in bundle.subject_labels
        subj = Week15TryNewData.load_subject_data(bundle.h5_path, subject_label)
        channel_idx = findfirst(==(channel_name), subj.channel_names)
        channel_idx === nothing && continue

        events_subset = Week15TryNewData.select_subject_events(bundle, subject_label)
        epoch_indices = Int.(events_subset.epoch_index)
        post_idx = post_stim_indices_from_times(subj.times_s)
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
            data_full_time_trials = Week15TryNewData.baseline_correct_time_trials(data_full_time_trials, subj.times_s)
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

    isempty(data_parts) && error("Channel $(channel_name) not found in dataset $(bundle.dataset_key).")
    data_time_trials = hcat(data_parts...)
    events_merged = vcat(event_parts...; cols = :union)
    @assert size(data_time_trials, 2) == nrow(events_merged) "Trial count mismatch after merging subjects."

    return (
        data_time_trials = data_time_trials,
        events = events_merged,
        subject_label = length(subject_labels) == 1 ? first(subject_labels) : "merged_experiment",
        channel_idx = first(channel_indices),
        n_trials = nrow(events_merged),
        n_timepoints_post = post_len,
        time_start_s = Float32(time_start_s),
        time_end_s = Float32(time_end_s),
        sampling_rate_hz = Float64(sfreq_hz),
    )
end

function load_subject_cache(bundle)
    caches = NamedTuple[]
    for subject_label in bundle.subject_labels
        subj = Week15TryNewData.load_subject_data(bundle.h5_path, subject_label)
        events_subset = Week15TryNewData.select_subject_events(bundle, subject_label)
        epoch_indices = Int.(events_subset.epoch_index)
        post_idx = post_stim_indices_from_times(subj.times_s)
        post_times_s = subj.times_s[post_idx]
        push!(caches, (
            subject_label = String(subject_label),
            epochs = subj.epochs,
            times_s = subj.times_s,
            channel_names = String.(subj.channel_names),
            n_timepoints = Int(subj.n_timepoints),
            sfreq_hz = Float64(subj.sfreq_hz),
            events = events_subset,
            epoch_indices = epoch_indices,
            post_idx = post_idx,
            post_time_start_s = Float32(first(post_times_s)),
            post_time_end_s = Float32(last(post_times_s)),
        ))
    end
    return caches
end

function merged_channel_trials_from_cache(bundle, subject_caches, channel_name::AbstractString; baseline_correct::Bool)
    data_parts = Matrix{Float32}[]
    event_parts = DataFrame[]
    subject_labels = String[]
    channel_indices = Int[]
    post_len = nothing
    sfreq_hz = nothing
    time_start_s = nothing
    time_end_s = nothing

    for cache in subject_caches
        channel_idx = findfirst(==(channel_name), cache.channel_names)
        channel_idx === nothing && continue

        post_len === nothing || post_len == length(cache.post_idx) ||
            error("Cannot merge $(bundle.dataset_key): post-stimulus length differs across subjects.")
        sfreq_hz === nothing || sfreq_hz == cache.sfreq_hz ||
            error("Cannot merge $(bundle.dataset_key): sampling rate differs across subjects.")
        time_start_s === nothing || time_start_s == cache.post_time_start_s ||
            error("Cannot merge $(bundle.dataset_key): preview start time differs across subjects.")
        time_end_s === nothing || time_end_s == cache.post_time_end_s ||
            error("Cannot merge $(bundle.dataset_key): preview end time differs across subjects.")

        post_len = length(cache.post_idx)
        sfreq_hz = cache.sfreq_hz
        time_start_s = cache.post_time_start_s
        time_end_s = cache.post_time_end_s

        data_full_time_trials = reshape(
            Float32.(cache.epochs[channel_idx, :, cache.epoch_indices]),
            cache.n_timepoints,
            length(cache.epoch_indices),
        )
        if baseline_correct
            data_full_time_trials = Week15TryNewData.baseline_correct_time_trials(data_full_time_trials, cache.times_s)
        end

        data_time_trials = reshape(
            Float32.(data_full_time_trials[cache.post_idx, :]),
            length(cache.post_idx),
            length(cache.epoch_indices),
        )

        push!(data_parts, data_time_trials)
        push!(event_parts, cache.events)
        push!(subject_labels, cache.subject_label)
        push!(channel_indices, Int(channel_idx))
    end

    isempty(data_parts) && error("Channel $(channel_name) not found in dataset $(bundle.dataset_key).")
    data_time_trials = hcat(data_parts...)
    events_merged = vcat(event_parts...; cols = :union)
    @assert size(data_time_trials, 2) == nrow(events_merged) "Trial count mismatch after merging subjects."

    return (
        data_time_trials = data_time_trials,
        events = events_merged,
        subject_label = length(subject_labels) == 1 ? first(subject_labels) : "merged_experiment",
        channel_idx = first(channel_indices),
        n_trials = nrow(events_merged),
        n_timepoints_post = post_len,
        time_start_s = Float32(time_start_s),
        time_end_s = Float32(time_end_s),
        sampling_rate_hz = Float64(sfreq_hz),
    )
end

function external_preprocess_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    img_trials_time = preprocess_sorted_zscore_image(data_time_trials, events_trials, sort_col)
    return apply_pipeline_to_image(
        img_trials_time;
        pipeline_name = :gaussian_reference,
        target_size = TARGET_SIZE,
        low_pass_sigma = LOWPASS_SIGMA,
        lowpass_kernel_size = LOWPASS_KERNEL_SIZE,
        filter_border = FILTER_BORDER,
    )
end

function materialize_target_dataset(spec)
    bundle = Week15TryNewData.load_clean_dataset_bundle(spec.dataset_key)
    @assert spec.sort_col in propertynames(bundle.events) "Sort column $(spec.sort_col) not found in $(spec.dataset_key)."
    subject_caches = load_subject_cache(bundle)

    rows = NamedTuple[]
    images = Matrix{Float32}[]

    for channel_name in String.(bundle.channel_names)
        origin = merged_channel_trials_from_cache(
            bundle,
            subject_caches,
            channel_name;
            baseline_correct = Bool(spec.baseline_correct),
        )
        origin_id = join([spec.dataset_key, origin.subject_label, channel_name, String(spec.sort_col)], "::")
        groups = spec.mod_split_k > 1 ?
            split_indices_sorted_modulo(origin.events, spec.sort_col, Int(spec.mod_split_k)) :
            [collect(1:nrow(origin.events))]

        for (part, idxs) in enumerate(groups)
            isempty(idxs) && continue
            events_part = origin.events[idxs, :]
            data_part = origin.data_time_trials[:, idxs]
            img = external_preprocess_image(data_part, events_part, spec.sort_col)
            variant = spec.mod_split_k > 1 ? "mod$(spec.mod_split_k)_part$(part)" : "full"
            image_id = "$(origin_id)::$(variant)"

            push!(rows, (
                dataset_key = String(spec.dataset_key),
                dataset_label = String(spec.label),
                component = String(bundle.metadata.component),
                subject_label = String(origin.subject_label),
                channel_name = String(channel_name),
                channel_idx = Int(origin.channel_idx),
                sort_col = String(spec.sort_col),
                mod_split_k = Int(spec.mod_split_k),
                split_part = Int(part),
                variant = variant,
                origin_id = origin_id,
                image_id = image_id,
                n_trials = length(idxs),
                n_origin_trials = Int(origin.n_trials),
                n_timepoints_post = Int(origin.n_timepoints_post),
                time_start_s = Float32(origin.time_start_s),
                time_end_s = Float32(origin.time_end_s),
                sampling_rate_hz = Float64(origin.sampling_rate_hz),
                baseline_correct = Bool(spec.baseline_correct),
                source_h5 = String(bundle.h5_path),
                source_events = String(bundle.events_path),
            ))
            push!(images, img)
        end
    end

    df = DataFrame(rows)
    df.processed_img = images
    return df
end

function materialize_all_target_datasets()
    dfs = DataFrame[]
    for spec in TARGET_DATASET_SPECS
        println("Materializing target dataset: ", spec.label)
        df = materialize_target_dataset(spec)
        println("  images: ", nrow(df), " | origins: ", length(unique(df.origin_id)))
        push!(dfs, df)
    end
    return vcat(dfs...; cols = :union)
end

function prepare_training_dataset()
    data_ctx = prepare_real_fixations_inputs(NOTEBOOK_DIR)
    sample_plan_df = build_single_channel_sample_plan(
        data_ctx.labels_df,
        data_ctx.events;
        positive_split_k = POSITIVE_SPLIT_K,
        no_class_split_k = NO_CLASS_SPLIT_K,
        no_class_pick_rng = MersenneTwister(NO_CLASS_PICK_SEED),
    )

    train_df = materialize_single_channel_dataset(
        data_ctx.erps,
        data_ctx.events,
        sample_plan_df;
        time_zero_idx = TIME_ZERO_IDX,
        pipeline_name = :gaussian_reference,
        target_size = TARGET_SIZE,
        low_pass_sigma = LOWPASS_SIGMA,
        lowpass_kernel_size = LOWPASS_KERNEL_SIZE,
        filter_border = FILTER_BORDER,
    )

    X = images_to_tensor(train_df.processed_img)
    y = Int.(train_df.binary_label)

    return (
        data_ctx = data_ctx,
        sample_plan_df = sample_plan_df,
        train_df = train_df,
        X = X,
        y = y,
    )
end

function training_metadata_tables(ctx)
    label_dist_df = combine(groupby(ctx.data_ctx.labels_df, :binary_label), nrow => :count)
    sort!(label_dist_df, :binary_label)

    augmented_dist_df = combine(groupby(ctx.train_df, [:binary_label, :variant]), nrow => :count)
    sort!(augmented_dist_df, [:binary_label, :variant])

    overview_df = DataFrame(
        metric = [
            "labeled_fixation_rows",
            "training_images_after_mod4_policy",
            "positive_training_images",
            "no_class_training_images",
            "target_height_trials_resized",
            "target_width_time_resized",
            "lowpass_sigma",
            "global_seed",
            "no_class_pick_seed",
        ],
        value = [
            string(nrow(ctx.data_ctx.labels_df)),
            string(nrow(ctx.train_df)),
            string(count(==(1), ctx.y)),
            string(count(==(0), ctx.y)),
            string(TARGET_SIZE[1]),
            string(TARGET_SIZE[2]),
            string(LOWPASS_SIGMA),
            string(GLOBAL_SEED),
            string(NO_CLASS_PICK_SEED),
        ],
    )

    return overview_df, label_dist_df, augmented_dist_df
end

function prediction_rows_for_dataset(model, model_name::AbstractString, target_df::DataFrame;
    batchsize::Int,
    device::Function)

    X = images_to_tensor(target_df.processed_img)
    logits, probs = predict_logits_probs(model, X; batchsize = batchsize, device = device)
    rows = NamedTuple[]

    for i in 1:nrow(target_df)
        prob_no = Float32(probs[1, i])
        prob_class = Float32(probs[2, i])
        pred_label = prob_class >= prob_no ? 1 : 0
        confidence = max(prob_no, prob_class)
        entropy = -sum(Float64[p * log(max(p, eps(Float32))) for p in (prob_no, prob_class)])
        r = target_df[i, :]

        push!(rows, (
            model_name = String(model_name),
            dataset_key = String(r.dataset_key),
            dataset_label = String(r.dataset_label),
            component = String(r.component),
            image_id = String(r.image_id),
            origin_id = String(r.origin_id),
            subject_label = String(r.subject_label),
            channel_name = String(r.channel_name),
            channel_idx = Int(r.channel_idx),
            sort_col = String(r.sort_col),
            mod_split_k = Int(r.mod_split_k),
            split_part = Int(r.split_part),
            variant = String(r.variant),
            n_trials = Int(r.n_trials),
            n_origin_trials = Int(r.n_origin_trials),
            n_timepoints_post = Int(r.n_timepoints_post),
            time_start_s = Float32(r.time_start_s),
            time_end_s = Float32(r.time_end_s),
            sampling_rate_hz = Float64(r.sampling_rate_hz),
            baseline_correct = Bool(r.baseline_correct),
            logit_no_class = Float32(logits[1, i]),
            logit_class = Float32(logits[2, i]),
            prob_no_class = prob_no,
            prob_class = prob_class,
            predicted_label = pred_label,
            predicted_class = class_name(pred_label),
            confidence = Float32(confidence),
            margin_abs = Float32(abs(prob_class - prob_no)),
            entropy = Float32(entropy),
        ))
    end

    return DataFrame(rows)
end

function summarize_predictions(pred_df::DataFrame)
    rows = NamedTuple[]
    for sdf in groupby(pred_df, [:model_name, :dataset_key, :dataset_label])
        n = nrow(sdf)
        pred_class_n = count(==(1), sdf.predicted_label)
        pred_no_n = count(==(0), sdf.predicted_label)
        probs = Float64.(sdf.prob_class)
        conf = Float64.(sdf.confidence)
        entropy = Float64.(sdf.entropy)
        dominant_label = pred_class_n >= pred_no_n ? 1 : 0
        dominant_n = max(pred_class_n, pred_no_n)
        push!(rows, (
            model_name = String(sdf.model_name[1]),
            dataset_key = String(sdf.dataset_key[1]),
            dataset_label = String(sdf.dataset_label[1]),
            n_images = n,
            n_origins = length(unique(sdf.origin_id)),
            predicted_class_n = pred_class_n,
            predicted_no_class_n = pred_no_n,
            predicted_class_rate = pred_class_n / n,
            predicted_no_class_rate = pred_no_n / n,
            dominant_class = class_name(dominant_label),
            dominant_class_rate = dominant_n / n,
            is_single_class = length(unique(sdf.predicted_label)) == 1,
            prob_class_mean = mean(probs),
            prob_class_std = std(probs),
            prob_class_min = minimum(probs),
            prob_class_max = maximum(probs),
            confidence_mean = mean(conf),
            confidence_min = minimum(conf),
            confidence_max = maximum(conf),
            entropy_mean = mean(entropy),
        ))
    end
    out = DataFrame(rows)
    sort!(out, [:model_name, :dataset_key])
    return out
end

function mod_split_consistency(pred_df::DataFrame)
    detail_rows = NamedTuple[]
    for sdf in groupby(pred_df, [:model_name, :dataset_key, :dataset_label, :origin_id])
        if maximum(Int.(sdf.mod_split_k)) <= 1 || nrow(sdf) <= 1
            continue
        end
        pred_labels = Int.(sdf.predicted_label)
        prob_range = maximum(Float64.(sdf.prob_class)) - minimum(Float64.(sdf.prob_class))
        conf_range = maximum(Float64.(sdf.confidence)) - minimum(Float64.(sdf.confidence))
        push!(detail_rows, (
            model_name = String(sdf.model_name[1]),
            dataset_key = String(sdf.dataset_key[1]),
            dataset_label = String(sdf.dataset_label[1]),
            origin_id = String(sdf.origin_id[1]),
            n_variants = nrow(sdf),
            predicted_labels = join(string.(pred_labels), "|"),
            predicted_classes = join(String.(sdf.predicted_class), "|"),
            all_variants_same_prediction = length(unique(pred_labels)) == 1,
            prob_class_min = minimum(Float64.(sdf.prob_class)),
            prob_class_max = maximum(Float64.(sdf.prob_class)),
            prob_class_range = prob_range,
            confidence_min = minimum(Float64.(sdf.confidence)),
            confidence_max = maximum(Float64.(sdf.confidence)),
            confidence_range = conf_range,
        ))
    end

    detail_df = DataFrame(detail_rows)
    if isempty(detail_rows)
        return detail_df, DataFrame()
    end

    summary_rows = NamedTuple[]
    for sdf in groupby(detail_df, [:model_name, :dataset_key, :dataset_label])
        n = nrow(sdf)
        same_n = count(sdf.all_variants_same_prediction)
        push!(summary_rows, (
            model_name = String(sdf.model_name[1]),
            dataset_key = String(sdf.dataset_key[1]),
            dataset_label = String(sdf.dataset_label[1]),
            n_modsplit_origins = n,
            n_same_prediction = same_n,
            n_disagree_prediction = n - same_n,
            same_prediction_rate = same_n / n,
            mean_prob_class_range = mean(Float64.(sdf.prob_class_range)),
            max_prob_class_range = maximum(Float64.(sdf.prob_class_range)),
            mean_confidence_range = mean(Float64.(sdf.confidence_range)),
        ))
    end

    summary_df = DataFrame(summary_rows)
    sort!(summary_df, [:model_name, :dataset_key])
    sort!(detail_df, [:model_name, :dataset_key, :origin_id])
    return detail_df, summary_df
end

function choose_unique_examples(pred_df::DataFrame, predicted_label::Int; n::Int = 4)
    sdf = pred_df[pred_df.predicted_label .== predicted_label, :]
    isempty(sdf) && return sdf
    sort!(sdf, [:confidence, :margin_abs]; rev = [true, true])
    keep = Int[]
    seen = Set{String}()
    for (row_idx, row) in enumerate(eachrow(sdf))
        origin_id = String(row.origin_id)
        origin_id in seen && continue
        push!(keep, row_idx)
        push!(seen, origin_id)
        length(keep) >= n && break
    end
    return sdf[keep, :]
end

function lookup_image(target_df::DataFrame, image_id::AbstractString)
    idx = findfirst(==(String(image_id)), String.(target_df.image_id))
    idx === nothing && error("Image id not found in target dataframe: $(image_id)")
    return target_df.processed_img[idx]
end

function plot_prediction_examples(model_name::String, dataset_label::String, pred_subset::DataFrame, target_df::DataFrame)
    class_examples = choose_unique_examples(pred_subset, 1; n = 4)
    no_class_examples = choose_unique_examples(pred_subset, 0; n = 4)
    rows = [
        (label = "high confidence class", df = class_examples),
        (label = "high confidence no_class", df = no_class_examples),
    ]

    n_cols = 4
    fig = Figure(size = (360 * n_cols + 90, 720), figure_padding = 18)
    Label(fig[0, 1:n_cols], "$(model_name) | $(dataset_label)", fontsize = 22, tellwidth = false)

    for (row_idx, row_spec) in enumerate(rows)
        Label(fig[row_idx, 0], row_spec.label, rotation = pi / 2, tellheight = false, fontsize = 16)
        examples = row_spec.df
        for col in 1:n_cols
            cell = GridLayout(fig[row_idx, col])
            if col > nrow(examples)
                ax = Axis(cell[1, 1]; title = "not available")
                hidedecorations!(ax)
                hidespines!(ax)
                text!(ax, 0.5, 0.5; text = "no unique example", space = :relative, align = (:center, :center), fontsize = 13)
                continue
            end

            r = examples[col, :]
            img = lookup_image(target_df, String(r.image_id))
            clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(Float32.(img))
            title = "$(r.channel_name) | $(r.variant)\nP(class)=$(round(Float64(r.prob_class); digits = 3)), conf=$(round(Float64(r.confidence); digits = 3))"
            ax = Axis(cell[1, 1];
                title = title,
                xlabel = "time (resized)",
                ylabel = "sorted trials (resized)",
                titlesize = 12,
                xlabelsize = 11,
                ylabelsize = 11,
                xticklabelsize = 9,
                yticklabelsize = 9,
            )
            hm = heatmap!(
                ax,
                1:size(clipped, 2),
                1:size(clipped, 1),
                permutedims(clipped, (2, 1));
                colormap = cmap,
                colorrange = colorrange,
            )
            Colorbar(
                cell[1, 2],
                hm;
                ticks = (tick_vals, tick_labels),
                ticklabelsize = 8,
                width = 12,
            )
        end
    end

    rowgap!(fig.layout, 24)
    colgap!(fig.layout, 12)
    return fig
end

function plot_prediction_distribution(summary_df::DataFrame, model_name::String)
    sdf = summary_df[String.(summary_df.model_name) .== model_name, :]
    sort!(sdf, :dataset_key)
    labels = String.(sdf.dataset_label)
    x = collect(1:nrow(sdf))
    fig = Figure(size = (1500, 560), figure_padding = 18)
    ax = Axis(
        fig[1, 1];
        title = "$(model_name): predicted class distribution by dataset",
        xlabel = "dataset",
        ylabel = "prediction rate",
        xticks = (x, labels),
        xticklabelrotation = pi / 7,
        titlesize = 22,
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 11,
        yticklabelsize = 12,
    )
    barplot!(ax, x .- 0.18, Float64.(sdf.predicted_class_rate); width = 0.34, label = "class", color = Makie.wong_colors()[2])
    barplot!(ax, x .+ 0.18, Float64.(sdf.predicted_no_class_rate); width = 0.34, label = "no_class", color = Makie.wong_colors()[1])
    ylims!(ax, 0, 1)
    axislegend(ax; position = :rt)
    return fig
end

function save_figures(pred_df::DataFrame, summary_df::DataFrame, target_df::DataFrame, output_dir::AbstractString)
    figure_paths = String[]
    for model_name in unique(String.(pred_df.model_name))
        fig = plot_prediction_distribution(summary_df, model_name)
        path = joinpath(output_dir, "$(stable_slug(model_name))_prediction_distribution.png")
        save(path, fig)
        push!(figure_paths, path)

        for dataset_key in unique(String.(pred_df.dataset_key))
            subset = pred_df[(String.(pred_df.model_name) .== model_name) .& (String.(pred_df.dataset_key) .== dataset_key), :]
            isempty(subset) && continue
            dataset_label = String(subset.dataset_label[1])
            fig_examples = plot_prediction_examples(model_name, dataset_label, subset, target_df)
            path_examples = joinpath(output_dir, "$(stable_slug(model_name))__$(stable_slug(dataset_key))__confidence_examples.png")
            save(path_examples, fig_examples)
            push!(figure_paths, path_examples)
        end
    end
    return DataFrame(file = figure_paths)
end

function save_source_documentation(output_dir::AbstractString)
    text = """
    ResNet fixation generalization experiment
    ========================================

    This run trains ImageNet-pretrained Metalhead ResNet-18 first and then
    ImageNet-pretrained Metalhead ResNet-34 on all currently labeled fixation
    ERP images.

    Training image policy:
    - labeled class rows keep all four mod-4 parts
    - labeled no_class rows are split mod-4 but only one seeded part is kept
    - each image is sorted by its labeled sort variable, z-scored over trials
      per time point, Gaussian-smoothed via notebooks/utils/erp_image_utils.jl,
      and resized to 64x64

    Target data policy:
    - ERP CORE N2PC and N170 use reaction_time_ms with a mod-2 split
    - fixation target datasets use fixation_duration_ms without a mod split
    - rows in the image matrix are sorted trials and columns are time

    Metalhead ResNet API used as reference:
    https://fluxml.ai/Metalhead.jl/dev/api/resnet/
    """
    path = joinpath(output_dir, "README.txt")
    write(path, text)
    return path
end

function run_experiment(; output_dir::AbstractString = OUTPUT_DIR)
    clean_outputs_dir!(output_dir)
    readme_path = save_source_documentation(output_dir)
    device, use_cuda = setup_device()
    batchsize = use_cuda ? TRAIN_BATCHSIZE_GPU : TRAIN_BATCHSIZE_CPU

    println("Output directory: ", output_dir)
    println("Training batch size: ", batchsize)

    println("Preparing labeled fixation training dataset.")
    train_ctx = prepare_training_dataset()
    overview_df, label_dist_df, augmented_dist_df = training_metadata_tables(train_ctx)
    CSV.write(joinpath(output_dir, "training_overview.csv"), overview_df)
    CSV.write(joinpath(output_dir, "training_label_distribution.csv"), label_dist_df)
    CSV.write(joinpath(output_dir, "training_augmented_distribution.csv"), augmented_dist_df)

    println("Preparing target datasets.")
    target_df = materialize_all_target_datasets()
    target_meta_df = select(target_df, Not(:processed_img))
    CSV.write(joinpath(output_dir, "target_images_metadata.csv"), target_meta_df)

    all_prediction_dfs = DataFrame[]
    all_history_dfs = DataFrame[]
    all_train_metric_rows = NamedTuple[]
    all_head_weight_dfs = DataFrame[]

    for (model_rank, spec) in enumerate(model_specs())
        model_name = String(spec.name)
        println()
        println(repeat("=", 88))
        println("Training ", model_name, " (", model_rank, "/", length(model_specs()), ")")
        println(repeat("=", 88))

        Random.seed!(TRAINING_SEED_BASE + 1_000 * model_rank)
        model, pretrained_params_loaded = spec.builder()
        model = device(model)

        model, history_df, train_time_s = train_full_model!(
            model,
            train_ctx.X,
            train_ctx.y;
            model_name = model_name,
            nepochs = TRAIN_EPOCHS,
            lr = TRAIN_LR,
            batchsize = batchsize,
            seed = TRAINING_SEED_BASE + 1_000 * model_rank,
            device = device,
        )
        push!(all_history_dfs, history_df)

        train_metrics = evaluate_training_fit(
            model,
            train_ctx.X,
            train_ctx.y;
            batchsize = PREDICT_BATCHSIZE,
            device = device,
        )
        push!(all_train_metric_rows, (
            model_name = model_name,
            depth = Int(spec.depth),
            nepochs = TRAIN_EPOCHS,
            lr = Float32(TRAIN_LR),
            batchsize = batchsize,
            train_time_s = train_time_s,
            pretrained_params_loaded = pretrained_params_loaded,
            accuracy = train_metrics.accuracy,
            balanced_accuracy = train_metrics.balanced_accuracy,
            macro_f1 = train_metrics.macro_f1,
            precision = train_metrics.precision,
            recall = train_metrics.recall,
            prob_class_mean = train_metrics.prob_class_mean,
            confidence_mean = train_metrics.confidence_mean,
        ))

        push!(all_head_weight_dfs, output_head_weights_df(model, model_name))

        println("Classifying target datasets with ", model_name)
        pred_df = prediction_rows_for_dataset(
            model,
            model_name,
            target_df;
            batchsize = PREDICT_BATCHSIZE,
            device = device,
        )
        push!(all_prediction_dfs, pred_df)

        model = nothing
        CUDA.functional() && CUDA.reclaim()
        GC.gc(true)
    end

    history_df = vcat(all_history_dfs...; cols = :union)
    train_metrics_df = DataFrame(all_train_metric_rows)
    prediction_df = vcat(all_prediction_dfs...; cols = :union)
    summary_df = summarize_predictions(prediction_df)
    consistency_detail_df, consistency_summary_df = mod_split_consistency(prediction_df)
    head_weights_df = vcat(all_head_weight_dfs...; cols = :union)

    CSV.write(joinpath(output_dir, "training_history.csv"), history_df)
    CSV.write(joinpath(output_dir, "training_fit_metrics.csv"), train_metrics_df)
    CSV.write(joinpath(output_dir, "target_predictions.csv"), prediction_df)
    CSV.write(joinpath(output_dir, "target_prediction_summary.csv"), summary_df)
    CSV.write(joinpath(output_dir, "mod_split_consistency_detail.csv"), consistency_detail_df)
    CSV.write(joinpath(output_dir, "mod_split_consistency_summary.csv"), consistency_summary_df)
    CSV.write(joinpath(output_dir, "model_output_head_weights.csv"), head_weights_df)

    figure_df = save_figures(prediction_df, summary_df, target_df, output_dir)
    CSV.write(joinpath(output_dir, "figure_index.csv"), figure_df)

    run_info_df = DataFrame(
        key = [
            "started_at",
            "finished_at",
            "repo_root",
            "output_dir",
            "readme_path",
            "julia_version",
            "flux_version",
            "metalhead_version",
            "cuda_functional",
            "target_size",
            "lowpass_sigma",
            "train_epochs",
            "train_lr",
            "global_seed",
        ],
        value = string.([
            now(),
            now(),
            REPO_ROOT,
            output_dir,
            readme_path,
            VERSION,
            Base.pkgversion(Flux),
            Base.pkgversion(Metalhead),
            CUDA.functional(),
            TARGET_SIZE,
            LOWPASS_SIGMA,
            TRAIN_EPOCHS,
            TRAIN_LR,
            GLOBAL_SEED,
        ]),
    )
    CSV.write(joinpath(output_dir, "run_info.csv"), run_info_df)

    println()
    println("Experiment finished.")
    println("Prediction summary rows: ", nrow(summary_df))
    println("Output directory: ", output_dir)

    return (
        output_dir = output_dir,
        train_overview_df = overview_df,
        train_metrics_df = train_metrics_df,
        target_metadata_df = target_meta_df,
        prediction_df = prediction_df,
        summary_df = summary_df,
        consistency_detail_df = consistency_detail_df,
        consistency_summary_df = consistency_summary_df,
        history_df = history_df,
        head_weights_df = head_weights_df,
        figure_df = figure_df,
    )
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .Week20ResNetFixationGeneralization
    Week20ResNetFixationGeneralization.run_experiment()
end
