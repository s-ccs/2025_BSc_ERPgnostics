module TestOutputLayerHelpers

using CSV
using CairoMakie
using CUDA
using DataFrames
using Flux
using Flux: onecold, onehotbatch
using HDF5
using ImageFiltering: imfilter
using Images: imresize
using LinearAlgebra
using MLUtils: DataLoader
using Metalhead
using PrettyTables
using Printf
using Random
using StatisticalMeasures
using StatisticalMeasures: macro_avg
using Statistics

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: gaussian_kernel, zscore_timepoints

export CLASS_NAMES
export instance_output_summary_df
export instance_neuron_breakdown_df
export instance_neuron_summary_df
export instance_vector_overview_df
export last_layer_overview_df
export model_architecture_trace_df
export output_neuron_summary_df
export output_neuron_weight_ranking_df
export run_outputlayer_analysis
export make_head_weight_fig
export make_instance_vector_fig
export make_output_panel_fig

const CLASS_NAMES = ["no_class", "erp_class"]
const RESULTS_CSV_PATHS = [
    joinpath(@__DIR__, "results", "project-14-at-2026-02-15-23-09-f5225e5c.csv"),
    joinpath(@__DIR__, "results", "project-15-at-2026-02-18-19-35-828515fe.csv"),
]
const FIXATIONS_DATA_DIR = joinpath(@__DIR__, "real_data_sets", "fixations_dataset")
const H5_PATH = joinpath(FIXATIONS_DATA_DIR, "data_fixations.hdf5")
const EVENTS_CSV_PATH = joinpath(FIXATIONS_DATA_DIR, "events.csv")

const SAMPLING_RATE = 512
const PRE_STIM_S = 0.5
const TIME_ZERO_IDX = Int(round(PRE_STIM_S * SAMPLING_RATE)) + 1
const TARGET_SIZE = (64, 64)
const LOWPASS_SIGMA = 75.0f0
const LOWPASS_KERNEL_SIZE = (21, 21)
const FILTER_BORDER = "reflect"

const TABLE_KWARGS = (
    fit_table_in_display_horizontally = false,
    fit_table_in_display_vertically = false,
    display_size = (10_000, 10_000),
    show_omitted_cell_summary = false,
)

const USE_CUDA = CUDA.functional()
if USE_CUDA
    CUDA.allowscalar(false)
end

device(x) = USE_CUDA ? cu(x) : x
to_cpu(x) = USE_CUDA ? cpu(x) : x

function maybe_cuda_reclaim!()
    USE_CUDA && CUDA.reclaim()
    return nothing
end

function load_erps_from_h5(path::AbstractString)
    return h5open(path, "r") do f
        candidates = ["erps", "/erps", "data", "/data/data_fixations.hdf5", "data/data_fixations.hdf5"]
        for key in candidates
            if haskey(f, key)
                obj = f[key]
                if obj isa HDF5.Dataset
                    return read(obj)
                end
            end
        end

        function first_dataset(g)
            for k in keys(g)
                obj = g[k]
                if obj isa HDF5.Dataset
                    return read(obj)
                elseif obj isa HDF5.Group
                    x = first_dataset(obj)
                    x === nothing || return x
                end
            end
            return nothing
        end

        x = first_dataset(f)
        x === nothing && error("No dataset found in HDF5 file: $path")
        return x
    end
end

function load_and_merge_label_sources(paths::Vector{String})
    dfs = DataFrame[]
    for p in paths
        df = CSV.read(p, DataFrame)
        df.source_csv = fill(basename(p), nrow(df))
        push!(dfs, df)
    end

    labels_all = vcat(dfs...; cols = :union)

    if :image in names(labels_all)
        labels_all.updated_at_str = :updated_at in names(labels_all) ? string.(coalesce.(labels_all.updated_at, "")) : fill("", nrow(labels_all))
        labels_all.created_at_str = :created_at in names(labels_all) ? string.(coalesce.(labels_all.created_at, "")) : fill("", nrow(labels_all))
        sort!(labels_all, [:image, :updated_at_str, :created_at_str], rev = [false, true, true])
        labels_merged = unique(labels_all, :image)
    else
        labels_merged = labels_all
    end

    return labels_all, labels_merged
end

parse_class_id(v) = begin
    t = tryparse(Int, strip(string(v)))
    t === nothing ? missing : t
end

function has_required_metadata(row)
    cols = propertynames(row)
    if !(:channel in cols && :sort_variable in cols)
        return false
    end
    return !ismissing(row.channel) && !ismissing(row.sort_variable)
end

function sortvalues_from(df::DataFrame, col::Symbol)
    v = df[!, col]
    if eltype(v) <: Number
        return Float64.(v)
    end
    return collect(v)
end

function extract_channel_trials(erps, events::DataFrame, channel::Int; time_zero_idx::Int = TIME_ZERO_IDX)
    @assert 1 <= channel <= size(erps, 1) "channel out of range"
    data = Float32.(erps[channel, time_zero_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], events[1:n, :]
end

function preprocess_erp_subset(
    data_time_trials::AbstractMatrix,
    events_trials::DataFrame,
    sort_col::Symbol;
    target_size::Tuple{Int, Int} = TARGET_SIZE,
    low_pass_sigma::Float32 = LOWPASS_SIGMA,
    lowpass_kernel_size::Tuple{Int, Int} = LOWPASS_KERNEL_SIZE,
    filter_border::String = FILTER_BORDER,
)
    @assert size(data_time_trials, 2) == nrow(events_trials) "trial count mismatch between matrix and events"
    @assert sort_col in propertynames(events_trials) "sort column not found: $(sort_col)"

    sortvals = sortvalues_from(events_trials, sort_col)
    order = sortperm(sortvals)
    data_sorted = data_time_trials[:, order]

    data_z = zscore_timepoints(data_sorted)
    img_trials_time = Float32.(permutedims(data_z, (2, 1)))

    kernel = gaussian_kernel(low_pass_sigma, size(img_trials_time), target_size, lowpass_kernel_size)
    img_lowpass = Float32.(imfilter(img_trials_time, kernel, filter_border))

    return Float32.(imresize(img_lowpass, target_size))
end

function split_indices_sorted_modulo(events_trials::DataFrame, sort_col::Symbol, k::Int)
    n = nrow(events_trials)
    n == 0 && return [Int[]]
    k_eff = min(max(k, 1), n)

    sortvals = sortvalues_from(events_trials, sort_col)
    order = sortperm(sortvals)

    groups = [Int[] for _ in 1:k_eff]
    for (i, idx) in enumerate(order)
        push!(groups[((i - 1) % k_eff) + 1], idx)
    end

    return groups
end

function image_id_from_row(row)
    cols = propertynames(row)
    if :image_file in cols && !ismissing(row.image_file)
        return String(row.image_file)
    end
    if :image in cols && !ismissing(row.image)
        return String(row.image)
    end
    return "unknown_image"
end

function build_training_dataset(
    erps,
    events::DataFrame,
    labels_df::DataFrame;
    positive_split_k::Int = 4,
    no_class_split_k::Int = 4,
    no_class_pick_rng::AbstractRNG = Random.GLOBAL_RNG,
    target_size::Tuple{Int, Int} = TARGET_SIZE,
)
    rows = NamedTuple[]
    imgs = Matrix{Float32}[]

    for (row_uid, row) in enumerate(eachrow(labels_df))
        channel = Int(row.channel_int)
        sort_col = row.sort_var_symbol
        class_id = Int(row.erp_class_id)
        binary = Int(class_id > 0)
        image_id = image_id_from_row(row)

        data_full, events_full = extract_channel_trials(erps, events, channel)

        if binary == 0
            groups = split_indices_sorted_modulo(events_full, sort_col, no_class_split_k)
            keep_part = rand(no_class_pick_rng, eachindex(groups))
            idxs = groups[keep_part]

            img_part = preprocess_erp_subset(data_full[:, idxs], events_full[idxs, :], sort_col; target_size = target_size)

            push!(rows, (
                sample_id = length(rows) + 1,
                group_id = row_uid,
                image_id = image_id,
                channel = channel,
                sort_var = String(sort_col),
                class_id = class_id,
                binary_label = binary,
                variant = "mod4_keep$(keep_part)",
                n_trials = length(idxs),
            ))
            push!(imgs, img_part)
        else
            groups = split_indices_sorted_modulo(events_full, sort_col, positive_split_k)
            for (part, idxs) in enumerate(groups)
                img_part = preprocess_erp_subset(data_full[:, idxs], events_full[idxs, :], sort_col; target_size = target_size)

                push!(rows, (
                    sample_id = length(rows) + 1,
                    group_id = row_uid,
                    image_id = image_id,
                    channel = channel,
                    sort_var = String(sort_col),
                    class_id = class_id,
                    binary_label = binary,
                    variant = "mod4_part$(part)",
                    n_trials = length(idxs),
                ))
                push!(imgs, img_part)
            end
        end
    end

    out_df = DataFrame(rows)
    out_df.processed_img = imgs
    return out_df
end

function images_to_tensor(imgs)
    h, w = size(imgs[1])
    n = length(imgs)
    x = Array{Float32}(undef, h, w, 1, n)
    for (i, img) in enumerate(imgs)
        x[:, :, 1, i] = Float32.(img)
    end
    return x
end

function make_group_kfolds(group_ids::Vector{Int}, y_binary::Vector{Int}, sort_vars::Vector{String}, k::Int; seed::Int = 20260220)
    @assert length(group_ids) == length(y_binary) == length(sort_vars)

    group_to_indices = Dict{Int, Vector{Int}}()
    group_to_label = Dict{Int, Int}()
    group_to_sort = Dict{Int, String}()
    idx_to_group = Dict{Int, Int}()

    for (idx, gid) in enumerate(group_ids)
        lbl = y_binary[idx]
        sv = sort_vars[idx]

        if haskey(group_to_label, gid)
            @assert group_to_label[gid] == lbl "Group label mismatch for group $(gid)"
            @assert group_to_sort[gid] == sv "Group sort-variable mismatch for group $(gid)"
        else
            group_to_label[gid] = lbl
            group_to_sort[gid] = sv
            group_to_indices[gid] = Int[]
        end

        push!(group_to_indices[gid], idx)
        idx_to_group[idx] = gid
    end

    split_positive_groups = Set{Int}()
    for gid in keys(group_to_indices)
        if group_to_label[gid] == 1 && length(group_to_indices[gid]) > 1
            push!(split_positive_groups, gid)
        end
    end

    locked_groups = [gid for gid in keys(group_to_indices) if !(gid in split_positive_groups)]
    rng = MersenneTwister(seed)

    fold_val_indices = [Int[] for _ in 1:k]
    fold_pos_counts = zeros(Int, k)
    fold_sort_counts = [Dict{String, Int}() for _ in 1:k]
    idx_to_fold = zeros(Int, length(group_ids))

    function add_index_to_fold!(fold::Int, idx::Int)
        push!(fold_val_indices[fold], idx)

        lbl = y_binary[idx]
        sv = sort_vars[idx]
        if lbl == 1
            fold_pos_counts[fold] += 1
        end
        fold_sort_counts[fold][sv] = get(fold_sort_counts[fold], sv, 0) + 1
        idx_to_fold[idx] = fold
    end

    function remove_index_from_fold!(fold::Int, idx::Int)
        pos = findfirst(==(idx), fold_val_indices[fold])
        pos === nothing && return false
        deleteat!(fold_val_indices[fold], pos)

        lbl = y_binary[idx]
        sv = sort_vars[idx]
        if lbl == 1
            fold_pos_counts[fold] -= 1
        end

        fold_sort_counts[fold][sv] = get(fold_sort_counts[fold], sv, 0) - 1
        if fold_sort_counts[fold][sv] <= 0
            delete!(fold_sort_counts[fold], sv)
        end

        idx_to_fold[idx] = 0
        return true
    end

    function add_group_to_fold!(fold::Int, gid::Int)
        for idx in group_to_indices[gid]
            add_index_to_fold!(fold, idx)
        end
    end

    stratum_to_locked_groups = Dict{Tuple{Int, String}, Vector{Int}}()
    for gid in locked_groups
        stratum = (group_to_label[gid], group_to_sort[gid])
        push!(get!(stratum_to_locked_groups, stratum, Int[]), gid)
    end

    for stratum in sort!(collect(keys(stratum_to_locked_groups)); by = x -> (x[1], x[2]))
        gids = shuffle(rng, stratum_to_locked_groups[stratum])
        for (i, gid) in enumerate(gids)
            add_group_to_fold!(((i - 1) % k) + 1, gid)
        end
    end

    sort_to_split_groups = Dict{String, Vector{Int}}()
    for gid in split_positive_groups
        push!(get!(sort_to_split_groups, group_to_sort[gid], Int[]), gid)
    end

    for sv in sort!(collect(keys(sort_to_split_groups)))
        gids = shuffle(rng, sort_to_split_groups[sv])

        for gid in gids
            idxs = shuffle(rng, copy(group_to_indices[gid]))
            used_folds = Set{Int}()

            for idx in idxs
                candidates = [f for f in 1:k if !(f in used_folds)]
                isempty(candidates) && (candidates = collect(1:k))

                best_fold = candidates[1]
                best_key = (typemax(Int), typemax(Int), typemax(Int), typemax(Int))

                for f in candidates
                    key = (
                        fold_pos_counts[f],
                        get(fold_sort_counts[f], sv, 0),
                        length(fold_val_indices[f]),
                        f,
                    )
                    if key < best_key
                        best_key = key
                        best_fold = f
                    end
                end

                add_index_to_fold!(best_fold, idx)
                push!(used_folds, best_fold)
            end
        end
    end

    max_iters = 5000
    iter = 0
    while iter < max_iters
        iter += 1

        src = argmax(fold_pos_counts)
        dst = argmin(fold_pos_counts)
        gap = fold_pos_counts[src] - fold_pos_counts[dst]
        gap <= 1 && break

        best_idx = nothing
        best_key = (typemax(Int), typemax(Int), typemax(Int))

        for idx in fold_val_indices[src]
            y_binary[idx] == 1 || continue
            gid = idx_to_group[idx]

            if gid in split_positive_groups
                dst_has_same_gid = any(idx_to_fold[j] == dst for j in group_to_indices[gid] if j != idx)
                dst_has_same_gid && continue
            elseif length(group_to_indices[gid]) > 1
                continue
            end

            sv = sort_vars[idx]
            key = (
                get(fold_sort_counts[dst], sv, 0),
                get(fold_sort_counts[src], sv, 0),
                idx,
            )

            if key < best_key
                best_key = key
                best_idx = idx
            end
        end

        best_idx === nothing && break

        moved = remove_index_from_fold!(src, best_idx)
        moved || break
        add_index_to_fold!(dst, best_idx)
    end

    for fold in 1:k
        sort!(fold_val_indices[fold])
    end

    return fold_val_indices
end

struct InspectableClassifier{F, P, H}
    feature_maps::F
    pre_logits::P
    head::H
end

Flux.@layer InspectableClassifier

(m::InspectableClassifier)(x) = m.head(m.pre_logits(m.feature_maps(x)))

function inspect_forward(m::InspectableClassifier, x)
    fmap = m.feature_maps(x)
    pooled = m.pre_logits(fmap)
    logits = m.head(pooled)
    return (feature_maps = fmap, pooled = pooled, logits = logits)
end

function build_cnn_3conv_inspectable(; in_channels::Int = 1, n_classes::Int = 2)
    feature_maps = Chain(
        Conv((3, 3), in_channels => 16, pad = 1), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, pad = 1), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, pad = 1), relu,
    )
    pre_logits = Chain(Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    head = Dense(64 => n_classes)
    return InspectableClassifier(feature_maps, pre_logits, head)
end

function collect_arrays_recursive(x, acc = Vector{Any}())
    if x isa AbstractArray
        push!(acc, x)
    elseif x isa NamedTuple
        for k in keys(x)
            collect_arrays_recursive(getfield(x, k), acc)
        end
    elseif x isa Tuple
        for xi in x
            collect_arrays_recursive(xi, acc)
        end
    end
    return acc
end

function load_resnet18_pretrained_greedy!(m)
    src_state = Metalhead.loadweights("resnet18-IMAGENET1K_V1")
    dst_arrays = Flux.trainables(m)
    src_arrays = collect_arrays_recursive(src_state)

    j = 1
    matched = 0
    for d in dst_arrays
        while j <= length(src_arrays) && size(src_arrays[j]) != size(d)
            j += 1
        end
        j <= length(src_arrays) || error("Failed to map pretrained weights for destination size $(size(d))")
        copyto!(d, src_arrays[j])
        matched += 1
        j += 1
    end

    return matched
end

function build_resnet18_pretrained_1ch(; n_classes::Int = 2)
    source = Metalhead.ResNet(18; pretrain = false, inchannels = 3, nclasses = 1000)
    source_matched = load_resnet18_pretrained_greedy!(source)
    source_backbone = Metalhead.backbone(source)

    target = Metalhead.ResNet(18; pretrain = false, inchannels = 1, nclasses = 1000)
    target_backbone = Metalhead.backbone(target)

    src_arrays = Flux.trainables(source_backbone)
    dst_arrays = Flux.trainables(target_backbone)
    @assert length(src_arrays) == length(dst_arrays) == 60

    dst_arrays[1] .= sum(src_arrays[1]; dims = 3)
    for i in 2:length(src_arrays)
        @assert size(src_arrays[i]) == size(dst_arrays[i]) "Backbone size mismatch at array $(i)"
        copyto!(dst_arrays[i], src_arrays[i])
    end

    pre_logits = Chain(Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    head = Dense(512 => n_classes)
    model = InspectableClassifier(target_backbone, pre_logits, head)
    return model, length(dst_arrays), source_matched
end

count_params(m) = sum(length, Flux.trainables(m))

loss_fn(model, x, y) = Flux.Losses.logitcrossentropy(model(x), y)

function stable_softmax(logits::AbstractMatrix)
    x = Float64.(logits)
    x_shift = x .- maximum(x; dims = 1)
    ex = exp.(x_shift)
    return ex ./ sum(ex; dims = 1)
end

function compute_metrics(y_pred::Vector{Int}, y_true::Vector{Int})
    return (
        accuracy = StatisticalMeasures.Accuracy()(y_pred, y_true),
        balanced_accuracy = StatisticalMeasures.BalancedAccuracy()(y_pred, y_true),
        macro_f1 = StatisticalMeasures.MulticlassFScore(; average = macro_avg)(y_pred, y_true),
    )
end

function train_model!(model, train_loader; nepochs::Int, lr::Float32, model_name::String)
    optim = Flux.Adam(lr)
    opt_state = Flux.setup(optim, model)
    Flux.trainmode!(model)

    train_time_s = @elapsed begin
        for epoch in 1:nepochs
            running_loss = 0f0
            n_batches = 0

            for (xb_cpu, yb_cpu) in train_loader
                xb = device(xb_cpu)
                yb = device(yb_cpu)

                loss_val, grads = Flux.withgradient(model) do m
                    loss_fn(m, xb, yb)
                end

                opt_state, model = Flux.update!(opt_state, model, grads[1])
                running_loss += loss_val
                n_batches += 1
            end

            avg_loss = running_loss / max(1, n_batches)
            @info "$(model_name) | epoch $(epoch)/$(nepochs) | train_loss=$(round(avg_loss, digits = 5))"
        end
    end

    return model, train_time_s
end

function run_single_fold_analysis(spec, X_train, y_train, X_val, y_val, val_meta_df, val_images)
    y_train_oh = onehotbatch(y_train, 0:1) |> Array{Float32}
    train_loader = DataLoader((X_train, y_train_oh); batchsize = spec.batchsize, shuffle = true)

    built = spec.builder()
    model = device(built.model)
    params_n = count_params(model)

    maybe_cuda_reclaim!()
    model, train_time_s = train_model!(model, train_loader; nepochs = spec.nepochs, lr = spec.lr, model_name = spec.name)

    Flux.testmode!(model)
    forward = inspect_forward(model, device(X_val))
    logits = Array(to_cpu(forward.logits))
    pooled = Array(to_cpu(forward.pooled))
    feature_maps = Array(to_cpu(forward.feature_maps))
    probs = stable_softmax(logits)
    y_pred = Int.(onecold(logits, 0:1))
    metrics = compute_metrics(y_pred, y_val)

    pred_df = copy(val_meta_df)
    pred_df.val_row = collect(1:nrow(pred_df))
    pred_df.true_label = y_val
    pred_df.true_name = [CLASS_NAMES[y + 1] for y in y_val]
    pred_df.pred_label = y_pred
    pred_df.pred_name = [CLASS_NAMES[y + 1] for y in y_pred]
    pred_df.correct = Int.(y_pred .== y_val)
    pred_df.logit_no_class = vec(Float64.(logits[1, :]))
    pred_df.logit_erp_class = vec(Float64.(logits[2, :]))
    pred_df.prob_no_class = vec(Float64.(probs[1, :]))
    pred_df.prob_erp_class = vec(Float64.(probs[2, :]))
    pred_df.margin = abs.(pred_df.prob_erp_class .- pred_df.prob_no_class)

    return (
        name = spec.name,
        model = cpu(model),
        predictions = pred_df,
        pooled_features = pooled,
        feature_maps = feature_maps,
        val_images = val_images,
        metrics = metrics,
        pretrained_loaded = built.pretrained_loaded,
        source_matched = built.source_matched,
        params_n = params_n,
        feature_dim = size(pooled, 1),
        train_time_s = train_time_s,
    )
end

shape_string(x) = string(size(x))

function layer_label(layer)
    if layer isa Dense
        return "Dense($(size(layer.weight, 2)) => $(size(layer.weight, 1)))"
    elseif layer isa Conv
        return "Conv($(size(layer.weight, 1))x$(size(layer.weight, 2)), $(size(layer.weight, 3)) => $(size(layer.weight, 4)))"
    elseif layer isa MaxPool
        return "MaxPool"
    elseif layer isa Flux.AdaptiveMeanPool
        return "AdaptiveMeanPool"
    elseif layer isa Chain
        return "Chain($(length(layer.layers)) layers)"
    elseif layer isa Function
        return sprint(show, layer)
    end
    return string(typeof(layer))
end

function model_architecture_trace_df(run; val_row::Int = 1)
    @assert 1 <= val_row <= length(run.val_images) "val_row out of range"

    img = Float32.(run.val_images[val_row])
    x = reshape(img, size(img, 1), size(img, 2), 1, 1)

    rows = NamedTuple[]
    push!(rows, (
        stage = "input",
        layer_idx = 0,
        layer_name = "ERP image",
        output_shape = shape_string(x),
        note = "one image as H x W x C x N",
    ))

    z = x
    for (i, layer) in enumerate(run.model.feature_maps.layers)
        z = layer(z)
        push!(rows, (
            stage = "feature_maps",
            layer_idx = i,
            layer_name = layer_label(layer),
            output_shape = shape_string(z),
            note = "",
        ))
    end

    for (i, layer) in enumerate(run.model.pre_logits.layers)
        z = layer(z)
        push!(rows, (
            stage = "pre_logits",
            layer_idx = i,
            layer_name = layer_label(layer),
            output_shape = shape_string(z),
            note = i == length(run.model.pre_logits.layers) ? "last feature vector before dense head" : "",
        ))
    end

    z = run.model.head(z)
    push!(rows, (
        stage = "output_head",
        layer_idx = 1,
        layer_name = layer_label(run.model.head),
        output_shape = shape_string(z),
        note = "2 output neurons = raw logits",
    ))

    return DataFrame(rows)
end

function last_layer_overview_df(run)
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)

    return DataFrame(
        component = [
            "pre_logits vector",
            "dense weight matrix",
            "dense bias vector",
            "output logits",
        ],
        shape = [
            string((size(W, 2), 1)),
            string(size(W)),
            string(size(b)),
            string((size(W, 1), 1)),
        ],
        role = [
            "feature vector entering the output layer",
            "one full weight vector per output neuron",
            "one bias value per output neuron",
            "raw class scores before softmax",
        ],
    )
end

function output_neuron_summary_df(run)
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)

    return DataFrame(
        neuron_idx = collect(1:size(W, 1)),
        class_name = CLASS_NAMES[1:size(W, 1)],
        bias = Float64.(b),
        weight_vector_length = fill(size(W, 2), size(W, 1)),
        weight_l2 = [norm(W[i, :]) for i in 1:size(W, 1)],
        weight_mean = [mean(W[i, :]) for i in 1:size(W, 1)],
        weight_std = [std(W[i, :]) for i in 1:size(W, 1)],
    )
end

function output_neuron_weight_ranking_df(run, neuron_idx::Int; top_k::Int = 16)
    W = Array(run.model.head.weight)
    @assert 1 <= neuron_idx <= size(W, 1) "neuron_idx out of range"

    out = DataFrame(
        neuron_idx = fill(neuron_idx, size(W, 2)),
        class_name = fill(CLASS_NAMES[neuron_idx], size(W, 2)),
        feature_idx = collect(1:size(W, 2)),
        weight = Float64.(vec(W[neuron_idx, :])),
    )
    out.abs_weight = abs.(out.weight)
    out.weight_sign = ifelse.(out.weight .>= 0, "positive", "negative")
    sort!(out, :abs_weight, rev = true)

    return first(out, min(top_k, nrow(out)))
end

function instance_neuron_summary_df(run, val_row::Int)
    @assert 1 <= val_row <= nrow(run.predictions) "val_row out of range"

    pred = run.predictions[val_row, :]
    pooled = vec(run.pooled_features[:, val_row])
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)
    manual = W * pooled .+ b
    probs = vec(stable_softmax(reshape(manual, :, 1)))

    return DataFrame(
        neuron_idx = collect(1:size(W, 1)),
        class_name = CLASS_NAMES[1:size(W, 1)],
        bias = Float64.(b),
        sum_feature_contrib = [Float64(sum(W[i, :] .* pooled)) for i in 1:size(W, 1)],
        manual_logit = Float64.(manual),
        stored_logit = [
            Float64(pred.logit_no_class),
            Float64(pred.logit_erp_class),
        ],
        softmax_probability = Float64.(probs),
    )
end

function instance_neuron_breakdown_df(
    run,
    val_row::Int,
    neuron_idx::Int;
    sort_by::Symbol = :feature_idx,
    top_k::Union{Nothing, Int} = nothing,
)
    @assert 1 <= val_row <= nrow(run.predictions) "val_row out of range"

    pooled = vec(run.pooled_features[:, val_row])
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)
    @assert 1 <= neuron_idx <= size(W, 1) "neuron_idx out of range"

    contributions = W[neuron_idx, :] .* pooled
    total_contrib = Float64(sum(contributions))
    final_logit = Float64(total_contrib + b[neuron_idx])

    out = DataFrame(
        neuron_idx = fill(neuron_idx, length(pooled)),
        class_name = fill(CLASS_NAMES[neuron_idx], length(pooled)),
        feature_idx = collect(1:length(pooled)),
        feature_value = Float64.(pooled),
        weight = Float64.(vec(W[neuron_idx, :])),
        contribution = Float64.(contributions),
    )
    out.abs_feature_value = abs.(out.feature_value)
    out.abs_weight = abs.(out.weight)
    out.abs_contribution = abs.(out.contribution)

    if sort_by != :feature_idx
        @assert sort_by in propertynames(out) "sort_by not found: $(sort_by)"
        sort!(out, sort_by, rev = true)
    end

    if !isnothing(top_k)
        out = first(out, min(top_k, nrow(out)))
    end

    out.bias = fill(Float64(b[neuron_idx]), nrow(out))
    out.logit_if_only_this_feature = out.bias .+ out.contribution
    out.running_logit_display_order = cumsum(out.contribution) .+ out.bias
    out.total_feature_contrib = fill(total_contrib, nrow(out))
    out.final_logit = fill(final_logit, nrow(out))

    return out
end

function head_weight_preview_df(run; max_features::Int = 12)
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)
    n_show = min(max_features, size(W, 2))

    out = DataFrame(
        class_name = CLASS_NAMES[1:size(W, 1)],
        bias = Float64.(b),
        weight_l2 = [norm(W[i, :]) for i in 1:size(W, 1)],
    )

    for j in 1:n_show
        out[!, Symbol("w_$(j)")] = Float64.(W[:, j])
    end

    return out
end

function manual_output_verification_df(run, val_rows::Vector{Int})
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)

    rows = NamedTuple[]
    for row_id in sort(val_rows)
        pooled = run.pooled_features[:, row_id]
        manual = W * pooled .+ b
        push!(rows, (
            val_row = row_id,
            sample_id = Int(run.predictions.sample_id[row_id]),
            manual_no_class = Float64(manual[1]),
            stored_no_class = Float64(run.predictions.logit_no_class[row_id]),
            diff_no_class = Float64(manual[1] - run.predictions.logit_no_class[row_id]),
            manual_erp_class = Float64(manual[2]),
            stored_erp_class = Float64(run.predictions.logit_erp_class[row_id]),
            diff_erp_class = Float64(manual[2] - run.predictions.logit_erp_class[row_id]),
        ))
    end

    return DataFrame(rows)
end

function pooled_feature_preview_df(run, val_rows::Vector{Int}; max_features::Int = 12)
    val_rows_sorted = sort(val_rows)
    out = copy(run.predictions[val_rows_sorted, [:val_row, :sample_id, :true_name, :pred_name, :prob_erp_class]])

    n_show = min(max_features, size(run.pooled_features, 1))
    for j in 1:n_show
        out[!, Symbol("feat_$(j)")] = Float64.(run.pooled_features[j, val_rows_sorted])
    end

    return out
end

function instance_output_summary_df(run, val_row::Int)
    pred = run.predictions[val_row, :]
    pooled = vec(run.pooled_features[:, val_row])
    W = Array(run.model.head.weight)
    b = Array(run.model.head.bias)
    manual = W * pooled .+ b
    contrib_no_class = W[1, :] .* pooled
    contrib_erp_class = W[2, :] .* pooled
    probs = vec(stable_softmax(reshape(manual, :, 1)))

    return DataFrame(
        component = [
            "bias",
            "sum_feature_contrib",
            "manual_logit",
            "stored_logit",
            "softmax_probability",
        ],
        no_class = [
            Float64(b[1]),
            Float64(sum(contrib_no_class)),
            Float64(manual[1]),
            Float64(pred.logit_no_class),
            Float64(probs[1]),
        ],
        erp_class = [
            Float64(b[2]),
            Float64(sum(contrib_erp_class)),
            Float64(manual[2]),
            Float64(pred.logit_erp_class),
            Float64(probs[2]),
        ],
    )
end

function instance_vector_overview_df(run, val_row::Int; sort_by::Symbol = :feature_idx, top_k::Union{Nothing, Int} = nothing)
    @assert 1 <= val_row <= nrow(run.predictions) "val_row out of range"

    pooled = vec(run.pooled_features[:, val_row])
    W = Array(run.model.head.weight)

    out = DataFrame(
        feature_idx = collect(1:length(pooled)),
        pooled_value = Float64.(pooled),
        weight_no_class = Float64.(W[1, :]),
        contrib_no_class = Float64.(W[1, :] .* pooled),
        weight_erp_class = Float64.(W[2, :]),
        contrib_erp_class = Float64.(W[2, :] .* pooled),
    )
    out.contrib_gap = out.contrib_erp_class .- out.contrib_no_class
    out.abs_pooled_value = abs.(out.pooled_value)
    out.abs_contrib_gap = abs.(out.contrib_gap)
    out.abs_contrib_no_class = abs.(out.contrib_no_class)
    out.abs_contrib_erp_class = abs.(out.contrib_erp_class)

    if sort_by != :feature_idx
        @assert sort_by in propertynames(out) "sort_by not found: $(sort_by)"
        sort!(out, sort_by, rev = true)
    end

    if !isnothing(top_k)
        out = first(out, min(top_k, nrow(out)))
    end

    return out
end

function pick_interesting_val_rows(runs::Dict{String, Any}; n_wrong::Int = 2, n_low_margin::Int = 2)
    picked = Int[]
    for run in values(runs)
        wrong_df = copy(run.predictions[run.predictions.correct .== 0, :])
        sort!(wrong_df, :margin)
        append!(picked, Int.(wrong_df.val_row[1:min(n_wrong, nrow(wrong_df))]))

        low_df = sort(copy(run.predictions), :margin)
        append!(picked, Int.(low_df.val_row[1:min(n_low_margin, nrow(low_df))]))
    end
    picked = unique(picked)
    sort!(picked)
    return picked
end

function make_prediction_compare_df(runs::Dict{String, Any})
    cnn_df = select(
        copy(runs["cnn_3conv"].predictions),
        :val_row,
        :sample_id,
        :sort_var,
        :variant,
        :true_name,
        :logit_no_class => :cnn3_logit_no_class,
        :logit_erp_class => :cnn3_logit_erp_class,
        :prob_no_class => :cnn3_prob_no_class,
        :prob_erp_class => :cnn3_prob_erp_class,
        :pred_name => :cnn3_pred_name,
    )

    resnet_df = select(
        copy(runs["resnet18_pretrained_1ch"].predictions),
        :val_row,
        :logit_no_class => :resnet_logit_no_class,
        :logit_erp_class => :resnet_logit_erp_class,
        :prob_no_class => :resnet_prob_no_class,
        :prob_erp_class => :resnet_prob_erp_class,
        :pred_name => :resnet_pred_name,
    )

    out = innerjoin(cnn_df, resnet_df, on = :val_row)
    sort!(out, :val_row)
    return out
end

function load_real_training_data(; data_split_seed::Int)
    erps = load_erps_from_h5(H5_PATH)
    events = CSV.read(EVENTS_CSV_PATH, DataFrame)
    labels_all_df, labels_merged_df = load_and_merge_label_sources(RESULTS_CSV_PATHS)

    labels_merged_df.erp_class_id = [parse_class_id(v) for v in labels_merged_df.erp_class]
    valid_mask = map(v -> !ismissing(v), labels_merged_df.erp_class_id)
    meta_mask = map(has_required_metadata, eachrow(labels_merged_df))
    labels_df = copy(labels_merged_df[valid_mask .& meta_mask, :])

    labels_df.channel_int = Int.(labels_df.channel)
    labels_df.sort_var_symbol = Symbol.(String.(labels_df.sort_variable))
    labels_df.binary_label = Int.(labels_df.erp_class_id .> 0)

    label_summary_df = combine(groupby(labels_df, :binary_label), nrow => :count)
    sort!(label_summary_df, :binary_label)

    no_class_rng = MersenneTwister(data_split_seed)
    train_df = build_training_dataset(erps, events, labels_df; no_class_pick_rng = no_class_rng)

    shuffle_rng = MersenneTwister(data_split_seed + 1)
    perm = randperm(shuffle_rng, nrow(train_df))
    train_df = train_df[perm, :]
    train_df.sample_id = collect(1:nrow(train_df))

    input_stats_df = DataFrame(
        metric = [
            "original_labels",
            "augmented_samples_total",
            "unique_source_groups",
            "no_class_samples",
            "class_samples",
        ],
        value = [
            nrow(labels_df),
            nrow(train_df),
            length(unique(train_df.group_id)),
            count(==(0), train_df.binary_label),
            count(==(1), train_df.binary_label),
        ],
    )

    variant_stats_df = combine(
        groupby(train_df, [:binary_label, :variant]),
        nrow => :count,
        :n_trials => mean => :mean_trials,
        :n_trials => minimum => :min_trials,
        :n_trials => maximum => :max_trials,
    )
    sort!(variant_stats_df, [:binary_label, :variant])

    return (
        erps = erps,
        events = events,
        labels_all_df = labels_all_df,
        labels_df = labels_df,
        label_summary_df = label_summary_df,
        train_df = train_df,
        input_stats_df = input_stats_df,
        variant_stats_df = variant_stats_df,
    )
end

function run_outputlayer_analysis(;
    analysis_fold::Int = 1,
    data_split_seed::Int = 20260308,
    fold_seed::Int = 20260220,
    cnn3_epochs::Int = 6,
    resnet_epochs::Int = 4,
    cnn3_lr::Float32 = 1f-3,
    resnet_lr::Float32 = 1f-4,
    cnn3_batchsize::Int = 32,
    resnet_batchsize::Int = 16,
)
    for p in vcat(RESULTS_CSV_PATHS, [H5_PATH, EVENTS_CSV_PATH])
        @assert isfile(p) "File not found: $p"
    end

    data_bundle = load_real_training_data(; data_split_seed = data_split_seed)
    train_df = data_bundle.train_df

    @assert nrow(train_df[train_df.variant .== "full", :]) == 0
    @assert nrow(train_df[(train_df.binary_label .== 0) .& .!startswith.(train_df.variant, "mod4_keep"), :]) == 0
    @assert nrow(train_df[(train_df.binary_label .== 1) .& .!startswith.(train_df.variant, "mod4_part"), :]) == 0

    X_gray = images_to_tensor(train_df.processed_img)
    y_binary = Int.(train_df.binary_label)
    group_ids = Int.(train_df.group_id)
    sort_vars = String.(train_df.sort_var)

    folds = make_group_kfolds(group_ids, y_binary, sort_vars, 5; seed = fold_seed)
    @assert 1 <= analysis_fold <= length(folds) "analysis_fold out of range"

    val_idx = folds[analysis_fold]
    train_mask = trues(length(y_binary))
    train_mask[val_idx] .= false
    train_idx = findall(train_mask)

    X_train_gray = X_gray[:, :, :, train_idx]
    X_val_gray = X_gray[:, :, :, val_idx]
    y_train = y_binary[train_idx]
    y_val = y_binary[val_idx]
    val_meta_df = train_df[val_idx, [:sample_id, :group_id, :image_id, :sort_var, :variant, :binary_label]]
    val_images = [Float32.(train_df.processed_img[i]) for i in val_idx]

    fold_summary_df = DataFrame(
        metric = ["analysis_fold", "n_train", "n_val", "positives_val", "negatives_val"],
        value = [
            analysis_fold,
            length(train_idx),
            length(val_idx),
            count(==(1), y_val),
            count(==(0), y_val),
        ],
    )

    model_specs = [
        (
            name = "cnn_3conv",
            nepochs = cnn3_epochs,
            lr = cnn3_lr,
            batchsize = cnn3_batchsize,
            builder = () -> (model = build_cnn_3conv_inspectable(), pretrained_loaded = 0, source_matched = 0),
        ),
        (
            name = "resnet18_pretrained_1ch",
            nepochs = resnet_epochs,
            lr = resnet_lr,
            batchsize = resnet_batchsize,
            builder = () -> begin
                model, pretrained_loaded, source_matched = build_resnet18_pretrained_1ch()
                (model = model, pretrained_loaded = pretrained_loaded, source_matched = source_matched)
            end,
        ),
    ]

    analysis_runs = Dict{String, Any}()
    summary_rows = NamedTuple[]

    for spec in model_specs
        @info "Running output-layer analysis for $(spec.name)"
        run = run_single_fold_analysis(spec, X_train_gray, y_train, X_val_gray, y_val, val_meta_df, val_images)
        analysis_runs[spec.name] = run

        push!(summary_rows, (
            model_name = spec.name,
            n_train = length(train_idx),
            n_val = length(val_idx),
            accuracy = run.metrics.accuracy,
            balanced_accuracy = run.metrics.balanced_accuracy,
            macro_f1 = run.metrics.macro_f1,
            feature_dim = run.feature_dim,
            params_n = run.params_n,
            train_time_s = run.train_time_s,
            pretrained_backbone_arrays_loaded = run.pretrained_loaded,
            source_arrays_matched = run.source_matched,
        ))
    end

    analysis_summary_df = DataFrame(summary_rows)
    sort!(analysis_summary_df, :model_name)

    interesting_val_rows = pick_interesting_val_rows(analysis_runs)
    prediction_compare_df = make_prediction_compare_df(analysis_runs)
    interesting_compare_df = prediction_compare_df[in.(prediction_compare_df.val_row, Ref(interesting_val_rows)), :]

    head_previews = Dict(
        "cnn_3conv" => head_weight_preview_df(analysis_runs["cnn_3conv"]),
        "resnet18_pretrained_1ch" => head_weight_preview_df(analysis_runs["resnet18_pretrained_1ch"]),
    )

    manual_output_dfs = Dict(
        "cnn_3conv" => manual_output_verification_df(analysis_runs["cnn_3conv"], interesting_val_rows),
        "resnet18_pretrained_1ch" => manual_output_verification_df(analysis_runs["resnet18_pretrained_1ch"], interesting_val_rows),
    )

    pooled_preview_dfs = Dict(
        "cnn_3conv" => pooled_feature_preview_df(analysis_runs["cnn_3conv"], interesting_val_rows),
        "resnet18_pretrained_1ch" => pooled_feature_preview_df(analysis_runs["resnet18_pretrained_1ch"], interesting_val_rows),
    )

    resnet_first_conv = analysis_runs["resnet18_pretrained_1ch"].model.feature_maps.layers[1].layers[1]
    resnet_head = analysis_runs["resnet18_pretrained_1ch"].model.head
    resnet_config_df = DataFrame(
        metric = [
            "input_channels",
            "stem_conv_weight_shape",
            "output_head_weight_shape",
            "pretrained_backbone_arrays_loaded",
            "source_arrays_matched",
            "adaptation_rule",
        ],
        value = [
            1,
            string(size(resnet_first_conv.weight)),
            string(size(resnet_head.weight)),
            analysis_runs["resnet18_pretrained_1ch"].pretrained_loaded,
            analysis_runs["resnet18_pretrained_1ch"].source_matched,
            "sum_rgb_channels",
        ],
    )

    return (
        use_cuda = USE_CUDA,
        label_summary_df = data_bundle.label_summary_df,
        input_stats_df = data_bundle.input_stats_df,
        variant_stats_df = data_bundle.variant_stats_df,
        fold_summary_df = fold_summary_df,
        analysis_summary_df = analysis_summary_df,
        analysis_runs = analysis_runs,
        interesting_val_rows = interesting_val_rows,
        prediction_compare_df = prediction_compare_df,
        interesting_compare_df = interesting_compare_df,
        head_previews = head_previews,
        manual_output_dfs = manual_output_dfs,
        pooled_preview_dfs = pooled_preview_dfs,
        resnet_config_df = resnet_config_df,
    )
end

function make_head_weight_fig(ctx)
    model_names = ["cnn_3conv", "resnet18_pretrained_1ch"]
    fig = Figure(size = (1200, 760))

    for (row_idx, model_name) in enumerate(model_names)
        run = ctx.analysis_runs[model_name]
        W = Array(run.model.head.weight)

        ax = Axis(
            fig[row_idx, 1],
            title = "$(model_name): dense output head",
            xlabel = "feature index",
            ylabel = "class",
            yticks = (1:2, CLASS_NAMES),
            titlesize = 22,
            xlabelsize = 18,
            ylabelsize = 18,
        )

        hm = heatmap!(ax, 1:size(W, 2), 1:size(W, 1), W; colormap = :balance)
        Colorbar(fig[row_idx, 2], hm, label = "weight")
    end

    return fig
end

function make_instance_vector_fig(run, val_row::Int; top_k::Int = 32, sort_by::Symbol = :abs_contrib_gap)
    df = instance_vector_overview_df(run, val_row; sort_by = sort_by, top_k = top_k)
    pred = run.predictions[val_row, :]

    x = collect(1:nrow(df))
    labels = string.(df.feature_idx)

    fig = Figure(size = (1450, 760))

    ax_vec = Axis(
        fig[1, 1],
        title = "$(run.name) | val_row=$(val_row) | pre-logit vector",
        xlabel = "feature index",
        ylabel = "pooled feature value",
        xticks = (x, labels),
        xticklabelrotation = 0.5,
        titlesize = 22,
        xlabelsize = 18,
        ylabelsize = 18,
    )
    barplot!(
        ax_vec,
        x,
        df.pooled_value;
        color = ifelse.(df.pooled_value .>= 0, :steelblue, :firebrick),
        strokecolor = :black,
        strokewidth = 0.35,
        width = 0.82,
    )

    ax_contrib = Axis(
        fig[2, 1],
        title = @sprintf(
            "Dense contributions | true=%s | pred=%s | logits=(%.3f, %.3f)",
            pred.true_name,
            pred.pred_name,
            pred.logit_no_class,
            pred.logit_erp_class,
        ),
        xlabel = "feature index",
        ylabel = "contribution = weight * feature",
        xticks = (x, labels),
        xticklabelrotation = 0.5,
        titlesize = 20,
        xlabelsize = 18,
        ylabelsize = 18,
    )
    barplot!(
        ax_contrib,
        x .- 0.18,
        df.contrib_no_class;
        color = :gray45,
        strokecolor = :black,
        strokewidth = 0.35,
        width = 0.34,
        label = "no_class",
    )
    barplot!(
        ax_contrib,
        x .+ 0.18,
        df.contrib_erp_class;
        color = :darkorange,
        strokecolor = :black,
        strokewidth = 0.35,
        width = 0.34,
        label = "erp_class",
    )
    axislegend(ax_contrib; position = :rb, labelsize = 14)

    return fig
end

function make_output_panel_fig(ctx; max_rows::Int = 4)
    model_names = ["cnn_3conv", "resnet18_pretrained_1ch"]
    plot_rows = ctx.interesting_val_rows[1:min(max_rows, length(ctx.interesting_val_rows))]
    n_examples = length(plot_rows)
    fig = Figure(size = (1550, max(1, 260 * n_examples * length(model_names))))

    colors_logits = [:gray45, :darkorange]
    colors_probs = [:gray70, :seagreen]

    for (model_idx, model_name) in enumerate(model_names)
        run = ctx.analysis_runs[model_name]
        for (sample_idx, row_id) in enumerate(plot_rows)
            grid_row = (model_idx - 1) * n_examples + sample_idx
            meta = run.predictions[row_id, :]

            ax_img = Axis(
                fig[grid_row, 1],
                title = "$(model_name) | val_row=$(row_id) | sample=$(meta.sample_id)",
                xlabel = "time",
                ylabel = "trial",
                titlesize = 16,
            )
            heatmap!(ax_img, run.val_images[row_id]; colormap = :balance)

            ax_logits = Axis(
                fig[grid_row, 2],
                title = "logits | true=$(meta.true_name) | pred=$(meta.pred_name)",
                ylabel = "logit",
                xticks = (1:2, CLASS_NAMES),
                xticklabelrotation = 0.12,
                titlesize = 16,
            )
            barplot!(
                ax_logits,
                1:2,
                [meta.logit_no_class, meta.logit_erp_class];
                color = colors_logits,
                strokecolor = :black,
                strokewidth = 0.4,
                width = 0.65,
            )

            ax_probs = Axis(
                fig[grid_row, 3],
                title = @sprintf("softmax | p(class)=%.3f", meta.prob_erp_class),
                ylabel = "probability",
                xticks = (1:2, CLASS_NAMES),
                xticklabelrotation = 0.12,
                titlesize = 16,
            )
            barplot!(
                ax_probs,
                1:2,
                [meta.prob_no_class, meta.prob_erp_class];
                color = colors_probs,
                strokecolor = :black,
                strokewidth = 0.4,
                width = 0.65,
            )
            ylims!(ax_probs, 0, 1)
        end
    end

    return fig
end

end
