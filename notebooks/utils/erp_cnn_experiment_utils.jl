module ERPCNNExperimentUtils

using CSV
using DataFrames
using Flux
using HDF5
using ImageFiltering: imfilter
using Images: imresize, dilate, erode, opening, closing, tophat, bothat, morphogradient, morpholaplace
using MLUtils: DataLoader
using Metalhead
using Random
using StatisticalMeasures
using StatisticalMeasures: macro_avg
using Statistics: mean, std

include(joinpath(@__DIR__, "erp_image_utils.jl"))
using .ERPImageUtils: gaussian_kernel, zscore_timepoints, clipped_color_stats_quantile_zero_ticks

export DEFAULT_RESULTS_CSV_BASENAMES
export apply_filter_n, apply_pipeline_to_image, build_single_channel_dataset
export build_single_channel_sample_plan, build_resnet18_single_channel_pretrained
export build_resnet18_single_channel_random, collect_arrays_recursive
export clipped_color_stats_quantile_zero_ticks
export compute_metrics, count_params, evaluate_one_fold, extract_channel_trials
export format_filter_setting, format_pipeline_setting, format_gaussian_setting
export fold_distribution_tables, images_to_tensor, load_and_merge_label_sources
export load_erps_from_h5, load_resnet18_pretrained_project_firstconv!
export make_filter_radius_setting_specs, make_gaussian_sigma_setting_specs
export make_group_kfolds, make_morphological_filter_specs
export make_resnet18_single_channel_model_specs, make_single_channel_pipeline_specs
export materialize_single_channel_dataset, prepare_real_fixations_inputs
export preprocess_pipeline_from_trials, preprocess_sorted_zscore_image
export prepare_training_labels, run_model_cv, sortvalues_from, split_indices_sorted_modulo
export summarize_cv_results, train_one_fold!

const DEFAULT_RESULTS_CSV_BASENAMES = [
    "project-14-at-2026-02-15-23-09-f5225e5c.csv",
    "project-15-at-2026-02-18-19-35-828515fe.csv",
]

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
        updated_at_str = :updated_at in names(labels_all) ? string.(coalesce.(labels_all.updated_at, "")) : fill("", nrow(labels_all))
        created_at_str = :created_at in names(labels_all) ? string.(coalesce.(labels_all.created_at, "")) : fill("", nrow(labels_all))
        labels_all.updated_at_str = updated_at_str
        labels_all.created_at_str = created_at_str

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

function prepare_training_labels(labels_merged_df::DataFrame)
    labels_merged_df.erp_class_id = [parse_class_id(v) for v in labels_merged_df.erp_class]
    valid_mask = map(v -> !ismissing(v), labels_merged_df.erp_class_id)
    meta_mask = map(has_required_metadata, eachrow(labels_merged_df))
    labels_df = copy(labels_merged_df[valid_mask .& meta_mask, :])

    labels_df.channel_int = Int.(labels_df.channel)
    labels_df.sort_var_symbol = Symbol.(String.(labels_df.sort_variable))
    labels_df.binary_label = Int.(labels_df.erp_class_id .> 0)

    return labels_df
end

function prepare_real_fixations_inputs(notebook_dir::AbstractString;
    results_csv_basenames::Vector{String} = DEFAULT_RESULTS_CSV_BASENAMES)

    model_test_dir = normpath(joinpath(notebook_dir, "..", "model_test"))
    results_dir = joinpath(model_test_dir, "results")
    fixations_data_dir = joinpath(model_test_dir, "real_data_sets", "fixations_dataset")

    results_csv_paths = [joinpath(results_dir, name) for name in results_csv_basenames]
    h5_path = joinpath(fixations_data_dir, "data_fixations.hdf5")
    events_csv_path = joinpath(fixations_data_dir, "events.csv")

    for p in vcat(results_csv_paths, [h5_path, events_csv_path])
        @assert isfile(p) "File not found: $p"
    end

    erps = load_erps_from_h5(h5_path)
    events = CSV.read(events_csv_path, DataFrame)
    labels_all_df, labels_merged_df = load_and_merge_label_sources(results_csv_paths)
    labels_df = prepare_training_labels(labels_merged_df)

    return (
        model_test_dir = model_test_dir,
        results_dir = results_dir,
        fixations_data_dir = fixations_data_dir,
        results_csv_paths = results_csv_paths,
        h5_path = h5_path,
        events_csv_path = events_csv_path,
        erps = erps,
        events = events,
        labels_all_df = labels_all_df,
        labels_merged_df = labels_merged_df,
        labels_df = labels_df,
    )
end

function sortvalues_from(df::DataFrame, col::Symbol)
    v = df[!, col]
    if eltype(v) <: Number
        return Float64.(v)
    end
    return collect(v)
end

function extract_channel_trials(erps, events::DataFrame, channel::Int; time_zero_idx::Int)
    @assert 1 <= channel <= size(erps, 1) "channel out of range"
    data = Float32.(erps[channel, time_zero_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], events[1:n, :]
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

function build_single_channel_sample_plan(labels_df::DataFrame, events::DataFrame;
    positive_split_k::Int = 4,
    no_class_split_k::Int = 4,
    no_class_pick_rng::AbstractRNG = Random.GLOBAL_RNG)

    rows = NamedTuple[]

    for (row_uid, row) in enumerate(eachrow(labels_df))
        channel = Int(row.channel_int)
        sort_col = row.sort_var_symbol
        class_id = Int(row.erp_class_id)
        binary = Int(class_id > 0)
        image_id = image_id_from_row(row)

        split_k = binary == 1 ? positive_split_k : no_class_split_k
        groups = split_indices_sorted_modulo(events, sort_col, split_k)

        if binary == 0
            keep_part = rand(no_class_pick_rng, eachindex(groups))
            idxs = groups[keep_part]

            push!(rows, (
                sample_id = length(rows) + 1,
                group_id = row_uid,
                image_id = image_id,
                channel = channel,
                sort_var = String(sort_col),
                sort_var_symbol = sort_col,
                class_id = class_id,
                binary_label = binary,
                variant = "mod$(split_k)_keep$(keep_part)",
                split_k = split_k,
                split_part = keep_part,
                n_trials = length(idxs),
                trial_indices = copy(idxs),
            ))
        else
            for (part, idxs) in enumerate(groups)
                push!(rows, (
                    sample_id = length(rows) + 1,
                    group_id = row_uid,
                    image_id = image_id,
                    channel = channel,
                    sort_var = String(sort_col),
                    sort_var_symbol = sort_col,
                    class_id = class_id,
                    binary_label = binary,
                    variant = "mod$(split_k)_part$(part)",
                    split_k = split_k,
                    split_part = part,
                    n_trials = length(idxs),
                    trial_indices = copy(idxs),
                ))
            end
        end
    end

    return DataFrame(rows)
end

function preprocess_sorted_zscore_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    @assert size(data_time_trials, 2) == nrow(events_trials) "trial count mismatch between matrix and events"
    @assert sort_col in propertynames(events_trials) "sort column not found: $(sort_col)"

    sortvals = sortvalues_from(events_trials, sort_col)
    order = sortperm(sortvals)
    data_sorted = data_time_trials[:, order]

    data_z = zscore_timepoints(data_sorted)
    # Keep the ERP image convention explicit across the project:
    # rows = trials (y-axis), columns = time (x-axis).
    return Float32.(permutedims(data_z, (2, 1)))
end

function apply_filter_n(data::AbstractMatrix, filter_fn::Function; repeats::Int = 1)
    out = Float32.(data)
    for _ in 1:repeats
        out = Float32.(filter_fn(out))
    end
    return out
end

function apply_gaussian_pre_resize(img_trials_time::AbstractMatrix;
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    kernel = gaussian_kernel(low_pass_sigma, size(img_trials_time), target_size, lowpass_kernel_size)
    return Float32.(imfilter(Float32.(img_trials_time), kernel, filter_border))
end

resize_processed_image(img::AbstractMatrix, target_size::Tuple{Int, Int}) = Float32.(imresize(Float32.(img), target_size))

function apply_pipeline_to_image(img_trials_time::AbstractMatrix;
    pipeline_name::Symbol,
    filter_fn::Union{Nothing, Function} = nothing,
    filter_repeats::Int = 1,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    if pipeline_name == :gaussian_reference
        smoothed = apply_gaussian_pre_resize(
            img_trials_time;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        return resize_processed_image(smoothed, target_size)
    end

    filter_fn === nothing && error("A filter function is required for pipeline $(pipeline_name).")

    if pipeline_name == :gaussian_then_filter
        smoothed = apply_gaussian_pre_resize(
            img_trials_time;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        filtered = apply_filter_n(smoothed, filter_fn; repeats = filter_repeats)
        return resize_processed_image(filtered, target_size)
    elseif pipeline_name == :filter_then_gaussian
        filtered = apply_filter_n(img_trials_time, filter_fn; repeats = filter_repeats)
        smoothed = apply_gaussian_pre_resize(
            filtered;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        return resize_processed_image(smoothed, target_size)
    elseif pipeline_name == :filter_only
        filtered = apply_filter_n(img_trials_time, filter_fn; repeats = filter_repeats)
        return resize_processed_image(filtered, target_size)
    else
        error("Unsupported pipeline name: $(pipeline_name)")
    end
end

function preprocess_pipeline_from_trials(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol;
    pipeline_name::Symbol,
    filter_fn::Union{Nothing, Function} = nothing,
    filter_repeats::Int = 1,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    img_trials_time = preprocess_sorted_zscore_image(data_time_trials, events_trials, sort_col)
    return apply_pipeline_to_image(
        img_trials_time;
        pipeline_name = pipeline_name,
        filter_fn = filter_fn,
        filter_repeats = filter_repeats,
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )
end

function materialize_single_channel_dataset(erps, events::DataFrame, sample_plan::DataFrame;
    time_zero_idx::Int,
    pipeline_name::Symbol,
    filter_fn::Union{Nothing, Function} = nothing,
    filter_repeats::Int = 1,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    out_df = copy(sample_plan)
    processed_imgs = Matrix{Float32}[]

    channel_cache = Dict{Int, Tuple{Matrix{Float32}, DataFrame}}()

    for row in eachrow(sample_plan)
        channel = Int(row.channel)
        if !haskey(channel_cache, channel)
            channel_cache[channel] = extract_channel_trials(erps, events, channel; time_zero_idx = time_zero_idx)
        end

        data_full, events_full = channel_cache[channel]
        idxs = row.trial_indices
        data_part = data_full[:, idxs]
        events_part = events_full[idxs, :]

        img = preprocess_pipeline_from_trials(
            data_part,
            events_part,
            row.sort_var_symbol;
            pipeline_name = pipeline_name,
            filter_fn = filter_fn,
            filter_repeats = filter_repeats,
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )

        push!(processed_imgs, img)
    end

    out_df.processed_img = processed_imgs
    return out_df
end

build_single_channel_dataset = materialize_single_channel_dataset

function images_to_tensor(imgs)
    h, w = size(imgs[1])
    n = length(imgs)
    x = Array{Float32}(undef, h, w, 1, n)
    for (i, img) in enumerate(imgs)
        @assert size(img) == (h, w) "All ERP images must share the same trial/time layout."
        # Preserve the matrix orientation for the CNN:
        # height = trials (y-axis), width = time (x-axis).
        x[:, :, 1, i] = Float32.(img)
    end
    return x
end

function make_group_kfolds(group_ids::Vector{Int}, y_binary::Vector{Int}, sort_vars::Vector{String}, k::Int; seed::Int = 20260220)
    @assert length(group_ids) == length(y_binary) == length(sort_vars)

    n = length(group_ids)

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
    idx_to_fold = zeros(Int, n)

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
        if !haskey(stratum_to_locked_groups, stratum)
            stratum_to_locked_groups[stratum] = Int[]
        end
        push!(stratum_to_locked_groups[stratum], gid)
    end

    for stratum in sort!(collect(keys(stratum_to_locked_groups)); by = x -> (x[1], x[2]))
        gids = shuffle(rng, stratum_to_locked_groups[stratum])
        for (i, gid) in enumerate(gids)
            fold = ((i - 1) % k) + 1
            add_group_to_fold!(fold, gid)
        end
    end

    sort_to_split_groups = Dict{String, Vector{Int}}()
    for gid in split_positive_groups
        sv = group_to_sort[gid]
        if !haskey(sort_to_split_groups, sv)
            sort_to_split_groups[sv] = Int[]
        end
        push!(sort_to_split_groups[sv], gid)
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
            else
                if length(group_to_indices[gid]) > 1
                    continue
                end
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

    for gid in split_positive_groups
        folds = unique(idx_to_fold[idx] for idx in group_to_indices[gid])
        expected = min(k, length(group_to_indices[gid]))
        @assert length(folds) == expected "Split constraint violated for positive group $(gid): got $(length(folds)) distinct folds, expected $(expected)."
    end

    for fold in 1:k
        sort!(fold_val_indices[fold])
    end

    return fold_val_indices
end

function fold_distribution_tables(fold_val_indices, y_binary::Vector{Int}, sort_vars::Vector{String})
    fold_rows = NamedTuple[]
    fold_sort_rows = NamedTuple[]

    for (fold, val_idx) in enumerate(fold_val_indices)
        y_fold = y_binary[val_idx]
        push!(fold_rows, (
            fold = fold,
            n_val = length(val_idx),
            n_pos = count(==(1), y_fold),
            n_neg = count(==(0), y_fold),
        ))

        sv_df = combine(groupby(DataFrame(sort_var = sort_vars[val_idx]), :sort_var), nrow => :count)
        sort!(sv_df, :sort_var)
        for r in eachrow(sv_df)
            push!(fold_sort_rows, (fold = fold, sort_var = String(r.sort_var), count = Int(r.count)))
        end
    end

    fold_stats_df = DataFrame(fold_rows)
    fold_sort_stats_df = DataFrame(fold_sort_rows)
    sort!(fold_sort_stats_df, [:fold, :sort_var])

    return fold_stats_df, fold_sort_stats_df
end

function compute_metrics(y_pred::Vector{Int}, y_true::Vector{Int})
    acc = StatisticalMeasures.Accuracy()(y_pred, y_true)
    bacc = StatisticalMeasures.BalancedAccuracy()(y_pred, y_true)
    macro_f1 = StatisticalMeasures.MulticlassFScore(; average = macro_avg)(y_pred, y_true)
    precision = StatisticalMeasures.MulticlassPositivePredictiveValue()(y_pred, y_true)
    recall = StatisticalMeasures.MulticlassTruePositiveRate()(y_pred, y_true)

    return (
        accuracy = acc,
        balanced_accuracy = bacc,
        macro_f1 = macro_f1,
        precision = precision,
        recall = recall,
    )
end

count_params(m) = sum(length, Flux.trainables(m))

loss_fn(model, x, y) = Flux.Losses.logitcrossentropy(model(x), y)

function train_one_fold!(model, train_loader;
    nepochs::Int,
    lr::Float32,
    model_name::String,
    fold_id::Int,
    device::Function = identity,
    show_epoch_logs::Bool = true)

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
            if show_epoch_logs
                @info "$(model_name) | fold $(fold_id) | epoch $(epoch)/$(nepochs) | train_loss=$(round(avg_loss, digits = 5))"
            end
        end
    end

    return model, train_time_s
end

function evaluate_one_fold(model, val_loader; device::Function = identity)
    Flux.testmode!(model)
    y_true = Int[]
    y_pred = Int[]

    val_time_s = @elapsed begin
        for (xb_cpu, yb_cpu) in val_loader
            logits = cpu(model(device(xb_cpu)))
            append!(y_pred, Flux.onecold(logits, 0:1))
            append!(y_true, Flux.onecold(yb_cpu, 0:1))
        end
    end

    metrics = compute_metrics(y_pred, y_true)
    return metrics, y_true, y_pred, val_time_s
end

function run_model_cv(model_name::String, model_builder::Function, X::Array{Float32, 4}, y_binary::Vector{Int}, fold_val_indices;
    nepochs::Int,
    lr::Float32,
    batchsize::Int,
    is_pretrained::Bool = false,
    seed::Int = 0,
    device::Function = identity,
    before_fold!::Function = () -> nothing,
    after_fold!::Function = () -> nothing,
    show_epoch_logs::Bool = true)

    n = size(X, 4)
    rows = NamedTuple[]

    for (fold_id, val_idx) in enumerate(fold_val_indices)
        Random.seed!(seed + fold_id)

        train_mask = trues(n)
        train_mask[val_idx] .= false
        train_idx = findall(train_mask)

        X_train = X[:, :, :, train_idx]
        y_train = Flux.onehotbatch(y_binary[train_idx], 0:1) |> Array{Float32}

        X_val = X[:, :, :, val_idx]
        y_val = Flux.onehotbatch(y_binary[val_idx], 0:1) |> Array{Float32}

        train_loader = DataLoader((X_train, y_train); batchsize = batchsize, shuffle = true)
        val_loader = DataLoader((X_val, y_val); batchsize = batchsize, shuffle = false)

        before_fold!()

        model, pretrain_matched = is_pretrained ? model_builder() : (model_builder(), 0)
        model = device(model)

        params_n = count_params(model)

        model, train_time_s = train_one_fold!(
            model,
            train_loader;
            nepochs = nepochs,
            lr = lr,
            model_name = model_name,
            fold_id = fold_id,
            device = device,
            show_epoch_logs = show_epoch_logs,
        )

        metrics, y_true, y_pred, val_time_s = evaluate_one_fold(model, val_loader; device = device)

        push!(rows, (
            model_name = model_name,
            fold = fold_id,
            n_train = length(train_idx),
            n_val = length(val_idx),
            accuracy = metrics.accuracy,
            balanced_accuracy = metrics.balanced_accuracy,
            macro_f1 = metrics.macro_f1,
            precision = metrics.precision,
            recall = metrics.recall,
            train_time_s = train_time_s,
            val_time_s = val_time_s,
            total_time_s = train_time_s + val_time_s,
            params_n = params_n,
            pretrained_params_loaded = pretrain_matched,
        ))

        after_fold!()
    end

    return DataFrame(rows)
end

function summarize_cv_results(cv_df::DataFrame; group_cols::Vector{Symbol})
    summary_df = combine(
        groupby(cv_df, group_cols),
        :accuracy => mean => :accuracy_mean,
        :accuracy => std => :accuracy_std,
        :balanced_accuracy => mean => :balanced_accuracy_mean,
        :balanced_accuracy => std => :balanced_accuracy_std,
        :macro_f1 => mean => :macro_f1_mean,
        :macro_f1 => std => :macro_f1_std,
        :precision => mean => :precision_mean,
        :recall => mean => :recall_mean,
        :train_time_s => mean => :train_time_mean_s,
        :train_time_s => std => :train_time_std_s,
        :val_time_s => mean => :val_time_mean_s,
        :val_time_s => std => :val_time_std_s,
        :total_time_s => mean => :total_time_mean_s,
        :total_time_s => std => :total_time_std_s,
        :params_n => first => :params_n,
        :pretrained_params_loaded => maximum => :pretrained_params_loaded,
    )
    sort!(summary_df, group_cols)

    single_value_df = combine(
        groupby(cv_df, group_cols),
        :accuracy => mean => :accuracy,
        :balanced_accuracy => mean => :balanced_accuracy,
        :macro_f1 => mean => :macro_f1,
        :precision => mean => :precision,
        :recall => mean => :recall,
        :train_time_s => mean => :train_time_s,
        :val_time_s => mean => :val_time_s,
        :total_time_s => mean => :total_time_s,
        :params_n => first => :params_n,
        :pretrained_params_loaded => maximum => :pretrained_params_loaded,
    )
    sort!(single_value_df, group_cols)

    return summary_df, single_value_df
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

function project_first_conv_weights(src_weight::AbstractArray, dst_inchannels::Int)
    @assert ndims(src_weight) == 4 "Expected a 4D convolution kernel."

    src_inchannels = size(src_weight, 3)
    if dst_inchannels == src_inchannels
        return copy(src_weight)
    end

    projected = mean(src_weight; dims = 3)
    return repeat(projected, 1, 1, dst_inchannels, 1)
end

function load_resnet18_pretrained_project_firstconv!(m)
    src_state = Metalhead.loadweights("resnet18-IMAGENET1K_V1")
    dst_arrays = Flux.trainables(m)
    src_arrays = collect_arrays_recursive(src_state)

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
        @assert size(projected) == size(first_dst) "Projected first conv weight has wrong size."
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
        j <= length(src_arrays) || error("Failed to map pretrained weights for destination size $(size(d))")
        copyto!(d, src_arrays[j])
        matched += 1
        j += 1
    end

    return matched
end

function resnet_backbone(model)
    if isdefined(Metalhead, :backbone)
        return getfield(Metalhead, :backbone)(model)
    end
    return model.layers.layers[1]
end

function resnet_classifier(model)
    if isdefined(Metalhead, :classifier)
        return getfield(Metalhead, :classifier)(model)
    end
    return model.layers.layers[2]
end

function build_resnet18_single_channel_random(; n_classes::Int = 2, in_channels::Int = 1)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 1000)

    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(
        old_head.layers[1],
        old_head.layers[2],
        Dense(in_dim => n_classes),
    )

    # Keep the model as a plain Chain, matching the working GPU path in the
    # other notebooks. Moving a raw Metalhead.ResNet directly to CUDA can leave
    # parts of the state on the CPU and later trigger scalar indexing errors.
    return Chain(features, new_head)
end

function build_resnet18_single_channel_pretrained(; n_classes::Int = 2, in_channels::Int = 1)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 1000)
    matched = load_resnet18_pretrained_project_firstconv!(base)

    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(
        old_head.layers[1],
        old_head.layers[2],
        Dense(in_dim => n_classes),
    )

    model = Chain(features, new_head)
    return model, matched
end

function make_resnet18_single_channel_model_specs(;
    n_classes::Int = 2,
    in_channels::Int = 1,
    nepochs::Int = 8,
    lr::Float32 = 3f-4,
    batchsize::Int = 32)

    return [
        (
            name = "resnet18_random",
            builder = () -> build_resnet18_single_channel_random(n_classes = n_classes, in_channels = in_channels),
            nepochs = nepochs,
            lr = lr,
            batchsize = batchsize,
            is_pretrained = false,
        ),
        (
            name = "resnet18_pretrained",
            builder = () -> build_resnet18_single_channel_pretrained(n_classes = n_classes, in_channels = in_channels),
            nepochs = nepochs,
            lr = lr,
            batchsize = batchsize,
            is_pretrained = true,
        ),
    ]
end

function make_single_channel_pipeline_specs()
    return [
        (
            name = :gaussian_reference,
            label = "sort -> zscore -> gaussian -> resize",
            requires_filter = false,
        ),
        (
            name = :gaussian_then_filter,
            label = "sort -> zscore -> gaussian -> filter -> resize",
            requires_filter = true,
        ),
        (
            name = :filter_then_gaussian,
            label = "sort -> zscore -> filter -> gaussian -> resize",
            requires_filter = true,
        ),
        (
            name = :filter_only,
            label = "sort -> zscore -> filter -> resize",
            requires_filter = true,
        ),
    ]
end

function make_filter_radius_setting_specs()
    return [
        (
            label = "low",
            radius = 0,
            description = "box radius 0 (identity neighborhood)",
        ),
        (
            label = "default",
            radius = 1,
            description = "box radius 1 (library default)",
        ),
        (
            label = "high",
            radius = 2,
            description = "box radius 2 (wider neighborhood)",
        ),
    ]
end

function make_gaussian_sigma_setting_specs()
    return [
        (
            label = "low_25",
            sigma = 25.0f0,
            description = "low-pass sigma factor 25",
        ),
        (
            label = "mid_50",
            sigma = 50.0f0,
            description = "low-pass sigma factor 50",
        ),
        (
            label = "default_75",
            sigma = 75.0f0,
            description = "low-pass sigma factor 75 (previous default)",
        ),
        (
            label = "high_100",
            sigma = 100.0f0,
            description = "low-pass sigma factor 100",
        ),
    ]
end

function format_filter_setting(filter_name::AbstractString, filter_family::AbstractString;
    filter_radius::Union{Nothing, Int} = nothing,
    filter_repeats::Int = 1)

    if filter_family == "reference"
        return "no morphological filter"
    end

    parts = String["family=$(filter_family)"]
    !isnothing(filter_radius) && push!(parts, "radius=$(filter_radius)")
    push!(parts, "repeats=$(filter_repeats)")
    return join(parts, ", ")
end

function format_gaussian_setting(low_pass_sigma::Union{Nothing, Float32};
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::AbstractString)

    if isnothing(low_pass_sigma)
        return "gaussian not applied"
    end

    return "sigma=$(low_pass_sigma), kernel=$(lowpass_kernel_size), border=$(filter_border)"
end

function format_pipeline_setting(pipeline_name::AbstractString, filter_name::AbstractString;
    filter_radius::Union{Nothing, Int} = nothing,
    filter_repeats::Int = 1,
    low_pass_sigma::Union{Nothing, Float32} = nothing,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::AbstractString,
    target_size::Tuple{Int, Int})

    gaussian_desc = isnothing(low_pass_sigma) ?
        "gaussian(not applied)" :
        "gaussian(sigma=$(low_pass_sigma), kernel=$(lowpass_kernel_size), border=$(filter_border))"

    filter_desc = filter_name == "Reference (no filter)" ?
        "no_filter" :
        "$(filter_name)(radius=$(something(filter_radius, 1)), repeats=$(filter_repeats))"

    resize_desc = "resize$(target_size)"

    if pipeline_name == "gaussian_reference"
        return "sort -> zscore -> $(gaussian_desc) -> $(resize_desc)"
    elseif pipeline_name == "gaussian_then_filter"
        return "sort -> zscore -> $(gaussian_desc) -> $(filter_desc) -> $(resize_desc)"
    elseif pipeline_name == "filter_then_gaussian"
        return "sort -> zscore -> $(filter_desc) -> $(gaussian_desc) -> $(resize_desc)"
    elseif pipeline_name == "filter_only"
        return "sort -> zscore -> $(filter_desc) -> $(resize_desc)"
    end

    error("Unsupported pipeline name: $(pipeline_name)")
end

function make_morphological_filter_specs(; radius::Int = 1)
    return [
        (
            name = "Erosion",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(erode(Float32.(data); r = radius)),
        ),
        (
            name = "Dilation",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(dilate(Float32.(data); r = radius)),
        ),
        (
            name = "Opening",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(opening(Float32.(data); r = radius)),
        ),
        (
            name = "Closing",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(closing(Float32.(data); r = radius)),
        ),
        (
            name = "Tophat",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(tophat(Float32.(data); r = radius)),
        ),
        (
            name = "Bothat",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(bothat(Float32.(data); r = radius)),
        ),
        (
            name = "Morphological Gradient",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(morphogradient(Float32.(data); r = radius)),
        ),
        (
            name = "Morphological Laplace",
            family = "morphological",
            radius = radius,
            fn = data -> Float32.(morpholaplace(Float32.(data); r = radius)),
        ),
    ]
end

end
