module ERPSSLExperimentUtils

using CairoMakie
using CSV
using CUDA
using DataFrames
using Flux
using HDF5
using ImageFiltering: imfilter
using Images: imresize
using MLUtils: DataLoader
using Metalhead
using Random
using Statistics: mean, std

using Flux: onecold, onehotbatch
using LinearAlgebra: transpose

include(joinpath(@__DIR__, "erp_cnn_experiment_utils.jl"))
using .ERPCNNExperimentUtils

export augment_erp_image
export build_binary_resnet18_classifier
export build_full_modulo_sample_plan
export build_simclr_resnet18
export build_unlabeled_candidate_dataset
export build_unlabeled_candidate_sample_plan
export count_params
export evaluate_classifier
export load_real_ssl_context
export plot_confidence_examples_grid
export plot_unlabeled_confidence_examples_grid
export plot_ssl_loss_history
export run_ssl_cv
export run_final_unlabeled_predictions
export select_confidence_examples
export select_unlabeled_confidence_examples
export train_classifier!
export train_simclr!

const DEFAULT_SSL_POOL_SPLIT_K = 4

count_params(m) = sum(length, Flux.trainables(m))
label_name(y::Integer) = y == 1 ? "pattern" : "no pattern"
erp_image_to_plot_matrix(img::AbstractMatrix) = Float32.(permutedims(img, (2, 1)))
short_label_name(y::Integer) = y == 1 ? "pattern" : "no pattern"

function build_full_modulo_sample_plan(labels_df::DataFrame, events::DataFrame; split_k::Int = DEFAULT_SSL_POOL_SPLIT_K)
    rows = NamedTuple[]

    for (row_uid, row) in enumerate(eachrow(labels_df))
        channel = Int(row.channel_int)
        sort_col = row.sort_var_symbol
        class_id = Int(row.erp_class_id)
        binary = Int(class_id > 0)
        image_id = ERPCNNExperimentUtils.image_id_from_row(row)
        groups = split_indices_sorted_modulo(events, sort_col, split_k)

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

    return DataFrame(rows)
end

function build_unlabeled_candidate_sample_plan(labels_df::DataFrame, events::DataFrame, erps;
    sort_vars::Vector{String} = String.(unique(labels_df.sort_variable)),
    max_channel::Int = maximum(Int.(labels_df.channel_int)),
    split_k::Int = DEFAULT_SSL_POOL_SPLIT_K)

    labeled_keys = Set((Int(row.channel_int), String(row.sort_variable)) for row in eachrow(labels_df))
    rows = NamedTuple[]
    group_id = 0

    for sort_var in sort(sort_vars), channel in 1:max_channel
        (channel, sort_var) in labeled_keys && continue
        channel > size(erps, 1) && continue
        Symbol(sort_var) in propertynames(events) || continue

        group_id += 1
        sort_col = Symbol(sort_var)
        image_id = "unlabeled_ch$(lpad(string(channel), 3, '0'))_$(sort_var)"
        groups = split_indices_sorted_modulo(events, sort_col, split_k)

        for (part, idxs) in enumerate(groups)
            push!(rows, (
                sample_id = length(rows) + 1,
                group_id = group_id,
                image_id = image_id,
                image_file = "$(image_id)_mod$(split_k)_part$(part).png",
                channel = channel,
                sort_var = sort_var,
                sort_var_symbol = sort_col,
                class_id = -1,
                binary_label = -1,
                variant = "mod$(split_k)_part$(part)",
                split_k = split_k,
                split_part = part,
                n_trials = length(idxs),
                trial_indices = copy(idxs),
            ))
        end
    end

    return DataFrame(rows)
end

function normalize_per_image(img::AbstractMatrix)
    x = Float32.(img)
    μ = mean(vec(x))
    σ = std(vec(x); corrected = true)
    σ_safe = σ > 1f-6 ? Float32(σ) : 1f0
    return Float32.((x .- Float32(μ)) ./ σ_safe)
end

function random_resized_crop(img::AbstractMatrix, target_size::Tuple{Int, Int}, rng::Random.AbstractRNG;
    scale_range::Tuple{Float32, Float32} = (0.70f0, 1.0f0))

    x = Float32.(img)
    h, w = size(x)
    lo, hi = scale_range
    scale_h = lo + rand(rng, Float32) * (hi - lo)
    scale_w = lo + rand(rng, Float32) * (hi - lo)
    crop_h = clamp(round(Int, h * scale_h), 8, h)
    crop_w = clamp(round(Int, w * scale_w), 8, w)
    top = rand(rng, 1:(h - crop_h + 1))
    left = rand(rng, 1:(w - crop_w + 1))
    crop = @view x[top:(top + crop_h - 1), left:(left + crop_w - 1)]
    return Float32.(imresize(crop, target_size))
end

function add_gaussian_noise(img::AbstractMatrix, rng::Random.AbstractRNG; sigma::Float32 = 0.08f0)
    return Float32.(img) .+ sigma .* randn(rng, Float32, size(img))
end

function random_amplitude_scale(img::AbstractMatrix, rng::Random.AbstractRNG;
    scale_range::Tuple{Float32, Float32} = (0.8f0, 1.2f0))

    lo, hi = scale_range
    factor = lo + rand(rng, Float32) * (hi - lo)
    return Float32.(img) .* factor
end

function random_axis_mask(img::AbstractMatrix, rng::Random.AbstractRNG;
    axis::Symbol = :time,
    max_frac::Float32 = 0.15f0)

    out = copy(Float32.(img))
    h, w = size(out)

    if axis == :time
        span = max(1, round(Int, w * (rand(rng, Float32) * max_frac)))
        start = rand(rng, 1:(w - span + 1))
        out[:, start:(start + span - 1)] .= 0f0
    elseif axis == :trial
        span = max(1, round(Int, h * (rand(rng, Float32) * max_frac)))
        start = rand(rng, 1:(h - span + 1))
        out[start:(start + span - 1), :] .= 0f0
    else
        error("Unknown axis: $axis")
    end

    return out
end

function augment_erp_image(img::AbstractMatrix, target_size::Tuple{Int, Int}, rng::Random.AbstractRNG)
    # Input ERP images follow the shared convention:
    # rows = trials, columns = time.
    x = random_resized_crop(img, target_size, rng)
    x = random_amplitude_scale(x, rng)
    rand(rng) < 0.75 && (x = add_gaussian_noise(x, rng))
    rand(rng) < 0.50 && (x = random_axis_mask(x, rng; axis = :time))
    rand(rng) < 0.35 && (x = random_axis_mask(x, rng; axis = :trial))
    return normalize_per_image(x)
end

function make_contrastive_batch(imgs::Vector{<:AbstractMatrix}, batch_indices::AbstractVector{<:Integer};
    target_size::Tuple{Int, Int},
    rng::Random.AbstractRNG)

    batch_size = length(batch_indices)
    h, w = target_size
    x1 = Array{Float32}(undef, h, w, 1, batch_size)
    x2 = Array{Float32}(undef, h, w, 1, batch_size)

    for (j, idx) in enumerate(batch_indices)
        img = imgs[idx]
        x1[:, :, 1, j] .= augment_erp_image(img, target_size, rng)
        x2[:, :, 1, j] .= augment_erp_image(img, target_size, rng)
    end

    return x1, x2
end

device_array(ref, x) = ref isa CuArray ? cu(x) : x

function nt_xent_loss(z1, z2; temperature::Float32 = 0.10f0)
    batch_size = size(z1, 2)
    z = hcat(z1, z2)
    z = z ./ sqrt.(sum(abs2, z; dims = 1) .+ 1f-8)

    sim = (transpose(z) * z) ./ temperature
    n_total = 2 * batch_size

    pos_idx = vcat((batch_size + 1):(2 * batch_size), 1:batch_size)
    idxs = collect(1:n_total)
    diagmask = 1f9 .* Float32.(reshape(idxs, :, 1) .== reshape(idxs, 1, :))
    pos_mask = Float32.(reshape(pos_idx, :, 1) .== reshape(idxs, 1, :))
    sim_masked = sim .- device_array(sim, diagmask)
    pos_mask = device_array(sim, pos_mask)

    numerator = sum(sim .* pos_mask; dims = 2)
    denominator = Flux.logsumexp(sim_masked; dims = 2)
    return -mean(numerator .- denominator)
end

function build_simclr_resnet18(; in_channels::Int = 1, projection_dim::Int = 128, hidden_dim::Int = 512)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = projection_dim)
    backbone = ERPCNNExperimentUtils.resnet_backbone(base)
    pool = Chain(Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    encoder = Chain(backbone, pool)
    projector = Chain(
        Dense(512 => hidden_dim),
        BatchNorm(hidden_dim, relu),
        Dense(hidden_dim => projection_dim),
    )
    return (encoder = encoder, projector = projector, model = Chain(encoder, projector))
end

function train_simclr!(simclr, imgs::Vector{<:AbstractMatrix};
    target_size::Tuple{Int, Int},
    batchsize::Int = 64,
    epochs::Int = 8,
    lr::Float32 = 1f-3,
    temperature::Float32 = 0.10f0,
    seed::Int = 20260408,
    show_epoch_logs::Bool = false)

    rng = Random.Xoshiro(seed)
    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu

    model = deepcopy(simclr.model) |> device
    opt_state = Flux.setup(Flux.Adam(lr), model)
    loss_history = Float32[]

    Flux.trainmode!(model)
    train_time_s = @elapsed begin
        for epoch in 1:epochs
            perm = randperm(rng, length(imgs))
            epoch_loss = 0f0
            n_batches = 0

            for start_idx in 1:batchsize:length(perm)
                batch_idx = perm[start_idx:min(start_idx + batchsize - 1, length(perm))]
                length(batch_idx) < 2 && continue

                x1, x2 = make_contrastive_batch(imgs, batch_idx; target_size = target_size, rng = rng)
                xb1 = device(x1)
                xb2 = device(x2)

                loss_val, grads = Flux.withgradient(model) do m
                    z1 = m(xb1)
                    z2 = m(xb2)
                    nt_xent_loss(z1, z2; temperature = temperature)
                end

                opt_state, model = Flux.update!(opt_state, model, grads[1])
                epoch_loss += Float32(loss_val)
                n_batches += 1
            end

            avg_loss = epoch_loss / max(1, n_batches)
            push!(loss_history, avg_loss)
            if show_epoch_logs
                println("simclr epoch $(epoch)/$(epochs) | loss=$(round(avg_loss, digits = 5))")
            end
        end
    end

    model_cpu = cpu(model)
    encoder_cpu = cpu(model_cpu.layers[1])
    projector_cpu = cpu(model_cpu.layers[2])

    GC.gc()
    use_gpu && CUDA.reclaim()

    return (
        encoder = encoder_cpu,
        projector = projector_cpu,
        model = model_cpu,
        used_gpu = use_gpu,
        loss_history = loss_history,
        params_n = count_params(model_cpu),
        train_time_s = train_time_s,
        n_images = length(imgs),
    )
end

function build_binary_resnet18_classifier(; in_channels::Int = 1, n_classes::Int = 2)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 128)
    backbone = ERPCNNExperimentUtils.resnet_backbone(base)
    encoder = Chain(backbone, Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    head = Dense(512 => n_classes)
    return (encoder = encoder, head = head, model = Chain(encoder, head))
end

classifier_loss(model, x, y) = Flux.Losses.logitcrossentropy(model(x), y)

function predict_classifier(model, x_cpu::Array{Float32, 4}; device::Function = identity)
    Flux.testmode!(model, true)

    pred_time_s = @elapsed logits_cpu = Float32.(Array(cpu(model(device(x_cpu)))))
    probs_cpu = Float32.(Flux.softmax(logits_cpu; dims = 1))
    prob_pattern = Float32.(vec(probs_cpu[2, :]))
    y_pred = Int.(onecold(logits_cpu, 0:1))
    pred_confidence = Float32.(max.(prob_pattern, 1f0 .- prob_pattern))

    return (
        logits = logits_cpu,
        prob_pattern = prob_pattern,
        y_pred = y_pred,
        pred_confidence = pred_confidence,
        pred_time_s = pred_time_s,
    )
end

function evaluate_classifier(model, x_cpu::Array{Float32, 4}, y::Vector{Int}; device::Function = identity)
    prediction = predict_classifier(model, x_cpu; device = device)
    y_true = Int.(y)
    metrics = compute_metrics(prediction.y_pred, y_true)
    return (
        metrics = metrics,
        y_true = y_true,
        y_pred = prediction.y_pred,
        prob_pattern = prediction.prob_pattern,
        pred_confidence = prediction.pred_confidence,
        val_time_s = prediction.pred_time_s,
    )
end

function train_classifier!(model_parts;
    x_train::Array{Float32, 4},
    y_train::Vector{Int},
    x_val::Array{Float32, 4},
    y_val::Vector{Int},
    batchsize::Int = 32,
    epochs::Int = 8,
    lr::Float32 = 1f-3,
    freeze_encoder::Bool = false,
    seed::Int = 20260408,
    model_name::String = "classifier",
    show_epoch_logs::Bool = false)

    Random.seed!(seed)
    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu

    encoder = deepcopy(model_parts.encoder) |> device
    head = deepcopy(model_parts.head) |> device
    model = Chain(encoder, head)

    y_train_oh = Float32.(Array(onehotbatch(y_train, 0:1)))
    train_loader = DataLoader((x_train, y_train_oh); batchsize = batchsize, shuffle = true)

    target_model = freeze_encoder ? head : model
    opt_state = Flux.setup(Flux.Adam(lr), target_model)
    loss_history = Float32[]

    if freeze_encoder
        Flux.testmode!(encoder, true)
        Flux.trainmode!(head)
    else
        Flux.trainmode!(model)
    end

    train_time_s = @elapsed begin
        for epoch in 1:epochs
            epoch_loss = 0f0
            n_batches = 0

            for (xb_cpu, yb_cpu) in train_loader
                xb = device(xb_cpu)
                yb = device(yb_cpu)

                if freeze_encoder
                    feats = encoder(xb)
                    loss_val, grads = Flux.withgradient(head) do h
                        Flux.Losses.logitcrossentropy(h(feats), yb)
                    end
                    opt_state, head = Flux.update!(opt_state, head, grads[1])
                    model = Chain(encoder, head)
                    epoch_loss += Float32(loss_val)
                else
                    loss_val, grads = Flux.withgradient(model) do m
                        classifier_loss(m, xb, yb)
                    end
                    opt_state, model = Flux.update!(opt_state, model, grads[1])
                    epoch_loss += Float32(loss_val)
                end

                n_batches += 1
            end

            avg_loss = epoch_loss / max(1, n_batches)
            push!(loss_history, avg_loss)
            if show_epoch_logs
                println("$(model_name) epoch $(epoch)/$(epochs) | loss=$(round(avg_loss, digits = 5))")
            end
        end
    end

    trained_model = cpu(model)
    eval_result = evaluate_classifier(trained_model, x_val, y_val; device = identity)

    GC.gc()
    use_gpu && CUDA.reclaim()

    return (
        model = trained_model,
        encoder = trained_model.layers[1],
        head = trained_model.layers[2],
        metrics = eval_result.metrics,
        y_true = eval_result.y_true,
        y_pred = eval_result.y_pred,
        prob_pattern = eval_result.prob_pattern,
        pred_confidence = eval_result.pred_confidence,
        loss_history = loss_history,
        used_gpu = use_gpu,
        params_n = count_params(trained_model),
        train_time_s = train_time_s,
        val_time_s = eval_result.val_time_s,
    )
end

function select_extra_unlabeled_pool(train_supervised_df::DataFrame, pool_train_df::DataFrame)
    labeled_keys = Set((Int(row.group_id), Int(row.split_part)) for row in eachrow(train_supervised_df))
    keep_mask = [!((Int(row.group_id), Int(row.split_part)) in labeled_keys) for row in eachrow(pool_train_df)]
    return copy(pool_train_df[keep_mask, :])
end

function select_pseudo_labels(model, unlabeled_df::DataFrame;
    confidence_threshold::Float32 = 0.90f0,
    min_keep::Int = 32)

    if nrow(unlabeled_df) == 0
        empty_df = copy(unlabeled_df)
        empty_df.pseudo_label = Int[]
        empty_df.prob_pattern = Float32[]
        empty_df.pred_confidence = Float32[]
        return (
            pseudo_df = empty_df,
            n_available = 0,
            n_selected = 0,
            selection_time_s = 0.0,
        )
    end

    x_unlabeled = images_to_tensor(unlabeled_df.processed_img)
    selection_time_s = @elapsed prediction = predict_classifier(model, x_unlabeled; device = identity)

    keep_mask = prediction.pred_confidence .>= confidence_threshold
    keep_n_min = min(min_keep, length(keep_mask))
    if sum(keep_mask) < keep_n_min
        keep_mask .= false
        keep_order = sortperm(prediction.pred_confidence; rev = true)
        keep_mask[keep_order[1:keep_n_min]] .= true
    end

    pseudo_df = copy(unlabeled_df[keep_mask, :])
    pseudo_df.pseudo_label = Int.(prediction.y_pred[keep_mask])
    pseudo_df.prob_pattern = Float32.(prediction.prob_pattern[keep_mask])
    pseudo_df.pred_confidence = Float32.(prediction.pred_confidence[keep_mask])

    return (
        pseudo_df = pseudo_df,
        n_available = nrow(unlabeled_df),
        n_selected = nrow(pseudo_df),
        selection_time_s = selection_time_s,
    )
end

function push_result_row!(rows::Vector{NamedTuple}; fold::Int, experiment::String, pretrain_source::String,
    transfer_mode::String, n_ssl_images::Int, n_train::Int, n_val::Int, metrics, ssl_pretrain_time_s::Real,
    classifier_train_time_s::Real, val_time_s::Real, total_time_s::Real, params_n::Int,
    n_pseudo_available::Int = 0, n_pseudo_selected::Int = 0, pseudo_confidence_threshold::Float32 = 0f0)

    push!(rows, (
        fold = fold,
        experiment = experiment,
        pretrain_source = pretrain_source,
        transfer_mode = transfer_mode,
        n_ssl_images = n_ssl_images,
        n_train = n_train,
        n_val = n_val,
        n_pseudo_available = n_pseudo_available,
        n_pseudo_selected = n_pseudo_selected,
        pseudo_confidence_threshold = pseudo_confidence_threshold,
        accuracy = metrics.accuracy,
        balanced_accuracy = metrics.balanced_accuracy,
        macro_f1 = metrics.macro_f1,
        precision = metrics.precision,
        recall = metrics.recall,
        ssl_pretrain_time_s = Float64(ssl_pretrain_time_s),
        classifier_train_time_s = Float64(classifier_train_time_s),
        val_time_s = Float64(val_time_s),
        total_time_s = Float64(total_time_s),
        params_n = params_n,
    ))
end

function append_prediction_rows!(prediction_rows::Vector{NamedTuple}, supervised_df::DataFrame,
    sample_indices::AbstractVector{<:Integer}, model_result, fold_id::Int;
    experiment::String, pretrain_source::String, transfer_mode::String)

    @assert length(sample_indices) == length(model_result.y_true) == length(model_result.y_pred) ==
        length(model_result.prob_pattern) == length(model_result.pred_confidence)

    for (local_idx, sample_idx) in enumerate(sample_indices)
        row = supervised_df[sample_idx, :]
        push!(prediction_rows, (
            fold = fold_id,
            experiment = experiment,
            pretrain_source = pretrain_source,
            transfer_mode = transfer_mode,
            sample_idx = Int(sample_idx),
            image_id = String(row.image_id),
            group_id = Int(row.group_id),
            split_part = Int(row.split_part),
            variant = String(row.variant),
            sort_var = String(row.sort_var),
            true_label = Int(model_result.y_true[local_idx]),
            pred_label = Int(model_result.y_pred[local_idx]),
            prob_pattern = Float32(model_result.prob_pattern[local_idx]),
            pred_confidence = Float32(model_result.pred_confidence[local_idx]),
            is_correct = Int(model_result.y_true[local_idx]) == Int(model_result.y_pred[local_idx]),
        ))
    end

    return nothing
end

function append_unlabeled_prediction_rows!(prediction_rows::Vector{NamedTuple}, unlabeled_df::DataFrame,
    prediction, experiment::String, pretrain_source::String, transfer_mode::String;
    n_pseudo_selected::Int = 0)

    @assert nrow(unlabeled_df) == length(prediction.y_pred) == length(prediction.prob_pattern) ==
        length(prediction.pred_confidence)

    for row_idx in 1:nrow(unlabeled_df)
        row = unlabeled_df[row_idx, :]
        push!(prediction_rows, (
            experiment = experiment,
            pretrain_source = pretrain_source,
            transfer_mode = transfer_mode,
            sample_idx = row_idx,
            image_id = String(row.image_id),
            image_file = (:image_file in propertynames(row)) ? String(row.image_file) : String(row.image_id),
            group_id = Int(row.group_id),
            split_part = Int(row.split_part),
            variant = String(row.variant),
            sort_var = String(row.sort_var),
            channel = Int(row.channel),
            pred_label = Int(prediction.y_pred[row_idx]),
            prob_pattern = Float32(prediction.prob_pattern[row_idx]),
            pred_confidence = Float32(prediction.pred_confidence[row_idx]),
            n_pseudo_selected = n_pseudo_selected,
        ))
    end

    return nothing
end

function push_unlabeled_summary_row!(rows::Vector{NamedTuple}, prediction, experiment::String,
    pretrain_source::String, transfer_mode::String; n_pseudo_available::Int = 0, n_pseudo_selected::Int = 0)

    pred_labels = Int.(prediction.y_pred)
    prob_pattern = Float32.(prediction.prob_pattern)
    confidence = Float32.(prediction.pred_confidence)

    push!(rows, (
        experiment = experiment,
        pretrain_source = pretrain_source,
        transfer_mode = transfer_mode,
        n_predictions = length(pred_labels),
        predicted_pattern_n = sum(pred_labels .== 1),
        predicted_no_pattern_n = sum(pred_labels .== 0),
        predicted_pattern_rate = mean(pred_labels .== 1),
        prob_pattern_mean = mean(prob_pattern),
        confidence_mean = mean(confidence),
        confidence_min = minimum(confidence),
        confidence_max = maximum(confidence),
        n_pseudo_available = n_pseudo_available,
        n_pseudo_selected = n_pseudo_selected,
    ))

    return nothing
end

function load_real_ssl_context(notebook_dir::AbstractString;
    target_size::Tuple{Int, Int} = (64, 64),
    low_pass_sigma::Float32 = 75.0f0,
    lowpass_kernel_size::Tuple{Int, Int} = (21, 21),
    filter_border::String = "reflect",
    positive_split_k::Int = 4,
    no_class_split_k::Int = 4,
    ssl_pool_split_k::Int = DEFAULT_SSL_POOL_SPLIT_K,
    no_class_pick_seed::Int = Int(mod(time_ns(), typemax(Int) - 2)),
    fold_seed::Int = Int(mod(time_ns(), typemax(Int) - 2)) + 1,
    k_folds::Int = 5)

    data_ctx = prepare_real_fixations_inputs(notebook_dir)
    labels_df = data_ctx.labels_df
    events = data_ctx.events

    no_class_pick_rng = MersenneTwister(no_class_pick_seed)
    supervised_plan_df = build_single_channel_sample_plan(
        labels_df,
        events;
        positive_split_k = positive_split_k,
        no_class_split_k = no_class_split_k,
        no_class_pick_rng = no_class_pick_rng,
    )
    ssl_pool_plan_df = build_full_modulo_sample_plan(labels_df, events; split_k = ssl_pool_split_k)

    supervised_df = materialize_single_channel_dataset(
        data_ctx.erps,
        events,
        supervised_plan_df;
        time_zero_idx = Int(round(0.5f0 * 512)) + 1,
        pipeline_name = :gaussian_reference,
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )

    ssl_pool_df = materialize_single_channel_dataset(
        data_ctx.erps,
        events,
        ssl_pool_plan_df;
        time_zero_idx = Int(round(0.5f0 * 512)) + 1,
        pipeline_name = :gaussian_reference,
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )

    X_supervised = images_to_tensor(supervised_df.processed_img)
    y_binary = Int.(supervised_df.binary_label)
    group_ids = Int.(supervised_df.group_id)
    sort_vars = String.(supervised_df.sort_var)

    fold_val_indices = make_group_kfolds(group_ids, y_binary, sort_vars, k_folds; seed = fold_seed)
    fold_stats_df, fold_sort_stats_df = fold_distribution_tables(fold_val_indices, y_binary, sort_vars)

    return (
        data_ctx = data_ctx,
        supervised_plan_df = supervised_plan_df,
        ssl_pool_plan_df = ssl_pool_plan_df,
        supervised_df = supervised_df,
        ssl_pool_df = ssl_pool_df,
        X_supervised = X_supervised,
        y_binary = y_binary,
        group_ids = group_ids,
        sort_vars = sort_vars,
        fold_val_indices = fold_val_indices,
        fold_stats_df = fold_stats_df,
        fold_sort_stats_df = fold_sort_stats_df,
        no_class_pick_seed = no_class_pick_seed,
        fold_seed = fold_seed,
        target_size = target_size,
    )
end

function build_unlabeled_candidate_dataset(data_ctx;
    target_size::Tuple{Int, Int} = (64, 64),
    low_pass_sigma::Float32 = 75.0f0,
    lowpass_kernel_size::Tuple{Int, Int} = (21, 21),
    filter_border::String = "reflect",
    split_k::Int = DEFAULT_SSL_POOL_SPLIT_K,
    sort_vars::Vector{String} = String.(unique(data_ctx.labels_df.sort_variable)),
    max_channel::Int = maximum(Int.(data_ctx.labels_df.channel_int)))

    sample_plan_df = build_unlabeled_candidate_sample_plan(
        data_ctx.labels_df,
        data_ctx.events,
        data_ctx.erps;
        sort_vars = sort_vars,
        max_channel = max_channel,
        split_k = split_k,
    )

    dataset_df = materialize_single_channel_dataset(
        data_ctx.erps,
        data_ctx.events,
        sample_plan_df;
        time_zero_idx = Int(round(0.5f0 * 512)) + 1,
        pipeline_name = :gaussian_reference,
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )

    return (
        sample_plan_df = sample_plan_df,
        dataset_df = dataset_df,
        X = images_to_tensor(dataset_df.processed_img),
    )
end

function run_ssl_cv(supervised_df::DataFrame, ssl_pool_df::DataFrame, X_supervised::Array{Float32, 4},
    y_binary::Vector{Int}, group_ids::Vector{Int}, fold_val_indices;
    ssl_batchsize::Int = 64,
    ssl_epochs::Int = 4,
    ssl_lr::Float32 = 1f-3,
    ssl_temperature::Float32 = 0.10f0,
    baseline_epochs::Int = 4,
    baseline_lr::Float32 = 1f-3,
    probe_epochs::Int = 6,
    probe_lr::Float32 = 2f-3,
    finetune_epochs::Int = 4,
    finetune_lr::Float32 = 3f-4,
    pseudo_label_threshold::Float32 = 0.90f0,
    pseudo_label_min_keep::Int = 32,
    pseudo_label_epochs::Int = 4,
    pseudo_label_lr::Float32 = 3f-4,
    classifier_batchsize::Int = 32,
    image_size::Tuple{Int, Int} = (64, 64),
    seed::Int = 20260408,
    show_epoch_logs::Bool = false)

    rows = NamedTuple[]
    prediction_rows = NamedTuple[]
    ssl_runs = Dict{Tuple{Int, Symbol}, NamedTuple}()
    n = size(X_supervised, 4)

    ssl_group_ids = Int.(ssl_pool_df.group_id)
    ssl_imgs_all = supervised_df.processed_img
    ssl_pool_imgs_all = ssl_pool_df.processed_img

    for (fold_id, val_idx) in enumerate(fold_val_indices)
        Random.seed!(seed + fold_id)

        train_mask = trues(n)
        train_mask[val_idx] .= false
        train_idx = findall(train_mask)

        x_train = X_supervised[:, :, :, train_idx]
        y_train = y_binary[train_idx]
        x_val = X_supervised[:, :, :, val_idx]
        y_val = y_binary[val_idx]
        train_supervised_df = supervised_df[train_idx, :]

        train_imgs = ssl_imgs_all[train_idx]
        train_group_set = Set(group_ids[train_idx])
        pool_train_idx = findall(gid -> gid in train_group_set, ssl_group_ids)
        pool_train_df = ssl_pool_df[pool_train_idx, :]
        pool_train_imgs = ssl_pool_imgs_all[pool_train_idx]

        use_gpu = CUDA.functional()
        use_gpu && CUDA.reclaim()

        baseline_init = build_binary_resnet18_classifier(in_channels = 1)
        baseline = train_classifier!(baseline_init;
            x_train = x_train,
            y_train = y_train,
            x_val = x_val,
            y_val = y_val,
            batchsize = classifier_batchsize,
            epochs = baseline_epochs,
            lr = baseline_lr,
            freeze_encoder = false,
            seed = seed + 10_000 * fold_id + 1,
            model_name = "supervised_resnet18",
            show_epoch_logs = show_epoch_logs,
        )

        push_result_row!(rows;
            fold = fold_id,
            experiment = "supervised_resnet18",
            pretrain_source = "none",
            transfer_mode = "supervised",
            n_ssl_images = 0,
            n_train = length(train_idx),
            n_val = length(val_idx),
            metrics = baseline.metrics,
            ssl_pretrain_time_s = 0.0,
            classifier_train_time_s = baseline.train_time_s,
            val_time_s = baseline.val_time_s,
            total_time_s = baseline.train_time_s + baseline.val_time_s,
            params_n = baseline.params_n,
        )
        append_prediction_rows!(prediction_rows, supervised_df, val_idx, baseline, fold_id;
            experiment = "supervised_resnet18",
            pretrain_source = "none",
            transfer_mode = "supervised",
        )

        ssl_trainfold = train_simclr!(build_simclr_resnet18(in_channels = 1), train_imgs;
            target_size = image_size,
            batchsize = ssl_batchsize,
            epochs = ssl_epochs,
            lr = ssl_lr,
            temperature = ssl_temperature,
            seed = seed + 10_000 * fold_id + 101,
            show_epoch_logs = show_epoch_logs,
        )
        ssl_runs[(fold_id, :trainfold)] = ssl_trainfold

        ssl_mod4pool = train_simclr!(build_simclr_resnet18(in_channels = 1), pool_train_imgs;
            target_size = image_size,
            batchsize = ssl_batchsize,
            epochs = ssl_epochs,
            lr = ssl_lr,
            temperature = ssl_temperature,
            seed = seed + 10_000 * fold_id + 201,
            show_epoch_logs = show_epoch_logs,
        )
        ssl_runs[(fold_id, :mod4pool)] = ssl_mod4pool

        for (source_key, source_label, ssl_run) in [
            (:trainfold, "trainfold_images", ssl_trainfold),
            (:mod4pool, "same_dataset_mod4_pool", ssl_mod4pool),
        ]
            probe_init = (
                encoder = deepcopy(ssl_run.encoder),
                head = Dense(512 => 2),
            )
            probe = train_classifier!(probe_init;
                x_train = x_train,
                y_train = y_train,
                x_val = x_val,
                y_val = y_val,
                batchsize = classifier_batchsize,
                epochs = probe_epochs,
                lr = probe_lr,
                freeze_encoder = true,
                seed = seed + 10_000 * fold_id + 300 + (source_key == :mod4pool ? 20 : 0),
                model_name = "ssl_linear_probe_$(source_label)",
                show_epoch_logs = show_epoch_logs,
            )

            push_result_row!(rows;
                fold = fold_id,
                experiment = "ssl_linear_probe_$(source_label)",
                pretrain_source = source_label,
                transfer_mode = "linear_probe",
                n_ssl_images = ssl_run.n_images,
                n_train = length(train_idx),
                n_val = length(val_idx),
                metrics = probe.metrics,
                ssl_pretrain_time_s = ssl_run.train_time_s,
                classifier_train_time_s = probe.train_time_s,
                val_time_s = probe.val_time_s,
                total_time_s = ssl_run.train_time_s + probe.train_time_s + probe.val_time_s,
                params_n = probe.params_n,
            )
            append_prediction_rows!(prediction_rows, supervised_df, val_idx, probe, fold_id;
                experiment = "ssl_linear_probe_$(source_label)",
                pretrain_source = source_label,
                transfer_mode = "linear_probe",
            )

            finetune_init = (
                encoder = deepcopy(ssl_run.encoder),
                head = Dense(512 => 2),
            )
            finetune = train_classifier!(finetune_init;
                x_train = x_train,
                y_train = y_train,
                x_val = x_val,
                y_val = y_val,
                batchsize = classifier_batchsize,
                epochs = finetune_epochs,
                lr = finetune_lr,
                freeze_encoder = false,
                seed = seed + 10_000 * fold_id + 400 + (source_key == :mod4pool ? 20 : 0),
                model_name = "ssl_finetune_$(source_label)",
                show_epoch_logs = show_epoch_logs,
            )

            push_result_row!(rows;
                fold = fold_id,
                experiment = "ssl_finetune_$(source_label)",
                pretrain_source = source_label,
                transfer_mode = "finetune",
                n_ssl_images = ssl_run.n_images,
                n_train = length(train_idx),
                n_val = length(val_idx),
                metrics = finetune.metrics,
                ssl_pretrain_time_s = ssl_run.train_time_s,
                classifier_train_time_s = finetune.train_time_s,
                val_time_s = finetune.val_time_s,
                total_time_s = ssl_run.train_time_s + finetune.train_time_s + finetune.val_time_s,
                params_n = finetune.params_n,
            )
            append_prediction_rows!(prediction_rows, supervised_df, val_idx, finetune, fold_id;
                experiment = "ssl_finetune_$(source_label)",
                pretrain_source = source_label,
                transfer_mode = "finetune",
            )

            if source_key == :mod4pool
                pool_extra_df = select_extra_unlabeled_pool(train_supervised_df, pool_train_df)
                pseudo_labels = select_pseudo_labels(finetune.model, pool_extra_df;
                    confidence_threshold = pseudo_label_threshold,
                    min_keep = pseudo_label_min_keep,
                )

                x_student_train = x_train
                y_student_train = y_train
                if pseudo_labels.n_selected > 0
                    x_pseudo = images_to_tensor(pseudo_labels.pseudo_df.processed_img)
                    y_pseudo = Int.(pseudo_labels.pseudo_df.pseudo_label)
                    x_student_train = cat(x_train, x_pseudo; dims = 4)
                    y_student_train = vcat(y_train, y_pseudo)
                end

                student_init = (
                    encoder = deepcopy(finetune.encoder),
                    head = deepcopy(finetune.head),
                )
                pseudo_student = train_classifier!(student_init;
                    x_train = x_student_train,
                    y_train = y_student_train,
                    x_val = x_val,
                    y_val = y_val,
                    batchsize = classifier_batchsize,
                    epochs = pseudo_label_epochs,
                    lr = pseudo_label_lr,
                    freeze_encoder = false,
                    seed = seed + 10_000 * fold_id + 520,
                    model_name = "semi_ssl_pseudolabel_$(source_label)",
                    show_epoch_logs = show_epoch_logs,
                )

                push_result_row!(rows;
                    fold = fold_id,
                    experiment = "semi_ssl_pseudolabel_$(source_label)",
                    pretrain_source = source_label,
                    transfer_mode = "pseudo_label_finetune",
                    n_ssl_images = ssl_run.n_images,
                    n_train = length(train_idx),
                    n_val = length(val_idx),
                    metrics = pseudo_student.metrics,
                    ssl_pretrain_time_s = ssl_run.train_time_s,
                    classifier_train_time_s = finetune.train_time_s + pseudo_student.train_time_s,
                    val_time_s = pseudo_student.val_time_s,
                    total_time_s = ssl_run.train_time_s + finetune.train_time_s +
                        pseudo_labels.selection_time_s + pseudo_student.train_time_s + pseudo_student.val_time_s,
                    params_n = pseudo_student.params_n,
                    n_pseudo_available = pseudo_labels.n_available,
                    n_pseudo_selected = pseudo_labels.n_selected,
                    pseudo_confidence_threshold = pseudo_label_threshold,
                )
                append_prediction_rows!(prediction_rows, supervised_df, val_idx, pseudo_student, fold_id;
                    experiment = "semi_ssl_pseudolabel_$(source_label)",
                    pretrain_source = source_label,
                    transfer_mode = "pseudo_label_finetune",
                )
            end
        end

        use_gpu && CUDA.reclaim()
        GC.gc(true)
    end

    cv_df = DataFrame(rows)
    sort!(cv_df, [:experiment, :fold])
    prediction_df = DataFrame(prediction_rows)
    sort!(prediction_df, [:experiment, :fold, :sample_idx])

    summary_df = combine(
        groupby(cv_df, [:experiment, :pretrain_source, :transfer_mode]),
        :accuracy => mean => :accuracy_mean,
        :accuracy => std => :accuracy_std,
        :balanced_accuracy => mean => :balanced_accuracy_mean,
        :balanced_accuracy => std => :balanced_accuracy_std,
        :macro_f1 => mean => :macro_f1_mean,
        :macro_f1 => std => :macro_f1_std,
        :precision => mean => :precision_mean,
        :recall => mean => :recall_mean,
        :ssl_pretrain_time_s => mean => :ssl_pretrain_time_mean_s,
        :classifier_train_time_s => mean => :classifier_train_time_mean_s,
        :val_time_s => mean => :val_time_mean_s,
        :total_time_s => mean => :total_time_mean_s,
        :n_ssl_images => mean => :n_ssl_images_mean,
        :n_pseudo_available => mean => :n_pseudo_available_mean,
        :n_pseudo_selected => mean => :n_pseudo_selected_mean,
        :pseudo_confidence_threshold => maximum => :pseudo_confidence_threshold,
        :params_n => first => :params_n,
    )
    sort!(summary_df, :balanced_accuracy_mean, rev = true)

    return (
        cv_df = cv_df,
        summary_df = summary_df,
        prediction_df = prediction_df,
        ssl_runs = ssl_runs,
    )
end

function run_final_unlabeled_predictions(supervised_df::DataFrame, ssl_pool_df::DataFrame,
    unlabeled_df::DataFrame, X_supervised::Array{Float32, 4}, y_binary::Vector{Int};
    ssl_batchsize::Int = 64,
    ssl_epochs::Int = 4,
    ssl_lr::Float32 = 1f-3,
    ssl_temperature::Float32 = 0.10f0,
    baseline_epochs::Int = 4,
    baseline_lr::Float32 = 1f-3,
    probe_epochs::Int = 6,
    probe_lr::Float32 = 2f-3,
    finetune_epochs::Int = 4,
    finetune_lr::Float32 = 3f-4,
    pseudo_label_threshold::Float32 = 0.90f0,
    pseudo_label_min_keep::Int = 32,
    pseudo_label_epochs::Int = 4,
    pseudo_label_lr::Float32 = 3f-4,
    classifier_batchsize::Int = 32,
    image_size::Tuple{Int, Int} = (64, 64),
    seed::Int = 20260409,
    show_epoch_logs::Bool = false)

    unlabeled_prediction_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    X_unlabeled = images_to_tensor(unlabeled_df.processed_img)
    X_train = X_supervised
    y_train = y_binary

    function record_predictions!(model, experiment::String, pretrain_source::String, transfer_mode::String;
        n_pseudo_available::Int = 0, n_pseudo_selected::Int = 0)

        prediction = predict_classifier(model, X_unlabeled; device = identity)
        append_unlabeled_prediction_rows!(
            unlabeled_prediction_rows,
            unlabeled_df,
            prediction,
            experiment,
            pretrain_source,
            transfer_mode;
            n_pseudo_selected = n_pseudo_selected,
        )
        push_unlabeled_summary_row!(
            summary_rows,
            prediction,
            experiment,
            pretrain_source,
            transfer_mode;
            n_pseudo_available = n_pseudo_available,
            n_pseudo_selected = n_pseudo_selected,
        )
    end

    baseline_init = build_binary_resnet18_classifier(in_channels = 1)
    baseline = train_classifier!(baseline_init;
        x_train = X_train,
        y_train = y_train,
        x_val = X_train,
        y_val = y_train,
        batchsize = classifier_batchsize,
        epochs = baseline_epochs,
        lr = baseline_lr,
        freeze_encoder = false,
        seed = seed + 1,
        model_name = "final_supervised_resnet18",
        show_epoch_logs = show_epoch_logs,
    )
    record_predictions!(baseline.model, "supervised_resnet18", "none", "supervised")

    train_imgs = supervised_df.processed_img
    pool_imgs = ssl_pool_df.processed_img

    ssl_trainfold = train_simclr!(build_simclr_resnet18(in_channels = 1), train_imgs;
        target_size = image_size,
        batchsize = ssl_batchsize,
        epochs = ssl_epochs,
        lr = ssl_lr,
        temperature = ssl_temperature,
        seed = seed + 101,
        show_epoch_logs = show_epoch_logs,
    )

    ssl_mod4pool = train_simclr!(build_simclr_resnet18(in_channels = 1), pool_imgs;
        target_size = image_size,
        batchsize = ssl_batchsize,
        epochs = ssl_epochs,
        lr = ssl_lr,
        temperature = ssl_temperature,
        seed = seed + 201,
        show_epoch_logs = show_epoch_logs,
    )

    for (source_key, source_label, ssl_run) in [
        (:trainfold, "trainfold_images", ssl_trainfold),
        (:mod4pool, "same_dataset_mod4_pool", ssl_mod4pool),
    ]
        probe_init = (
            encoder = deepcopy(ssl_run.encoder),
            head = Dense(512 => 2),
        )
        probe = train_classifier!(probe_init;
            x_train = X_train,
            y_train = y_train,
            x_val = X_train,
            y_val = y_train,
            batchsize = classifier_batchsize,
            epochs = probe_epochs,
            lr = probe_lr,
            freeze_encoder = true,
            seed = seed + 300 + (source_key == :mod4pool ? 20 : 0),
            model_name = "final_ssl_linear_probe_$(source_label)",
            show_epoch_logs = show_epoch_logs,
        )
        record_predictions!(
            probe.model,
            "ssl_linear_probe_$(source_label)",
            source_label,
            "linear_probe",
        )

        finetune_init = (
            encoder = deepcopy(ssl_run.encoder),
            head = Dense(512 => 2),
        )
        finetune = train_classifier!(finetune_init;
            x_train = X_train,
            y_train = y_train,
            x_val = X_train,
            y_val = y_train,
            batchsize = classifier_batchsize,
            epochs = finetune_epochs,
            lr = finetune_lr,
            freeze_encoder = false,
            seed = seed + 400 + (source_key == :mod4pool ? 20 : 0),
            model_name = "final_ssl_finetune_$(source_label)",
            show_epoch_logs = show_epoch_logs,
        )
        record_predictions!(
            finetune.model,
            "ssl_finetune_$(source_label)",
            source_label,
            "finetune",
        )

        if source_key == :mod4pool
            pseudo_pool = select_extra_unlabeled_pool(supervised_df, ssl_pool_df)
            pseudo_labels = select_pseudo_labels(finetune.model, pseudo_pool;
                confidence_threshold = pseudo_label_threshold,
                min_keep = pseudo_label_min_keep,
            )

            x_student_train = X_train
            y_student_train = y_train
            if pseudo_labels.n_selected > 0
                x_pseudo = images_to_tensor(pseudo_labels.pseudo_df.processed_img)
                y_pseudo = Int.(pseudo_labels.pseudo_df.pseudo_label)
                x_student_train = cat(X_train, x_pseudo; dims = 4)
                y_student_train = vcat(y_train, y_pseudo)
            end

            student_init = (
                encoder = deepcopy(finetune.encoder),
                head = deepcopy(finetune.head),
            )
            pseudo_student = train_classifier!(student_init;
                x_train = x_student_train,
                y_train = y_student_train,
                x_val = X_train,
                y_val = y_train,
                batchsize = classifier_batchsize,
                epochs = pseudo_label_epochs,
                lr = pseudo_label_lr,
                freeze_encoder = false,
                seed = seed + 520,
                model_name = "final_semi_ssl_pseudolabel_$(source_label)",
                show_epoch_logs = show_epoch_logs,
            )
            record_predictions!(
                pseudo_student.model,
                "semi_ssl_pseudolabel_$(source_label)",
                source_label,
                "pseudo_label_finetune";
                n_pseudo_available = pseudo_labels.n_available,
                n_pseudo_selected = pseudo_labels.n_selected,
            )
        end

        CUDA.functional() && CUDA.reclaim()
        GC.gc(true)
    end

    prediction_df = DataFrame(unlabeled_prediction_rows)
    sort!(prediction_df, [:experiment, :sample_idx])
    summary_df = DataFrame(summary_rows)
    sort!(summary_df, :confidence_mean, rev = true)

    return (
        prediction_df = prediction_df,
        summary_df = summary_df,
    )
end

function take_unique_examples(sorted_df::DataFrame, used_image_ids::Set{String}, n::Int)
    keep_idx = Int[]
    for row_idx in 1:nrow(sorted_df)
        image_id = String(sorted_df.image_id[row_idx])
        image_id in used_image_ids && continue
        push!(keep_idx, row_idx)
        push!(used_image_ids, image_id)
        length(keep_idx) >= n && break
    end
    return sorted_df[keep_idx, :]
end

function select_confidence_examples(prediction_df::DataFrame, experiment::AbstractString; n_per_bucket::Int = 4)
    pred_sub = copy(prediction_df[prediction_df.experiment .== experiment, :])
    pred_sub.true_bucket = map(pred_sub.true_label, pred_sub.prob_pattern) do y, p
        if y == 1
            p >= 0.5f0 ? "true_pattern" : "true_pattern"
        else
            p < 0.5f0 ? "true_no_pattern" : "true_no_pattern"
        end
    end

    buckets = [
        ("high_confidence_true_pattern", pred_sub[pred_sub.true_label .== 1, :], :prob_pattern, true),
        ("low_confidence_true_pattern", pred_sub[pred_sub.true_label .== 1, :], :prob_pattern, false),
        ("high_confidence_true_no_pattern", pred_sub[pred_sub.true_label .== 0, :], :prob_pattern, false),
        ("low_confidence_true_no_pattern", pred_sub[pred_sub.true_label .== 0, :], :prob_pattern, true),
    ]

    used_image_ids = Set{String}()
    out = DataFrame[]
    bucket_names = String[]
    for (bucket_name, bucket_df, score_col, rev) in buckets
        bucket_sorted = sort(bucket_df, score_col; rev = rev)
        bucket_selected = take_unique_examples(bucket_sorted, used_image_ids, n_per_bucket)
        bucket_selected.bucket = fill(bucket_name, nrow(bucket_selected))
        push!(out, bucket_selected)
        push!(bucket_names, bucket_name)
    end

    return (
        selected_df = isempty(out) ? DataFrame() : vcat(out...; cols = :union),
        bucket_names = bucket_names,
    )
end

function select_unlabeled_confidence_examples(prediction_df::DataFrame, experiment::AbstractString; n_per_bucket::Int = 4)
    pred_sub = copy(prediction_df[prediction_df.experiment .== experiment, :])

    buckets = [
        ("high_confidence_pred_pattern", pred_sub[pred_sub.pred_label .== 1, :], :prob_pattern, true),
        ("low_confidence_pred_pattern", pred_sub[pred_sub.pred_label .== 1, :], :pred_confidence, false),
        ("high_confidence_pred_no_pattern", pred_sub[pred_sub.pred_label .== 0, :], :prob_pattern, false),
        ("low_confidence_pred_no_pattern", pred_sub[pred_sub.pred_label .== 0, :], :pred_confidence, false),
    ]

    used_image_ids = Set{String}()
    out = DataFrame[]
    bucket_names = String[]
    for (bucket_name, bucket_df, score_col, rev) in buckets
        bucket_sorted = sort(bucket_df, score_col; rev = rev)
        bucket_selected = take_unique_examples(bucket_sorted, used_image_ids, n_per_bucket)
        bucket_selected.bucket = fill(bucket_name, nrow(bucket_selected))
        push!(out, bucket_selected)
        push!(bucket_names, bucket_name)
    end

    return (
        selected_df = isempty(out) ? DataFrame() : vcat(out...; cols = :union),
        bucket_names = bucket_names,
    )
end

function plot_confidence_examples_grid(prediction_df::DataFrame, supervised_df::DataFrame, experiment::AbstractString;
    n_per_bucket::Int = 4,
    figure_title::Union{Nothing, String} = nothing)

    selection = select_confidence_examples(prediction_df, experiment; n_per_bucket = n_per_bucket)
    selected_df = selection.selected_df
    bucket_names = selection.bucket_names

    bucket_labels = Dict(
        "high_confidence_true_pattern" => "High confidence: true pattern",
        "low_confidence_true_pattern" => "Low confidence: true pattern",
        "high_confidence_true_no_pattern" => "High confidence: true no pattern",
        "low_confidence_true_no_pattern" => "Low confidence: true no pattern",
    )

    fig = Figure(size = (420 * (n_per_bucket + 1), 300 * length(bucket_names)))
    title_text = something(figure_title, "Confidence examples for $(experiment)")
    Label(fig[0, 1:(n_per_bucket + 1)], title_text;
        fontsize = 24,
        font = :bold,
        tellwidth = false)

    for (row_idx, bucket_name) in enumerate(bucket_names)
        Label(fig[row_idx, 1], bucket_labels[bucket_name];
            rotation = pi / 2,
            tellheight = false,
            fontsize = 18)

        bucket_df = selected_df[selected_df.bucket .== bucket_name, :]
        for col_idx in 1:n_per_bucket
            ax = Axis(fig[row_idx, col_idx + 1];
                aspect = AxisAspect(1),
                xlabel = "time",
                ylabel = "trials",
                title = "",
                titlesize = 12,
            )

            if col_idx <= nrow(bucket_df)
                sample_idx = Int(bucket_df.sample_idx[col_idx])
                img = supervised_df.processed_img[sample_idx]
                clipped_img, colorrange, _, _, cmap = ERPCNNExperimentUtils.clipped_color_stats_quantile_zero_ticks(
                    img; q_low = 0.01, q_high = 0.99
                )
                plot_img = erp_image_to_plot_matrix(clipped_img)
                heatmap!(
                    ax,
                    1:size(clipped_img, 2),
                    1:size(clipped_img, 1),
                    plot_img;
                    colormap = cmap,
                    colorrange = colorrange,
                )

                prob = round(Float64(bucket_df.prob_pattern[col_idx]), digits = 3)
                conf = round(Float64(bucket_df.pred_confidence[col_idx]), digits = 3)
                true_name = short_label_name(Int(bucket_df.true_label[col_idx]))
                pred_name = short_label_name(Int(bucket_df.pred_label[col_idx]))
                ax.title = "true: $(true_name) | pred: $(pred_name)\np=$(prob) | conf=$(conf)"
            else
                text!(ax, 0.5, 0.5;
                    text = "Not enough\nunique images",
                    align = (:center, :center),
                    fontsize = 18,
                    space = :relative)
            end

            ax.xticks = [1, size(supervised_df.processed_img[1], 2)]
            ax.yticks = [1, size(supervised_df.processed_img[1], 1)]
        end
    end

    return fig
end

function plot_unlabeled_confidence_examples_grid(prediction_df::DataFrame, unlabeled_df::DataFrame, experiment::AbstractString;
    n_per_bucket::Int = 4,
    figure_title::Union{Nothing, String} = nothing)

    selection = select_unlabeled_confidence_examples(prediction_df, experiment; n_per_bucket = n_per_bucket)
    selected_df = selection.selected_df
    bucket_names = selection.bucket_names

    bucket_labels = Dict(
        "high_confidence_pred_pattern" => "High confidence: predicted pattern",
        "low_confidence_pred_pattern" => "Low confidence: predicted pattern",
        "high_confidence_pred_no_pattern" => "High confidence: predicted no pattern",
        "low_confidence_pred_no_pattern" => "Low confidence: predicted no pattern",
    )

    fig = Figure(size = (420 * (n_per_bucket + 1), 300 * length(bucket_names)))
    title_text = something(figure_title, "Unlabeled confidence examples for $(experiment)")
    Label(fig[0, 1:(n_per_bucket + 1)], title_text;
        fontsize = 24,
        font = :bold,
        tellwidth = false)

    for (row_idx, bucket_name) in enumerate(bucket_names)
        Label(fig[row_idx, 1], bucket_labels[bucket_name];
            rotation = pi / 2,
            tellheight = false,
            fontsize = 18)

        bucket_df = selected_df[selected_df.bucket .== bucket_name, :]
        for col_idx in 1:n_per_bucket
            ax = Axis(fig[row_idx, col_idx + 1];
                aspect = AxisAspect(1),
                xlabel = "time",
                ylabel = "trials",
                title = "",
                titlesize = 12,
            )

            if col_idx <= nrow(bucket_df)
                sample_idx = Int(bucket_df.sample_idx[col_idx])
                img = unlabeled_df.processed_img[sample_idx]
                clipped_img, colorrange, _, _, cmap = ERPCNNExperimentUtils.clipped_color_stats_quantile_zero_ticks(
                    img; q_low = 0.01, q_high = 0.99
                )
                plot_img = erp_image_to_plot_matrix(clipped_img)
                heatmap!(
                    ax,
                    1:size(clipped_img, 2),
                    1:size(clipped_img, 1),
                    plot_img;
                    colormap = cmap,
                    colorrange = colorrange,
                )

                prob = round(Float64(bucket_df.prob_pattern[col_idx]), digits = 3)
                conf = round(Float64(bucket_df.pred_confidence[col_idx]), digits = 3)
                pred_name = short_label_name(Int(bucket_df.pred_label[col_idx]))
                ax.title = "pred: $(pred_name)\np=$(prob) | conf=$(conf)"
            else
                text!(ax, 0.5, 0.5;
                    text = "Not enough\nunique images",
                    align = (:center, :center),
                    fontsize = 18,
                    space = :relative)
            end

            ax.xticks = [1, size(unlabeled_df.processed_img[1], 2)]
            ax.yticks = [1, size(unlabeled_df.processed_img[1], 1)]
        end
    end

    return fig
end

function plot_ssl_loss_history(loss_dict::AbstractDict{<:AbstractString, <:AbstractVector})
    fig = Figure(size = (900, 480))
    ax = Axis(fig[1, 1], xlabel = "epoch", ylabel = "loss", title = "SSL / transfer loss histories")
    colors = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple, :black]
    for (idx, (name, values)) in enumerate(pairs(loss_dict))
        lines!(ax, 1:length(values), Float32.(values);
            label = name,
            color = colors[mod1(idx, length(colors))],
            linewidth = 3)
    end
    axislegend(ax; position = :rt)
    return fig
end

end
