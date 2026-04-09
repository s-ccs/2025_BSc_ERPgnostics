module ERPSupervisedAugmentationUtils

using CUDA
using CairoMakie
using DataFrames
using Flux
using ImageFiltering: imfilter
using Images: imresize
using Random
using Statistics: mean, std

if isdefined(Main, :ERPCNNExperimentUtils)
    const ERPCNNExperimentUtils = Main.ERPCNNExperimentUtils
else
    include(joinpath(@__DIR__, "erp_cnn_experiment_utils.jl"))
end

const CNNUtils = ERPCNNExperimentUtils

export augmentation_specs, imbalance_strategy_specs
export baseline_corrected_channel_trials
export build_reference_supervised_dataset
export erp_image_to_plot_matrix
export materialize_augmented_training_tensor
export plot_augmentation_preview_grid
export run_supervised_augmentation_cv
export summarize_supervised_augmentation_results

const DEFAULT_BASELINE_WINDOW = nothing
const DEFAULT_POSITIVE_SPLIT_K = 4
const DEFAULT_NO_CLASS_SPLIT_K = 4
const DEFAULT_K_FOLDS = 5

"""
Return the augmentation settings used in the supervised-only experiment.

The `stage` value documents where an operation is applied:
- `:pre_resize_image`: after sorting and z-scoring, before Gaussian smoothing and resizing.
- `:raw_baseline`: before the standard post-stimulus image pipeline.
"""
function augmentation_specs()
    return [
        (
            name = "none",
            label = "Reference only",
            stage = :none,
            copies_pattern = 0,
            copies_no_pattern = 0,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "No augmented copies; standard sort -> z-score -> Gaussian -> resize reference.",
        ),
        (
            name = "trial_shuffle",
            label = "Trial shuffling",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Randomly permute the trial axis after sorting. This is a stress-test because it destroys the sorted-trial structure. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "amplitude_scaling",
            label = "Amplitude scaling",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Multiply the ERP image by a global factor sampled from [0.7, 1.3]. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "time_jitter",
            label = "Small time jitter",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Shift the time axis by +/-5 to +/-10 samples before resizing. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "pink_noise",
            label = "Pink noise",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Add approximate 1/f noise with sigma sampled from [0.1, 0.3]. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "trial_dropout",
            label = "Trial dropout",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Drop 10-20% of rows from the sorted trial image and resize back to 64x64. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "baseline_m100_0",
            label = "Baseline -100..0 ms",
            stage = :raw_baseline,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = (-100, 0),
            description = "Add an alternate view after subtracting each trial's -100..0 ms pre-stimulus baseline. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "baseline_m200_0",
            label = "Baseline -200..0 ms",
            stage = :raw_baseline,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = (-200, 0),
            description = "Add an alternate view after subtracting each trial's -200..0 ms pre-stimulus baseline. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
        (
            name = "safe_combo",
            label = "Safe ERP combo",
            stage = :pre_resize_image,
            copies_pattern = 4,
            copies_no_pattern = 1,
            baseline_window_ms = DEFAULT_BASELINE_WINDOW,
            description = "Combine amplitude scaling, small time jitter, pink noise, and mild trial dropout. Each labeled pattern sample gets 4 augmented views, each labeled no-pattern sample gets 1.",
        ),
    ]
end

augmentation_copy_count(spec, binary_label::Int) = binary_label == 1 ? Int(spec.copies_pattern) : Int(spec.copies_no_pattern)
has_augmentation_copies(spec) = max(Int(spec.copies_pattern), Int(spec.copies_no_pattern)) > 0

function imbalance_strategy_specs()
    return [
        (
            name = "standard_ce",
            label = "Standard CE",
            loss = :crossentropy,
            use_class_weights = false,
            balanced_batches = false,
            focal_gamma = 0.0f0,
            description = "Standard logit cross entropy without explicit imbalance correction.",
        ),
        (
            name = "class_weighted_ce",
            label = "Class-weighted CE",
            loss = :crossentropy,
            use_class_weights = true,
            balanced_batches = false,
            focal_gamma = 0.0f0,
            description = "Logit cross entropy with inverse-frequency class weights.",
        ),
        (
            name = "focal_loss",
            label = "Focal loss",
            loss = :focal,
            use_class_weights = true,
            balanced_batches = false,
            focal_gamma = 2.0f0,
            description = "Focal loss with gamma=2 and inverse-frequency alpha weights.",
        ),
        (
            name = "balanced_batches",
            label = "Balanced batches",
            loss = :crossentropy,
            use_class_weights = false,
            balanced_batches = true,
            focal_gamma = 0.0f0,
            description = "Each mini-batch is sampled with an approximately equal number of pattern and no-pattern samples.",
        ),
    ]
end

function class_weights_inverse_frequency(y::Vector{Int})
    counts = [count(==(cls), y) for cls in 0:1]
    n = length(y)
    weights = Float32[
        counts[i] == 0 ? 0f0 : Float32(n / (2 * counts[i]))
        for i in 1:2
    ]
    m = mean(weights[weights .> 0f0])
    m > 0f0 && (weights ./= m)
    return weights
end

function baseline_indices(time_zero_idx::Int, sampling_rate::Int, window_ms::Tuple{Int, Int})
    start_ms, stop_ms = window_ms
    start_idx = time_zero_idx + Int(round(start_ms / 1000 * sampling_rate))
    stop_idx = time_zero_idx + Int(round(stop_ms / 1000 * sampling_rate)) - 1
    start_idx = max(1, start_idx)
    stop_idx = min(time_zero_idx - 1, stop_idx)
    @assert start_idx <= stop_idx "Invalid baseline window $(window_ms) for time_zero_idx=$(time_zero_idx)."
    return start_idx:stop_idx
end

function baseline_corrected_channel_trials(erps, events::DataFrame, channel::Int;
    time_zero_idx::Int,
    sampling_rate::Int,
    baseline_window_ms::Union{Nothing, Tuple{Int, Int}} = nothing)

    @assert 1 <= channel <= size(erps, 1) "channel out of range"
    n = min(size(erps, 3), nrow(events))
    post = Float32.(erps[channel, time_zero_idx:end, 1:n])

    if baseline_window_ms !== nothing
        bidx = baseline_indices(time_zero_idx, sampling_rate, baseline_window_ms)
        baseline = Float32.(mean(Float32.(erps[channel, bidx, 1:n]); dims = 1))
        post = post .- baseline
    end

    return post, events[1:n, :]
end

function pre_resize_reference_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    return CNNUtils.preprocess_sorted_zscore_image(data_time_trials, events_trials, sort_col)
end

function finish_reference_pipeline(img_trials_time::AbstractMatrix;
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    kernel = CNNUtils.gaussian_kernel(low_pass_sigma, size(img_trials_time), target_size, lowpass_kernel_size)
    smoothed = Float32.(imfilter(Float32.(img_trials_time), kernel, filter_border))
    return Float32.(imresize(smoothed, target_size))
end

function pink_noise_vector(rng::AbstractRNG, n::Int)
    # Voss-McCartney style octave sum. It is an approximation, but it keeps the
    # intended low-frequency dominance without adding another dependency.
    out = zeros(Float32, n)
    block = 1
    while block <= n
        n_blocks = cld(n, block)
        values = randn(rng, Float32, n_blocks)
        for i in 1:n
            out[i] += values[cld(i, block)]
        end
        block *= 2
    end
    s = std(out)
    s > 0f0 && (out ./= s)
    return out
end

function pink_noise_matrix(rng::AbstractRNG, h::Int, w::Int)
    out = Matrix{Float32}(undef, h, w)
    for r in 1:h
        out[r, :] .= pink_noise_vector(rng, w)
    end
    return out
end

function time_shift_no_wrap(img::AbstractMatrix, shift::Int)
    h, w = size(img)
    out = zeros(Float32, h, w)
    if shift == 0
        out .= Float32.(img)
    elseif shift > 0
        out[:, (shift + 1):end] .= Float32.(img[:, 1:(end - shift)])
    else
        s = -shift
        out[:, 1:(end - s)] .= Float32.(img[:, (s + 1):end])
    end
    return out
end

function apply_pre_resize_augmentation(img::AbstractMatrix, spec, rng::AbstractRNG)
    name = Symbol(spec.name)
    out = Float32.(img)

    if name == :trial_shuffle
        out = out[shuffle(rng, 1:size(out, 1)), :]
    elseif name == :amplitude_scaling
        out .*= rand(rng, Float32) * 0.6f0 + 0.7f0
    elseif name == :time_jitter
        mag = rand(rng, 5:10)
        shift = rand(rng, Bool) ? mag : -mag
        shift = clamp(shift, -size(out, 2) + 1, size(out, 2) - 1)
        out = time_shift_no_wrap(out, shift)
    elseif name == :pink_noise
        sigma = rand(rng, Float32) * 0.2f0 + 0.1f0
        out .+= sigma .* pink_noise_matrix(rng, size(out, 1), size(out, 2))
    elseif name == :trial_dropout
        keep_fraction = 1f0 - (rand(rng, Float32) * 0.1f0 + 0.1f0)
        keep_n = max(4, Int(round(size(out, 1) * keep_fraction)))
        keep_idx = sort(shuffle(rng, 1:size(out, 1))[1:keep_n])
        out = out[keep_idx, :]
    elseif name == :safe_combo
        out .*= rand(rng, Float32) * 0.4f0 + 0.8f0
        mag = rand(rng, 5:8)
        out = time_shift_no_wrap(out, rand(rng, Bool) ? mag : -mag)
        out .+= (rand(rng, Float32) * 0.1f0 + 0.05f0) .* pink_noise_matrix(rng, size(out, 1), size(out, 2))
        keep_fraction = 1f0 - (rand(rng, Float32) * 0.05f0 + 0.05f0)
        keep_n = max(4, Int(round(size(out, 1) * keep_fraction)))
        keep_idx = sort(shuffle(rng, 1:size(out, 1))[1:keep_n])
        out = out[keep_idx, :]
    elseif name == :none
        return out
    else
        error("Unsupported pre-resize augmentation: $(spec.name)")
    end

    return Float32.(out)
end

function build_reference_supervised_dataset(notebook_dir::AbstractString;
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String,
    time_zero_idx::Int,
    positive_split_k::Int = DEFAULT_POSITIVE_SPLIT_K,
    no_class_split_k::Int = DEFAULT_NO_CLASS_SPLIT_K,
    no_class_pick_seed::Int,
    fold_seed::Int,
    k_folds::Int = DEFAULT_K_FOLDS)

    data_ctx = CNNUtils.prepare_real_fixations_inputs(notebook_dir)
    sample_plan_df = CNNUtils.build_single_channel_sample_plan(
        data_ctx.labels_df,
        data_ctx.events;
        positive_split_k = positive_split_k,
        no_class_split_k = no_class_split_k,
        no_class_pick_rng = MersenneTwister(no_class_pick_seed),
    )

    supervised_df = CNNUtils.materialize_single_channel_dataset(
        data_ctx.erps,
        data_ctx.events,
        sample_plan_df;
        time_zero_idx = time_zero_idx,
        pipeline_name = :gaussian_reference,
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )

    X = CNNUtils.images_to_tensor(supervised_df.processed_img)
    y = Int.(supervised_df.binary_label)
    group_ids = Int.(supervised_df.group_id)
    sort_vars = String.(supervised_df.sort_var)
    folds = CNNUtils.make_group_kfolds(group_ids, y, sort_vars, k_folds; seed = fold_seed)
    fold_stats_df, fold_sort_stats_df = CNNUtils.fold_distribution_tables(folds, y, sort_vars)

    return (
        data_ctx = data_ctx,
        sample_plan_df = sample_plan_df,
        supervised_df = supervised_df,
        X = X,
        y = y,
        group_ids = group_ids,
        sort_vars = sort_vars,
        fold_val_indices = folds,
        fold_stats_df = fold_stats_df,
        fold_sort_stats_df = fold_sort_stats_df,
    )
end

function materialize_augmented_image_for_row(erps, events::DataFrame, row;
    spec,
    rng::AbstractRNG,
    sampling_rate::Int,
    time_zero_idx::Int,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    data_full, events_full = baseline_corrected_channel_trials(
        erps,
        events,
        Int(row.channel);
        time_zero_idx = time_zero_idx,
        sampling_rate = sampling_rate,
        baseline_window_ms = spec.baseline_window_ms,
    )

    idxs = row.trial_indices
    data_part = data_full[:, idxs]
    events_part = events_full[idxs, :]
    img = pre_resize_reference_image(data_part, events_part, row.sort_var_symbol)

    if spec.stage == :pre_resize_image
        img = apply_pre_resize_augmentation(img, spec, rng)
    elseif spec.stage == :raw_baseline || spec.stage == :none
        # The raw-baseline view has already been changed before z-scoring.
    else
        error("Unsupported augmentation stage: $(spec.stage)")
    end

    return finish_reference_pipeline(
        img;
        target_size = target_size,
        low_pass_sigma = low_pass_sigma,
        lowpass_kernel_size = lowpass_kernel_size,
        filter_border = filter_border,
    )
end

function materialize_augmented_training_tensor(erps, events::DataFrame, supervised_df::DataFrame,
    train_indices::Vector{Int}, spec;
    rng::AbstractRNG,
    sampling_rate::Int,
    time_zero_idx::Int,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    !has_augmentation_copies(spec) && return Array{Float32, 4}(undef, target_size[1], target_size[2], 1, 0), Int[]

    imgs = Matrix{Float32}[]
    y_aug = Int[]
    for idx in train_indices
        row = supervised_df[idx, :]
        n_copies = augmentation_copy_count(spec, Int(row.binary_label))
        for _ in 1:n_copies
            push!(imgs, materialize_augmented_image_for_row(
                erps,
                events,
                row;
                spec = spec,
                rng = rng,
                sampling_rate = sampling_rate,
                time_zero_idx = time_zero_idx,
                target_size = target_size,
                low_pass_sigma = low_pass_sigma,
                lowpass_kernel_size = lowpass_kernel_size,
                filter_border = filter_border,
            ))
            push!(y_aug, Int(row.binary_label))
        end
    end

    return CNNUtils.images_to_tensor(imgs), y_aug
end

function stratified_inner_split(train_idx::Vector{Int}, y::Vector{Int};
    tune_fraction::Float64,
    seed::Int)

    rng = MersenneTwister(seed)
    tune = Int[]
    fit = Int[]

    for cls in 0:1
        cls_idx = [idx for idx in train_idx if y[idx] == cls]
        shuffled = shuffle(rng, cls_idx)
        n_tune = max(1, Int(round(length(shuffled) * tune_fraction)))
        n_tune = min(n_tune, max(length(shuffled) - 1, 1))
        append!(tune, shuffled[1:n_tune])
        append!(fit, shuffled[(n_tune + 1):end])
    end

    sort!(fit)
    sort!(tune)
    return fit, tune
end

function onehot_labels(y::Vector{Int})
    return Float32.(Array(Flux.onehotbatch(y, 0:1)))
end

function make_epoch_batches(rng::AbstractRNG, y::Vector{Int}, batchsize::Int; balanced_batches::Bool)
    n = length(y)
    if !balanced_batches
        idx = shuffle(rng, 1:n)
        return [idx[i:min(i + batchsize - 1, n)] for i in 1:batchsize:n]
    end

    pos = findall(==(1), y)
    neg = findall(==(0), y)
    if isempty(pos) || isempty(neg)
        idx = shuffle(rng, 1:n)
        return [idx[i:min(i + batchsize - 1, n)] for i in 1:batchsize:n]
    end

    half = max(1, batchsize ÷ 2)
    n_batches = max(1, cld(2 * max(length(pos), length(neg)), batchsize))
    batches = Vector{Vector{Int}}(undef, n_batches)
    for b in 1:n_batches
        bpos = rand(rng, pos, half)
        bneg = rand(rng, neg, batchsize - half)
        batches[b] = shuffle(rng, vcat(bpos, bneg))
    end
    return batches
end

function weighted_logitcrossentropy(logits, y; class_weights = nothing)
    logp = Flux.logsoftmax(logits; dims = 1)
    ce = vec(-sum(y .* logp; dims = 1))
    if class_weights === nothing
        return mean(ce)
    end
    sample_weights = vec(sum(y .* class_weights; dims = 1))
    return sum(sample_weights .* ce) / (sum(sample_weights) + eps(Float32))
end

function weighted_focal_loss(logits, y; class_weights = nothing, gamma::Float32 = 2.0f0)
    logp = Flux.logsoftmax(logits; dims = 1)
    logpt = vec(sum(y .* logp; dims = 1))
    pt = exp.(logpt)
    focal = .-((1f0 .- pt) .^ gamma) .* logpt
    if class_weights === nothing
        return mean(focal)
    end
    sample_weights = vec(sum(y .* class_weights; dims = 1))
    return sum(sample_weights .* focal) / (sum(sample_weights) + eps(Float32))
end

function classifier_loss_for_strategy(model, x, y, strategy, class_weights_dev)
    logits = model(x)
    weights = strategy.use_class_weights ? class_weights_dev : nothing
    if strategy.loss == :crossentropy
        return weighted_logitcrossentropy(logits, y; class_weights = weights)
    elseif strategy.loss == :focal
        return weighted_focal_loss(logits, y; class_weights = weights, gamma = Float32(strategy.focal_gamma))
    else
        error("Unsupported loss kind: $(strategy.loss)")
    end
end

function predict_probabilities(model, X::Array{Float32, 4}; batchsize::Int = 64, device::Function = identity)
    Flux.testmode!(model, true)
    n = size(X, 4)
    probs = Vector{Float32}(undef, n)
    for start_idx in 1:batchsize:n
        idx = start_idx:min(start_idx + batchsize - 1, n)
        logits = Array(cpu(model(device(X[:, :, :, idx]))))
        p = Flux.softmax(Float32.(logits); dims = 1)
        probs[idx] .= Float32.(vec(p[2, :]))
    end
    return probs
end

function binary_pr_auc(y_true::Vector{Int}, scores::Vector{Float32})
    order = sortperm(scores; rev = true)
    y_sorted = y_true[order]
    positives = count(==(1), y_true)
    positives == 0 && return Float32(NaN)

    tp = 0
    fp = 0
    last_recall = 0.0
    auc = 0.0
    last_precision = 1.0

    for y in y_sorted
        if y == 1
            tp += 1
        else
            fp += 1
        end
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        auc += (recall - last_recall) * ((precision + last_precision) / 2)
        last_recall = recall
        last_precision = precision
    end

    return Float32(auc)
end

function metrics_at_threshold(y_true::Vector{Int}, prob_pattern::Vector{Float32}, threshold::Float32)
    y_pred = Int.(prob_pattern .>= threshold)
    metrics = CNNUtils.compute_metrics(y_pred, y_true)
    return (
        y_pred = y_pred,
        threshold = threshold,
        accuracy = Float64(metrics.accuracy),
        balanced_accuracy = Float64(metrics.balanced_accuracy),
        macro_f1 = Float64(metrics.macro_f1),
        precision = Float64(metrics.precision),
        recall = Float64(metrics.recall),
        pr_auc = Float64(binary_pr_auc(y_true, prob_pattern)),
    )
end

function tune_threshold(y_true::Vector{Int}, prob_pattern::Vector{Float32};
    thresholds = Float32.(0.05:0.01:0.95))

    best = metrics_at_threshold(y_true, prob_pattern, 0.5f0)
    best_key = (best.balanced_accuracy, best.macro_f1, best.recall)

    for threshold in thresholds
        m = metrics_at_threshold(y_true, prob_pattern, threshold)
        key = (m.balanced_accuracy, m.macro_f1, m.recall)
        if key > best_key
            best = m
            best_key = key
        end
    end

    return best
end

function train_supervised_resnet18!(X_fit::Array{Float32, 4}, y_fit::Vector{Int},
    X_tune::Array{Float32, 4}, y_tune::Vector{Int}, strategy;
    batchsize::Int,
    max_epochs::Int,
    lr::Float32,
    weight_decay::Float32,
    patience::Int,
    seed::Int,
    show_epoch_logs::Bool)

    rng = MersenneTwister(seed)
    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu
    model = CNNUtils.build_resnet18_single_channel_random(n_classes = 2, in_channels = 1) |> device

    opt = Flux.AdamW(lr, (0.9f0, 0.999f0), weight_decay)
    opt_state = Flux.setup(opt, model)
    class_weights = class_weights_inverse_frequency(y_fit)
    class_weights_dev = reshape(device(class_weights), :, 1)
    y_fit_oh = onehot_labels(y_fit)

    best_model = nothing
    best_score = -Inf
    best_epoch = 0
    wait = 0
    history_rows = NamedTuple[]

    train_time_s = @elapsed begin
        for epoch in 1:max_epochs
            Flux.trainmode!(model)
            epoch_loss = 0f0
            n_batches = 0

            for idx in make_epoch_batches(rng, y_fit, batchsize; balanced_batches = strategy.balanced_batches)
                xb = device(X_fit[:, :, :, idx])
                yb = device(y_fit_oh[:, idx])

                loss_val, grads = Flux.withgradient(model) do m
                    classifier_loss_for_strategy(m, xb, yb, strategy, class_weights_dev)
                end
                opt_state, model = Flux.update!(opt_state, model, grads[1])
                epoch_loss += Float32(loss_val)
                n_batches += 1
            end

            tune_probs = predict_probabilities(cpu(model), X_tune; batchsize = batchsize, device = identity)
            tune_metrics = metrics_at_threshold(y_tune, tune_probs, 0.5f0)
            avg_loss = epoch_loss / max(1, n_batches)
            push!(history_rows, (
                epoch = epoch,
                train_loss = Float64(avg_loss),
                tune_balanced_accuracy = tune_metrics.balanced_accuracy,
                tune_macro_f1 = tune_metrics.macro_f1,
            ))

            if show_epoch_logs
                println("$(strategy.name) epoch $(epoch)/$(max_epochs) | loss=$(round(avg_loss, digits = 5)) | tune_bacc=$(round(tune_metrics.balanced_accuracy, digits = 4))")
            end

            if tune_metrics.balanced_accuracy > best_score + 1e-6
                best_score = tune_metrics.balanced_accuracy
                best_epoch = epoch
                best_model = deepcopy(cpu(model))
                wait = 0
            else
                wait += 1
                wait >= patience && break
            end
        end
    end

    if best_model === nothing
        best_model = deepcopy(cpu(model))
        best_epoch = max_epochs
    end

    GC.gc()
    use_gpu && CUDA.reclaim()

    return (
        model = best_model,
        best_epoch = best_epoch,
        best_tune_balanced_accuracy = Float64(best_score),
        train_time_s = train_time_s,
        class_weight_no_pattern = Float64(class_weights[1]),
        class_weight_pattern = Float64(class_weights[2]),
        history_df = DataFrame(history_rows),
    )
end

function experiment_grid(; include_augmentation::Bool = true, include_imbalance::Bool = true)
    rows = NamedTuple[]

    if include_imbalance
        for strategy in imbalance_strategy_specs()
            push!(rows, (
                experiment_group = "imbalance",
                experiment = strategy.name,
                label = strategy.label,
                augmentation_name = "none",
                strategy_name = strategy.name,
                augmentation = augmentation_specs()[1],
                strategy = strategy,
            ))
        end
    end

    if include_augmentation
        base_strategy = imbalance_strategy_specs()[1]
        for aug in augmentation_specs()
            aug.name == "none" && continue
            push!(rows, (
                experiment_group = "augmentation",
                experiment = aug.name,
                label = aug.label,
                augmentation_name = aug.name,
                strategy_name = base_strategy.name,
                augmentation = aug,
                strategy = base_strategy,
            ))
        end
    end

    return rows
end

function run_supervised_augmentation_cv(ctx;
    include_augmentation::Bool = true,
    include_imbalance::Bool = true,
    selected_experiments::Union{Nothing, Vector{String}} = nothing,
    sampling_rate::Int,
    time_zero_idx::Int,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String,
    batchsize::Int = 32,
    max_epochs::Int = 4,
    lr::Float32 = 3f-4,
    weight_decay::Float32 = 1f-4,
    patience::Int = 2,
    tune_fraction::Float64 = 0.2,
    seed::Int = 0,
    show_epoch_logs::Bool = false)

    all_specs = experiment_grid(include_augmentation = include_augmentation, include_imbalance = include_imbalance)
    specs = selected_experiments === nothing ? all_specs : [s for s in all_specs if s.experiment in selected_experiments]
    @assert !isempty(specs) "No experiment specs selected."

    rows = NamedTuple[]
    threshold_rows = NamedTuple[]
    history_rows = NamedTuple[]

    n = size(ctx.X, 4)

    for spec in specs
        experiment_seed_offset = Int(mod(hash(String(spec.experiment)), UInt(50_000)))

        println()
        println(repeat("=", 88))
        println("Experiment: $(spec.experiment_group) / $(spec.experiment)")
        println("  label       : $(spec.label)")
        println("  augmentation: $(spec.augmentation.description)")
        println("  strategy    : $(spec.strategy.description)")

        for (fold_id, val_idx) in enumerate(ctx.fold_val_indices)
            train_mask = trues(n)
            train_mask[val_idx] .= false
            train_idx = findall(train_mask)
            fit_idx, tune_idx = stratified_inner_split(train_idx, ctx.y; tune_fraction = tune_fraction, seed = seed + 10_000 * fold_id + 13)

            X_fit = ctx.X[:, :, :, fit_idx]
            y_fit = ctx.y[fit_idx]

            if has_augmentation_copies(spec.augmentation)
                aug_rng = MersenneTwister(seed + 100_000 * fold_id + experiment_seed_offset)
                X_aug, y_aug = materialize_augmented_training_tensor(
                    ctx.data_ctx.erps,
                    ctx.data_ctx.events,
                    ctx.supervised_df,
                    fit_idx,
                    spec.augmentation;
                    rng = aug_rng,
                    sampling_rate = sampling_rate,
                    time_zero_idx = time_zero_idx,
                    target_size = target_size,
                    low_pass_sigma = low_pass_sigma,
                    lowpass_kernel_size = lowpass_kernel_size,
                    filter_border = filter_border,
                )
                if size(X_aug, 4) > 0
                    X_fit = cat(X_fit, X_aug; dims = 4)
                    y_fit = vcat(y_fit, y_aug)
                end
            end

            X_tune = ctx.X[:, :, :, tune_idx]
            y_tune = ctx.y[tune_idx]
            X_val = ctx.X[:, :, :, val_idx]
            y_val = ctx.y[val_idx]

            trained = train_supervised_resnet18!(
                X_fit,
                y_fit,
                X_tune,
                y_tune,
                spec.strategy;
                batchsize = batchsize,
                max_epochs = max_epochs,
                lr = lr,
                weight_decay = weight_decay,
                patience = patience,
                seed = seed + 10_000 * fold_id + experiment_seed_offset,
                show_epoch_logs = show_epoch_logs,
            )

            tune_probs = predict_probabilities(trained.model, X_tune; batchsize = batchsize, device = identity)
            tuned = tune_threshold(y_tune, tune_probs)
            val_probs = predict_probabilities(trained.model, X_val; batchsize = batchsize, device = identity)
            default_metrics = metrics_at_threshold(y_val, val_probs, 0.5f0)
            tuned_metrics = metrics_at_threshold(y_val, val_probs, Float32(tuned.threshold))

            push!(rows, (
                experiment_group = spec.experiment_group,
                experiment = spec.experiment,
                label = spec.label,
                fold = fold_id,
                n_fit_reference = length(fit_idx),
                n_fit_total = length(y_fit),
                n_tune = length(tune_idx),
                n_val = length(val_idx),
                n_pattern_reference_fit = count(==(1), ctx.y[fit_idx]),
                n_no_pattern_reference_fit = count(==(0), ctx.y[fit_idx]),
                n_pattern_fit = count(==(1), y_fit),
                n_no_pattern_fit = count(==(0), y_fit),
                n_pattern_augmented = count(==(1), y_fit) - count(==(1), ctx.y[fit_idx]),
                n_no_pattern_augmented = count(==(0), y_fit) - count(==(0), ctx.y[fit_idx]),
                augmentation_name = spec.augmentation_name,
                augmentation_copies_pattern = Int(spec.augmentation.copies_pattern),
                augmentation_copies_no_pattern = Int(spec.augmentation.copies_no_pattern),
                augmentation_stage = String(spec.augmentation.stage),
                strategy_name = spec.strategy_name,
                loss = String(spec.strategy.loss),
                use_class_weights = spec.strategy.use_class_weights,
                balanced_batches = spec.strategy.balanced_batches,
                focal_gamma = Float64(spec.strategy.focal_gamma),
                threshold_tuned = Float64(tuned.threshold),
                best_epoch = trained.best_epoch,
                class_weight_no_pattern = trained.class_weight_no_pattern,
                class_weight_pattern = trained.class_weight_pattern,
                train_time_s = Float64(trained.train_time_s),
                accuracy_default = default_metrics.accuracy,
                balanced_accuracy_default = default_metrics.balanced_accuracy,
                macro_f1_default = default_metrics.macro_f1,
                precision_default = default_metrics.precision,
                recall_default = default_metrics.recall,
                pr_auc = default_metrics.pr_auc,
                accuracy_tuned = tuned_metrics.accuracy,
                balanced_accuracy_tuned = tuned_metrics.balanced_accuracy,
                macro_f1_tuned = tuned_metrics.macro_f1,
                precision_tuned = tuned_metrics.precision,
                recall_tuned = tuned_metrics.recall,
            ))

            push!(threshold_rows, (
                experiment_group = spec.experiment_group,
                experiment = spec.experiment,
                fold = fold_id,
                tuned_threshold = Float64(tuned.threshold),
                tune_balanced_accuracy = tuned.balanced_accuracy,
                tune_macro_f1 = tuned.macro_f1,
                tune_precision = tuned.precision,
                tune_recall = tuned.recall,
            ))

            for r in eachrow(trained.history_df)
                push!(history_rows, (
                    experiment_group = spec.experiment_group,
                    experiment = spec.experiment,
                    fold = fold_id,
                    epoch = Int(r.epoch),
                    train_loss = Float64(r.train_loss),
                    tune_balanced_accuracy = Float64(r.tune_balanced_accuracy),
                    tune_macro_f1 = Float64(r.tune_macro_f1),
                ))
            end
        end
    end

    cv_df = DataFrame(rows)
    threshold_df = DataFrame(threshold_rows)
    history_df = DataFrame(history_rows)
    summary_df = summarize_supervised_augmentation_results(cv_df)

    return (
        cv_df = cv_df,
        summary_df = summary_df,
        threshold_df = threshold_df,
        history_df = history_df,
    )
end

function summarize_supervised_augmentation_results(cv_df::DataFrame)
    summary_df = combine(
        groupby(cv_df, [:experiment_group, :experiment, :label]),
        :balanced_accuracy_default => mean => :balanced_accuracy_default_mean,
        :balanced_accuracy_default => std => :balanced_accuracy_default_std,
        :balanced_accuracy_tuned => mean => :balanced_accuracy_tuned_mean,
        :balanced_accuracy_tuned => std => :balanced_accuracy_tuned_std,
        :macro_f1_tuned => mean => :macro_f1_tuned_mean,
        :macro_f1_tuned => std => :macro_f1_tuned_std,
        :precision_tuned => mean => :precision_tuned_mean,
        :recall_tuned => mean => :recall_tuned_mean,
        :pr_auc => mean => :pr_auc_mean,
        :threshold_tuned => mean => :threshold_tuned_mean,
        :best_epoch => mean => :best_epoch_mean,
        :augmentation_copies_pattern => mean => :augmentation_copies_pattern_mean,
        :augmentation_copies_no_pattern => mean => :augmentation_copies_no_pattern_mean,
        :n_pattern_reference_fit => mean => :n_pattern_reference_fit_mean,
        :n_no_pattern_reference_fit => mean => :n_no_pattern_reference_fit_mean,
        :n_pattern_augmented => mean => :n_pattern_augmented_mean,
        :n_no_pattern_augmented => mean => :n_no_pattern_augmented_mean,
        :n_fit_total => mean => :n_fit_total_mean,
        :train_time_s => mean => :train_time_mean_s,
    )
    sort!(summary_df, :balanced_accuracy_tuned_mean, rev = true)
    return summary_df
end

erp_image_to_plot_matrix(img::AbstractMatrix) = Float32.(permutedims(img, (2, 1)))

function plot_augmentation_preview_grid(ctx, spec_names::Vector{String};
    sample_index::Union{Nothing, Int} = nothing,
    sampling_rate::Int,
    time_zero_idx::Int,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String,
    seed::Int)

    specs = Dict(spec.name => spec for spec in augmentation_specs())
    rng = MersenneTwister(seed)
    idx = sample_index === nothing ? findfirst(==(1), ctx.y) : sample_index
    @assert idx !== nothing "No positive sample found for augmentation preview."

    row = ctx.supervised_df[Int(idx), :]
    pairs = Pair{String, Matrix{Float32}}[]
    push!(pairs, "reference" => ctx.supervised_df.processed_img[Int(idx)])

    for name in spec_names
        spec = specs[name]
        img = materialize_augmented_image_for_row(
            ctx.data_ctx.erps,
            ctx.data_ctx.events,
            row;
            spec = spec,
            rng = rng,
            sampling_rate = sampling_rate,
            time_zero_idx = time_zero_idx,
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        push!(pairs, name => img)
    end

    ncols = min(3, length(pairs))
    nrows = cld(length(pairs), ncols)
    fig = Figure(size = (390 * ncols, 330 * nrows))

    for (i, pair) in enumerate(pairs)
        r = cld(i, ncols)
        c = ((i - 1) % ncols) + 1
        img_col = 2 * c - 1
        colorbar_col = 2 * c
        img = pair.second
        clipped, crange, tick_vals, tick_labels, cmap = CNNUtils.clipped_color_stats_quantile_zero_ticks(img)
        ax = Axis(fig[r, img_col], title = pair.first, xlabel = "time", ylabel = "trials", aspect = DataAspect())
        hm = heatmap!(ax, 1:size(img, 2), 1:size(img, 1), erp_image_to_plot_matrix(clipped);
            colormap = cmap,
            colorrange = crange,
        )
        Colorbar(fig[r, colorbar_col], hm; ticks = (tick_vals, tick_labels))
    end

    Label(fig[0, 1:(2 * ncols)], "Training-only augmentation previews; validation images stay unaugmented", fontsize = 18, font = :bold)
    return fig
end

end # module
