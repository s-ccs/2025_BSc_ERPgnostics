# =============================================================================
# Simulated training data, ResNet18 training, and candidate scoring
#
# For each simulator candidate this module generates a balanced set of simulated
# `sigmoid` / `no_class` images, trains a ResNet18 on them, and scores the
# trained network on the real validation set. A strict "collapse" check rejects
# degenerate runs that predict a single class.
# =============================================================================

"""
    simulated_sort_column(events::DataFrame, pattern::Symbol, rng)

Return the per-trial sort key for a simulated pattern. The `sigmoid` pattern is
sorted by its latency structure; `no_class` is sorted by a random permutation so
no pattern emerges along the trial axis.

# Arguments
- `events::DataFrame`: the simulated event table.
- `pattern::Symbol`: `:sigmoid` or `:no_class`.
- `rng::Random.AbstractRNG`: random source for the `no_class` permutation.

# Returns
- A vector of sort keys, one per trial.
"""
function simulated_sort_column(events::DataFrame, pattern::Symbol, rng::Random.AbstractRNG)
    pattern == :sigmoid && return collect(zip(events[!, ERPGen.DELTA_LATENCY], events.latency))
    pattern == :no_class && return randperm(rng, nrow(events))
    error("Unsupported simulated pattern: $(pattern)")
end

"""
    preprocess_simulated_pattern(data_time_trials, events, pattern, rng, config)

Sort the simulated trials by the pattern's sort key and run the shared image
pipeline, producing one `config.target_size` image.

# Arguments
- `data_time_trials::AbstractMatrix`: simulated raw matrix (timepoints x trials).
- `events::DataFrame`: simulated event table.
- `pattern::Symbol`: `:sigmoid` or `:no_class`.
- `rng::Random.AbstractRNG`: random source for the `no_class` permutation.
- `config::RunConfig`: image pipeline settings.

# Returns
- `Matrix{Float32}`: the preprocessed pattern image.
"""
function preprocess_simulated_pattern(data_time_trials::AbstractMatrix, events::DataFrame, pattern::Symbol, rng::Random.AbstractRNG, config::RunConfig)
    events_work = copy(events)
    events_work[!, :strategy_sort_key] = simulated_sort_column(events_work, pattern, rng)
    return shared_preprocess(data_time_trials, events_work, :strategy_sort_key, config)
end

"""
    simulate_sigmoid_no_class_pair(cfg, config, real)

Simulate one raw ERP matrix and turn it into a `sigmoid` and a `no_class` image.
Both images come from the same raw trials, differing only in how the trials are
ordered, which isolates the visual pattern as the only class signal.

The simulator draws its own random parameters on every call (seeded from the
system clock), so each pair is an independent sample. To keep the simulated
images identical in size and processing to the real fixation data, **no trial
dropout and no cropping** are applied: the raw simulated trials are used as-is.

# Arguments
- `cfg::ERPGen.GenerationConfig`: the candidate simulator configuration.
- `config::RunConfig`: image pipeline settings.
- `real::RealValidationData`: provides the expected raw matrix dimensions.

# Returns
- `Tuple{Matrix{Float32},Matrix{Float32}}`: `(sigmoid_img, no_class_img)`.
"""
function simulate_sigmoid_no_class_pair(cfg::ERPGen.GenerationConfig, config::RunConfig, real::RealValidationData)
    raw = ERPGen.simulate_raw_erp(cfg, Random.Xoshiro(new_seed()))

    # No dropout, no cropping: keep the simulated trials 1:1 with the real
    # recording so dimensions and the downstream image pipeline stay identical.
    data_time_trials = Float32.(raw.data)
    events_trials = raw.events
    size(data_time_trials) == (real.n_timepoints, real.n_trials) ||
        error("Simulated raw dimensions $(size(data_time_trials)) do not match the fixations dataset $((real.n_timepoints, real.n_trials)).")

    # Fresh, independent RNGs for the two trial orderings.
    sigmoid_img = preprocess_simulated_pattern(data_time_trials, events_trials, :sigmoid, Random.Xoshiro(new_seed()), config)
    no_class_img = preprocess_simulated_pattern(data_time_trials, events_trials, :no_class, Random.Xoshiro(new_seed()), config)
    return sigmoid_img, no_class_img
end

"""
    generate_simulated_training_dataset(cfg, config, real)

Generate a shuffled, balanced training set of simulated `sigmoid` and
`no_class` images for one candidate. The `config.n_per_pattern` image pairs are
simulated in parallel across the available Julia threads (the simulator runs on
the CPU; only the network runs on the GPU).

# Arguments
- `cfg::ERPGen.GenerationConfig`: the candidate simulator configuration.
- `config::RunConfig`: supplies `n_per_pattern` and pipeline settings.
- `real::RealValidationData`: provides the raw matrix dimensions.

# Returns
- `DataFrame`: columns `sample_id`, `binary_label`, and `processed_img`.
"""
function generate_simulated_training_dataset(cfg::ERPGen.GenerationConfig, config::RunConfig, real::RealValidationData)
    n = config.n_per_pattern
    imgs = Vector{Matrix{Float32}}(undef, 2 * n)
    labels = Vector{Int}(undef, 2 * n)

    # Each pair is independent, so simulate them on whatever threads are present.
    # The simulator's RNG counter is atomic, which keeps the parallel draws safe.
    Threads.@threads for i in 1:n
        sigmoid_img, no_class_img = simulate_sigmoid_no_class_pair(cfg, config, real)
        imgs[2 * i - 1] = Float32.(sigmoid_img); labels[2 * i - 1] = 1
        imgs[2 * i] = Float32.(no_class_img); labels[2 * i] = 0
    end

    order = randperm(Random.Xoshiro(new_seed()), 2 * n)
    out = DataFrame(sample_id = 1:(2 * n), binary_label = labels[order])
    out.processed_img = imgs[order]
    return out
end

"""
    smooth_onehot(y_oh, epsilon)

Apply label smoothing to a one-hot label matrix, moving `epsilon` of the mass
uniformly across classes. A non-positive `epsilon` returns the labels unchanged.

# Arguments
- `y_oh::AbstractMatrix{<:Real}`: one-hot labels (classes x samples).
- `epsilon::Real`: smoothing strength.

# Returns
- `Array{Float32}`: the smoothed labels.
"""
function smooth_onehot(y_oh::AbstractMatrix{<:Real}, epsilon::Real)
    epsilon <= 0 && return Array{Float32}(y_oh)
    n_classes = size(y_oh, 1)
    return Float32.((1 - epsilon) .* y_oh .+ epsilon / n_classes)
end

"""
    weighted_logitcrossentropy(logits, y_oh, class_weights)

Compute a class-weighted cross-entropy from raw logits and one-hot targets.

# Arguments
- `logits`: model outputs (classes x samples).
- `y_oh`: one-hot targets (classes x samples).
- `class_weights`: per-class weights (classes x 1).

# Returns
- The scalar mean weighted cross-entropy.
"""
function weighted_logitcrossentropy(logits, y_oh, class_weights)
    logp = Flux.logsoftmax(logits; dims = 1)
    return mean(-sum(class_weights .* y_oh .* logp; dims = 1))
end

"""
    train_resnet18!(model, X_train, y_train, profile, use_gpu; run_tag)

Train `model` in place for the profile's epochs with class-weighted, label-
smoothed cross-entropy and the Adam optimiser.

# Arguments
- `model`: the ResNet18 to train.
- `X_train::Array{Float32,4}`: training images (`h x w x 1 x n`).
- `y_train::Vector{Int}`: binary training labels.
- `profile`: named tuple with `nepochs`, `lr`, `batchsize`, `class_weights`,
  `label_smoothing`.
- `use_gpu::Bool`: whether to train on CUDA.
- `run_tag::AbstractString`: label used in per-epoch log messages (keyword).

# Returns
- `Tuple{model, Float64}`: the trained model and the training wall-time in seconds.
"""
function train_resnet18!(model, X_train::Array{Float32, 4}, y_train::Vector{Int}, profile, use_gpu::Bool; run_tag::AbstractString)
    y_train_oh = smooth_onehot(onehotbatch(y_train, 0:1) |> Array{Float32}, profile.label_smoothing)
    train_loader = DataLoader((X_train, y_train_oh); batchsize = profile.batchsize, shuffle = true)
    model = to_device(model, use_gpu)
    opt_state = Flux.setup(Flux.Adam(profile.lr), model)
    class_weights = to_device(reshape(Float32.(profile.class_weights), :, 1), use_gpu)
    Flux.trainmode!(model)

    train_time_s = @elapsed begin
        for epoch in 1:profile.nepochs
            running_loss = 0.0f0
            n_batches = 0
            for (xb_cpu, yb_cpu) in train_loader
                xb = to_device(xb_cpu, use_gpu)
                yb = to_device(yb_cpu, use_gpu)
                loss_val, grads = Flux.withgradient(model) do m
                    weighted_logitcrossentropy(m(xb), yb, class_weights)
                end
                opt_state, model = Flux.update!(opt_state, model, grads[1])
                running_loss += loss_val
                n_batches += 1
            end
            @info "$(run_tag) | epoch $(epoch)/$(profile.nepochs) | train_loss=$(@sprintf("%.5f", Float64(running_loss / max(1, n_batches))))"
        end
    end
    return model, Float64(train_time_s)
end

"""
    binary_metrics(y_pred, y_true)

Compute balanced accuracy and macro F1 for binary predictions.

# Arguments
- `y_pred::Vector{Int}`: predicted labels.
- `y_true::Vector{Int}`: true labels.

# Returns
- `NamedTuple`: `balanced_accuracy` and `macro_f1`.
"""
function binary_metrics(y_pred::Vector{Int}, y_true::Vector{Int})
    return (
        balanced_accuracy = StatisticalMeasures.BalancedAccuracy()(y_pred, y_true),
        macro_f1 = StatisticalMeasures.MulticlassFScore(; average = macro_avg)(y_pred, y_true),
    )
end

"""
    evaluate_resnet18(model, X_val, y_val, use_gpu; batchsize)

Run inference on the validation set and return metrics alongside the predicted
and true labels.

# Arguments
- `model`: the trained ResNet18.
- `X_val::Array{Float32,4}`: validation images.
- `y_val::Vector{Int}`: validation labels.
- `use_gpu::Bool`: whether inference runs on CUDA.
- `batchsize::Int`: inference batch size (keyword).

# Returns
- `Tuple`: `(metrics, y_true, y_pred, classification_time_s)`.
"""
function evaluate_resnet18(model, X_val::Array{Float32, 4}, y_val::Vector{Int}, use_gpu::Bool; batchsize::Int)
    y_val_oh = onehotbatch(y_val, 0:1) |> Array{Float32}
    val_loader = DataLoader((X_val, y_val_oh); batchsize = batchsize, shuffle = false)
    Flux.testmode!(model)
    y_true = Int[]
    y_pred = Int[]
    classification_time_s = @elapsed begin
        for (xb_cpu, yb_cpu) in val_loader
            logits = Flux.cpu(model(to_device(xb_cpu, use_gpu)))
            append!(y_pred, onecold(logits, 0:1))
            append!(y_true, onecold(yb_cpu, 0:1))
        end
    end
    return binary_metrics(y_pred, y_true), y_true, y_pred, Float64(classification_time_s)
end

"""
    is_noncollapsed_result(result, config) -> Bool

Decide whether a run separated the two classes. A run counts as non-collapsed
when its balanced accuracy clears `config.sanity_bacc_min` and each predicted
class reaches `config.sanity_class_balance_frac` of the validation set.

# Arguments
- `result`: named tuple with prediction and balanced-accuracy fields.
- `config::RunConfig`: collapse thresholds.

# Returns
- `Bool`: `true` if the run is non-degenerate.
"""
function is_noncollapsed_result(result, config::RunConfig)
    n_validation = result.true_count_0 + result.true_count_1
    min_pred = min(result.pred_count_0, result.pred_count_1)
    required = ceil(Int, config.sanity_class_balance_frac * n_validation)
    return result.balanced_accuracy >= config.sanity_bacc_min && min_pred >= required
end

"""
    train_and_score_candidate(cfg, real, profile, config, seed; run_tag)

Generate simulated training data for one candidate, train a ResNet18, and score
it on the real validation set. Returns a flat result with metrics, timings,
prediction counts, and a `collapsed` flag.

# Arguments
- `cfg::ERPGen.GenerationConfig`: the candidate simulator configuration.
- `real::RealValidationData`: the real validation set and dimensions.
- `profile`: the training profile to use.
- `config::RunConfig`: experiment settings.
- `seed`: seed for this evaluation's model initialisation and shuffling. The
  simulated images themselves are drawn independently from the system clock.
- `run_tag::AbstractString`: label for logging (keyword).

# Returns
- `NamedTuple`: metrics, timings, counts, profile info, and `collapsed`.
"""
function train_and_score_candidate(cfg::ERPGen.GenerationConfig, real::RealValidationData, profile, config::RunConfig, seed; run_tag::AbstractString)
    generation_time_s = @elapsed train_df = generate_simulated_training_dataset(cfg, config, real)
    X_train = images_to_tensor(train_df.processed_img)
    y_train = Int.(train_df.binary_label)

    # Seed the global RNG so model initialisation and training shuffles are tied
    # to this repeat's recorded seed.
    set_all_seeds!(seed, config.use_gpu)
    model, pretrained_loaded = build_resnet18_for_profile(profile)
    model, train_time_s = train_resnet18!(model, X_train, y_train, profile, config.use_gpu; run_tag = run_tag)
    metrics, y_true, y_pred, classification_time_s = evaluate_resnet18(model, real.tensor, real.labels, config.use_gpu; batchsize = profile.batchsize)

    result = (
        balanced_accuracy = Float64(metrics.balanced_accuracy),
        macro_f1 = Float64(metrics.macro_f1),
        train_time_s = train_time_s,
        classification_time_s = classification_time_s,
        generation_time_s = Float64(generation_time_s),
        n_train = length(y_train),
        n_val_real = length(y_true),
        train_pos = count(==(1), y_train),
        train_neg = count(==(0), y_train),
        pred_count_0 = count(==(0), y_pred),
        pred_count_1 = count(==(1), y_pred),
        true_count_0 = count(==(0), y_true),
        true_count_1 = count(==(1), y_true),
        pretrained_params_loaded = Int(pretrained_loaded),
        training_profile = String(profile.name),
        model_init = String(profile.model_init),
    )
    cleanup_device!(config.use_gpu)
    return merge(result, (collapsed = !is_noncollapsed_result(result, config),))
end

"""
    run_sanity_gate(base_cfg, real, config)

Try the training profiles in order and return the first that yields a
non-collapsed run on the real validation set. Errors if every profile collapses,
which refuses to run the strategy sweep on a degenerate setup.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: the baseline configuration to evaluate.
- `real::RealValidationData`: the real validation set.
- `config::RunConfig`: experiment settings and training profiles.

# Returns
- `Tuple`: `(selected_profile, sanity_df)` with one row per tried profile.
"""
function run_sanity_gate(base_cfg::ERPGen.GenerationConfig, real::RealValidationData, config::RunConfig)
    profiles = default_training_profiles(config)
    rows = NamedTuple[]
    for (profile_index, profile) in enumerate(profiles)
        println("\n=== Sanity profile $(profile_index)/$(length(profiles)): $(profile.name) ===")
        result = train_and_score_candidate(
            base_cfg, real, profile, config, new_seed();
            run_tag = "sanity/$(profile.name)",
        )
        push!(rows, merge((profile_index = profile_index,), result))
        println("Sanity result | BAcc=", round(result.balanced_accuracy, digits = 4),
            " | macro_F1=", round(result.macro_f1, digits = 4),
            " | predicted no_class=", result.pred_count_0,
            " | predicted sigmoid=", result.pred_count_1,
            " | collapsed=", result.collapsed)
        result.collapsed || return profile, DataFrame(rows)
    end
    error("Strict sanity gate failed for every training profile. Refusing to run the strategy sweep or write CSV output.")
end
