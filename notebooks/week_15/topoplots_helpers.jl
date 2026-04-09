module Week15Topoplots

using CairoMakie
using CSV
using DataFrames
using Flux
using JLD2
using JSON3
using LinearAlgebra
using Printf: @sprintf
using Random
using Statistics

using Main.Week15TryNewData
const TD = Main.Week15TryNewData

if !isdefined(Main, :TestOutputLayerHelpers)
    include(joinpath(@__DIR__, "..", "model_test", "test_outputlayer_helpers.jl"))
end
const OL = isdefined(Main, :TestOutputLayerHelpers) ? Main.TestOutputLayerHelpers : TestOutputLayerHelpers

export activate_topoplot_backend!
export build_fixation_duration_topoplot_context
export build_fixation_duration_topoplot_app
export closest_electrode_index
export display_fixation_duration_topoplot!
export plot_fixation_duration_topoplot

const FIXATION_POSITIONS_PATH = joinpath(TD.FIXATIONS_DATASET_DIR, "positions_128.jld2")
const ADDITIONAL_DURATION_TASKS_PATH = joinpath(
    @__DIR__,
    "..",
    "model_test",
    "label_studio_data_unlabelled_additional_400",
    "tasks_unlabelled_400.json",
)
const DURATION_SORT_NAME = "duration"
const PATTERN_THRESHOLD = 0.5
const MOD4_SPLIT_K = 4
const DUMMY_FEATURE_NAMES = [
    :peak_trend_abs,
    :peak_span,
    :diag_advantage,
    :late_energy_shift,
    :row_energy_flatness,
]
const DUMMY_PATTERN_WEIGHTS = Dict(
    :peak_trend_abs => 1.15,
    :peak_span => 0.90,
    :diag_advantage => 0.80,
    :late_energy_shift => 0.55,
    :row_energy_flatness => -0.45,
)
const DUMMY_PATTERN_BIAS = -0.10
const DUMMY_LOGIT_TEMPERATURE = 3.0
const DUMMY_PROBABILITY_SPAN = 0.20

normalize_sort_name(v) = lowercase(strip(String(v)))
pattern_label(prob::Real) = Float64(prob) >= PATTERN_THRESHOLD ? "pattern" : "no_pattern"
sigmoid_scalar(x::Real) = 1.0 / (1.0 + exp(-Float64(x)))
dummy_probability_from_logit(x::Real) = 0.5 + DUMMY_PROBABILITY_SPAN * tanh(Float64(x) / DUMMY_LOGIT_TEMPERATURE)

function activate_topoplot_backend!()
    try
        Base.eval(@__MODULE__, :(import WGLMakie))
        wglmakie = Base.invokelatest(getfield, @__MODULE__, :WGLMakie)
        activate_backend = getfield(wglmakie, :activate!)
        Base.invokelatest(activate_backend; resize_to = :parent)
        return :WGLMakie
    catch err
        CairoMakie.activate!()
        @warn "WGLMakie is unavailable. Falling back to CairoMakie. The figure renders, but click interaction needs WGLMakie or GLMakie." exception = (err, catch_backtrace())
        return :CairoMakie
    end
end

function resolve_data_split_seed(data_split_seed::Union{Nothing, Integer})
    if isnothing(data_split_seed)
        seed = Int(time_ns() % UInt64(typemax(Int)))
        seed == 0 && (seed = 1)
        return (seed = seed, source = "time_ns()")
    end
    return (seed = Int(data_split_seed), source = "fixed")
end

function inverse_frequency_class_weights(binary_labels::AbstractVector{<:Integer})
    n_total = max(length(binary_labels), 1)
    n_no_class = count(==(0), Int.(binary_labels))
    n_pattern = count(==(1), Int.(binary_labels))

    return Float32[
        n_total / (2f0 * max(n_no_class, 1)),
        n_total / (2f0 * max(n_pattern, 1)),
    ]
end

function safe_std(x::AbstractVector{<:Real})
    sigma = std(Float64.(x))
    return isfinite(sigma) && sigma > 0 ? sigma : 1.0
end

function safe_zscore(x::AbstractVector{<:Real})
    mu = mean(Float64.(x))
    sigma = safe_std(x)
    return (Float64.(x) .- mu) ./ sigma
end

function safe_cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x_f = Float64.(x)
    y_f = Float64.(y)
    if length(x_f) <= 1 || std(x_f) <= eps(Float64) || std(y_f) <= eps(Float64)
        return 0.0
    end
    return cor(x_f, y_f)
end

function weighted_logitcrossentropy(logits, y, class_weights)
    log_probs = Flux.logsoftmax(logits)
    losses = -vec(sum(y .* log_probs; dims = 1))
    sample_weights = vec(sum(class_weights .* y; dims = 1))
    normalizer = max(sum(sample_weights), eps(Float32))
    return sum(losses .* sample_weights) / normalizer
end

function probability_colorrange(values::AbstractVector; min_span::Float64 = 0.08, pad_frac::Float64 = 0.15)
    vv = clamp.(Float64.(collect(skipmissing(values))), 0.0, 1.0)
    isempty(vv) && return (0.0, 1.0)
    lo = minimum(vv)
    hi = maximum(vv)
    span = max(hi - lo, min_span)
    pad = pad_frac * span
    lo2 = max(0.0, lo - pad)
    hi2 = min(1.0, hi + pad)

    if hi2 - lo2 < min_span
        center = clamp((lo + hi) / 2, min_span / 2, 1 - min_span / 2)
        lo2 = center - min_span / 2
        hi2 = center + min_span / 2
    end

    return (lo2, hi2)
end

function centered_rank_score(values::AbstractVector{<:Real})
    n = length(values)
    n == 0 && return Float64[]
    n == 1 && return [0.0]

    vv = Float64.(values)
    perm = sortperm(vv)
    ranks = zeros(Float64, n)

    for (rank_idx, row_idx) in enumerate(perm)
        ranks[row_idx] = (rank_idx - 1) / (n - 1)
    end

    return 2 .* (ranks .- 0.5)
end

function relative_pattern_score(values::AbstractVector{<:Real}; z_scale::Float64 = 1.5)
    vv = Float64.(values)
    isempty(vv) && return Float64[]
    length(vv) == 1 && return [0.0]

    sigma = std(vv)
    if !isfinite(sigma) || sigma < 1e-8
        return centered_rank_score(vv)
    end

    z = (vv .- mean(vv)) ./ sigma
    score = tanh.(z ./ z_scale)
    max_abs = maximum(abs, score)
    max_abs > 0 || return centered_rank_score(vv)
    return score ./ max_abs
end

function symmetric_colorrange(values::AbstractVector; min_abs::Float64 = 0.35, pad_frac::Float64 = 0.08)
    vv = abs.(Float64.(collect(skipmissing(values))))
    isempty(vv) && return (-1.0, 1.0)
    hi = max(maximum(vv) * (1 + pad_frac), min_abs)
    return (-hi, hi)
end

function colorbar_tick_spec(colorrange::Tuple{<:Real, <:Real}; n_ticks::Int = 5)
    vals = collect(range(Float64(colorrange[1]), Float64(colorrange[2]); length = n_ticks))
    labels = [@sprintf("%.3f", v) for v in vals]
    return (vals, labels)
end

function bonito_module()
    backend = activate_topoplot_backend!()
    backend == :WGLMakie || error("WGLMakie is unavailable in the current environment, so the interactive Bonito app cannot be built.")
    wglmakie = Base.invokelatest(getfield, @__MODULE__, :WGLMakie)
    return getfield(wglmakie, :Bonito)
end

function load_normalized_fixation_positions()
    @assert isfile(FIXATION_POSITIONS_PATH) "File not found: $FIXATION_POSITIONS_PATH"
    raw_positions = Point2f.(JLD2.load_object(FIXATION_POSITIONS_PATH))
    xs = Float64[first(p) for p in raw_positions]
    ys = Float64[last(p) for p in raw_positions]

    center_x = mean(xs)
    center_y = mean(ys)
    shifted = Point2f[(p[1] - center_x, p[2] - center_y) for p in raw_positions]
    radius = maximum(norm, shifted)
    radius <= 0 && (radius = 1.0)

    return Point2f[(p[1] / radius, p[2] / radius) for p in shifted]
end

function interpolate_topomap(
        positions::AbstractVector,
        values::AbstractVector{<:Real};
        grid_size::Int = 180,
        power::Float64 = 3.0,
        head_radius::Float64 = 1.0)
    xs = collect(range(-1.08, 1.08; length = grid_size))
    ys = collect(range(-1.08, 1.08; length = grid_size))
    grid = Matrix{Float32}(undef, length(ys), length(xs))

    px = Float64[first(p) for p in positions]
    py = Float64[last(p) for p in positions]
    vv = Float64.(values)
    eps2 = 1e-6

    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        if hypot(x, y) > head_radius
            grid[j, i] = NaN32
            continue
        end

        dist2 = (px .- x) .^ 2 .+ (py .- y) .^ 2
        nearest = argmin(dist2)
        if dist2[nearest] < eps2
            grid[j, i] = Float32(vv[nearest])
            continue
        end

        weights = 1.0 ./ (dist2 .+ eps2) .^ (power / 2)
        grid[j, i] = Float32(sum(weights .* vv) / sum(weights))
    end

    return (xs = xs, ys = ys, values = grid)
end

function head_outline_points(; n::Int = 361, radius::Float64 = 1.0)
    angles = range(0, 2pi; length = n)
    return Point2f[(radius * cos(a), radius * sin(a)) for a in angles]
end

function nose_outline_points()
    return Point2f[
        (-0.11f0, 0.98f0),
        (0.00f0, 1.11f0),
        (0.11f0, 0.98f0),
    ]
end

function ear_outline_points(side::Symbol)
    x_sign = side === :left ? -1.0f0 : 1.0f0
    return Point2f[
        (x_sign * 0.97f0, 0.18f0),
        (x_sign * 1.08f0, 0.10f0),
        (x_sign * 1.10f0, -0.06f0),
        (x_sign * 1.02f0, -0.18f0),
    ]
end

function closest_electrode_index(mouse_pos, positions)
    best_idx = 1
    best_dist_sq = Inf

    for i in eachindex(positions)
        dx = positions[i][1] - mouse_pos[1]
        dy = positions[i][2] - mouse_pos[2]
        dist_sq = dx * dx + dy * dy

        if dist_sq < best_dist_sq
            best_dist_sq = dist_sq
            best_idx = i
        end
    end

    return best_idx
end

function local_band_abs_mean(img::AbstractMatrix, row::Int, col::Int; band_radius::Int = 3)
    lo = max(1, col - band_radius)
    hi = min(size(img, 2), col + band_radius)
    return mean(abs, @view img[row, lo:hi])
end

function dummy_channel_features(img::AbstractMatrix)
    n_rows, n_cols = size(img)
    row_axis = collect(range(-1.0, 1.0; length = n_rows))
    peak_cols = Vector{Float64}(undef, n_rows)
    row_energy = Vector{Float64}(undef, n_rows)
    diag_vals = Vector{Float64}(undef, n_rows)
    anti_vals = Vector{Float64}(undef, n_rows)

    span_den = max(n_cols - 1, 1)
    for row in 1:n_rows
        row_view = @view img[row, :]
        _, peak_idx = findmax(abs.(row_view))
        peak_cols[row] = Float64(peak_idx)
        row_energy[row] = mean(abs, row_view)

        frac = n_rows == 1 ? 0.0 : (row - 1) / (n_rows - 1)
        main_col = Int(round(1 + frac * (n_cols - 1)))
        anti_col = Int(round(n_cols - frac * (n_cols - 1)))
        diag_vals[row] = local_band_abs_mean(img, row, main_col)
        anti_vals[row] = local_band_abs_mean(img, row, anti_col)
    end

    peak_cols_norm = (peak_cols .- 1.0) ./ span_den
    late_start = max(1, Int(round(0.62 * n_cols)))
    early_stop = min(n_cols, Int(round(0.38 * n_cols)))

    return (
        peak_trend_abs = abs(safe_cor(row_axis, peak_cols_norm)),
        peak_span = quantile(peak_cols_norm, 0.90) - quantile(peak_cols_norm, 0.10),
        diag_advantage = mean(diag_vals) - mean(anti_vals),
        late_energy_shift = mean(abs, @view img[:, late_start:end]) - mean(abs, @view img[:, 1:early_stop]),
        row_energy_flatness = std(row_energy),
    )
end

function duration_image_from_dataset(
        erps,
        duration_order::AbstractVector{<:Integer},
        channel::Int;
        start_idx::Int = TD.FIXATION_TIME_ZERO_IDX,
        n_trials::Int,
        target_size::Tuple{Int, Int},
        lowpass::Bool)
    data_time_trials = Float32.(erps[channel, start_idx:end, 1:n_trials])
    data_sorted = data_time_trials[:, duration_order]
    base_img = Float32.(permutedims(TD.zscore_timepoints(data_sorted), (2, 1)))
    return TD.process_erp_image(base_img, target_size; lowpass = lowpass)
end

function duration_image_from_subset(
        data_time_trials::AbstractMatrix;
        target_size::Tuple{Int, Int},
        lowpass::Bool)
    base_img = Float32.(permutedims(TD.zscore_timepoints(data_time_trials), (2, 1)))
    return TD.process_erp_image(base_img, target_size; lowpass = lowpass)
end

function load_duration_label_metadata()
    for path in vcat(OL.RESULTS_CSV_PATHS, [TD.FIXATION_H5_PATH, TD.FIXATION_EVENTS_CSV_PATH])
        @assert isfile(path) "File not found: $path"
    end

    events = CSV.read(TD.FIXATION_EVENTS_CSV_PATH, DataFrame)
    labels_all_df, labels_merged_df = OL.load_and_merge_label_sources(OL.RESULTS_CSV_PATHS)

    labels_merged_df.erp_class_id = [OL.parse_class_id(v) for v in labels_merged_df.erp_class]
    valid_mask = map(v -> !ismissing(v), labels_merged_df.erp_class_id)
    meta_mask = map(OL.has_required_metadata, eachrow(labels_merged_df))
    duration_mask = [normalize_sort_name(v) == DURATION_SORT_NAME for v in labels_merged_df.sort_variable]
    labels_df = copy(labels_merged_df[valid_mask .& meta_mask .& duration_mask, :])

    labels_df.channel_int = Int.(labels_df.channel)
    labels_df.sort_var_symbol = fill(:duration, nrow(labels_df))
    labels_df.binary_label = Int.(labels_df.erp_class_id .> 0)
    @assert nrow(labels_df) > 0 "No labelled fixation duration samples found."

    return (
        events = events,
        labels_all_df = labels_all_df,
        labels_df = labels_df,
    )
end

function load_positive_sort_mix_label_metadata()
    for path in vcat(OL.RESULTS_CSV_PATHS, [TD.FIXATION_H5_PATH, TD.FIXATION_EVENTS_CSV_PATH])
        @assert isfile(path) "File not found: $path"
    end

    events = CSV.read(TD.FIXATION_EVENTS_CSV_PATH, DataFrame)
    labels_all_df, labels_merged_df = OL.load_and_merge_label_sources(OL.RESULTS_CSV_PATHS)

    labels_merged_df.erp_class_id = [OL.parse_class_id(v) for v in labels_merged_df.erp_class]
    valid_mask = map(v -> !ismissing(v), labels_merged_df.erp_class_id)
    meta_mask = map(OL.has_required_metadata, eachrow(labels_merged_df))
    labels_df = copy(labels_merged_df[valid_mask .& meta_mask, :])

    labels_df.channel_int = Int.(labels_df.channel)
    labels_df.sort_var_symbol = Symbol.(String.(labels_df.sort_variable))
    labels_df.binary_label = Int.(labels_df.erp_class_id .> 0)

    selected_sort_names = sort(unique(normalize_sort_name(v) for v in labels_df.sort_variable[labels_df.binary_label .== 1]))
    selected_sort_set = Set(selected_sort_names)
    labels_df = copy(labels_df[in.(normalize_sort_name.(labels_df.sort_variable), Ref(selected_sort_set)), :])
    @assert nrow(labels_df) > 0 "No labelled fixation samples remain for positive-sort mix training."

    return (
        events = events,
        labels_all_df = labels_all_df,
        labels_df = labels_df,
        selected_sort_names = selected_sort_names,
    )
end

function build_duration_training_bundle(
        label_meta;
        exclude_channels::AbstractVector{<:Integer} = Int[],
        target_size::Tuple{Int, Int},
        data_split_seed::Int,
        positive_split_k::Int = MOD4_SPLIT_K,
        no_class_split_k::Int = MOD4_SPLIT_K)
    exclude_set = Set(Int.(collect(exclude_channels)))
    labels_df = isempty(exclude_set) ? copy(label_meta.labels_df) :
        copy(label_meta.labels_df[.!in.(Int.(label_meta.labels_df.channel_int), Ref(exclude_set)), :])

    @assert nrow(labels_df) > 0 "No training labels remain after excluding inference channels."
    @assert length(unique(Int.(labels_df.binary_label))) == 2 "Training labels must still contain both classes after channel exclusion."

    no_class_rng = MersenneTwister(data_split_seed)
    train_df = TD.with_erps_dataset(TD.FIXATION_H5_PATH) do erps
        OL.build_training_dataset(
            erps,
            label_meta.events,
            labels_df;
            positive_split_k = positive_split_k,
            no_class_split_k = no_class_split_k,
            no_class_pick_rng = no_class_rng,
            target_size = target_size,
        )
    end

    shuffle_rng = MersenneTwister(data_split_seed + 1)
    perm = randperm(shuffle_rng, nrow(train_df))
    train_df = copy(train_df[perm, :])
    train_df.sample_id = collect(1:nrow(train_df))

    return (
        events = label_meta.events,
        labels_all_df = label_meta.labels_all_df,
        labels_df = labels_df,
        train_df = train_df,
    )
end

function unseen_duration_task_df(
        labels_df::DataFrame;
        tasks_path::AbstractString,
        max_channels::Int,
        channels::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    @assert max_channels > 0 "max_channels must be positive."

    labelled_channels = Set(Int.(labels_df.channel_int))

    if channels !== nothing
        rows = NamedTuple[]
        seen = Set{Int}()
        for channel in Int.(collect(channels))
            channel in seen && continue
            channel in labelled_channels && continue
            push!(seen, channel)
            push!(rows, (
                task_id = missing,
                channel = channel,
                image_file = missing,
                source = "manual_channels",
            ))
            length(rows) >= max_channels && break
        end
        return DataFrame(rows)
    end

    @assert isfile(tasks_path) "File not found: $tasks_path"
    tasks = JSON3.read(read(tasks_path, String))

    rows = NamedTuple[]
    seen = Set{Int}()
    for task in tasks
        data = task["data"]
        normalize_sort_name(data["sort_variable"]) == DURATION_SORT_NAME || continue

        channel = Int(data["channel"])
        channel in seen && continue
        channel in labelled_channels && continue

        push!(seen, channel)
        push!(rows, (
            task_id = Int(task["id"]),
            channel = channel,
            image_file = haskey(data, "image_file") ? String(data["image_file"]) : missing,
            source = basename(tasks_path),
        ))
        length(rows) >= max_channels && break
    end

    return DataFrame(rows)
end

function heldout_labelled_task_df(
        labels_df::DataFrame;
        max_channels::Int,
        data_split_seed::Int,
        minimum_train_channels::Int = 16)
    channel_df = unique(select(labels_df, :channel_int, :binary_label, :image_file), :channel_int)
    n_channels_total = nrow(channel_df)
    @assert n_channels_total >= 3 "At least three labelled duration channels are required for holdout inference."

    desired_holdout = min(max_channels, max(1, n_channels_total - minimum_train_channels))
    desired_holdout = min(desired_holdout, max(1, n_channels_total - 2))

    rng = MersenneTwister(data_split_seed + 17)
    unique_labels = sort(unique(Int.(channel_df.binary_label)))
    selected = Int[]
    selected_set = Set{Int}()
    taken_by_label = Dict{Int, Int}()
    shuffled_by_label = Dict{Int, Vector{Int}}()

    for label in unique_labels
        label_channels = collect(Int.(channel_df.channel_int[channel_df.binary_label .== label]))
        shuffle!(rng, label_channels)
        shuffled_by_label[label] = label_channels

        max_take = max(0, length(label_channels) - 1)
        target_take = min(max_take, round(Int, desired_holdout * length(label_channels) / n_channels_total))
        taken_by_label[label] = target_take

        for channel in label_channels[1:target_take]
            push!(selected, channel)
            push!(selected_set, channel)
        end
    end

    if length(selected) < desired_holdout
        extras = Int[]
        for label in unique_labels
            label_channels = shuffled_by_label[label]
            max_take = max(0, length(label_channels) - 1)
            already = get(taken_by_label, label, 0)
            if already < max_take
                append!(extras, label_channels[(already + 1):max_take])
            end
        end
        shuffle!(rng, extras)
        for channel in extras
            channel in selected_set && continue
            push!(selected, channel)
            push!(selected_set, channel)
            length(selected) >= desired_holdout && break
        end
    end

    rows = NamedTuple[]
    for channel in selected
        row = only(channel_df[Int.(channel_df.channel_int) .== channel, :])
        push!(rows, (
            task_id = missing,
            channel = channel,
            image_file = row.image_file,
            source = "heldout_labelled_channels",
            true_binary_label = Int(row.binary_label),
            true_label = Int(row.binary_label) == 1 ? "pattern" : "no_pattern",
        ))
    end

    out = DataFrame(rows)
    @assert nrow(out) > 0 "Failed to sample heldout labelled duration channels."
    return out
end

function select_modulo_split(groups::Vector{Vector{Int}}, channel::Int, rng::AbstractRNG)
    isempty(groups) && error("No modulo groups available for channel $channel.")

    nonempty_parts = findall(group -> !isempty(group), groups)
    isempty(nonempty_parts) && error("All modulo groups are empty for channel $channel.")
    keep_part = rand(rng, nonempty_parts)
    idxs = groups[keep_part]
    return (keep_part = keep_part, idxs = idxs)
end

function build_unseen_duration_images(
        events::DataFrame,
        channels::AbstractVector{<:Integer};
        target_size::Tuple{Int, Int},
        lowpass::Bool,
        inference_split_k::Int = MOD4_SPLIT_K)
    images = Matrix{Float32}[]
    rows = NamedTuple[]
    n_timepoints_post = 0

    TD.with_erps_dataset(TD.FIXATION_H5_PATH) do erps
        n_timepoints_post = size(erps, 2) - TD.FIXATION_TIME_ZERO_IDX + 1
        for channel in Int.(collect(channels))
            data_full, events_full = OL.extract_channel_trials(
                erps,
                events,
                channel;
                time_zero_idx = TD.FIXATION_TIME_ZERO_IDX,
            )
            groups = OL.split_indices_sorted_modulo(events_full, :duration, inference_split_k)
            for keep_part in eachindex(groups)
                idxs = groups[keep_part]
                isempty(idxs) && continue

                push!(images, duration_image_from_subset(
                    data_full[:, idxs];
                    target_size = target_size,
                    lowpass = lowpass,
                ))
                push!(rows, (
                    image_index = length(images),
                    channel = channel,
                    inference_variant = @sprintf("mod%d_keep%d", length(groups), keep_part),
                    inference_keep_part = keep_part,
                    inference_n_trials = length(idxs),
                ))
            end
        end
    end

    return (
        images = images,
        split_meta_df = DataFrame(rows),
        n_timepoints_post = n_timepoints_post,
    )
end

function train_resnet18_duration_model(
        train_df::DataFrame;
        train_epochs::Int,
        train_lr::Float32,
        train_batchsize::Int,
        use_class_weights::Bool = false,
        train_scope::Symbol = :head_only)
    @assert nrow(train_df) > 0 "Training dataframe is empty."
    @assert train_scope in (:full, :head_only) "train_scope must be :full or :head_only."

    X_train = OL.images_to_tensor(Vector{Matrix{Float32}}(train_df.processed_img))
    y_train = Int.(train_df.binary_label)
    y_train_oh = OL.onehotbatch(y_train, 0:1) |> Array{Float32}
    train_loader = OL.DataLoader((X_train, y_train_oh); batchsize = train_batchsize, shuffle = true)
    class_weights = use_class_weights ? inverse_frequency_class_weights(y_train) : Float32[1, 1]

    model, pretrained_loaded, source_matched = OL.build_resnet18_pretrained_1ch()
    model = OL.device(model)
    class_weights_dev = OL.device(reshape(class_weights, :, 1))
    OL.maybe_cuda_reclaim!()
    optim = Flux.Adam(train_lr)
    train_target = train_scope == :head_only ? (head = model.head,) : model
    opt_state = Flux.setup(optim, train_target)
    Flux.trainmode!(model)

    train_time_s = @elapsed begin
        for epoch in 1:train_epochs
            running_loss = 0f0
            n_batches = 0

            for (xb_cpu, yb_cpu) in train_loader
                xb = OL.device(xb_cpu)
                yb = OL.device(yb_cpu)

                loss_val, grads = Flux.withgradient(model) do m
                    logits = if train_scope == :head_only
                        pooled = Flux.Zygote.ignore() do
                            m.pre_logits(m.feature_maps(xb))
                        end
                        m.head(pooled)
                    else
                        m(xb)
                    end
                    weighted_logitcrossentropy(logits, yb, class_weights_dev)
                end

                if train_scope == :head_only
                    head_target = (head = model.head,)
                    head_grads = (head = grads[1].head,)
                    opt_state, updated = Flux.update!(opt_state, head_target, head_grads)
                    model = OL.InspectableClassifier(model.feature_maps, model.pre_logits, updated.head)
                else
                    opt_state, model = Flux.update!(opt_state, model, grads[1])
                end
                running_loss += loss_val
                n_batches += 1
            end

            avg_loss = running_loss / max(1, n_batches)
            @info "resnet18_fixation_duration_topoplot | epoch $(epoch)/$(train_epochs) | train_loss=$(round(avg_loss, digits = 5))"
        end
    end
    Flux.testmode!(model)

    return (
        model = model,
        y_train = y_train,
        train_time_s = train_time_s,
        pretrained_loaded = pretrained_loaded,
        source_matched = source_matched,
        class_weights = class_weights,
        train_scope = String(train_scope),
    )
end

function score_unseen_duration_images(
        model,
        task_df::DataFrame,
        split_meta_df::DataFrame,
        images::Vector{Matrix{Float32}})
    X_infer = OL.images_to_tensor(images)
    forward = OL.inspect_forward(model, OL.device(X_infer))
    logits = Array(OL.to_cpu(forward.logits))
    probs = OL.stable_softmax(logits)
    pooled = Array(OL.to_cpu(forward.pooled))

    split_score_df = copy(split_meta_df)
    split_score_df.logit_no_class = vec(Float64.(logits[1, :]))
    split_score_df.logit_erp_class = vec(Float64.(logits[2, :]))
    split_score_df.prob_no_pattern = vec(Float64.(probs[1, :]))
    split_score_df.prob_pattern = vec(Float64.(probs[2, :]))
    split_score_df.pattern_margin = split_score_df.prob_pattern .- split_score_df.prob_no_pattern
    split_score_df.predicted_label = pattern_label.(split_score_df.prob_pattern)

    selected_rows = Int[]
    for channel in Int.(task_df.channel)
        rows = findall(==(channel), Int.(split_score_df.channel))
        isempty(rows) && continue

        best_pos = argmax([
            (
                split_score_df.prob_pattern[row],
                split_score_df.pattern_margin[row],
                -split_score_df.inference_keep_part[row],
            )
            for row in rows
        ])
        push!(selected_rows, rows[best_pos])
    end

    selected_split_df = copy(split_score_df[selected_rows, :])
    sort!(selected_split_df, :channel)

    score_df = leftjoin(copy(task_df), selected_split_df; on = :channel)
    score_df.channel_name = [@sprintf("ch%03d", ch) for ch in Int.(score_df.channel)]
    for col in [:image_index, :inference_keep_part, :inference_n_trials]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = Int.(score_df[!, col])
    end
    for col in [:logit_no_class, :logit_erp_class, :prob_no_pattern, :prob_pattern, :pattern_margin]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = Float64.(score_df[!, col])
    end
    for col in [:inference_variant, :predicted_label]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = String.(score_df[!, col])
    end
    selected_image_indices = Int.(score_df.image_index)
    selected_pooled = pooled[:, selected_image_indices]

    return (
        score_df = score_df,
        split_score_df = split_score_df,
        pooled_features = selected_pooled,
        selected_image_indices = selected_image_indices,
    )
end

function score_unseen_duration_images_dummy(
        task_df::DataFrame,
        split_meta_df::DataFrame,
        images::Vector{Matrix{Float32}})
    @assert length(images) == nrow(split_meta_df) "Image count and split metadata row count must match."

    feature_rows = [dummy_channel_features(img) for img in images]
    split_score_df = hcat(copy(split_meta_df), DataFrame(feature_rows))

    dummy_logit = fill(DUMMY_PATTERN_BIAS, nrow(split_score_df))
    for feature_name in DUMMY_FEATURE_NAMES
        raw_values = Float64.(split_score_df[!, feature_name])
        z_col = Symbol(string(feature_name), "_z")
        contrib_col = Symbol(string(feature_name), "_contrib")
        split_score_df[!, z_col] = safe_zscore(raw_values)
        split_score_df[!, contrib_col] = DUMMY_PATTERN_WEIGHTS[feature_name] .* Float64.(split_score_df[!, z_col])
        dummy_logit .+= Float64.(split_score_df[!, contrib_col])
    end

    split_score_df.logit_erp_class = dummy_logit
    split_score_df.logit_no_class = .-dummy_logit
    split_score_df.prob_pattern = dummy_probability_from_logit.(dummy_logit)
    split_score_df.prob_no_pattern = 1 .- split_score_df.prob_pattern
    split_score_df.pattern_margin = split_score_df.prob_pattern .- split_score_df.prob_no_pattern
    split_score_df.predicted_label = pattern_label.(split_score_df.prob_pattern)

    selected_rows = Int[]
    for channel in Int.(task_df.channel)
        rows = findall(==(channel), Int.(split_score_df.channel))
        isempty(rows) && continue

        best_pos = argmax([
            (
                split_score_df.prob_pattern[row],
                split_score_df.pattern_margin[row],
                -split_score_df.inference_keep_part[row],
            )
            for row in rows
        ])
        push!(selected_rows, rows[best_pos])
    end

    selected_split_df = copy(split_score_df[selected_rows, :])
    sort!(selected_split_df, :channel)

    score_df = leftjoin(copy(task_df), selected_split_df; on = :channel)
    score_df.channel_name = [@sprintf("ch%03d", ch) for ch in Int.(score_df.channel)]
    for col in [:image_index, :inference_keep_part, :inference_n_trials]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = Int.(score_df[!, col])
    end
    for col in [:logit_no_class, :logit_erp_class, :prob_no_pattern, :prob_pattern, :pattern_margin]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = Float64.(score_df[!, col])
    end
    for col in [:inference_variant, :predicted_label]
        @assert !any(ismissing, score_df[!, col]) "Missing values encountered in $(col) after channel aggregation."
        score_df[!, col] = String.(score_df[!, col])
    end
    selected_image_indices = Int.(score_df.image_index)
    selected_pooled = permutedims(Matrix{Float64}(selected_split_df[:, DUMMY_FEATURE_NAMES]))

    return (
        score_df = score_df,
        split_score_df = split_score_df,
        pooled_features = selected_pooled,
        selected_image_indices = selected_image_indices,
    )
end

function build_fixation_duration_topoplot_context(;
        target_size::Tuple{Int, Int} = TD.REAL_TARGET_SIZE,
        lowpass::Bool = true,
        channels::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        max_inference_channels::Int = 64,
        unseen_tasks_path::AbstractString = ADDITIONAL_DURATION_TASKS_PATH,
        data_split_seed::Union{Nothing, Int} = 20260308,
        score_mode::Symbol = :dummy,
        training_sort_mode::Symbol = :positive_sort_mix,
        positive_split_k::Int = MOD4_SPLIT_K,
        no_class_split_k::Int = MOD4_SPLIT_K,
        inference_split_k::Int = MOD4_SPLIT_K,
        use_class_weights::Bool = false,
        train_scope::Symbol = :head_only,
        train_epochs::Int = 3,
        train_lr::Float32 = 1f-4,
        train_batchsize::Int = 16)
    seed_info = resolve_data_split_seed(data_split_seed)
    resolved_seed = seed_info.seed
    @assert score_mode in (:dummy, :resnet18) "score_mode must be :dummy or :resnet18."
    @assert training_sort_mode in (:duration_only, :positive_sort_mix) "training_sort_mode must be :duration_only or :positive_sort_mix."
    duration_label_meta = load_duration_label_metadata()
    task_df = unseen_duration_task_df(
        duration_label_meta.labels_df;
        tasks_path = unseen_tasks_path,
        max_channels = max_inference_channels,
        channels = channels,
    )
    inference_source = "additional_unlabelled_duration_tasks"
    heldout_channels = Int[]

    if nrow(task_df) == 0
        channels === nothing || error("No unseen channels remain after excluding labelled duration channels from the manual channel list.")
        task_df = heldout_labelled_task_df(
            duration_label_meta.labels_df;
            max_channels = max_inference_channels,
            data_split_seed = resolved_seed,
        )
        heldout_channels = Int.(task_df.channel)
        inference_source = "heldout_labelled_duration_channels"
    end

    unseen = build_unseen_duration_images(
        duration_label_meta.events,
        Int.(task_df.channel);
        target_size = target_size,
        lowpass = lowpass,
        inference_split_k = inference_split_k,
    )
    training_labels_df = DataFrame()
    training_df = DataFrame()

    if score_mode == :dummy
        scored = score_unseen_duration_images_dummy(task_df, unseen.split_meta_df, unseen.images)
        model_summary_df = DataFrame(
            metric = [
                "score_mode",
                "duration_label_rows",
                "duration_label_channels",
                "unseen_duration_channels",
                "inference_split_images_total",
                "dummy_pattern_bias",
                "dummy_logit_temperature",
                "dummy_probability_span",
                "dummy_feature_weights",
                "inference_split_k",
                "resolved_data_split_seed",
                "seed_source",
                "inference_channel_source",
            ],
            value = Any[
                String(score_mode),
                nrow(duration_label_meta.labels_df),
                length(unique(Int.(duration_label_meta.labels_df.channel_int))),
                nrow(task_df),
                nrow(unseen.split_meta_df),
                DUMMY_PATTERN_BIAS,
                DUMMY_LOGIT_TEMPERATURE,
                DUMMY_PROBABILITY_SPAN,
                join([@sprintf("%s=%+.2f", String(name), DUMMY_PATTERN_WEIGHTS[name]) for name in DUMMY_FEATURE_NAMES], " | "),
                inference_split_k,
                resolved_seed,
                seed_info.source,
                inference_source,
            ],
        )
    else
        training_label_meta = training_sort_mode == :positive_sort_mix ? load_positive_sort_mix_label_metadata() : duration_label_meta
        training_sort_names = training_sort_mode == :positive_sort_mix ? training_label_meta.selected_sort_names : [DURATION_SORT_NAME]
        training = build_duration_training_bundle(
            training_label_meta;
            exclude_channels = heldout_channels,
            target_size = target_size,
            data_split_seed = resolved_seed,
            positive_split_k = positive_split_k,
            no_class_split_k = no_class_split_k,
        )
        fitted = train_resnet18_duration_model(
            training.train_df;
            train_epochs = train_epochs,
            train_lr = train_lr,
            train_batchsize = train_batchsize,
            use_class_weights = use_class_weights,
            train_scope = train_scope,
        )
        scored = score_unseen_duration_images(fitted.model, task_df, unseen.split_meta_df, unseen.images)
        training_labels_df = training.labels_df
        training_df = training.train_df

        model_summary_df = DataFrame(
            metric = [
                "score_mode",
                "training_label_rows",
                "training_label_channels",
                "duration_label_rows",
                "duration_label_channels",
                "train_samples",
                "train_positive_samples",
                "train_negative_samples",
                "unseen_duration_channels",
                "inference_split_images_total",
                "training_sort_mode",
                "training_sort_vars",
                "train_positive_split_k",
                "train_no_class_split_k",
                "inference_split_k",
                "class_weight_no_class",
                "class_weight_pattern",
                "use_class_weights",
                "train_scope",
                "resolved_data_split_seed",
                "seed_source",
                "train_epochs",
                "train_batchsize",
                "train_lr",
                "train_time_s",
                "pretrained_backbone_arrays_loaded",
                "source_arrays_matched",
                "inference_channel_source",
            ],
            value = Any[
                String(score_mode),
                nrow(training_label_meta.labels_df),
                length(unique(Int.(training_label_meta.labels_df.channel_int))),
                nrow(duration_label_meta.labels_df),
                length(unique(Int.(duration_label_meta.labels_df.channel_int))),
                nrow(training.train_df),
                count(==(1), Int.(training.train_df.binary_label)),
                count(==(0), Int.(training.train_df.binary_label)),
                nrow(task_df),
                nrow(unseen.split_meta_df),
                String(training_sort_mode),
                join(training_sort_names, ", "),
                positive_split_k,
                no_class_split_k,
                inference_split_k,
                Float64(fitted.class_weights[1]),
                Float64(fitted.class_weights[2]),
                use_class_weights,
                fitted.train_scope,
                resolved_seed,
                seed_info.source,
                train_epochs,
                train_batchsize,
                train_lr,
                fitted.train_time_s,
                fitted.pretrained_loaded,
                fitted.source_matched,
                inference_source,
            ],
        )
    end

    score_df = copy(scored.score_df)
    score_df.topomap_relative_score = relative_pattern_score(score_df.prob_pattern)
    selected_images = [unseen.images[idx] for idx in scored.selected_image_indices]
    positions_all = load_normalized_fixation_positions()
    selected_positions = positions_all[Int.(score_df.channel)]
    image_stats = [TD.image_color_stats(img) for img in selected_images]
    topo_grid = interpolate_topomap(selected_positions, Float64.(score_df.topomap_relative_score))

    GC.gc()
    OL.maybe_cuda_reclaim!()

    return (
        sort_var = :duration,
        lowpass = lowpass,
        target_size = target_size,
        n_timepoints_post = unseen.n_timepoints_post,
        positions = selected_positions,
        topo_grid = topo_grid,
        score_df = score_df,
        split_score_df = scored.split_score_df,
        images = selected_images,
        image_stats = image_stats,
        image_trial_counts = Int.(score_df.inference_n_trials),
        image_variants = String.(score_df.inference_variant),
        pooled_features = scored.pooled_features,
        resized_size = size(first(selected_images)),
        model_summary_df = model_summary_df,
        training_labels_df = training_labels_df,
        duration_labels_df = duration_label_meta.labels_df,
        training_df = training_df,
        unseen_task_df = task_df,
        score_mode = score_mode,
    )
end

function plot_fixation_duration_topoplot(ctx; initial_channel::Union{Nothing, Int} = nothing)
    @assert nrow(ctx.score_df) > 0 "No channels available for plotting."
    score_mode = hasproperty(ctx, :score_mode) ? ctx.score_mode : :resnet18
    mode_title = score_mode == :dummy ?
        "dummy class probabilities from ERP image heuristics" :
        "resnet18 on real labelled data"
    left_title = score_mode == :dummy ?
        "unseen fixation duration channels | relative dummy pattern evidence within the current channel set" :
        "unseen fixation duration channels | relative pattern evidence within the current channel set"
    prob_label_pattern = score_mode == :dummy ? "dummy p(pattern)" : "p(pattern)"
    prob_label_no_pattern = score_mode == :dummy ? "dummy p(no class)" : "p(no class)"

    initial_row = if initial_channel === nothing
        argmax(Float64.(ctx.score_df.prob_pattern))
    else
        found = findfirst(==(initial_channel), Int.(ctx.score_df.channel))
        found === nothing && error("Initial channel $(initial_channel) is not part of the current context.")
        found
    end

    selected_row = Observable(initial_row)
    selected_name = @lift(String(ctx.score_df.channel_name[$selected_row]))
    selected_prob_pattern = @lift(Float64(ctx.score_df.prob_pattern[$selected_row]))
    selected_prob_no_pattern = @lift(Float64(ctx.score_df.prob_no_pattern[$selected_row]))
    selected_variant = @lift(String(ctx.score_df.inference_variant[$selected_row]))
    selected_trials = @lift(Int(ctx.score_df.inference_n_trials[$selected_row]))
    image_obs = @lift(ctx.image_stats[$selected_row].clipped)
    cmap_obs = @lift(ctx.image_stats[$selected_row].cmap)
    colorrange_obs = @lift(ctx.image_stats[$selected_row].colorrange)
    ticks_obs = @lift((ctx.image_stats[$selected_row].tick_vals, ctx.image_stats[$selected_row].tick_labels))
    selected_pos_obs = @lift([ctx.positions[$selected_row]])
    ytick_obs = @lift(TD.fixation_axis_ticks((
        resized_size = ctx.resized_size,
        n_timepoints_post = ctx.n_timepoints_post,
        n_trials = ctx.image_trial_counts[$selected_row],
    )).yticks)
    topomap_range = symmetric_colorrange(ctx.score_df.topomap_relative_score)
    topomap_ticks = colorbar_tick_spec(topomap_range)

    topomap_cmap = cgrad([:dodgerblue4, :white, :firebrick3], [0.0, 0.5, 1.0])

    fig = Figure(size = (1450, 720), figure_padding = 18)
    Label(
        fig[0, 1:2],
        "Fixation duration topoplot prototype | $(mode_title) | click a channel to open one unseen mod4 ERP image";
        fontsize = 20,
        tellwidth = false,
    )

    left = GridLayout(fig[1, 1])
    ax_topo = Axis(
        left[1, 1];
        title = left_title,
        aspect = DataAspect(),
        xlabel = "",
        ylabel = "",
        xgridvisible = false,
        ygridvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
        backgroundcolor = :white,
    )
    hidedecorations!(ax_topo)

    hm_topo = heatmap!(
        ax_topo,
        ctx.topo_grid.xs,
        ctx.topo_grid.ys,
        ctx.topo_grid.values;
        colormap = topomap_cmap,
        colorrange = topomap_range,
        interpolate = true,
        inspectable = false,
        nan_color = RGBAf(0, 0, 0, 0),
    )

    lines!(ax_topo, head_outline_points(); color = :black, linewidth = 2.0)
    lines!(ax_topo, nose_outline_points(); color = :black, linewidth = 2.0)
    lines!(ax_topo, ear_outline_points(:left); color = :black, linewidth = 2.0)
    lines!(ax_topo, ear_outline_points(:right); color = :black, linewidth = 2.0)

    marker_sizes = 12 .+ 16 .* abs.(Float32.(ctx.score_df.topomap_relative_score))
    topo_scatter = scatter!(
        ax_topo,
        first.(ctx.positions),
        last.(ctx.positions);
        color = Float32.(ctx.score_df.topomap_relative_score),
        colormap = topomap_cmap,
        colorrange = topomap_range,
        markersize = marker_sizes,
        strokecolor = :black,
        strokewidth = 1.1,
    )
    picker_plot = scatter!(
        ax_topo,
        first.(ctx.positions),
        last.(ctx.positions);
        color = RGBAf(0, 0, 0, 0),
        strokecolor = RGBAf(0, 0, 0, 0),
        markersize = max.(marker_sizes, 28),
    )
    scatter!(
        ax_topo,
        selected_pos_obs;
        color = :transparent,
        markersize = 31,
        strokecolor = :white,
        strokewidth = 3.2,
    )
    scatter!(
        ax_topo,
        selected_pos_obs;
        color = :transparent,
        markersize = 36,
        strokecolor = :black,
        strokewidth = 1.6,
    )

    xlims!(ax_topo, -1.18, 1.18)
    ylims!(ax_topo, -1.12, 1.20)
    Colorbar(
        left[1, 2],
        hm_topo;
        label = "relative pattern evidence\n(0 = channel-set mean, positive = above mean)",
        ticks = topomap_ticks,
        width = 16,
    )

    right = GridLayout(fig[1, 2])
    ticks = TD.fixation_axis_ticks((
        resized_size = ctx.resized_size,
        n_timepoints_post = ctx.n_timepoints_post,
        n_trials = ctx.image_trial_counts[initial_row],
    ))

    ax_img = Axis(
        right[1, 1];
        title = @lift(@sprintf(
            "%s | unseen duration ERP image | %s | n=%d | %s=%.3f | %s=%.3f",
            $selected_name,
            $selected_variant,
            $selected_trials,
            prob_label_pattern,
            $selected_prob_pattern,
            prob_label_no_pattern,
            $selected_prob_no_pattern,
        )),
        xlabel = "post-stimulus timepoints\nelapsed time after stimulus",
        ylabel = "trial rank",
        xticks = ticks.xticks,
        yticks = ytick_obs,
        titlesize = 15,
        xlabelsize = 12,
        ylabelsize = 12,
        xticklabelsize = 10,
        yticklabelsize = 10,
    )

    hm_img = heatmap!(
        ax_img,
        1:ctx.resized_size[2],
        1:ctx.resized_size[1],
        @lift(permutedims($image_obs, (2, 1)));
        colormap = cmap_obs,
        colorrange = colorrange_obs,
    )
    Colorbar(
        right[1, 2],
        hm_img;
        label = "duration ERP image\n(asymmetric zero-anchored)",
        ticks = ticks_obs,
        ticklabelsize = 9,
        width = 16,
    )

    on(events(ax_topo.scene).mousebutton, priority = 20) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plt, idx = pick(ax_topo.scene)
            if (plt === picker_plot || plt === topo_scatter) && idx > 0
                selected_row[] = idx
                return Consume(true)
            end
            CairoMakie.Makie.is_mouseinside(ax_topo.scene) || return Consume(false)
            selected_row[] = closest_electrode_index(mouseposition(ax_topo.scene), ctx.positions)
            return Consume(true)
        end
        return Consume(false)
    end

    return (
        figure = fig,
        selected_row = selected_row,
    )
end

function build_fixation_duration_topoplot_app(
        ctx;
        initial_channel::Union{Nothing, Int} = nothing,
        title::AbstractString = "Fixation duration topoplot")
    bonito = bonito_module()
    page_reset = getfield(bonito, :Page)
    app_ctor = getfield(bonito, :App)

    Base.invokelatest(page_reset)
    topo = plot_fixation_duration_topoplot(ctx; initial_channel = initial_channel)
    app = Base.invokelatest(app_ctor, topo.figure; title = title)
    return (
        topo = topo,
        app = app,
    )
end

function display_fixation_duration_topoplot!(
        ctx;
        target::Symbol = :browser,
        initial_channel::Union{Nothing, Int} = nothing)
    bundle = build_fixation_duration_topoplot_app(ctx; initial_channel = initial_channel)
    bonito = bonito_module()

    if target == :browser
        browser_display = getfield(bonito, :browser_display)
        Base.invokelatest(browser_display)
        display(bundle.app)
    elseif target == :inline
        display(bundle.app)
    else
        error("Unsupported target $(target). Use :browser or :inline.")
    end

    return bundle
end

end
