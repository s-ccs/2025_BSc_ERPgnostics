module Week15SelfSupervisedLearning

using CSV
using CUDA
using CairoMakie
using DataFrames
using Flux
using HDF5
using Images: imresize
using ImageFiltering: imfilter
using JSON3
using MLUtils: DataLoader
using Metalhead
using Printf: @sprintf
using Random
using Statistics
using StatsBase: mean_and_std, quantile

using Flux: onecold, onehotbatch
using LinearAlgebra: diagind

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: gaussian_kernel, zscore_timepoints, clipped_color_stats_quantile_zero_ticks

export OPENNEURO_REPO_DIR
export OPENNEURO_DERIVED_DIR
export OPENNEURO_H5_PATH
export OPENNEURO_EVENTS_PATH
export FIXATION_RESULTS_CSV_PATHS
export ensure_openneuro_8bit_dataset!
export summarize_8bit_events
export load_8bit_unlabelled_cache
export load_fixation_binary_cache
export build_simclr_resnet18
export train_simclr!
export run_transfer_experiment
export plot_image_grid
export plot_loss_history

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OPENNEURO_REPO_DIR = joinpath(REPO_ROOT, "notebooks", "datasets", "ds003517")
const OPENNEURO_DERIVED_DIR = joinpath(REPO_ROOT, "notebooks", "datasets", "ds003517_sub001_derived")
const OPENNEURO_H5_PATH = joinpath(OPENNEURO_DERIVED_DIR, "epochs.hdf5")
const OPENNEURO_EVENTS_PATH = joinpath(OPENNEURO_DERIVED_DIR, "events.csv")
const OPENNEURO_METADATA_PATH = joinpath(OPENNEURO_DERIVED_DIR, "metadata.json")
const OPENNEURO_PREPARE_SCRIPT = joinpath(REPO_ROOT, "scripts", "prepare_openneuro_8bit_dataset.py")
const OPENNEURO_PREPARE_PYTHON = let venv_python = joinpath(REPO_ROOT, ".venv_8bit", "bin", "python")
    isfile(venv_python) ? venv_python : "python"
end

const FIXATIONS_DATASET_DIR = joinpath(@__DIR__, "..", "model_test", "real_data_sets", "fixations_dataset")
const FIXATION_RESULTS_CSV_PATHS = [
    joinpath(@__DIR__, "..", "model_test", "results", "project-14-at-2026-02-15-23-09-f5225e5c.csv"),
    joinpath(@__DIR__, "..", "model_test", "results", "project-15-at-2026-02-18-19-35-828515fe.csv"),
]
const FIXATION_H5_PATH = joinpath(FIXATIONS_DATASET_DIR, "data_fixations.hdf5")
const FIXATION_EVENTS_CSV_PATH = joinpath(FIXATIONS_DATASET_DIR, "events.csv")

const SIGMOID_CLASS_ID = 1
const NO_CLASS_ID = 0
const REAL_LOWPASS_SIGMA = 75.0f0
const REAL_LOWPASS_KERNEL_SIZE = (21, 21)
const FILTER_BORDER = "reflect"

const FIXATION_PRE_STIM_S = 0.5
const FIXATION_SAMPLING_RATE = 512
const FIXATION_TIME_ZERO_IDX = Int(round(FIXATION_PRE_STIM_S * FIXATION_SAMPLING_RATE)) + 1
const SORT_TIEBREAKER_COLUMNS = [
    :source_part_index,
    :source_epoch_index,
    :epoch_index,
    :sample_index,
    :event_rank_within_type,
    :flash_index_within_run,
    :flash_index_within_trial,
    :onset_s,
    :stimulus_onset_s,
]

safe_div(num::Real, den::Real) = den == 0 ? 0.0 : Float64(num) / Float64(den)

function ensure_openneuro_8bit_dataset!()
    if isfile(OPENNEURO_H5_PATH) && isfile(OPENNEURO_EVENTS_PATH)
        return nothing
    end

    @assert isdir(OPENNEURO_REPO_DIR) "Expected cloned OpenNeuro dataset not found: $OPENNEURO_REPO_DIR"
    @assert isfile(OPENNEURO_PREPARE_SCRIPT) "Preprocessing script not found: $OPENNEURO_PREPARE_SCRIPT"
    cmd = Cmd([
        OPENNEURO_PREPARE_PYTHON,
        OPENNEURO_PREPARE_SCRIPT,
        "--source-root", OPENNEURO_REPO_DIR,
        "--output-dir", OPENNEURO_DERIVED_DIR,
    ])
    run(cmd)

    @assert isfile(OPENNEURO_H5_PATH) "Expected derived HDF5 not found after preprocessing: $OPENNEURO_H5_PATH"
    @assert isfile(OPENNEURO_EVENTS_PATH) "Expected events CSV not found after preprocessing: $OPENNEURO_EVENTS_PATH"
    return nothing
end

function summarize_8bit_events()
    ensure_openneuro_8bit_dataset!()
    events = CSV.read(OPENNEURO_EVENTS_PATH, DataFrame)
    counts = combine(groupby(events, [:run, :trial_type]), nrow => :count)
    sort!(counts, [:run, :trial_type])
    return counts
end

function find_erps_dataset(file)
    candidates = ["epochs", "/epochs", "erps", "/erps", "data", "/data/data_fixations.hdf5", "data/data_fixations.hdf5"]
    for key in candidates
        if haskey(file, key)
            obj = file[key]
            if obj isa HDF5.Dataset
                return obj
            end
        end
    end

    function first_dataset(group)
        for key in keys(group)
            obj = group[key]
            if obj isa HDF5.Dataset
                return obj
            elseif obj isa HDF5.Group
                nested = first_dataset(obj)
                nested === nothing || return nested
            end
        end
        return nothing
    end

    dataset = first_dataset(file)
    dataset === nothing && error("No dataset found in HDF5 file.")
    return dataset
end

function with_erps_dataset(func::Function, path::AbstractString)
    return h5open(path, "r") do file
        dataset = find_erps_dataset(file)
        return func(dataset)
    end
end

with_erps_dataset(path::AbstractString, func::Function) = with_erps_dataset(func, path)

function load_h5_metadata(path::AbstractString)
    return h5open(path, "r") do file
        dataset = find_erps_dataset(file)
        times_s = haskey(file, "times_s") ? read(file["times_s"]) : Float32[]
        channel_names = haskey(file, "channel_names") ? String.(read(file["channel_names"])) : String[]
        attrs = Dict{String, Any}()
        for key in keys(HDF5.attributes(dataset))
            attrs[string(key)] = HDF5.read_attribute(dataset, key)
        end
        for key in keys(HDF5.attributes(file))
            attrs[string(key)] = HDF5.read_attribute(file, key)
        end
        return (times_s = Float32.(times_s), channel_names = channel_names, attrs = attrs)
    end
end

function load_and_merge_label_sources(paths::Vector{String})
    dfs = DataFrame[]
    for path in paths
        df = CSV.read(path, DataFrame)
        df.source_csv = fill(basename(path), nrow(df))
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
    parsed = tryparse(Int, strip(string(v)))
    parsed === nothing ? missing : parsed
end

function has_required_metadata(row)
    cols = propertynames(row)
    if !(:channel in cols && :sort_variable in cols)
        return false
    end
    return !ismissing(row.channel) && !ismissing(row.sort_variable)
end

function sortvalues_from(df::DataFrame, col::Symbol)
    values = df[!, col]
    if eltype(values) <: Number
        return Float64.(values)
    end
    return collect(values)
end

function trial_sort_order(df::DataFrame, sort_col::Symbol)
    row_col = :__row_idx__
    sort_cols = Symbol[sort_col]
    for col in SORT_TIEBREAKER_COLUMNS
        col == sort_col && continue
        col in propertynames(df) || continue
        push!(sort_cols, col)
    end

    order_df = DataFrame()
    order_df[!, row_col] = collect(1:nrow(df))
    for col in sort_cols
        order_df[!, col] = df[!, col]
    end

    sort!(order_df, sort_cols)
    return Int.(order_df[!, row_col])
end

function build_base_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    @assert size(data_time_trials, 2) == nrow(events_trials) "Trial count mismatch between matrix and events."
    @assert sort_col in propertynames(events_trials) "Sort column not found: $sort_col"

    order = trial_sort_order(events_trials, sort_col)
    data_sorted = Float32.(data_time_trials[:, order])
    data_z = zscore_timepoints(data_sorted)
    return Float32.(permutedims(data_z, (2, 1)))
end

function process_erp_image(img_trials_time::AbstractMatrix, target_size::Tuple{Int, Int};
        lowpass::Bool,
        low_pass_sigma::Float32 = REAL_LOWPASS_SIGMA,
        kernel_size::Tuple{Int, Int} = REAL_LOWPASS_KERNEL_SIZE)
    filtered = Float32.(img_trials_time)
    if lowpass && low_pass_sigma > 0f0 && min(size(filtered)...) > 1
        kernel = gaussian_kernel(low_pass_sigma, size(filtered), target_size, kernel_size)
        filtered = Float32.(imfilter(filtered, kernel, FILTER_BORDER))
    end
    return size(filtered) == target_size ? filtered : Float32.(imresize(filtered, target_size))
end

function normalize_per_image(img::AbstractMatrix)
    x = Float32.(img)
    μ, σ = mean_and_std(vec(x); corrected = true)
    σ_safe = σ > 1f-6 ? Float32(σ) : 1f0
    return Float32.((x .- Float32(μ)) ./ σ_safe)
end

function images_to_tensor(imgs::Vector{<:AbstractMatrix}; normalize::Bool = true)
    @assert !isempty(imgs) "At least one image is required."
    h, w = size(imgs[1])
    X = Array{Float32}(undef, h, w, 1, length(imgs))
    for (idx, img) in enumerate(imgs)
        x = normalize ? normalize_per_image(img) : Float32.(img)
        X[:, :, 1, idx] .= x
    end
    return X
end

function extract_fixation_channel_trials(erps, events::DataFrame, channel::Int; post_stim_only::Bool = true)
    @assert 1 <= channel <= size(erps, 1) "Channel out of range: $channel"
    start_idx = post_stim_only ? FIXATION_TIME_ZERO_IDX : 1
    data = Float32.(erps[channel, start_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], copy(events[1:n, :])
end

function load_fixation_binary_records(; post_stim_only::Bool = true)
    for path in vcat(FIXATION_RESULTS_CSV_PATHS, [FIXATION_H5_PATH, FIXATION_EVENTS_CSV_PATH])
        @assert isfile(path) "File not found: $path"
    end

    events = CSV.read(FIXATION_EVENTS_CSV_PATH, DataFrame)
    _, labels_merged_df = load_and_merge_label_sources(String.(FIXATION_RESULTS_CSV_PATHS))

    labels_merged_df.erp_class_id = [parse_class_id(v) for v in labels_merged_df.erp_class]
    valid_mask = map(v -> !ismissing(v), labels_merged_df.erp_class_id)
    meta_mask = map(has_required_metadata, eachrow(labels_merged_df))
    labels_df = copy(labels_merged_df[valid_mask .& meta_mask, :])

    labels_df.channel_int = Int.(labels_df.channel)
    labels_df.sort_var_symbol = Symbol.(String.(labels_df.sort_variable))

    keep_ids = Set([SIGMOID_CLASS_ID, NO_CLASS_ID])
    keep_mask = Int.(labels_df.erp_class_id) .∈ Ref(keep_ids)
    labels_df = copy(labels_df[keep_mask, :])

    rows = NamedTuple[]
    base_imgs = Matrix{Float32}[]

    with_erps_dataset(FIXATION_H5_PATH) do erps
        for channel_df in groupby(labels_df, :channel_int)
            channel = Int(channel_df.channel_int[1])
            data_full, events_full = extract_fixation_channel_trials(erps, events, channel; post_stim_only = post_stim_only)

            for row in eachrow(channel_df)
                base_img = build_base_image(data_full, events_full, row.sort_var_symbol)
                push!(rows, (
                    channel = channel,
                    sort_var = String(row.sort_var_symbol),
                    class_id = Int(row.erp_class_id),
                    binary_label = Int(Int(row.erp_class_id) == SIGMOID_CLASS_ID),
                    image_file = hasproperty(row, :image_file) && !ismissing(row.image_file) ? String(row.image_file) : "unknown_image",
                    base_shape = size(base_img),
                ))
                push!(base_imgs, base_img)
            end
        end
    end

    out_df = DataFrame(rows)
    out_df.base_img = base_imgs
    return out_df
end

function load_fixation_binary_cache(; target_size::Tuple{Int, Int} = (64, 64), lowpass::Bool = true, post_stim_only::Bool = true)
    records = load_fixation_binary_records(; post_stim_only = post_stim_only)
    imgs = [process_erp_image(img, target_size; lowpass = lowpass) for img in records.base_img]
    return (
        images = imgs,
        labels = Int.(records.binary_label),
        tensor = images_to_tensor(imgs),
        meta = select(records, Not(:base_img)),
    )
end

function eightbit_time_zero_index(times_s::AbstractVector{<:Real})
    idx = findfirst(t -> t >= 0, times_s)
    idx === nothing && error("No non-negative timepoint found in 8bit dataset.")
    return Int(idx)
end

function extract_8bit_channel_trials(erps, events::DataFrame, channel::Int, time_zero_idx::Int; post_stim_only::Bool = true)
    @assert 1 <= channel <= size(erps, 1) "Channel out of range: $channel"
    start_idx = post_stim_only ? time_zero_idx : 1
    data = Float32.(erps[channel, start_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], copy(events[1:n, :])
end

function split_indices_sorted_modulo(events_trials::DataFrame, sort_col::Symbol, k::Int)
    @assert k >= 1 "k must be at least 1."
    order = trial_sort_order(events_trials, sort_col)
    groups = [Int[] for _ in 1:k]
    for (rank, idx) in enumerate(order)
        push!(groups[mod1(rank, k)], idx)
    end
    return groups
end

parts_for_trial_count(n_trials::Int; min_trials_per_part::Int = 8, max_parts::Int = 4) =
    clamp(fld(n_trials, min_trials_per_part), 1, max_parts)

function load_8bit_unlabelled_cache(;
        target_size::Tuple{Int, Int} = (64, 64),
        lowpass::Bool = true,
        post_stim_only::Bool = true,
        sort_col::Symbol = :onset_s,
        min_trials_per_part::Int = 8,
        max_parts::Int = 4,
        selected_trial_types = nothing)
    ensure_openneuro_8bit_dataset!()

    events = CSV.read(OPENNEURO_EVENTS_PATH, DataFrame)
    meta = load_h5_metadata(OPENNEURO_H5_PATH)
    times_s = meta.times_s
    channel_names = meta.channel_names
    time_zero_idx = eightbit_time_zero_index(times_s)

    trial_types = isnothing(selected_trial_types) ? unique(String.(events.trial_type)) : collect(String.(selected_trial_types))

    rows = NamedTuple[]
    imgs = Matrix{Float32}[]

    with_erps_dataset(OPENNEURO_H5_PATH) do erps
        for channel in 1:length(channel_names)
            data_full, events_full = extract_8bit_channel_trials(erps, events, channel, time_zero_idx; post_stim_only = post_stim_only)

            for trial_type in trial_types
                mask = String.(events_full.trial_type) .== trial_type
                any(mask) || continue
                events_subset = copy(events_full[mask, :])
                data_subset = data_full[:, mask]
                n_trials = nrow(events_subset)
                n_trials == 0 && continue

                n_parts = parts_for_trial_count(n_trials; min_trials_per_part = min_trials_per_part, max_parts = max_parts)
                groups = split_indices_sorted_modulo(events_subset, sort_col, n_parts)

                for (part_idx, idxs) in enumerate(groups)
                    isempty(idxs) && continue
                    img = build_base_image(data_subset[:, idxs], events_subset[idxs, :], sort_col)
                    proc = process_erp_image(img, target_size; lowpass = lowpass)
                    push!(rows, (
                        channel = channel,
                        channel_name = channel_names[channel],
                        trial_type = trial_type,
                        part = part_idx,
                        n_parts = n_parts,
                        n_trials = length(idxs),
                        target_height = target_size[1],
                        target_width = target_size[2],
                        lowpass = lowpass,
                    ))
                    push!(imgs, proc)
                end
            end
        end
    end

    meta_df = DataFrame(rows)
    return (
        images = imgs,
        tensor = images_to_tensor(imgs),
        meta = meta_df,
        event_counts = combine(groupby(meta_df, :trial_type), nrow => :n_images),
    )
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

function device_array(ref, x)
    return ref isa CuArray ? cu(x) : x
end

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
    backbone = base.layers.layers[1]
    pool = Chain(Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    encoder = Chain(backbone, pool)
    projector = Chain(
        Dense(512 => hidden_dim),
        BatchNorm(hidden_dim, relu),
        Dense(hidden_dim => projection_dim),
    )
    return (encoder = encoder, projector = projector, model = Chain(encoder, projector))
end

function count_params(m)
    return sum(length, Flux.trainables(m))
end

function train_simclr!(simclr, imgs::Vector{<:AbstractMatrix};
        target_size::Tuple{Int, Int},
        batchsize::Int = 64,
        epochs::Int = 20,
        lr::Float32 = 1f-3,
        temperature::Float32 = 0.10f0,
        seed::Int = 20260317)
    rng = Random.Xoshiro(seed)
    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu

    model = simclr.model |> device
    opt_state = Flux.setup(Flux.Adam(lr), model)
    loss_history = Float32[]

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

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float32(loss_val)
            n_batches += 1
        end

        avg_loss = epoch_loss / max(1, n_batches)
        push!(loss_history, avg_loss)
        println(@sprintf("simclr epoch %d/%d | loss=%.5f", epoch, epochs, avg_loss))
    end

    model_cpu = cpu(model)
    encoder_cpu = cpu(model_cpu.layers[1])
    projector_cpu = cpu(model_cpu.layers[2])

    GC.gc()
    if use_gpu
        CUDA.reclaim()
    end

    return (
        encoder = encoder_cpu,
        projector = projector_cpu,
        model = model_cpu,
        used_gpu = use_gpu,
        loss_history = loss_history,
        params_n = count_params(model_cpu),
    )
end

function build_binary_resnet18_classifier(; in_channels::Int = 1, n_classes::Int = 2)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 128)
    backbone = base.layers.layers[1]
    encoder = Chain(backbone, Flux.AdaptiveMeanPool((1, 1)), Flux.flatten)
    head = Dense(512 => n_classes)
    return (encoder = encoder, head = head, model = Chain(encoder, head))
end

function classifier_loss(model, x, y)
    return Flux.Losses.logitcrossentropy(model(x), y)
end

function evaluate_binary_metrics(y_true::AbstractVector{<:Integer}, y_pred::AbstractVector{<:Integer})
    classes = (0, 1)
    recalls = Float64[]
    precisions = Float64[]
    f1s = Float64[]

    for cls in classes
        tp = count(i -> y_true[i] == cls && y_pred[i] == cls, eachindex(y_true))
        fp = count(i -> y_true[i] != cls && y_pred[i] == cls, eachindex(y_true))
        fn = count(i -> y_true[i] == cls && y_pred[i] != cls, eachindex(y_true))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
        push!(precisions, precision)
        push!(recalls, recall)
        push!(f1s, f1)
    end

    return (
        accuracy = mean(y_true .== y_pred),
        balanced_accuracy = mean(recalls),
        macro_f1 = mean(f1s),
        precision = mean(precisions),
        recall = mean(recalls),
        support_no_class = count(==(0), y_true),
        support_sigmoid = count(==(1), y_true),
    )
end

function evaluate_classifier(model, x, y)
    Flux.testmode!(model, true)
    logits = cpu(model(x))
    y_pred = Int.(onecold(Array(logits), 0:1))
    y_true = Int.(onecold(cpu(y), 0:1))
    return evaluate_binary_metrics(y_true, y_pred), y_true, y_pred
end

function train_classifier!(model_parts;
        x_train::Array{Float32, 4},
        y_train::Vector{Int},
        x_val::Array{Float32, 4},
        y_val::Vector{Int},
        batchsize::Int = 32,
        epochs::Int = 16,
        lr::Float32 = 1f-3,
        freeze_encoder::Bool = false,
        seed::Int = 20260317,
        model_name::String = "classifier")
    Random.seed!(seed)
    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu

    encoder = model_parts.encoder |> device
    head = model_parts.head |> device
    model = Chain(encoder, head)

    y_train_oh = Float32.(Array(onehotbatch(y_train, 0:1)))
    y_val_oh = Float32.(Array(onehotbatch(y_val, 0:1)))

    train_loader = DataLoader((x_train, y_train_oh); batchsize = batchsize, shuffle = true)
    val_x = device(x_val)
    val_y = device(y_val_oh)

    Flux.testmode!(encoder, freeze_encoder)
    Flux.testmode!(head, false)

    target_model = freeze_encoder ? head : model
    opt_state = Flux.setup(Flux.Adam(lr), target_model)
    loss_history = Float32[]

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
                Flux.update!(opt_state, head, grads[1])
                epoch_loss += Float32(loss_val)
            else
                loss_val, grads = Flux.withgradient(model) do m
                    classifier_loss(m, xb, yb)
                end
                Flux.update!(opt_state, model, grads[1])
                epoch_loss += Float32(loss_val)
            end

            n_batches += 1
        end

        avg_loss = epoch_loss / max(1, n_batches)
        push!(loss_history, avg_loss)
        println(@sprintf("%s epoch %d/%d | loss=%.5f", model_name, epoch, epochs, avg_loss))
    end

    metrics, y_true, y_pred = evaluate_classifier(cpu(model), x_val, val_y)
    trained_model = cpu(model)

    GC.gc()
    if use_gpu
        CUDA.reclaim()
    end

    return (
        model = trained_model,
        encoder = cpu(trained_model.layers[1]),
        head = cpu(trained_model.layers[2]),
        metrics = metrics,
        y_true = y_true,
        y_pred = y_pred,
        loss_history = loss_history,
        used_gpu = use_gpu,
        params_n = count_params(trained_model),
    )
end

function stratified_train_val_split(labels::AbstractVector{<:Integer}; train_frac::Float64 = 0.8, seed::Int = 20260317)
    rng = Random.Xoshiro(seed)
    train_idx = Int[]
    val_idx = Int[]

    for cls in sort(unique(labels))
        cls_idx = findall(==(cls), labels)
        shuffle!(rng, cls_idx)
        n_train = clamp(round(Int, train_frac * length(cls_idx)), 1, length(cls_idx) - 1)
        append!(train_idx, cls_idx[1:n_train])
        append!(val_idx, cls_idx[(n_train + 1):end])
    end

    sort!(train_idx)
    sort!(val_idx)
    return train_idx, val_idx
end

function run_transfer_experiment(;
        image_size::Tuple{Int, Int} = (64, 64),
        lowpass::Bool = true,
        ssl_batchsize::Int = 64,
        ssl_epochs::Int = 20,
        ssl_lr::Float32 = 1f-3,
        ssl_temperature::Float32 = 0.10f0,
        baseline_epochs::Int = 16,
        baseline_lr::Float32 = 1f-3,
        probe_epochs::Int = 20,
        probe_lr::Float32 = 2f-3,
        finetune_epochs::Int = 12,
        finetune_lr::Float32 = 3f-4,
        classifier_batchsize::Int = 32,
        seed::Int = 20260317)
    unlabeled = load_8bit_unlabelled_cache(target_size = image_size, lowpass = lowpass)
    labeled = load_fixation_binary_cache(target_size = image_size, lowpass = lowpass)

    train_idx, val_idx = stratified_train_val_split(labeled.labels; seed = seed)
    x_train = labeled.tensor[:, :, :, train_idx]
    y_train = labeled.labels[train_idx]
    x_val = labeled.tensor[:, :, :, val_idx]
    y_val = labeled.labels[val_idx]

    simclr = build_simclr_resnet18(in_channels = 1)
    ssl = train_simclr!(simclr, unlabeled.images;
        target_size = image_size,
        batchsize = ssl_batchsize,
        epochs = ssl_epochs,
        lr = ssl_lr,
        temperature = ssl_temperature,
        seed = seed,
    )

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
        seed = seed + 1,
        model_name = "baseline_resnet18",
    )

    probe_init = (
        encoder = deepcopy(ssl.encoder),
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
        seed = seed + 2,
        model_name = "ssl_linear_probe",
    )

    finetune_init = (
        encoder = deepcopy(ssl.encoder),
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
        seed = seed + 3,
        model_name = "ssl_finetune",
    )

    results_rows = [
        (
            experiment = "baseline_resnet18",
            evaluation_dataset = "fixation_holdout",
            balanced_accuracy = baseline.metrics.balanced_accuracy,
            macro_f1 = baseline.metrics.macro_f1,
            accuracy = baseline.metrics.accuracy,
            precision = baseline.metrics.precision,
            recall = baseline.metrics.recall,
            params_n = baseline.params_n,
        ),
        (
            experiment = "ssl_linear_probe",
            evaluation_dataset = "fixation_holdout",
            balanced_accuracy = probe.metrics.balanced_accuracy,
            macro_f1 = probe.metrics.macro_f1,
            accuracy = probe.metrics.accuracy,
            precision = probe.metrics.precision,
            recall = probe.metrics.recall,
            params_n = probe.params_n,
        ),
        (
            experiment = "ssl_finetune",
            evaluation_dataset = "fixation_holdout",
            balanced_accuracy = finetune.metrics.balanced_accuracy,
            macro_f1 = finetune.metrics.macro_f1,
            accuracy = finetune.metrics.accuracy,
            precision = finetune.metrics.precision,
            recall = finetune.metrics.recall,
            params_n = finetune.params_n,
        ),
    ]

    return (
        results_df = DataFrame(results_rows),
        unlabeled = unlabeled,
        labeled = labeled,
        split = (train_idx = train_idx, test_idx = val_idx, val_idx = val_idx),
        ssl = ssl,
        baseline = baseline,
        probe = probe,
        finetune = finetune,
    )
end

function plot_image_grid(imgs::Vector{<:AbstractMatrix}, meta::DataFrame; n::Int = 9, seed::Int = 20260317, title::AbstractString = "ERP images")
    @assert !isempty(imgs) "No images available for plotting."
    rng = Random.Xoshiro(seed)
    n = min(n, length(imgs))
    chosen = randperm(rng, length(imgs))[1:n]
    ncols = min(3, n)
    nrows = cld(n, ncols)

    fig = Figure(size = (420 * ncols, 320 * nrows))
    for (plot_idx, img_idx) in enumerate(chosen)
        row = cld(plot_idx, ncols)
        col = mod1(plot_idx, ncols)
        ax = Axis(fig[row, col], title = begin
            if :trial_type in propertynames(meta)
                if :n_parts in propertynames(meta) && meta.n_parts[img_idx] > 1
                    "$(meta.trial_type[img_idx]) | ch$(meta.channel[img_idx]) | part $(meta.part[img_idx])"
                else
                    "$(meta.trial_type[img_idx]) | ch$(meta.channel[img_idx])"
                end
            else
                "$(meta.sort_var[img_idx]) | ch$(meta.channel[img_idx]) | y=$(meta.binary_label[img_idx])"
            end
        end)
        clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(Float32.(imgs[img_idx]))
        hm = heatmap!(ax, permutedims(clipped, (2, 1)); colormap = cmap, colorrange = colorrange)
        Colorbar(fig[row, col + ncols], hm; width = 12, ticks = (tick_vals, tick_labels))
    end
    Label(fig[0, :], title, fontsize = 24)
    return fig
end

function plot_loss_history(loss_dict::AbstractDict{<:AbstractString, <:AbstractVector})
    fig = Figure(size = (900, 480))
    ax = Axis(fig[1, 1], xlabel = "epoch", ylabel = "loss", title = "Training loss histories")
    colors = [:steelblue, :darkorange, :forestgreen, :firebrick, :purple]
    for (idx, (name, values)) in enumerate(pairs(loss_dict))
        lines!(ax, 1:length(values), Float32.(values); label = name, color = colors[mod1(idx, length(colors))], linewidth = 3)
    end
    axislegend(ax; position = :rt)
    return fig
end

end
