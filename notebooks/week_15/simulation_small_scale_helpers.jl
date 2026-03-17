module SmallScaleERPClassification

using CSV
using CairoMakie
using CUDA
using DataFrames
using DecisionTree
using Distributions
using Flux
using HDF5
using ImageFiltering: KernelFactors, imfilter
using Images: imresize
using LIBSVM
using MLUtils: DataLoader
using Random
using Statistics
using StatsBase
using cuDNN

using Flux: onecold

import Distributions: Distribution, mean, std

include(joinpath(@__DIR__, "..", "data_generation", "erpgen.jl"))
using .ERPGen

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: clipped_color_stats_quantile_zero_ticks, gaussian_kernel, zscore_timepoints

export TARGET_SIZES
export LOWPASS_CHOICES
export CALIBRATION_TARGET_SIZE
export CALIBRATION_LOWPASS
export RESULTS_CSV_PATHS
export H5_PATH
export EVENTS_CSV_PATH
export make_base_config
export load_real_condition_cache
export make_strategy_bundle
export run_small_scale_experiment
export summarize_results
export select_best_strategy
export plot_metric_heatmap
export plot_lowpass_effect
export plot_example_images
export build_ranked_settings_table
export run_dense_nn_lowpass_search
export build_confusion_example_bundle
export plot_confusion_examples

const TARGET_SIZES = [(2, 2), (4, 4), (8, 8), (16, 16)]
const LOWPASS_CHOICES = (true, false)
const CALIBRATION_TARGET_SIZE = (16, 16)
const CALIBRATION_LOWPASS = true
const LOWPASS_FACTOR = 0.75f0
const FILTER_BORDER = "reflect"

const FIXATIONS_DATASET_DIR = joinpath(@__DIR__, "..", "model_test", "real_data_sets", "fixations_dataset")
const RESULTS_CSV_PATHS = [
    joinpath(@__DIR__, "..", "model_test", "results", "project-14-at-2026-02-15-23-09-f5225e5c.csv"),
    joinpath(@__DIR__, "..", "model_test", "results", "project-15-at-2026-02-18-19-35-828515fe.csv"),
]
const H5_PATH = joinpath(FIXATIONS_DATASET_DIR, "data_fixations.hdf5")
const EVENTS_CSV_PATH = joinpath(FIXATIONS_DATASET_DIR, "events.csv")

const PRE_STIM_S = 0.5
const SAMPLING_RATE = 512
const TIME_ZERO_IDX = Int(round(PRE_STIM_S * SAMPLING_RATE)) + 1
const SIGMOID_CLASS_ID = 1
const NO_CLASS_ID = 0

Base.@kwdef struct SimulationTask
    cfg::ERPGen.GenerationConfig
    seed::Int
    tag::String
end

struct FixedDist <: Distribution{Distributions.Univariate, Distributions.Continuous}
    value::Float64
end

Random.rand(rng::Random.AbstractRNG, d::FixedDist) = d.value
Random.rand(d::FixedDist) = d.value
mean(d::FixedDist) = d.value
std(::FixedDist) = 0.0

fixed_dist(x::Real) = FixedDist(Float64(x))

condition_key(target_size::Tuple{Int, Int}, lowpass::Bool) = (target_size = target_size, lowpass = lowpass)
resolution_label(target_size::Tuple{Int, Int}) = "$(target_size[1])x$(target_size[2])"
lowpass_label(lowpass::Bool) = lowpass ? "with low-pass" : "without low-pass"

function parse_resolution_label(resolution::AbstractString)
    parts = split(strip(String(resolution)), 'x')
    length(parts) == 2 || error("Resolution must look like HxW, got: $(resolution)")
    return (parse(Int, parts[1]), parse(Int, parts[2]))
end

safe_div(num::Real, den::Real) = den == 0 ? 0.0 : Float64(num) / Float64(den)

function ensure_paths!()
    for path in vcat(RESULTS_CSV_PATHS, [H5_PATH, EVENTS_CSV_PATH])
        @assert isfile(path) "File not found: $path"
    end
    return nothing
end

function make_base_config(; target_size::Tuple{Int, Int} = CALIBRATION_TARGET_SIZE, apply_lowpass::Bool = true)
    base = ERPGen.GenerationConfig()

    sim = ERPGen.SimulationConfig(
        mu_dist = Normal(4.5, 0.3),
        sigma_dist = base.sim.sigma_dist,
        epoch_duration_dist = fixed_dist(0.998046875),
        sampling_rate_dist = fixed_dist(512.0),
        n_trials_dist = fixed_dist(2508.0),
    )

    patterns = ERPGen.PatternConfig(
        patterns = [:sigmoid],
        loaded_patterns = ERPGen.DEFAULT_PATTERN_LIST,
        covariate_dists = base.patterns.covariate_dists,
        diverging_bar_levels = base.patterns.diverging_bar_levels,
    )

    processing = ERPGen.ProcessingConfig(
        dropout_trials_rate_dist = fixed_dist(0.0),
        crop_start_dist = fixed_dist(0.0),
        crop_end_dist = fixed_dist(0.0),
        zscore_timepoints = true,
        resize_antialias = apply_lowpass,
        low_pass_factor = LOWPASS_FACTOR,
        resize_method = base.processing.resize_method,
        target_height = target_size[1],
        target_width = target_size[2],
    )

    runtime = ERPGen.RuntimeConfig(
        threaded = false,
        show_progress = true,
        blas_threads = 1,
        progress_every = 50,
    )

    return ERPGen.GenerationConfig(
        sim = sim,
        components = base.components,
        patterns = patterns,
        noise = base.noise,
        processing = processing,
        runtime = runtime,
    )
end

function load_erps_from_h5(path::AbstractString)
    return h5open(path, "r") do file
        dataset = find_erps_dataset(file)
        return read(dataset)
    end
end

function find_erps_dataset(file)
    candidates = ["erps", "/erps", "data", "/data/data_fixations.hdf5", "data/data_fixations.hdf5"]
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

function extract_channel_trials(erps, events::DataFrame, channel::Int; post_stim_only::Bool = true)
    @assert 1 <= channel <= size(erps, 1) "Channel out of range: $channel"
    start_idx = post_stim_only ? TIME_ZERO_IDX : 1
    data = Float32.(erps[channel, start_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], copy(events[1:n, :])
end

function build_base_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    @assert size(data_time_trials, 2) == nrow(events_trials) "Trial count mismatch between matrix and events"
    @assert sort_col in propertynames(events_trials) "Sort column not found: $(sort_col)"

    data_z = zscore_timepoints(data_time_trials)
    sortvals = sortvalues_from(events_trials, sort_col)
    order = sortperm(sortvals)
    return Float32.(permutedims(data_z[:, order], (2, 1)))
end

function process_small_scale_image(img_trials_time::AbstractMatrix, target_size::Tuple{Int, Int}; lowpass::Bool, low_pass_factor::Float32 = LOWPASS_FACTOR)
    filtered = Float32.(img_trials_time)

    if lowpass && low_pass_factor > 0f0 && size(filtered) != target_size && min(size(filtered)...) > 1
        kernel = gaussian_kernel(low_pass_factor, size(filtered), target_size)
        filtered = Float32.(imfilter(filtered, kernel, FILTER_BORDER))
    end

    return size(filtered) == target_size ? filtered : Float32.(imresize(filtered, target_size))
end

function flatten_images(imgs::Vector{<:AbstractMatrix})
    n = length(imgs)
    @assert n > 0 "At least one image is required."
    n_features = length(vec(imgs[1]))
    X = Matrix{Float32}(undef, n, n_features)
    for (idx, img) in enumerate(imgs)
        X[idx, :] .= vec(Float32.(img))
    end
    return X
end

function load_real_labelled_records(; post_stim_only::Bool = true)
    ensure_paths!()

    events = CSV.read(EVENTS_CSV_PATH, DataFrame)
    labels_all_df, labels_merged_df = load_and_merge_label_sources(String.(RESULTS_CSV_PATHS))

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

    with_erps_dataset(H5_PATH) do erps
        row_idx = 0
        for channel_df in groupby(labels_df, :channel_int)
            channel = Int(channel_df.channel_int[1])
            data_full, events_full = extract_channel_trials(erps, events, channel; post_stim_only = post_stim_only)

            for row in eachrow(channel_df)
                row_idx += 1
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

                if row_idx % 50 == 0 || row_idx == nrow(labels_df)
                    println("  real images processed: $(row_idx)/$(nrow(labels_df))")
                end
            end
        end
    end

    out_df = DataFrame(rows)
    out_df.base_img = base_imgs
    return out_df
end

function load_real_condition_cache(; target_sizes = TARGET_SIZES, lowpass_choices = LOWPASS_CHOICES, post_stim_only::Bool = true)
    records = load_real_labelled_records(; post_stim_only = post_stim_only)
    cache = Dict{NamedTuple, NamedTuple}()

    for target_size in target_sizes, lowpass in lowpass_choices
        key = condition_key(target_size, lowpass)
        imgs = [process_small_scale_image(img, target_size; lowpass = lowpass) for img in records.base_img]
        cache[key] = (
            images = imgs,
            labels = Int.(records.binary_label),
            features = flatten_images(imgs),
            meta = select(records, Not(:base_img)),
        )
    end

    return (records = records, cache = cache)
end

function simulate_base_pair(cfg::ERPGen.GenerationConfig, rng::Random.AbstractRNG)
    raw = ERPGen.simulate_raw_erp(cfg, rng)
    dropped = ERPGen.apply_trial_dropout(raw.data, raw.events, cfg.processing, rng)
    cropped = ERPGen.apply_cropping(dropped.data, cfg.processing, rng, raw.params.sampling_rate)

    data_z = cfg.processing.zscore_timepoints ? zscore_timepoints(cropped.data) : Float32.(cropped.data)
    out = Dict{Symbol, Matrix{Float32}}()
    for pattern_name in (:sigmoid, :no_class)
        sortvals = ERPGen.pattern_sort_values(pattern_name, dropped.events, rng)
        order = sortperm(sortvals)
        out[pattern_name] = Float32.(permutedims(data_z[:, order], (2, 1)))
    end
    return out
end

function generate_single_condition_images(cfg::ERPGen.GenerationConfig, n_per_pattern::Int;
        target_size::Tuple{Int, Int},
        lowpass::Bool,
        seed::Int)
    rng = Random.Xoshiro(seed)
    total = 2 * n_per_pattern
    imgs = Vector{Matrix{Float32}}(undef, total)
    labels = Vector{Int}(undef, total)

    out_idx = 1
    for _ in 1:n_per_pattern
        base_pair = simulate_base_pair(cfg, rng)
        for (pattern_name, label) in ((:sigmoid, 1), (:no_class, 0))
            imgs[out_idx] = process_small_scale_image(base_pair[pattern_name], target_size; lowpass = lowpass)
            labels[out_idx] = label
            out_idx += 1
        end
    end

    return imgs, labels
end

function sigmoid_shape_features(img::AbstractMatrix{<:Real})
    x = Float32.(img)
    h, w = size(x)

    gx = diff(x; dims = 2)
    gy = diff(x; dims = 1)

    feat_global_std = std(x)
    feat_mean_gx = mean(abs.(gx))
    feat_mean_gy = mean(abs.(gy))

    patch_size = min(8, min(h, w))
    patch_stds = Float32[]
    if patch_size >= 2
        for i in 1:patch_size:(h - patch_size + 1)
            for j in 1:patch_size:(w - patch_size + 1)
                push!(patch_stds, std(@view x[i:(i + patch_size - 1), j:(j + patch_size - 1)]))
            end
        end
    end
    isempty(patch_stds) && push!(patch_stds, std(x))

    feat_patch_std_mean = mean(patch_stds)
    feat_patch_std_std = std(patch_stds)
    feat_patch_std_skew = if length(patch_stds) > 2
        z = (patch_stds .- feat_patch_std_mean) ./ max(feat_patch_std_std, 1f-6)
        mean(z .^ 3)
    else
        0f0
    end

    autocorrs = Float32[]
    for i in 1:h
        row = @view x[i, :]
        if length(row) > 1
            c = cor(row[1:(end - 1)], row[2:end])
            isfinite(c) && push!(autocorrs, Float32(c))
        end
    end
    feat_autocorr_mean = isempty(autocorrs) ? 0f0 : mean(autocorrs)

    trial_grad_profile = vec(mean(abs.(gy); dims = 1))
    if length(trial_grad_profile) > 1
        feat_grad_concentration = maximum(trial_grad_profile) / (mean(trial_grad_profile) + 1f-6)
        feat_grad_peak_pos = Float32(argmax(trial_grad_profile)) / Float32(length(trial_grad_profile))
    else
        feat_grad_concentration = 1f0
        feat_grad_peak_pos = 0.5f0
    end

    q25_cols = Float32[quantile(vec(@view x[:, j]), 0.25f0) for j in 1:w]
    q75_cols = Float32[quantile(vec(@view x[:, j]), 0.75f0) for j in 1:w]
    iqr_profile = q75_cols .- q25_cols
    feat_iqr_std = std(iqr_profile)
    feat_iqr_trend = if length(iqr_profile) > 1
        t = collect(range(1f0, Float32(length(iqr_profile)); length = length(iqr_profile)))
        c = cor(t, iqr_profile)
        isfinite(c) ? Float32(c) : 0f0
    else
        0f0
    end

    row_energies = vec(sum(x .^ 2; dims = 2))
    col_energies = vec(sum(x .^ 2; dims = 1))
    feat_row_energy_cv = std(row_energies) / (mean(row_energies) + 1f-6)
    feat_col_energy_cv = std(col_energies) / (mean(col_energies) + 1f-6)

    return Float32[
        feat_global_std,
        feat_mean_gx,
        feat_mean_gy,
        feat_patch_std_mean,
        feat_patch_std_std,
        feat_patch_std_skew,
        feat_autocorr_mean,
        feat_grad_concentration,
        feat_grad_peak_pos,
        feat_iqr_std,
        feat_iqr_trend,
        feat_row_energy_cv,
        feat_col_energy_cv,
    ]
end

function feature_matrix(imgs::Vector{<:AbstractMatrix})
    @assert !isempty(imgs) "At least one image is required."
    first_feat = sigmoid_shape_features(imgs[1])
    F = Matrix{Float32}(undef, length(first_feat), length(imgs))
    F[:, 1] = first_feat
    for i in 2:length(imgs)
        F[:, i] = sigmoid_shape_features(imgs[i])
    end
    return F
end

function feature_summary(F::AbstractMatrix)
    mu = vec(mean(F; dims = 2))
    sigma = max.(vec(std(F; dims = 2)), 1f-5)
    return (mu = mu, sigma = sigma)
end

function mmd2_rbf(X::AbstractMatrix, Y::AbstractMatrix; gamma::Union{Nothing, Float64} = nothing)
    nx = size(X, 2)
    ny = size(Y, 2)
    @assert nx > 1 && ny > 1 "At least two samples per set are required."

    if isnothing(gamma)
        Z = hcat(Float64.(X), Float64.(Y))
        d2 = Float64[]
        n = size(Z, 2)
        for i in 1:(n - 1), j in (i + 1):n
            push!(d2, sum((Z[:, i] .- Z[:, j]) .^ 2))
        end
        med = isempty(d2) ? 1.0 : median(d2)
        gamma = 1.0 / max(med, 1e-6)
    end

    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0

    for i in 1:nx, j in 1:nx
        i == j && continue
        Kxx += exp(-gamma * sum((Float64.(X[:, i]) .- Float64.(X[:, j])) .^ 2))
    end
    for i in 1:ny, j in 1:ny
        i == j && continue
        Kyy += exp(-gamma * sum((Float64.(Y[:, i]) .- Float64.(Y[:, j])) .^ 2))
    end
    for i in 1:nx, j in 1:ny
        Kxy += exp(-gamma * sum((Float64.(X[:, i]) .- Float64.(Y[:, j])) .^ 2))
    end

    return Kxx / (nx * (nx - 1)) + Kyy / (ny * (ny - 1)) - 2 * Kxy / (nx * ny)
end

function latin_hypercube(n::Int, d::Int, rng::Random.AbstractRNG)
    X = Matrix{Float64}(undef, n, d)
    for j in 1:d
        p = randperm(rng, n)
        u = rand(rng, n)
        X[:, j] = (p .- u) ./ n
    end
    return X
end

function parameterized_config(base_cfg::ERPGen.GenerationConfig;
        mu_dist::Distribution = base_cfg.sim.mu_dist,
        sigma_dist::Distribution = base_cfg.sim.sigma_dist,
        p100_width_dist::Distribution = base_cfg.components.p100_width_dist,
        p100_n170_gap_dist::Distribution = base_cfg.components.p100_n170_gap_dist,
        n170_p300_gap_dist::Distribution = base_cfg.components.n170_p300_gap_dist,
        n170_width_dist::Distribution = base_cfg.components.n170_width_dist,
        p300_width_dist::Distribution = base_cfg.components.p300_width_dist,
        p1_beta_dist::Distribution = base_cfg.components.p1_beta_dist,
        p3_beta_dist::Distribution = base_cfg.components.p3_beta_dist,
        n1_beta1_dist::Distribution = base_cfg.components.n1_beta1_dist,
        n1_beta2_dist::Distribution = base_cfg.components.n1_beta2_dist,
        n1_beta3_dist::Distribution = base_cfg.components.n1_beta3_dist,
        componentA_amp_dist::Distribution = base_cfg.components.componentA_amp_dist,
        componentB_amp_dist::Distribution = base_cfg.components.componentB_amp_dist,
        componentC_amp_dist::Distribution = base_cfg.components.componentC_amp_dist,
        pink_noise_dist::Distribution = base_cfg.noise.noiselevel_dists[ERPGen.PinkNoise],
        white_noise_dist::Distribution = base_cfg.noise.noiselevel_dists[ERPGen.WhiteNoise],
        red_noise_dist::Distribution = base_cfg.noise.noiselevel_dists[ERPGen.RedNoise],
        exponential_noise_dist::Distribution = base_cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise])
    sim = ERPGen.SimulationConfig(
        mu_dist = mu_dist,
        sigma_dist = sigma_dist,
        epoch_duration_dist = base_cfg.sim.epoch_duration_dist,
        sampling_rate_dist = base_cfg.sim.sampling_rate_dist,
        n_trials_dist = base_cfg.sim.n_trials_dist,
    )

    components = ERPGen.ComponentConfig(
        p100_width_dist = p100_width_dist,
        p100_window_offset_dist = base_cfg.components.p100_window_offset_dist,
        p100_n170_gap_dist = p100_n170_gap_dist,
        n170_p300_gap_dist = n170_p300_gap_dist,
        n170_width_dist = n170_width_dist,
        p300_width_dist = p300_width_dist,
        p1_beta_dist = p1_beta_dist,
        p3_beta_dist = p3_beta_dist,
        n1_beta1_dist = n1_beta1_dist,
        n1_beta2_dist = n1_beta2_dist,
        n1_beta3_dist = n1_beta3_dist,
        componentA_amp_dist = componentA_amp_dist,
        componentB_amp_dist = componentB_amp_dist,
        componentC_amp_dist = componentC_amp_dist,
    )

    noiselevel_dists = Dict{DataType, Distribution}(
        ERPGen.PinkNoise => pink_noise_dist,
        ERPGen.WhiteNoise => white_noise_dist,
        ERPGen.RedNoise => red_noise_dist,
        ERPGen.ExponentialNoise => exponential_noise_dist,
    )
    noise = ERPGen.NoiseConfig(noise_pool = base_cfg.noise.noise_pool, noiselevel_dists = noiselevel_dists)

    return ERPGen.GenerationConfig(
        sim = sim,
        components = components,
        patterns = base_cfg.patterns,
        noise = noise,
        processing = base_cfg.processing,
        runtime = base_cfg.runtime,
    )
end

function range_from_scale(base_value::Real, lo_scale::Real, hi_scale::Real; min_value::Float64 = -Inf)
    a, b = minmax(Float64(base_value) * Float64(lo_scale), Float64(base_value) * Float64(hi_scale))
    return (max(a, min_value), max(b, min_value))
end

function parameter_spec(key::Symbol, label::AbstractString, base_dist::Normal, lo_scale::Real, hi_scale::Real;
        min_mean::Float64 = -Inf,
        min_std::Float64 = 1e-5)
    return (
        key = key,
        label = String(label),
        mean_symbol = Symbol("$(key)_mean"),
        std_symbol = Symbol("$(key)_std"),
        base_dist = base_dist,
        mean_range = range_from_scale(mean(base_dist), lo_scale, hi_scale; min_value = min_mean),
        std_range = range_from_scale(std(base_dist), abs(lo_scale), abs(hi_scale); min_value = min_std),
    )
end

function parameter_specs(base_cfg::ERPGen.GenerationConfig)
    return [
        parameter_spec(:sim_mu, "sim.mu_dist", base_cfg.sim.mu_dist, 0.70, 1.30; min_mean = 0.1, min_std = 1e-4),
        parameter_spec(:sim_sigma, "sim.sigma_dist", base_cfg.sim.sigma_dist, 0.40, 2.00; min_mean = 0.01),
        parameter_spec(:p100_width, "components.p100_width_dist", base_cfg.components.p100_width_dist, 0.40, 2.20; min_mean = 0.002),
        parameter_spec(:n170_width, "components.n170_width_dist", base_cfg.components.n170_width_dist, 0.40, 2.20; min_mean = 0.002),
        parameter_spec(:p300_width, "components.p300_width_dist", base_cfg.components.p300_width_dist, 0.40, 2.20; min_mean = 0.002),
        parameter_spec(:p100_n170_gap, "components.p100_n170_gap_dist", base_cfg.components.p100_n170_gap_dist, 0.50, 2.00; min_mean = 0.002),
        parameter_spec(:n170_p300_gap, "components.n170_p300_gap_dist", base_cfg.components.n170_p300_gap_dist, 0.50, 2.00; min_mean = 0.002),
        parameter_spec(:p1_beta, "components.p1_beta_dist", base_cfg.components.p1_beta_dist, 0.40, 2.00),
        parameter_spec(:p3_beta, "components.p3_beta_dist", base_cfg.components.p3_beta_dist, 0.40, 2.00),
        parameter_spec(:n1_beta1, "components.n1_beta1_dist", base_cfg.components.n1_beta1_dist, 0.40, 2.00),
        parameter_spec(:n1_beta2, "components.n1_beta2_dist", base_cfg.components.n1_beta2_dist, 0.50, 2.00),
        parameter_spec(:n1_beta3, "components.n1_beta3_dist", base_cfg.components.n1_beta3_dist, 0.50, 2.00),
        parameter_spec(:componentA_amp, "components.componentA_amp_dist", base_cfg.components.componentA_amp_dist, 0.50, 2.00),
        parameter_spec(:componentB_amp, "components.componentB_amp_dist", base_cfg.components.componentB_amp_dist, 0.50, 2.00),
        parameter_spec(:componentC_amp, "components.componentC_amp_dist", base_cfg.components.componentC_amp_dist, 0.50, 2.00),
        parameter_spec(:noise_pink, "noise.noiselevel_dists[PinkNoise]", base_cfg.noise.noiselevel_dists[ERPGen.PinkNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_white, "noise.noiselevel_dists[WhiteNoise]", base_cfg.noise.noiselevel_dists[ERPGen.WhiteNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_red, "noise.noiselevel_dists[RedNoise]", base_cfg.noise.noiselevel_dists[ERPGen.RedNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_exponential, "noise.noiselevel_dists[ExponentialNoise]", base_cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise], 0.30, 2.50; min_mean = 0.01),
    ]
end

function parameter_infos(base_cfg::ERPGen.GenerationConfig)
    infos = NamedTuple[]
    for spec in parameter_specs(base_cfg)
        push!(infos, (symbol = spec.mean_symbol, label = "$(spec.label).mu", range = spec.mean_range, key = spec.key, field = :mean))
        push!(infos, (symbol = spec.std_symbol, label = "$(spec.label).sigma", range = spec.std_range, key = spec.key, field = :std))
    end
    return infos
end

parameter_ranges(base_cfg::ERPGen.GenerationConfig) = [info.range for info in parameter_infos(base_cfg)]
parameter_symbols(base_cfg::ERPGen.GenerationConfig) = [info.symbol for info in parameter_infos(base_cfg)]

function build_cfg_from_params(base_cfg::ERPGen.GenerationConfig, params::AbstractVector{<:Real})
    specs = parameter_specs(base_cfg)
    @assert length(params) == 2 * length(specs) "Parameter vector length mismatch."

    dists = Dict{Symbol, Distribution}()
    idx = 1
    for spec in specs
        dists[spec.key] = Normal(Float64(params[idx]), Float64(params[idx + 1]))
        idx += 2
    end

    return parameterized_config(base_cfg;
        mu_dist = dists[:sim_mu],
        sigma_dist = dists[:sim_sigma],
        p100_width_dist = dists[:p100_width],
        p100_n170_gap_dist = dists[:p100_n170_gap],
        n170_p300_gap_dist = dists[:n170_p300_gap],
        n170_width_dist = dists[:n170_width],
        p300_width_dist = dists[:p300_width],
        p1_beta_dist = dists[:p1_beta],
        p3_beta_dist = dists[:p3_beta],
        n1_beta1_dist = dists[:n1_beta1],
        n1_beta2_dist = dists[:n1_beta2],
        n1_beta3_dist = dists[:n1_beta3],
        componentA_amp_dist = dists[:componentA_amp],
        componentB_amp_dist = dists[:componentB_amp],
        componentC_amp_dist = dists[:componentC_amp],
        pink_noise_dist = dists[:noise_pink],
        white_noise_dist = dists[:noise_white],
        red_noise_dist = dists[:noise_red],
        exponential_noise_dist = dists[:noise_exponential],
    )
end

function sample_param_vector(rng::Random.AbstractRNG, ranges)
    params = Float64[]
    for (lo, hi) in ranges
        push!(params, lo + rand(rng) * (hi - lo))
    end
    return params
end

function probe_config_score(cfg::ERPGen.GenerationConfig, real_sigmoid_features::AbstractMatrix;
        probe_n_per_pattern::Int = 128,
        seed::Int = Int(time_ns()))
    probe_n_per_pattern = max(2, probe_n_per_pattern)
    imgs, labels = generate_single_condition_images(
        cfg,
        probe_n_per_pattern;
        target_size = CALIBRATION_TARGET_SIZE,
        lowpass = CALIBRATION_LOWPASS,
        seed = seed,
    )

    sigmoid_imgs = [imgs[i] for i in eachindex(imgs) if labels[i] == 1]
    syn_feats = feature_matrix(sigmoid_imgs)
    real_summary = feature_summary(real_sigmoid_features)
    syn_summary = feature_summary(syn_feats)

    zdist = mean(abs.((syn_summary.mu .- real_summary.mu) ./ real_summary.sigma))
    mmd2 = mmd2_rbf(real_sigmoid_features, syn_feats)
    score = Float64(zdist + 0.5 * mmd2)

    return (score = score, zdist = Float64(zdist), mmd2 = Float64(mmd2))
end

function calibrate_lhs(real_sigmoid_imgs::Vector{<:AbstractMatrix};
        n_candidates::Int = 18,
        top_k::Int = 4,
        probe_n_per_pattern::Int = 128,
        seed::Int = Int(time_ns()))
    rng = Random.Xoshiro(seed)
    base_cfg = make_base_config()
    real_feats = feature_matrix(real_sigmoid_imgs)
    ranges = parameter_ranges(base_cfg)
    param_symbols = parameter_symbols(base_cfg)
    lhs = latin_hypercube(n_candidates, length(ranges), rng)

    rows = NamedTuple[]
    cfgs = ERPGen.GenerationConfig[]
    for i in 1:n_candidates
        params = [ranges[j][1] + lhs[i, j] * (ranges[j][2] - ranges[j][1]) for j in eachindex(ranges)]
        cfg = build_cfg_from_params(base_cfg, params)
        score = probe_config_score(cfg, real_feats; probe_n_per_pattern = probe_n_per_pattern, seed = seed + 1000 * i)
        row_pairs = Pair{Symbol, Any}[:candidate => i]
        append!(row_pairs, [param_symbols[j] => params[j] for j in eachindex(params)])
        push!(row_pairs, :score => score.score)
        push!(row_pairs, :zdist => score.zdist)
        push!(row_pairs, :mmd2 => score.mmd2)
        push!(rows, (; row_pairs...))
        push!(cfgs, cfg)
    end

    candidates_df = DataFrame(rows)
    sort!(candidates_df, :score)
    top_k = min(top_k, nrow(candidates_df))
    top_rows = candidates_df[1:top_k, :]
    top_cfgs = [cfgs[Int(top_rows.candidate[i])] for i in 1:nrow(top_rows)]

    return (base_cfg = base_cfg, candidates_df = candidates_df, top_df = top_rows, top_cfgs = top_cfgs)
end

function calibrate_mc(real_sigmoid_imgs::Vector{<:AbstractMatrix};
        n_candidates::Int = 18,
        top_k::Int = 4,
        probe_n_per_pattern::Int = 128,
        seed::Int = Int(time_ns()))
    rng = Random.Xoshiro(seed)
    base_cfg = make_base_config()
    real_feats = feature_matrix(real_sigmoid_imgs)
    ranges = parameter_ranges(base_cfg)
    param_symbols = parameter_symbols(base_cfg)

    rows = NamedTuple[]
    cfgs = ERPGen.GenerationConfig[]
    for i in 1:n_candidates
        params = sample_param_vector(rng, ranges)
        cfg = build_cfg_from_params(base_cfg, params)
        score = probe_config_score(cfg, real_feats; probe_n_per_pattern = probe_n_per_pattern, seed = seed + 2000 * i)
        row_pairs = Pair{Symbol, Any}[:candidate => i]
        append!(row_pairs, [param_symbols[j] => params[j] for j in eachindex(params)])
        push!(row_pairs, :score => score.score)
        push!(row_pairs, :zdist => score.zdist)
        push!(row_pairs, :mmd2 => score.mmd2)
        push!(rows, (; row_pairs...))
        push!(cfgs, cfg)
    end

    candidates_df = DataFrame(rows)
    sort!(candidates_df, :score)
    top_k = min(top_k, nrow(candidates_df))
    top_rows = candidates_df[1:top_k, :]
    top_cfgs = [cfgs[Int(top_rows.candidate[i])] for i in 1:nrow(top_rows)]

    return (base_cfg = base_cfg, candidates_df = candidates_df, top_df = top_rows, top_cfgs = top_cfgs)
end

function sample_random_cfgs(base_cfg::ERPGen.GenerationConfig; n_cfgs::Int, seed::Int)
    rng = Random.Xoshiro(seed)
    ranges = parameter_ranges(base_cfg)
    cfgs = ERPGen.GenerationConfig[]
    for _ in 1:n_cfgs
        push!(cfgs, build_cfg_from_params(base_cfg, sample_param_vector(rng, ranges)))
    end
    return cfgs
end

function config_distribution_map(cfg::ERPGen.GenerationConfig)
    return Dict{Symbol, Distribution}(
        :sim_mu => cfg.sim.mu_dist,
        :sim_sigma => cfg.sim.sigma_dist,
        :p100_width => cfg.components.p100_width_dist,
        :n170_width => cfg.components.n170_width_dist,
        :p300_width => cfg.components.p300_width_dist,
        :p100_n170_gap => cfg.components.p100_n170_gap_dist,
        :n170_p300_gap => cfg.components.n170_p300_gap_dist,
        :p1_beta => cfg.components.p1_beta_dist,
        :p3_beta => cfg.components.p3_beta_dist,
        :n1_beta1 => cfg.components.n1_beta1_dist,
        :n1_beta2 => cfg.components.n1_beta2_dist,
        :n1_beta3 => cfg.components.n1_beta3_dist,
        :componentA_amp => cfg.components.componentA_amp_dist,
        :componentB_amp => cfg.components.componentB_amp_dist,
        :componentC_amp => cfg.components.componentC_amp_dist,
        :noise_pink => cfg.noise.noiselevel_dists[ERPGen.PinkNoise],
        :noise_white => cfg.noise.noiselevel_dists[ERPGen.WhiteNoise],
        :noise_red => cfg.noise.noiselevel_dists[ERPGen.RedNoise],
        :noise_exponential => cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise],
    )
end

function config_parameter_namedtuple(base_cfg::ERPGen.GenerationConfig, cfg::ERPGen.GenerationConfig)
    dists = config_distribution_map(cfg)
    row_pairs = Pair{Symbol, Any}[]
    for spec in parameter_specs(base_cfg)
        dist = dists[spec.key]
        push!(row_pairs, spec.mean_symbol => Float64(mean(dist)))
        push!(row_pairs, spec.std_symbol => Float64(std(dist)))
    end
    return (; row_pairs...)
end

function parameter_vector_from_cfg(base_cfg::ERPGen.GenerationConfig, cfg::ERPGen.GenerationConfig)
    dists = config_distribution_map(cfg)
    params = Float64[]
    for spec in parameter_specs(base_cfg)
        dist = dists[spec.key]
        push!(params, Float64(mean(dist)))
        push!(params, Float64(std(dist)))
    end
    return params
end

function build_cfg_from_settings_row(base_cfg::ERPGen.GenerationConfig, row)
    params = Float64[Float64(row[sym]) for sym in parameter_symbols(base_cfg)]
    return build_cfg_from_params(base_cfg, params)
end

function config_parameter_row(base_cfg::ERPGen.GenerationConfig, cfg::ERPGen.GenerationConfig;
        strategy::AbstractString,
        setting_rank::Int,
        setting_source::AbstractString,
        setting_candidate = missing,
        setting_score = missing,
        setting_zdist = missing,
        setting_mmd2 = missing)
    row_pairs = Pair{Symbol, Any}[
        :strategy => String(strategy),
        :setting_rank => setting_rank,
        :setting_source => String(setting_source),
        :setting_candidate => setting_candidate,
        :setting_score => setting_score,
        :setting_zdist => setting_zdist,
        :setting_mmd2 => setting_mmd2,
    ]
    append!(row_pairs, pairs(config_parameter_namedtuple(base_cfg, cfg)))

    return (; row_pairs...)
end

function strategy_settings_from_cfgs(base_cfg::ERPGen.GenerationConfig, strategy_name::Symbol, cfgs::Vector{ERPGen.GenerationConfig};
        setting_source::AbstractString)
    rows = NamedTuple[]
    for (setting_rank, cfg) in enumerate(cfgs)
        push!(rows, config_parameter_row(base_cfg, cfg;
            strategy = String(strategy_name),
            setting_rank = setting_rank,
            setting_source = setting_source,
        ))
    end
    return DataFrame(rows)
end

function strategy_settings_from_top_df(strategy_name::Symbol, top_df::DataFrame; setting_source::AbstractString)
    out = copy(top_df)
    :candidate in propertynames(out) && rename!(out, :candidate => :setting_candidate)
    :score in propertynames(out) && rename!(out, :score => :setting_score)
    :zdist in propertynames(out) && rename!(out, :zdist => :setting_zdist)
    :mmd2 in propertynames(out) && rename!(out, :mmd2 => :setting_mmd2)

    out.strategy = fill(String(strategy_name), nrow(out))
    out.setting_rank = collect(1:nrow(out))
    out.setting_source = fill(String(setting_source), nrow(out))

    front_cols = Symbol[:strategy, :setting_rank, :setting_source]
    for col in (:setting_candidate, :setting_score, :setting_zdist, :setting_mmd2)
        col in propertynames(out) && push!(front_cols, col)
    end
    remaining = [col for col in propertynames(out) if col ∉ Set(front_cols)]
    return out[:, vcat(front_cols, remaining)]
end

function allocate_counts(weights::AbstractVector{<:Real}, total::Int)
    w = Float64.(weights)
    w ./= sum(w)
    counts = floor.(Int, total .* w)
    remainder = total - sum(counts)
    order = sortperm(w; rev = true)
    for i in 1:remainder
        counts[order[mod1(i, length(order))]] += 1
    end
    return counts
end

function build_task_list(cfgs::Vector{ERPGen.GenerationConfig}, weights::AbstractVector{<:Real}, n_per_pattern::Int;
        seed::Int,
        tag_prefix::AbstractString)
    rng = Random.Xoshiro(seed)
    counts = allocate_counts(weights, n_per_pattern)
    tasks = SimulationTask[]

    for (cfg_idx, cfg) in enumerate(cfgs)
        for rep in 1:counts[cfg_idx]
            push!(tasks, SimulationTask(
                cfg = cfg,
                seed = rand(rng, 1:2_000_000_000),
                tag = "$(tag_prefix)$(cfg_idx)_$(rep)",
            ))
        end
    end

    shuffle!(rng, tasks)
    return tasks
end

function make_strategy_bundle(real_sigmoid_imgs::Vector{<:AbstractMatrix};
        n_per_pattern::Int = 1000,
        n_candidates::Int = 18,
        top_k::Int = 4,
        probe_n_per_pattern::Int = 128,
        seed::Int = Int(time_ns()))
    lhs = calibrate_lhs(real_sigmoid_imgs;
        n_candidates = n_candidates,
        top_k = top_k,
        probe_n_per_pattern = probe_n_per_pattern,
        seed = seed,
    )

    mc = calibrate_mc(real_sigmoid_imgs;
        n_candidates = n_candidates,
        top_k = top_k,
        probe_n_per_pattern = probe_n_per_pattern,
        seed = seed + 50_000,
    )

    broad_cfgs = sample_random_cfgs(lhs.base_cfg; n_cfgs = max(6, top_k), seed = seed + 90_000)
    broad_weights = fill(1.0 / length(broad_cfgs), length(broad_cfgs))
    lhs_weights = fill(1.0 / length(lhs.top_cfgs), length(lhs.top_cfgs))
    mc_weights = fill(1.0 / length(mc.top_cfgs), length(mc.top_cfgs))
    broad_settings_df = strategy_settings_from_cfgs(lhs.base_cfg, :broad_random, broad_cfgs; setting_source = "random_sample")
    lhs_settings_df = strategy_settings_from_top_df(:lhs_calibrated, lhs.top_df; setting_source = "lhs_top_candidate")
    mc_settings_df = strategy_settings_from_top_df(:mc_random_search, mc.top_df; setting_source = "mc_top_candidate")
    strategy_settings_df = vcat(broad_settings_df, lhs_settings_df, mc_settings_df; cols = :union)

    strategies = Dict{Symbol, NamedTuple}(
        :broad_random => (
            name = :broad_random,
            tasks = build_task_list(broad_cfgs, broad_weights, n_per_pattern; seed = seed + 100_000, tag_prefix = "broad_"),
            calibration_df = broad_settings_df,
        ),
        :lhs_calibrated => (
            name = :lhs_calibrated,
            tasks = build_task_list(lhs.top_cfgs, lhs_weights, n_per_pattern; seed = seed + 110_000, tag_prefix = "lhs_"),
            calibration_df = lhs_settings_df,
        ),
        :mc_random_search => (
            name = :mc_random_search,
            tasks = build_task_list(mc.top_cfgs, mc_weights, n_per_pattern; seed = seed + 120_000, tag_prefix = "mc_"),
            calibration_df = mc_settings_df,
        ),
    )

    return (
        strategies = strategies,
        calibration_lhs_df = lhs.candidates_df,
        calibration_mc_df = mc.candidates_df,
        strategy_settings_df = strategy_settings_df,
    )
end

function generate_strategy_datasets(tasks::Vector{SimulationTask};
        target_sizes = TARGET_SIZES,
        lowpass_choices = LOWPASS_CHOICES,
        progress_every::Int = 100)
    n_total = 2 * length(tasks)
    image_buffers = Dict{NamedTuple, Vector{Matrix{Float32}}}()
    for target_size in target_sizes, lowpass in lowpass_choices
        image_buffers[condition_key(target_size, lowpass)] = Vector{Matrix{Float32}}(undef, n_total)
    end

    labels = Vector{Int}(undef, n_total)
    tags = Vector{String}(undef, n_total)

    out_idx = 1
    for (task_idx, task) in enumerate(tasks)
        rng = Random.Xoshiro(task.seed)
        base_pair = simulate_base_pair(task.cfg, rng)
        for (pattern_name, label) in ((:sigmoid, 1), (:no_class, 0))
            base_img = base_pair[pattern_name]
            for target_size in target_sizes, lowpass in lowpass_choices
                key = condition_key(target_size, lowpass)
                image_buffers[key][out_idx] = process_small_scale_image(base_img, target_size; lowpass = lowpass)
            end
            labels[out_idx] = label
            tags[out_idx] = task.tag
            out_idx += 1
        end

        if progress_every > 0 && (task_idx % progress_every == 0 || task_idx == length(tasks))
            println("  processed repetitions: $(task_idx)/$(length(tasks))")
        end
    end

    datasets = Dict{NamedTuple, NamedTuple}()
    for (key, imgs) in image_buffers
        datasets[key] = (
            images = imgs,
            labels = copy(labels),
            features = flatten_images(imgs),
            tags = copy(tags),
        )
    end

    return datasets
end

function onehot01(labels::AbstractVector{<:Integer})
    Y = zeros(Float32, 2, length(labels))
    for (idx, label) in enumerate(labels)
        Y[label + 1, idx] = 1f0
    end
    return Y
end

function build_dense_nn(input_dim::Int)
    hidden_dims = if input_dim <= 4
        (16, 8)
    elseif input_dim <= 16
        (32, 16)
    elseif input_dim <= 64
        (64, 32, 16)
    else
        (128, 64, 32)
    end

    layers = Any[]
    in_dim = input_dim
    for (idx, hidden_dim) in enumerate(hidden_dims)
        if idx <= 2
            push!(layers, Dense(in_dim, hidden_dim))
            push!(layers, BatchNorm(hidden_dim, relu))
        else
            push!(layers, Dense(in_dim, hidden_dim, relu))
        end
        in_dim = hidden_dim
    end
    push!(layers, Dense(in_dim, 2))
    return Chain(layers...)
end

function train_dense_nn(train_features::AbstractMatrix, train_labels::AbstractVector{<:Integer}, test_features::AbstractMatrix;
        batch_size::Int = 32,
        epochs::Int = 30,
        lr::Float32 = 1f-3,
        seed::Int = Int(time_ns()))
    Random.seed!(seed)

    X_train = permutedims(Float32.(train_features))
    Y_train = onehot01(train_labels)
    X_test = permutedims(Float32.(test_features))

    use_gpu = CUDA.functional()
    device = use_gpu ? gpu : cpu

    model = build_dense_nn(size(train_features, 2)) |> device
    X_train = device(X_train)
    Y_train = device(Y_train)

    loader = DataLoader((X_train, Y_train); batchsize = batch_size, shuffle = true)
    opt_state = Flux.setup(Flux.Adam(lr), model)
    loss_history = Float32[]

    report_every = max(1, fld(epochs, 5))
    for epoch in 1:epochs
        epoch_loss = 0f0
        n_batches = 0
        for (xb, yb) in loader
            loss, grads = Flux.withgradient(model) do nn
                Flux.Losses.logitcrossentropy(nn(xb), yb)
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += Float32(loss)
            n_batches += 1
        end
        avg_loss = epoch_loss / max(n_batches, 1)
        push!(loss_history, avg_loss)
        if epoch == 1 || epoch == epochs || epoch % report_every == 0
            println("    dense_nn epoch $(epoch)/$(epochs) | loss=$(round(avg_loss, digits = 5))")
        end
    end

    model_cpu = cpu(model)
    logits = model_cpu(X_test)
    pred = Int.(onecold(Array(logits), 0:1))

    GC.gc()
    if use_gpu
        CUDA.reclaim()
    end

    return pred, (used_gpu = use_gpu, loss_history = loss_history)
end

function train_random_forest(train_features::AbstractMatrix, train_labels::AbstractVector{<:Integer}, test_features::AbstractMatrix;
        n_trees::Int = 200,
        partial_sampling::Float64 = 0.7,
        seed::Int = Int(time_ns()))
    rng = Random.Xoshiro(seed)
    n_subfeatures = max(1, round(Int, sqrt(size(train_features, 2))))

    model = DecisionTree.build_forest(
        Int.(train_labels),
        Matrix{Float32}(train_features),
        n_subfeatures,
        n_trees,
        partial_sampling,
        -1,
        1,
        2,
        0.0;
        rng = rng,
        impurity_importance = false,
    )

    pred = DecisionTree.apply_forest(model, Matrix{Float32}(test_features))
    return Int.(pred), (n_trees = n_trees, n_subfeatures = n_subfeatures)
end

function train_svm_rbf(train_features::AbstractMatrix, train_labels::AbstractVector{<:Integer}, test_features::AbstractMatrix;
        cost::Float64 = 1.0,
        gamma::Union{Nothing, Float64} = nothing)
    gamma_val = isnothing(gamma) ? 1.0 / max(1, size(train_features, 2)) : gamma
    X_train = permutedims(Float64.(train_features))
    X_test = permutedims(Float64.(test_features))
    model = LIBSVM.svmtrain(X_train, Int.(train_labels); kernel = LIBSVM.Kernel.RadialBasis, cost = cost, gamma = gamma_val)
    pred, _ = LIBSVM.svmpredict(model, X_test)
    return Int.(round.(pred)), (cost = cost, gamma = gamma_val)
end

function evaluate_binary_metrics(y_true::AbstractVector{<:Integer}, y_pred::AbstractVector{<:Integer})
    classes = (0, 1)
    recalls = Float64[]
    f1s = Float64[]

    for cls in classes
        tp = count(i -> y_true[i] == cls && y_pred[i] == cls, eachindex(y_true))
        fp = count(i -> y_true[i] != cls && y_pred[i] == cls, eachindex(y_true))
        fn = count(i -> y_true[i] == cls && y_pred[i] != cls, eachindex(y_true))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
        push!(recalls, recall)
        push!(f1s, f1)
    end

    return (
        balanced_accuracy = mean(recalls),
        macro_f1 = mean(f1s),
        recall_no_class = recalls[1],
        recall_sigmoid = recalls[2],
        support_no_class = count(==(0), y_true),
        support_sigmoid = count(==(1), y_true),
    )
end

function train_and_predict(model_name::Symbol, train_features::AbstractMatrix, train_labels::AbstractVector{<:Integer}, test_features::AbstractMatrix;
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        rf_trees::Int = 200,
        seed::Int = Int(time_ns()))
    if model_name == :dense_nn_gpu
        return train_dense_nn(train_features, train_labels, test_features;
            batch_size = nn_batch_size,
            epochs = nn_epochs,
            lr = nn_lr,
            seed = seed,
        )
    elseif model_name == :random_forest
        return train_random_forest(train_features, train_labels, test_features;
            n_trees = rf_trees,
            seed = seed,
        )
    elseif model_name == :svm_rbf
        return train_svm_rbf(train_features, train_labels, test_features)
    end

    error("Unknown model name: $model_name")
end

function run_small_scale_experiment(;
        n_per_pattern::Int = 1000,
        target_sizes = TARGET_SIZES,
        lowpass_choices = LOWPASS_CHOICES,
        strategy_names = [:broad_random, :lhs_calibrated, :mc_random_search],
        seed::Int = Int(time_ns()),
        calibration_candidates::Int = 18,
        calibration_top_k::Int = 4,
        probe_n_per_pattern::Int = 128,
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        rf_trees::Int = 200,
        post_stim_only::Bool = true)
    requested_target_sizes = collect(target_sizes)
    requested_lowpass_choices = collect(lowpass_choices)
    real_target_sizes = unique(vcat(requested_target_sizes, [CALIBRATION_TARGET_SIZE]))
    real_lowpass_choices = Tuple(unique(vcat(requested_lowpass_choices, [CALIBRATION_LOWPASS])))

    println("Loading real labelled ERP images...")
    real_bundle = load_real_condition_cache(; target_sizes = real_target_sizes, lowpass_choices = real_lowpass_choices, post_stim_only = post_stim_only)
    real_calibration = real_bundle.cache[condition_key(CALIBRATION_TARGET_SIZE, CALIBRATION_LOWPASS)]
    real_sigmoid_imgs = [real_calibration.images[i] for i in eachindex(real_calibration.images) if real_calibration.labels[i] == 1]
    println("Real calibration sigmoid images: $(length(real_sigmoid_imgs))")

    println("Searching simulation strategies...")
    strategy_bundle = make_strategy_bundle(real_sigmoid_imgs;
        n_per_pattern = n_per_pattern,
        n_candidates = calibration_candidates,
        top_k = calibration_top_k,
        probe_n_per_pattern = probe_n_per_pattern,
        seed = seed,
    )

    simulated_cache = Dict{Symbol, Dict{NamedTuple, NamedTuple}}()
    results_rows = NamedTuple[]
    model_names = [:dense_nn_gpu, :random_forest, :svm_rbf]

    for (strategy_idx, strategy_name) in enumerate(strategy_names)
        strategy_spec = strategy_bundle.strategies[strategy_name]
        println("\n=== Strategy $(strategy_name) | repetitions=$(length(strategy_spec.tasks)) ===")
        sim_cache = generate_strategy_datasets(strategy_spec.tasks;
            target_sizes = target_sizes,
            lowpass_choices = lowpass_choices,
            progress_every = max(1, fld(length(strategy_spec.tasks), 10)),
        )
        simulated_cache[strategy_name] = sim_cache

        for target_size in requested_target_sizes, lowpass in requested_lowpass_choices
            key = condition_key(target_size, lowpass)
            sim_ds = sim_cache[key]
            real_ds = real_bundle.cache[key]

            println("\nCondition $(resolution_label(target_size)) | $(lowpass_label(lowpass))")
            for (model_idx, model_name) in enumerate(model_names)
                start_time = time()
                pred, info = train_and_predict(
                    model_name,
                    sim_ds.features,
                    sim_ds.labels,
                    real_ds.features;
                    nn_batch_size = nn_batch_size,
                    nn_epochs = nn_epochs,
                    nn_lr = nn_lr,
                    rf_trees = rf_trees,
                    seed = seed + 10_000 * strategy_idx + 100 * model_idx,
                )
                elapsed = time() - start_time
                metrics = evaluate_binary_metrics(real_ds.labels, pred)
                push!(results_rows, (
                    strategy = String(strategy_name),
                    resolution = resolution_label(target_size),
                    width = target_size[2],
                    height = target_size[1],
                    lowpass = lowpass,
                    model = String(model_name),
                    balanced_accuracy = metrics.balanced_accuracy,
                    macro_f1 = metrics.macro_f1,
                    recall_no_class = metrics.recall_no_class,
                    recall_sigmoid = metrics.recall_sigmoid,
                    support_no_class = metrics.support_no_class,
                    support_sigmoid = metrics.support_sigmoid,
                    n_train = length(sim_ds.labels),
                    n_real = length(real_ds.labels),
                    train_time_s = elapsed,
                    info = string(info),
                ))
                println("  model=$(model_name) | balanced_accuracy=$(round(metrics.balanced_accuracy, digits = 4)) | macro_f1=$(round(metrics.macro_f1, digits = 4)) | seconds=$(round(elapsed, digits = 2))")
            end

            GC.gc()
            if CUDA.functional()
                CUDA.reclaim()
            end
        end
    end

    results_df = DataFrame(results_rows)
    return (
        results_df = results_df,
        real_bundle = real_bundle,
        strategy_bundle = strategy_bundle,
        simulated_cache = simulated_cache,
    )
end

function summarize_results(results_df::DataFrame)
    ranked = sort(copy(results_df), [:balanced_accuracy, :macro_f1, :strategy, :resolution, :lowpass, :model];
        rev = [true, true, false, false, true, false])
    summary = combine(
        groupby(results_df, [:strategy, :model]),
        :balanced_accuracy => mean => :mean_balanced_accuracy,
        :macro_f1 => mean => :mean_macro_f1,
    )
    sort!(summary, [:mean_balanced_accuracy, :mean_macro_f1], rev = [true, true])
    return (ranked = ranked, summary = summary)
end

function select_best_strategy(results_df::DataFrame; metric::Symbol = :balanced_accuracy)
    agg = combine(groupby(results_df, :strategy), metric => mean => :mean_metric)
    sort!(agg, :mean_metric, rev = true)
    return agg.strategy[1], agg
end

function plot_metric_heatmap(results_df::DataFrame; metric::Symbol = :balanced_accuracy)
    strategies = unique(results_df.strategy)
    resolutions = [resolution_label(size) for size in TARGET_SIZES]
    lowpass_levels = [true, false]
    models = ["dense_nn_gpu", "random_forest", "svm_rbf"]
    metric_values = Float64.(results_df[!, metric])
    colorrange = (minimum(metric_values), maximum(metric_values))

    fig = Figure(size = (1200, 300 * length(strategies)), figure_padding = 20)

    for (row_idx, strategy) in enumerate(strategies)
        for (col_idx, lowpass) in enumerate(lowpass_levels)
            mat = fill(NaN, length(models), length(resolutions))
            for (model_idx, model_name) in enumerate(models)
                for (res_idx, resolution) in enumerate(resolutions)
                    mask = (results_df.strategy .== strategy) .&
                           (results_df.lowpass .== lowpass) .&
                           (results_df.model .== model_name) .&
                           (results_df.resolution .== resolution)
                    any(mask) || continue
                    mat[model_idx, res_idx] = Float64(results_df[findfirst(mask), metric])
                end
            end

            ax = Axis(
                fig[row_idx, col_idx];
                title = "$(strategy) | $(lowpass_label(lowpass))",
                xlabel = "resolution",
                ylabel = "model",
                xticks = (1:length(resolutions), resolutions),
                yticks = (1:length(models), models),
            )
            hm = heatmap!(ax, 1:length(resolutions), 1:length(models), mat; colorrange = colorrange, colormap = :viridis)
            Colorbar(fig[row_idx, col_idx + 2], hm; label = String(metric))
        end
    end

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    return fig
end

function plot_lowpass_effect(results_df::DataFrame; metric::Symbol = :balanced_accuracy)
    strategies = unique(results_df.strategy)
    resolutions = [resolution_label(size) for size in TARGET_SIZES]
    models = ["dense_nn_gpu", "random_forest", "svm_rbf"]

    fig = Figure(size = (1100, 280 * length(strategies)), figure_padding = 20)
    palette = [:steelblue, :darkorange, :seagreen]

    for (row_idx, strategy) in enumerate(strategies)
        ax = Axis(
            fig[row_idx, 1];
            title = "$(strategy) | low-pass minus no low-pass",
            xlabel = "resolution",
            ylabel = String(metric),
            xticks = (1:length(resolutions), resolutions),
        )

        for (model_idx, model_name) in enumerate(models)
            deltas = Float64[]
            for resolution in resolutions
                lp_mask = (results_df.strategy .== strategy) .&
                          (results_df.model .== model_name) .&
                          (results_df.resolution .== resolution) .&
                          (results_df.lowpass .== true)
                no_mask = (results_df.strategy .== strategy) .&
                          (results_df.model .== model_name) .&
                          (results_df.resolution .== resolution) .&
                          (results_df.lowpass .== false)
                lp_val = any(lp_mask) ? Float64(results_df[findfirst(lp_mask), metric]) : NaN
                no_val = any(no_mask) ? Float64(results_df[findfirst(no_mask), metric]) : NaN
                push!(deltas, lp_val - no_val)
            end

            lines!(ax, 1:length(resolutions), deltas; color = palette[model_idx], linewidth = 3, label = model_name)
            scatter!(ax, 1:length(resolutions), deltas; color = palette[model_idx], markersize = 12)
        end

        axislegend(ax; position = :rb)
    end

    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    return fig
end

function plot_example_images(simulated_cache::Dict{Symbol, Dict{NamedTuple, NamedTuple}}, strategy_name::Symbol; lowpass::Bool = true)
    strategy_cache = simulated_cache[strategy_name]
    fig = Figure(size = (1200, 560), figure_padding = 20)

    row_specs = [(1, "sigmoid"), (0, "no_class")]
    for (col_idx, target_size) in enumerate(TARGET_SIZES)
        key = condition_key(target_size, lowpass)
        ds = strategy_cache[key]
        for (row_pos, (label_value, row_title)) in enumerate(row_specs)
            img_idx = findfirst(==(label_value), ds.labels)
            img_idx === nothing && continue
            img = ds.images[img_idx]
            clipped, colorrange, _, _, cmap = clipped_color_stats_quantile_zero_ticks(img; q_low = 0.01, q_high = 0.99)

            ax = Axis(
                fig[row_pos, col_idx];
                title = row_pos == 1 ? resolution_label(target_size) : "",
                xlabel = "time",
                ylabel = col_idx == 1 ? row_title : "",
            )
            heatmap!(ax, permutedims(clipped, (2, 1)); colormap = cmap, colorrange = colorrange)
        end
    end

    Label(fig[0, :], "$(strategy_name) | $(lowpass_label(lowpass))", fontsize = 24)
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    return fig
end

function build_ranked_settings_table(experiment;
        ranked_results::Union{Nothing, DataFrame} = nothing,
        setting_rank::Int = 1)
    ranked = isnothing(ranked_results) ? summarize_results(experiment.results_df).ranked : ranked_results
    settings_df = experiment.strategy_bundle.strategy_settings_df
    nrow(ranked) > 0 || error("No ranked results available.")
    nrow(settings_df) > 0 || error("No strategy settings available.")

    setting_cols = [col for col in propertynames(settings_df) if col != :strategy]
    missing_settings = Dict{Symbol, Any}(col => missing for col in setting_cols)
    rows = NamedTuple[]

    for (result_rank, row) in enumerate(eachrow(ranked))
        row_settings = missing_settings
        mask = settings_df.strategy .== String(row.strategy)
        :setting_rank in propertynames(settings_df) && (mask .&= settings_df.setting_rank .== setting_rank)

        setting_idx = findfirst(mask)
        if setting_idx !== nothing
            row_settings = Dict{Symbol, Any}(col => settings_df[setting_idx, col] for col in setting_cols)
        end

        row_pairs = Pair{Symbol, Any}[
            :result_rank => result_rank,
            :strategy => String(row.strategy),
            :resolution => String(row.resolution),
            :lowpass => Bool(row.lowpass),
            :model => String(row.model),
            :balanced_accuracy => Float64(row.balanced_accuracy),
            :macro_f1 => Float64(row.macro_f1),
            :recall_no_class => Float64(row.recall_no_class),
            :recall_sigmoid => Float64(row.recall_sigmoid),
            :train_time_s => Float64(row.train_time_s),
        ]
        append!(row_pairs, [col => row_settings[col] for col in setting_cols])
        push!(rows, (; row_pairs...))
    end

    out = DataFrame(rows)
    front_cols = Symbol[
        :result_rank,
        :strategy,
        :resolution,
        :lowpass,
        :model,
        :balanced_accuracy,
        :macro_f1,
        :recall_no_class,
        :recall_sigmoid,
        :train_time_s,
        :setting_source,
        :setting_rank,
        :setting_candidate,
        :setting_score,
        :setting_zdist,
        :setting_mmd2,
    ]
    ordered_front = [col for col in front_cols if col in propertynames(out)]
    remaining = [col for col in propertynames(out) if col ∉ Set(ordered_front)]
    return out[:, vcat(ordered_front, remaining)]
end

function centered_parameter_ranges(center_params::AbstractVector{<:Real}, global_ranges;
        local_range_scale::Float64 = 0.25)
    @assert length(center_params) == length(global_ranges) "Center parameter length mismatch."
    scale = clamp(Float64(local_range_scale), 1e-6, 1.0)
    local_ranges = Tuple{Float64, Float64}[]

    for (center_raw, (lo_raw, hi_raw)) in zip(center_params, global_ranges)
        lo = Float64(lo_raw)
        hi = Float64(hi_raw)
        center = clamp(Float64(center_raw), lo, hi)
        half_width = 0.5 * scale * (hi - lo)
        a = clamp(center - half_width, lo, hi)
        b = clamp(center + half_width, lo, hi)
        if b <= a
            delta = max(abs(center), 1.0) * 1e-6
            a = clamp(center - delta, lo, hi)
            b = clamp(center + delta, lo, hi)
            b <= a && (b = min(hi, nextfloat(a)))
            b <= a && (a = max(lo, prevfloat(b)))
        end
        push!(local_ranges, (a, b))
    end

    return local_ranges
end

function lhs_param_vectors(ranges, n::Int, rng::Random.AbstractRNG)
    n <= 0 && return Vector{Vector{Float64}}()
    lhs = latin_hypercube(n, length(ranges), rng)
    out = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        out[i] = [ranges[j][1] + lhs[i, j] * (ranges[j][2] - ranges[j][1]) for j in eachindex(ranges)]
    end
    return out
end

function collect_previous_dense_nn_seed_settings(experiment;
        ranked_results::Union{Nothing, DataFrame} = nothing,
        target_size::Union{Nothing, Tuple{Int, Int}} = nothing,
        lowpass::Bool = true,
        top_settings_per_strategy::Int = 2,
        max_seed_strategies::Int = 3)
    ranked = isnothing(ranked_results) ? summarize_results(experiment.results_df).ranked : ranked_results
    settings_df = experiment.strategy_bundle.strategy_settings_df
    nrow(settings_df) == 0 && return DataFrame()

    mask = (ranked.model .== "dense_nn_gpu") .& (ranked.lowpass .== lowpass)
    ranked_dense = copy(ranked[mask, :])
    if !isnothing(target_size)
        target_mask = ranked_dense.resolution .== resolution_label(target_size)
        any(target_mask) && (ranked_dense = copy(ranked_dense[target_mask, :]))
    end
    nrow(ranked_dense) == 0 && (ranked_dense = copy(ranked))

    strategy_order = unique(String.(ranked_dense.strategy))
    strategy_order = strategy_order[1:min(length(strategy_order), max_seed_strategies)]

    rows = NamedTuple[]
    for (strategy_priority, strategy) in enumerate(strategy_order)
        strat_settings = copy(settings_df[settings_df.strategy .== strategy, :])
        nrow(strat_settings) == 0 && continue
        :setting_rank in propertynames(strat_settings) && sort!(strat_settings, :setting_rank)
        take_n = min(top_settings_per_strategy, nrow(strat_settings))

        for i in 1:take_n
            row = strat_settings[i, :]
            row_pairs = Pair{Symbol, Any}[
                :strategy_priority => strategy_priority,
                :seed_label => "$(strategy)#$(Int(row.setting_rank))",
                :seed_origin => "previous_run",
            ]
            append!(row_pairs, [col => row[col] for col in propertynames(row)])
            push!(rows, (; row_pairs...))
        end
    end

    return DataFrame(rows)
end

function build_dense_nn_seed_specs(base_cfg::ERPGen.GenerationConfig, previous_seed_settings_df::DataFrame)
    seed_specs = NamedTuple[]
    push!(seed_specs, (
        seed_label = "base_cfg",
        seed_origin = "baseline",
        source_strategy = "base_cfg",
        source_setting_rank = 0,
        cfg = base_cfg,
    ))

    if nrow(previous_seed_settings_df) == 0
        return seed_specs
    end

    for row in eachrow(previous_seed_settings_df)
        push!(seed_specs, (
            seed_label = String(row.seed_label),
            seed_origin = String(row.seed_origin),
            source_strategy = String(row.strategy),
            source_setting_rank = Int(row.setting_rank),
            cfg = build_cfg_from_settings_row(base_cfg, row),
        ))
    end

    return seed_specs
end

function build_dense_nn_search_candidate_specs(method::Symbol, seed_specs, base_cfg::ERPGen.GenerationConfig;
        n_candidates::Int,
        local_range_scale::Float64 = 0.25,
        seed::Int = Int(time_ns()))
    rng = Random.Xoshiro(seed)
    global_ranges = parameter_ranges(base_cfg)
    candidate_specs = NamedTuple[]

    for seed_spec in seed_specs
        push!(candidate_specs, (
            candidate_source = "seed_config",
            seed_label = seed_spec.seed_label,
            seed_origin = seed_spec.seed_origin,
            source_strategy = seed_spec.source_strategy,
            source_setting_rank = seed_spec.source_setting_rank,
            is_seed_config = true,
            cfg = seed_spec.cfg,
        ))
    end

    extra_candidates = max(n_candidates - length(candidate_specs), 0)
    extra_candidates == 0 && return candidate_specs

    if method == :broad_random
        for _ in 1:extra_candidates
            cfg = build_cfg_from_params(base_cfg, sample_param_vector(rng, global_ranges))
            push!(candidate_specs, (
                candidate_source = "global_random",
                seed_label = "global_random",
                seed_origin = "global_random",
                source_strategy = "global_random",
                source_setting_rank = 0,
                is_seed_config = false,
                cfg = cfg,
            ))
        end
    elseif method == :latin_hypercube || method == :monte_carlo
        seed_counts = allocate_counts(fill(1.0, length(seed_specs)), extra_candidates)
        for (seed_idx, count) in enumerate(seed_counts)
            count == 0 && continue
            seed_spec = seed_specs[seed_idx]
            center_params = parameter_vector_from_cfg(base_cfg, seed_spec.cfg)
            local_ranges = centered_parameter_ranges(center_params, global_ranges; local_range_scale = local_range_scale)
            param_vectors = if method == :latin_hypercube
                lhs_param_vectors(local_ranges, count, rng)
            else
                [sample_param_vector(rng, local_ranges) for _ in 1:count]
            end

            for params in param_vectors
                cfg = build_cfg_from_params(base_cfg, params)
                push!(candidate_specs, (
                    candidate_source = method == :latin_hypercube ? "lhs_local" : "mc_local",
                    seed_label = seed_spec.seed_label,
                    seed_origin = seed_spec.seed_origin,
                    source_strategy = seed_spec.source_strategy,
                    source_setting_rank = seed_spec.source_setting_rank,
                    is_seed_config = false,
                    cfg = cfg,
                ))
            end
        end
    else
        error("Unknown dense NN search method: $(method)")
    end

    return candidate_specs
end

function evaluate_dense_nn_search_candidate(cfg::ERPGen.GenerationConfig, real_ds;
        target_size::Tuple{Int, Int},
        lowpass::Bool = true,
        n_per_pattern::Int = 256,
        eval_repeats::Int = 1,
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        seed::Int = Int(time_ns()))
    balanced_accuracies = Float64[]
    macro_f1s = Float64[]
    recall_no_classes = Float64[]
    recall_sigmoids = Float64[]
    sim_balanced_accuracies = Float64[]
    sim_macro_f1s = Float64[]
    sim_recall_no_classes = Float64[]
    sim_recall_sigmoids = Float64[]
    train_times = Float64[]
    n_train = 0
    n_real = length(real_ds.labels)

    for repeat_idx in 1:max(eval_repeats, 1)
        sim_seed = seed + 10_000 * repeat_idx
        train_seed = seed + 100_000 * repeat_idx
        imgs, labels = generate_single_condition_images(
            cfg,
            n_per_pattern;
            target_size = target_size,
            lowpass = lowpass,
            seed = sim_seed,
        )
        train_features = flatten_images(imgs)
        n_train = length(labels)
        joint_features = vcat(real_ds.features, train_features)

        start_time = time()
        pred_joint, _ = train_and_predict(
            :dense_nn_gpu,
            train_features,
            labels,
            joint_features;
            nn_batch_size = nn_batch_size,
            nn_epochs = nn_epochs,
            nn_lr = nn_lr,
            seed = train_seed,
        )
        elapsed = time() - start_time
        pred_real = pred_joint[1:n_real]
        pred_sim = pred_joint[(n_real + 1):end]
        metrics = evaluate_binary_metrics(real_ds.labels, pred_real)
        sim_metrics = evaluate_binary_metrics(labels, pred_sim)

        push!(balanced_accuracies, metrics.balanced_accuracy)
        push!(macro_f1s, metrics.macro_f1)
        push!(recall_no_classes, metrics.recall_no_class)
        push!(recall_sigmoids, metrics.recall_sigmoid)
        push!(sim_balanced_accuracies, sim_metrics.balanced_accuracy)
        push!(sim_macro_f1s, sim_metrics.macro_f1)
        push!(sim_recall_no_classes, sim_metrics.recall_no_class)
        push!(sim_recall_sigmoids, sim_metrics.recall_sigmoid)
        push!(train_times, elapsed)

        GC.gc()
        if CUDA.functional()
            CUDA.reclaim()
        end
    end

    return (
        mean_balanced_accuracy = mean(balanced_accuracies),
        std_balanced_accuracy = length(balanced_accuracies) > 1 ? std(balanced_accuracies) : 0.0,
        mean_macro_f1 = mean(macro_f1s),
        std_macro_f1 = length(macro_f1s) > 1 ? std(macro_f1s) : 0.0,
        mean_recall_no_class = mean(recall_no_classes),
        mean_recall_sigmoid = mean(recall_sigmoids),
        mean_sim_balanced_accuracy = mean(sim_balanced_accuracies),
        mean_sim_macro_f1 = mean(sim_macro_f1s),
        mean_sim_recall_no_class = mean(sim_recall_no_classes),
        mean_sim_recall_sigmoid = mean(sim_recall_sigmoids),
        mean_train_time_s = mean(train_times),
        n_train = n_train,
        n_real = n_real,
        eval_repeats = max(eval_repeats, 1),
    )
end

function run_dense_nn_lowpass_search(experiment;
        target_size::Tuple{Int, Int} = (16, 16),
        lowpass::Bool = true,
        ranked_results::Union{Nothing, DataFrame} = nothing,
        method_candidates = Dict(:broad_random => 12, :latin_hypercube => 12, :monte_carlo => 12),
        n_per_pattern::Int = 256,
        eval_repeats::Int = 1,
        top_settings_per_strategy::Int = 2,
        max_seed_strategies::Int = 3,
        local_range_scale::Float64 = 0.25,
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        seed::Int = Int(time_ns()))
    lowpass || error("This workflow is intended for lowpass = true.")

    key = condition_key(target_size, lowpass)
    @assert haskey(experiment.real_bundle.cache, key) "Target size $(target_size) with lowpass=$(lowpass) not found in experiment.real_bundle.cache."

    base_cfg = make_base_config(target_size = target_size, apply_lowpass = lowpass)
    real_ds = experiment.real_bundle.cache[key]
    previous_seed_settings_df = collect_previous_dense_nn_seed_settings(
        experiment;
        ranked_results = ranked_results,
        target_size = target_size,
        lowpass = lowpass,
        top_settings_per_strategy = top_settings_per_strategy,
        max_seed_strategies = max_seed_strategies,
    )
    seed_specs = build_dense_nn_seed_specs(base_cfg, previous_seed_settings_df)

    methods = [:broad_random, :latin_hypercube, :monte_carlo]
    results_rows = NamedTuple[]
    method_offset = 0

    for method in methods
        requested_candidates = Int(get(method_candidates, method, length(seed_specs)))
        requested_candidates = max(requested_candidates, length(seed_specs))
        println("\nDense NN low-pass search | method=$(method) | target=$(resolution_label(target_size)) | candidates=$(requested_candidates)")

        candidate_specs = build_dense_nn_search_candidate_specs(
            method,
            seed_specs,
            base_cfg;
            n_candidates = requested_candidates,
            local_range_scale = local_range_scale,
            seed = seed + method_offset,
        )

        for (candidate_idx, candidate_spec) in enumerate(candidate_specs)
            println("  evaluating candidate $(candidate_idx)/$(length(candidate_specs)) | source=$(candidate_spec.candidate_source) | seed=$(candidate_spec.seed_label)")
            metrics = evaluate_dense_nn_search_candidate(
                candidate_spec.cfg,
                real_ds;
                target_size = target_size,
                lowpass = lowpass,
                n_per_pattern = n_per_pattern,
                eval_repeats = eval_repeats,
                nn_batch_size = nn_batch_size,
                nn_epochs = nn_epochs,
                nn_lr = nn_lr,
                seed = seed + method_offset + 1_000 * candidate_idx,
            )

            row_pairs = Pair{Symbol, Any}[
                :search_method => String(method),
                :resolution => resolution_label(target_size),
                :lowpass => lowpass,
                :candidate_index => candidate_idx,
                :candidate_source => candidate_spec.candidate_source,
                :seed_label => candidate_spec.seed_label,
                :seed_origin => candidate_spec.seed_origin,
                :source_strategy => candidate_spec.source_strategy,
                :source_setting_rank => candidate_spec.source_setting_rank,
                :is_seed_config => candidate_spec.is_seed_config,
                :n_per_pattern => n_per_pattern,
                :eval_repeats => metrics.eval_repeats,
                :mean_balanced_accuracy => metrics.mean_balanced_accuracy,
                :std_balanced_accuracy => metrics.std_balanced_accuracy,
                :mean_macro_f1 => metrics.mean_macro_f1,
                :std_macro_f1 => metrics.std_macro_f1,
                :mean_recall_no_class => metrics.mean_recall_no_class,
                :mean_recall_sigmoid => metrics.mean_recall_sigmoid,
                :mean_sim_balanced_accuracy => metrics.mean_sim_balanced_accuracy,
                :mean_sim_macro_f1 => metrics.mean_sim_macro_f1,
                :mean_sim_recall_no_class => metrics.mean_sim_recall_no_class,
                :mean_sim_recall_sigmoid => metrics.mean_sim_recall_sigmoid,
                :mean_train_time_s => metrics.mean_train_time_s,
                :n_train => metrics.n_train,
                :n_real => metrics.n_real,
            ]
            append!(row_pairs, pairs(config_parameter_namedtuple(base_cfg, candidate_spec.cfg)))
            push!(results_rows, (; row_pairs...))
        end

        method_offset += 1_000_000
    end

    results_df = DataFrame(results_rows)
    sort!(results_df, [:search_method, :mean_balanced_accuracy, :mean_macro_f1, :std_balanced_accuracy];
        rev = [false, true, true, false])
    results_df.method_rank = zeros(Int, nrow(results_df))
    for group in groupby(results_df, :search_method)
        group.method_rank = collect(1:nrow(group))
    end

    summary_df = combine(
        groupby(results_df, :search_method),
        :mean_balanced_accuracy => maximum => :best_balanced_accuracy,
        :mean_macro_f1 => maximum => :best_macro_f1,
        :candidate_index => length => :n_evaluated,
    )
    sort!(summary_df, :best_balanced_accuracy, rev = true)

    return (
        results_df = results_df,
        summary_df = summary_df,
        previous_seed_settings_df = previous_seed_settings_df,
        seed_specs = seed_specs,
        target_size = target_size,
        lowpass = lowpass,
        n_per_pattern = n_per_pattern,
        eval_repeats = eval_repeats,
        method_candidates = method_candidates,
        local_range_scale = local_range_scale,
    )
end

function confusion_category_indices(y_true::AbstractVector{<:Integer}, y_pred::AbstractVector{<:Integer})
    return Dict{Symbol, Vector{Int}}(
        :tp => findall(i -> y_true[i] == 1 && y_pred[i] == 1, eachindex(y_true)),
        :tn => findall(i -> y_true[i] == 0 && y_pred[i] == 0, eachindex(y_true)),
        :fp => findall(i -> y_true[i] == 0 && y_pred[i] == 1, eachindex(y_true)),
        :fn => findall(i -> y_true[i] == 1 && y_pred[i] == 0, eachindex(y_true)),
    )
end

function confusion_counts(indices::Dict{Symbol, Vector{Int}})
    return (
        tp = length(indices[:tp]),
        tn = length(indices[:tn]),
        fp = length(indices[:fp]),
        fn = length(indices[:fn]),
    )
end

has_complete_confusion(counts::NamedTuple) = all(values(counts) .> 0)

function replay_training_seed(results_df::DataFrame, row, base_seed::Int)
    strategy_names = unique(String.(results_df.strategy))
    model_names = unique(String.(results_df.model))
    strategy_idx = findfirst(==(String(row.strategy)), strategy_names)
    model_idx = findfirst(==(String(row.model)), model_names)
    strategy_idx === nothing && error("Strategy not found in results table: $(row.strategy)")
    model_idx === nothing && error("Model not found in results table: $(row.model)")
    return base_seed + 10_000 * strategy_idx + 100 * model_idx
end

function build_result_prediction_bundle(experiment, row;
        rank::Int,
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        rf_trees::Int = 200,
        seed::Int = Int(time_ns()))
    strategy_name = Symbol(String(row.strategy))
    model_name = Symbol(String(row.model))
    target_size = parse_resolution_label(String(row.resolution))
    lowpass = Bool(row.lowpass)

    key = condition_key(target_size, lowpass)
    sim_ds = experiment.simulated_cache[strategy_name][key]
    real_ds = experiment.real_bundle.cache[key]
    training_seed = replay_training_seed(experiment.results_df, row, seed)

    pred, train_info = train_and_predict(
        model_name,
        sim_ds.features,
        sim_ds.labels,
        real_ds.features;
        nn_batch_size = nn_batch_size,
        nn_epochs = nn_epochs,
        nn_lr = nn_lr,
        rf_trees = rf_trees,
        seed = training_seed,
    )

    y_true = Int.(real_ds.labels)
    y_pred = Int.(pred)
    category_indices = confusion_category_indices(y_true, y_pred)
    counts = confusion_counts(category_indices)

    selected_config = (
        rank = rank,
        strategy = String(row.strategy),
        resolution = String(row.resolution),
        target_size = target_size,
        lowpass = lowpass,
        model = String(row.model),
        balanced_accuracy = Float64(row.balanced_accuracy),
        macro_f1 = Float64(row.macro_f1),
        n_train = Int(row.n_train),
        n_real = Int(row.n_real),
    )

    return (
        selected_config = selected_config,
        train_info = train_info,
        y_true = y_true,
        y_pred = y_pred,
        images = real_ds.images,
        meta = copy(real_ds.meta),
        category_indices = category_indices,
        counts = counts,
        complete_confusion = has_complete_confusion(counts),
    )
end

function build_confusion_example_bundle(experiment;
        ranked_results::Union{Nothing, DataFrame} = nothing,
        prefer_complete_confusion::Bool = true,
        nn_batch_size::Int = 32,
        nn_epochs::Int = 30,
        nn_lr::Float32 = 1f-3,
        rf_trees::Int = 200,
        seed::Int = Int(time_ns()))
    ranked = isnothing(ranked_results) ? summarize_results(experiment.results_df).ranked : ranked_results
    nrow(ranked) > 0 || error("No ranked results available.")

    fallback_bundle = nothing
    for (rank, row) in enumerate(eachrow(ranked))
        bundle = build_result_prediction_bundle(experiment, row;
            rank = rank,
            nn_batch_size = nn_batch_size,
            nn_epochs = nn_epochs,
            nn_lr = nn_lr,
            rf_trees = rf_trees,
            seed = seed,
        )

        fallback_bundle === nothing && (fallback_bundle = bundle)
        if !prefer_complete_confusion || bundle.complete_confusion
            return merge(bundle, (prefer_complete_confusion = prefer_complete_confusion,))
        end
    end

    return merge(fallback_bundle, (prefer_complete_confusion = prefer_complete_confusion,))
end

function plot_confusion_examples(bundle;
        rng::Random.AbstractRNG = Random.Xoshiro(rand(Random.RandomDevice(), UInt64)),
        q_low::Float64 = 0.01,
        q_high::Float64 = 0.99,
        panel_px::Int = 380,
        cb_px::Int = 64)
    fig = Figure(size = (2 * panel_px + 2 * cb_px + 240, 2 * panel_px + 220), figure_padding = 24)
    specs = [
        (key = :tp, short = "TP", subtitle = "true=sigmoid | pred=sigmoid"),
        (key = :tn, short = "TN", subtitle = "true=no_class | pred=no_class"),
        (key = :fp, short = "FP", subtitle = "true=no_class | pred=sigmoid"),
        (key = :fn, short = "FN", subtitle = "true=sigmoid | pred=no_class"),
    ]

    cfg = bundle.selected_config
    header = "Confusion examples | rank $(cfg.rank) | $(cfg.strategy) | $(cfg.resolution) | $(lowpass_label(cfg.lowpass)) | $(cfg.model)"
    counts_text = "TP=$(bundle.counts.tp)  TN=$(bundle.counts.tn)  FP=$(bundle.counts.fp)  FN=$(bundle.counts.fn)"
    Label(fig[0, :], "$(header)\n$(counts_text)", fontsize = 18)

    for (panel_idx, spec) in enumerate(specs)
        row = panel_idx <= 2 ? 1 : 2
        img_col = isodd(panel_idx) ? 1 : 3
        cb_col = img_col + 1

        indices = bundle.category_indices[spec.key]
        if isempty(indices)
            ax = Axis(
                fig[row, img_col];
                title = spec.short,
                subtitle = "$(spec.subtitle)\nno sampled instance available",
                xlabel = "time index",
                ylabel = "sorted trial",
                xticklabelsize = 11,
                yticklabelsize = 11,
                titlealign = :left,
            )
            text!(ax, 0.5, 0.5; text = "Keine Instanz\nverfugbar", space = :relative, align = (:center, :center), fontsize = 18)
            hidespines!(ax)
            hidedecorations!(ax)
            continue
        end

        sample_idx = rand(rng, indices)
        img = bundle.images[sample_idx]
        clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(img; q_low = q_low, q_high = q_high)

        meta_row = bundle.meta[sample_idx, :]
        details = "idx=$(sample_idx) | ch=$(meta_row.channel) | sort=$(meta_row.sort_var)"
        ax = Axis(
            fig[row, img_col];
            title = "$(spec.short) | $(details)",
            subtitle = spec.subtitle,
            xlabel = "time index",
            ylabel = "sorted trial",
            xticklabelsize = 11,
            yticklabelsize = 11,
            titlesize = 15,
            subtitlesize = 12,
            titlealign = :left,
        )
        hm = heatmap!(ax, permutedims(clipped, (2, 1)); colormap = cmap, colorrange = colorrange)

        Colorbar(fig[row, cb_col], hm;
            width = 16,
            ticklabelsize = 11,
            ticks = (tick_vals, tick_labels),
        )

        colsize!(fig.layout, img_col, Fixed(panel_px))
        colsize!(fig.layout, cb_col, Fixed(cb_px))
    end

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 16)
    resize_to_layout!(fig)
    return fig
end

end
