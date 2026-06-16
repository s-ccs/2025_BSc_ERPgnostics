# =============================================================================
# Simulator parameter space
#
# The search optimises 24 Normal-valued simulator distributions, each described
# by a mean and a standard deviation, giving a 48-dimensional parameter vector.
# This file defines the base configuration matched to the real dataset, the
# per-parameter search ranges, and the conversions between a flat parameter
# vector and an ERPGen `GenerationConfig`.
# =============================================================================

"""
    FixedDist(value)

A degenerate distribution that always returns `value`. Used to pin simulator
quantities (epoch duration, sampling rate, trial count) to the exact values of
the real dataset so simulated and real images share the same dimensions.
"""
struct FixedDist <: Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}
    value::Float64
end

Random.rand(::Random.AbstractRNG, d::FixedDist) = d.value
Random.rand(d::FixedDist) = d.value
Distributions.mean(d::FixedDist) = d.value
Distributions.std(::FixedDist) = 0.0

"""
    fixed_dist(x) -> FixedDist

Convenience constructor wrapping `x` in a [`FixedDist`](@ref).

# Arguments
- `x`: the constant value the distribution should always return.

# Returns
- `FixedDist`: a distribution pinned to `Float64(x)`.
"""
fixed_dist(x::Real) = FixedDist(Float64(x))

"""
    build_base_config(config::RunConfig, real) -> ERPGen.GenerationConfig

Build the hand-crafted baseline simulator configuration, matched to the real
fixation dataset. The simulation distributions for epoch duration, sampling
rate, and trial count are pinned to the real recording so that every simulated
raw matrix has the same shape as the real channels.

# Arguments
- `config::RunConfig`: provides target size and Gaussian smoothing factor.
- `real`: named tuple with `n_trials`, `n_timepoints`, `sampling_rate`, and
  `epoch_duration_s` describing the real dataset dimensions.

# Returns
- `ERPGen.GenerationConfig`: the dataset-matched baseline ("starting parameters").
"""
function build_base_config(config::RunConfig, real)
    defaults = ERPGen.GenerationConfig()

    # Pin the simulation geometry to the real recording; keep the calibrated
    # amplitude/latency means (mu) as the hand-crafted starting point.
    sim = ERPGen.SimulationConfig(
        mu_dist = Normal(4.5, 0.3),
        sigma_dist = defaults.sim.sigma_dist,
        epoch_duration_dist = fixed_dist(real.epoch_duration_s),
        sampling_rate_dist = fixed_dist(real.sampling_rate),
        n_trials_dist = fixed_dist(Float64(real.n_trials)),
    )

    patterns = ERPGen.PatternConfig(
        patterns = [:sigmoid],
        loaded_patterns = ERPGen.DEFAULT_PATTERN_LIST,
        covariate_dists = defaults.patterns.covariate_dists,
        diverging_bar_levels = defaults.patterns.diverging_bar_levels,
    )

    # No dropout or cropping: the simulated raw matrix must match the real size
    # exactly. Smoothing and resizing happen later in the shared image pipeline.
    processing = ERPGen.ProcessingConfig(
        dropout_trials_rate_dist = fixed_dist(0.0),
        crop_start_dist = fixed_dist(0.0),
        crop_end_dist = fixed_dist(0.0),
        zscore_timepoints = true,
        resize_antialias = true,
        low_pass_factor = config.low_pass_factor,
        resize_method = defaults.processing.resize_method,
        target_height = config.target_size[1],
        target_width = config.target_size[2],
    )

    runtime = ERPGen.RuntimeConfig(threaded = false, show_progress = false, blas_threads = 1, progress_every = 50)

    return ERPGen.GenerationConfig(
        sim = sim,
        components = defaults.components,
        patterns = patterns,
        noise = defaults.noise,
        processing = processing,
        runtime = runtime,
    )
end

"""
    parameterized_config(base_cfg; kwargs...) -> ERPGen.GenerationConfig

Return a copy of `base_cfg` with the listed simulator distributions replaced.
Every keyword defaults to the matching distribution in `base_cfg`, so callers
only pass the distributions they want to change.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: configuration to start from.
- `kwargs...`: replacement `Distribution`s for individual components, the
  simulation means, and the four noise levels.

# Returns
- `ERPGen.GenerationConfig`: the updated configuration.
"""
function parameterized_config(base_cfg::ERPGen.GenerationConfig;
        mu_dist = base_cfg.sim.mu_dist,
        sigma_dist = base_cfg.sim.sigma_dist,
        p100_width_dist = base_cfg.components.p100_width_dist,
        p100_n170_gap_dist = base_cfg.components.p100_n170_gap_dist,
        n170_p300_gap_dist = base_cfg.components.n170_p300_gap_dist,
        n170_width_dist = base_cfg.components.n170_width_dist,
        p300_width_dist = base_cfg.components.p300_width_dist,
        p1_beta_dist = base_cfg.components.p1_beta_dist,
        p3_beta_dist = base_cfg.components.p3_beta_dist,
        n1_beta1_dist = base_cfg.components.n1_beta1_dist,
        n1_beta2_dist = base_cfg.components.n1_beta2_dist,
        n1_beta3_dist = base_cfg.components.n1_beta3_dist,
        componentA_amp_dist = base_cfg.components.componentA_amp_dist,
        componentB_amp_dist = base_cfg.components.componentB_amp_dist,
        componentC_amp_dist = base_cfg.components.componentC_amp_dist,
        tilted_bar_hanning_length_dist = base_cfg.components.tilted_bar_hanning_length_dist,
        one_sided_fan_duration_divisor_dist = base_cfg.components.one_sided_fan_duration_divisor_dist,
        one_sided_fan_log_mu_offset_dist = base_cfg.components.one_sided_fan_log_mu_offset_dist,
        one_sided_fan_log_sigma_dist = base_cfg.components.one_sided_fan_log_sigma_dist,
        one_sided_fan_support_max_dist = base_cfg.components.one_sided_fan_support_max_dist,
        pink_noise_dist = base_cfg.noise.noiselevel_dists[ERPGen.PinkNoise],
        white_noise_dist = base_cfg.noise.noiselevel_dists[ERPGen.WhiteNoise],
        red_noise_dist = base_cfg.noise.noiselevel_dists[ERPGen.RedNoise],
        exponential_noise_dist = base_cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise])
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
        tilted_bar_hanning_length_dist = tilted_bar_hanning_length_dist,
        one_sided_fan_duration_divisor_dist = one_sided_fan_duration_divisor_dist,
        one_sided_fan_log_mu_offset_dist = one_sided_fan_log_mu_offset_dist,
        one_sided_fan_log_sigma_dist = one_sided_fan_log_sigma_dist,
        one_sided_fan_support_max_dist = one_sided_fan_support_max_dist,
    )

    noiselevel_dists = Dict{DataType, Distributions.Distribution}(
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

"""
    range_from_scale(base_value, lo_scale, hi_scale; min_value = -Inf)

Turn a base value into a `(low, high)` search range by scaling it. The bounds
are ordered and clamped to `min_value`.

# Arguments
- `base_value`: the value to scale (a distribution mean or std).
- `lo_scale`, `hi_scale`: multiplicative factors for the range bounds.
- `min_value`: lower clamp applied to both bounds (keyword).

# Returns
- `Tuple{Float64,Float64}`: the ordered, clamped `(low, high)` range.
"""
function range_from_scale(base_value::Real, lo_scale::Real, hi_scale::Real; min_value::Float64 = -Inf)
    a, b = minmax(Float64(base_value) * Float64(lo_scale), Float64(base_value) * Float64(hi_scale))
    return (max(a, min_value), max(b, min_value))
end

"""
    parameter_spec(key, label, base_dist, lo_scale, hi_scale; min_mean, min_std)

Describe one searchable simulator distribution: its symbolic keys, its base
`Normal`, and the search ranges for its mean and standard deviation.

# Arguments
- `key::Symbol`: identifier of the parameter.
- `label::AbstractString`: human-readable config path for diagnostics.
- `base_dist::Normal`: the baseline distribution.
- `lo_scale`, `hi_scale`: scaling factors for the mean range.
- `min_mean`, `min_std`: lower clamps for the mean and std ranges (keywords).

# Returns
- `NamedTuple`: spec with `key`, `label`, `mean_symbol`, `std_symbol`,
  `base_dist`, `mean_range`, and `std_range`.
"""
function parameter_spec(key::Symbol, label::AbstractString, base_dist::Normal, lo_scale::Real, hi_scale::Real;
        min_mean::Float64 = -Inf, min_std::Float64 = 1.0e-5)
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

"""
    parameter_specs(base_cfg::ERPGen.GenerationConfig)

Return the 24 parameter specs that define the search space, in a fixed order.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: configuration whose distributions provide
  the base values that the search ranges are scaled from.

# Returns
- `Vector{<:NamedTuple}`: the ordered parameter specs.
"""
function parameter_specs(base_cfg::ERPGen.GenerationConfig)
    return [
        parameter_spec(:sim_mu, "sim.mu_dist", base_cfg.sim.mu_dist, 0.70, 1.30; min_mean = 0.1, min_std = 1.0e-4),
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
        parameter_spec(:tilted_bar_hanning_length, "components.tilted_bar_hanning_length_dist", base_cfg.components.tilted_bar_hanning_length_dist, 0.50, 2.00; min_mean = 2.0),
        parameter_spec(:one_sided_fan_duration_divisor, "components.one_sided_fan_duration_divisor_dist", base_cfg.components.one_sided_fan_duration_divisor_dist, 0.50, 2.00; min_mean = 0.01),
        parameter_spec(:one_sided_fan_log_mu_offset, "components.one_sided_fan_log_mu_offset_dist", base_cfg.components.one_sided_fan_log_mu_offset_dist, 0.40, 2.50; min_mean = 0.001),
        parameter_spec(:one_sided_fan_log_sigma, "components.one_sided_fan_log_sigma_dist", base_cfg.components.one_sided_fan_log_sigma_dist, 0.40, 2.00; min_mean = 0.01),
        parameter_spec(:one_sided_fan_support_max, "components.one_sided_fan_support_max_dist", base_cfg.components.one_sided_fan_support_max_dist, 0.50, 2.00; min_mean = 0.01),
        parameter_spec(:noise_pink, "noise.noiselevel_dists[PinkNoise]", base_cfg.noise.noiselevel_dists[ERPGen.PinkNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_white, "noise.noiselevel_dists[WhiteNoise]", base_cfg.noise.noiselevel_dists[ERPGen.WhiteNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_red, "noise.noiselevel_dists[RedNoise]", base_cfg.noise.noiselevel_dists[ERPGen.RedNoise], 0.30, 2.50; min_mean = 0.01),
        parameter_spec(:noise_exponential, "noise.noiselevel_dists[ExponentialNoise]", base_cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise], 0.30, 2.50; min_mean = 0.01),
    ]
end

"""
    parameter_infos(base_cfg::ERPGen.GenerationConfig)

Expand the 24 specs into 48 scalar entries (mean and std per parameter), each
with its flat-vector symbol, label, range, owning key, and field.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: provides the base distributions.

# Returns
- `Vector{<:NamedTuple}`: 48 entries with `symbol`, `label`, `range`, `key`, `field`.
"""
function parameter_infos(base_cfg::ERPGen.GenerationConfig)
    infos = NamedTuple[]
    for spec in parameter_specs(base_cfg)
        push!(infos, (symbol = spec.mean_symbol, label = "$(spec.label).mu", range = spec.mean_range, key = spec.key, field = :mean))
        push!(infos, (symbol = spec.std_symbol, label = "$(spec.label).sigma", range = spec.std_range, key = spec.key, field = :std))
    end
    return infos
end

"""
    parameter_ranges(base_cfg)

Return the 48 `(low, high)` search ranges in flat-vector order.
"""
parameter_ranges(base_cfg::ERPGen.GenerationConfig) = [info.range for info in parameter_infos(base_cfg)]

"""
    parameter_symbols(base_cfg)

Return the 48 parameter symbols (e.g. `:sim_mu_mean`, `:sim_mu_std`) in
flat-vector order.
"""
parameter_symbols(base_cfg::ERPGen.GenerationConfig) = [info.symbol for info in parameter_infos(base_cfg)]

"""
    build_cfg_from_params(base_cfg, params) -> ERPGen.GenerationConfig

Rebuild a simulator configuration from a flat 48-element parameter vector. Each
consecutive `(mean, std)` pair becomes one `Normal` distribution.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: configuration to copy unset fields from.
- `params::AbstractVector{<:Real}`: 48-element vector of means and stds.

# Returns
- `ERPGen.GenerationConfig`: the configuration described by `params`.
"""
function build_cfg_from_params(base_cfg::ERPGen.GenerationConfig, params::AbstractVector{<:Real})
    specs = parameter_specs(base_cfg)
    length(params) == 2 * length(specs) || error("Parameter vector length mismatch.")

    # Decode each consecutive (mean, std) pair into a Normal distribution.
    dists = Dict{Symbol, Distributions.Distribution}()
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
        tilted_bar_hanning_length_dist = dists[:tilted_bar_hanning_length],
        one_sided_fan_duration_divisor_dist = dists[:one_sided_fan_duration_divisor],
        one_sided_fan_log_mu_offset_dist = dists[:one_sided_fan_log_mu_offset],
        one_sided_fan_log_sigma_dist = dists[:one_sided_fan_log_sigma],
        one_sided_fan_support_max_dist = dists[:one_sided_fan_support_max],
        pink_noise_dist = dists[:noise_pink],
        white_noise_dist = dists[:noise_white],
        red_noise_dist = dists[:noise_red],
        exponential_noise_dist = dists[:noise_exponential],
    )
end

"""
    sample_param_vector(rng, ranges)

Draw one parameter vector by sampling each range uniformly and independently.
This is the Monte Carlo / broad random sampling primitive.

# Arguments
- `rng::Random.AbstractRNG`: random source.
- `ranges`: iterable of `(low, high)` bounds.

# Returns
- `Vector{Float64}`: one uniform sample per range.
"""
function sample_param_vector(rng::Random.AbstractRNG, ranges)
    return Float64[lo + rand(rng) * (hi - lo) for (lo, hi) in ranges]
end

"""
    latin_hypercube(n, d, rng) -> Matrix{Float64}

Generate an `n`-by-`d` Latin hypercube design in the unit cube. Each column is a
random permutation of stratified, jittered samples so candidates spread evenly
across every dimension.

# Arguments
- `n::Int`: number of candidates (rows).
- `d::Int`: number of dimensions (columns).
- `rng::Random.AbstractRNG`: random source.

# Returns
- `Matrix{Float64}`: design matrix with entries in `[0, 1)`.
"""
function latin_hypercube(n::Int, d::Int, rng::Random.AbstractRNG)
    design = Matrix{Float64}(undef, n, d)
    for j in 1:d
        perm = randperm(rng, n)
        jitter = rand(rng, n)
        design[:, j] = (perm .- jitter) ./ n
    end
    return design
end

"""
    config_distribution_map(cfg::ERPGen.GenerationConfig)

Map each parameter key to its current distribution in `cfg`. Used to read a
configuration back into a flat parameter vector.

# Arguments
- `cfg::ERPGen.GenerationConfig`: configuration to inspect.

# Returns
- `Dict{Symbol,Distribution}`: distribution per parameter key.
"""
function config_distribution_map(cfg::ERPGen.GenerationConfig)
    return Dict{Symbol, Distributions.Distribution}(
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
        :tilted_bar_hanning_length => cfg.components.tilted_bar_hanning_length_dist,
        :one_sided_fan_duration_divisor => cfg.components.one_sided_fan_duration_divisor_dist,
        :one_sided_fan_log_mu_offset => cfg.components.one_sided_fan_log_mu_offset_dist,
        :one_sided_fan_log_sigma => cfg.components.one_sided_fan_log_sigma_dist,
        :one_sided_fan_support_max => cfg.components.one_sided_fan_support_max_dist,
        :noise_pink => cfg.noise.noiselevel_dists[ERPGen.PinkNoise],
        :noise_white => cfg.noise.noiselevel_dists[ERPGen.WhiteNoise],
        :noise_red => cfg.noise.noiselevel_dists[ERPGen.RedNoise],
        :noise_exponential => cfg.noise.noiselevel_dists[ERPGen.ExponentialNoise],
    )
end

"""
    parameter_vector_from_cfg(base_cfg, cfg) -> Vector{Float64}

Read configuration `cfg` back into a flat 48-element parameter vector, ordered
the same way as [`parameter_symbols`](@ref).

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: defines parameter order via its specs.
- `cfg::ERPGen.GenerationConfig`: configuration whose values are extracted.

# Returns
- `Vector{Float64}`: the 48 mean/std values of `cfg`.
"""
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
