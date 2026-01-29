# Clone noise settings while overriding noiselevel when supported.
function noise_with_level(noise, noiselevel::Real)
    if hasproperty(noise, :noiselevel)
        fields = fieldnames(typeof(noise))
        kwargs = (; (f => getfield(noise, f) for f in fields if f != :noiselevel)..., noiselevel = noiselevel)
        return typeof(noise)(; kwargs...)
    end
    return noise
end

# Simulate ERP patterns and return data, events, and parameter metadata.
function simulate_pattern_trials(rng::AbstractRNG, mu::Real, sigma::Real,
        n_trials::Int, sampling_rate::Int, epoch_duration_s::Real,
        p100_width_dist::Distribution,
        p100_offset_dist::Distribution,
        p300_width_dist::Distribution,
        p300_offset_dist::Distribution,
        n170_width_dist::Distribution,
        n170_offset_dist::Distribution,
        p1_beta_dist::Distribution,
        p3_beta_dist::Distribution,
        n1_beta1_dist::Distribution,
        n1_beta2_dist::Distribution,
        n1_beta3_dist::Distribution,
        componentA_amp_dist::Distribution,
        componentB_amp_dist::Distribution,
        componentC_amp_dist::Distribution,
        noise::NoiseConfig,
        patterns::AbstractVector{Symbol},
        covariate_dists::AbstractDict{Symbol, <:Distribution})

    ensure_latest_unfoldsim!(propagate = false)

    patterns = collect(patterns)
    covariates = covariates_for_patterns(patterns, covariate_dists)
    conditions = Dict{Symbol, Any}()
    if :diverging_bar in patterns
        conditions[PATTERN_CATEGORICALS[:diverging_bar]] = ["car", "face"]
    end

    base_design = isempty(conditions) ? nothing : SingleSubjectDesign(; 
        conditions = conditions,
        event_order_function = (rng, x) -> shuffle(rng, x),
    )

    n_trials = max(100, n_trials)
    n_trials += isodd(n_trials) ? 1 : 0
    design = CovariateDesign(; design = base_design, n_trials = n_trials, covariates = covariates)

    sr = max(1, sampling_rate)
    epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))
    signal_len = max(2, Int(round(epoch_duration_s * sr)) + 1)

    min_sample = 1 / sr

    p100_width = max(min_sample, rand(rng, p100_width_dist))
    p100_offset = rand(rng, p100_offset_dist)
    min_p100_offset = (p100_width / 2) + min_sample
    p100_offset = max(p100_offset, min_p100_offset)

    p300_width = max(min_sample, rand(rng, p300_width_dist))
    p300_offset = rand(rng, p300_offset_dist)
    min_p300_offset = (p300_width / 2) + min_sample
    p300_offset = max(p300_offset, min_p300_offset)

    n170_width = max(min_sample, rand(rng, n170_width_dist))
    n170_offset = rand(rng, n170_offset_dist)
    min_n170_offset = (n170_width / 2) + min_sample
    n170_offset = max(n170_offset, min_n170_offset)

    p1_beta = rand(rng, p1_beta_dist)
    p1_basis = UnfoldSim.hanning(p100_width, p100_offset, sr)
    p1 = LinearModelComponent(; basis = p1_basis, formula = @formula(0 ~ 1), β = [p1_beta])

    has_diverging = :diverging_bar in patterns
    has_hourglass = :hourglass in patterns

    n1_beta1 = rand(rng, n1_beta1_dist)
    n1_beta2 = has_diverging ? rand(rng, n1_beta2_dist) : NaN
    n1_beta3 = has_hourglass ? rand(rng, n1_beta3_dist) : NaN
    n1_basis = -UnfoldSim.hanning(n170_width, n170_offset, sr)

    n1_betas_vec = Float64[n1_beta1]
    if has_diverging
        push!(n1_betas_vec, n1_beta2)
    end
    if has_hourglass
        push!(n1_betas_vec, n1_beta3)
    end

    if has_diverging && has_hourglass
        n1_formula = @formula(0 ~ 1 + diverging_bar_condition + hourglass_continuous)
    elseif has_diverging
        n1_formula = @formula(0 ~ 1 + diverging_bar_condition)
    elseif has_hourglass
        n1_formula = @formula(0 ~ 1 + hourglass_continuous)
    else
        n1_formula = @formula(0 ~ 1)
    end

    n1 = LinearModelComponent(; 
        basis = n1_basis,
        formula = n1_formula,
        β = n1_betas_vec,
    )

    p3_basis = UnfoldSim.hanning(p300_width, p300_offset, sr)
    p3_beta = rand(rng, p3_beta_dist)
    p3 = LinearModelComponent(; basis = p3_basis, formula = @formula(0 ~ 1), β = [p3_beta])

    components_vec = AbstractComponent[p1, n1, p3]

    if :one_sided_fan in patterns
        componentA_amp = rand(rng, componentA_amp_dist)
        componentA = TimeVaryingComponent(basis_one_sided_fan, signal_len, componentA_amp)
        push!(components_vec, componentA)
    end

    if :two_sided_fan in patterns
        componentB_amp = rand(rng, componentB_amp_dist)
        componentB = TimeVaryingComponent(basis_two_sided_fan, signal_len, componentB_amp)
        push!(components_vec, componentB)
    end

    if :tilted_bar in patterns
        componentC_amp = rand(rng, componentC_amp_dist)
        componentC = TimeVaryingComponent(basis_tilted_bar, signal_len, componentC_amp)
        push!(components_vec, componentC)
    end

    data, simulated_events = simulate(
        rng,
        design,
        components_vec,
        LogNormalOnset(; μ = Float64(mu), σ = Float64(sigma)),
        NoNoise(),
        return_epoched = true,
    )

    local_noise_pool = map(noise.noise_pool) do noise_inst
        if noise_inst isa ExponentialNoise
            return ExponentialNoise(τ = sr)
        end
        return noise_inst
    end

    noiselevels = Dict{Symbol, Float64}()
    for noise_inst in local_noise_pool
        noise_type = typeof(noise_inst)
        if !haskey(noise.noiselevel_dists, noise_type)
            throw(ArgumentError("Missing noiselevel_dists entry for $(noise_type)."))
        end
        dist = noise.noiselevel_dists[noise_type]
        noiselevel = max(0.0, rand(rng, dist))
        noiselevels[Symbol(nameof(noise_type))] = noiselevel
        noise_with = noise_with_level(noise_inst, noiselevel)
        for trial in axes(data, 2)
            data[:, trial] .+= UnfoldSim.simulate_noise(rng, noise_with, size(data, 1))
        end
    end

    simulated_events[!, DELTA_LATENCY] = vcat(diff(simulated_events.latency), 0)

    hanning_params = (
        p100_width = p100_width,
        p100_offset = p100_offset,
        p300_width = p300_width,
        p300_offset = p300_offset,
        n170_width = n170_width,
        n170_offset = n170_offset,
    )

    n1_beta_map = (
        baseline = Float64(n1_beta1),
        diverging_bar = has_diverging ? Float64(n1_beta2) : missing,
        hourglass = has_hourglass ? Float64(n1_beta3) : missing,
    )
    return (
        data = data,
        events = simulated_events,
        noiselevels = noiselevels,
        p1_beta = Float64(p1_beta),
        p3_beta = Float64(p3_beta),
        n1_betas = n1_beta_map,
        hanning_params = hanning_params,
    )
end

# Simulate ERP data (time x trials) with diagnostics.
function simulate_erp_trials(rng::AbstractRNG, mu::Real, sigma::Real,
        n_trials::Int, sampling_rate::Int, epoch_duration_s::Real,
        p100_width_dist::Distribution,
        p100_offset_dist::Distribution,
        p300_width_dist::Distribution,
        p300_offset_dist::Distribution,
        n170_width_dist::Distribution,
        n170_offset_dist::Distribution,
        p1_beta_dist::Distribution,
        p3_beta_dist::Distribution,
        n1_beta1_dist::Distribution,
        n1_beta2_dist::Distribution,
        n1_beta3_dist::Distribution,
        componentA_amp_dist::Distribution,
        componentB_amp_dist::Distribution,
        componentC_amp_dist::Distribution,
        noise::NoiseConfig,
        patterns::AbstractVector{Symbol},
        covariate_dists::AbstractDict{Symbol, <:Distribution})
    return diag_call(:simulate_pattern_trials) do
        with_logger(NullLogger()) do
            return simulate_pattern_trials(rng, mu, sigma, n_trials, sampling_rate, epoch_duration_s,
                p100_width_dist, p100_offset_dist, p300_width_dist, p300_offset_dist,
                n170_width_dist, n170_offset_dist, p1_beta_dist, p3_beta_dist,
                n1_beta1_dist, n1_beta2_dist, n1_beta3_dist,
                componentA_amp_dist, componentB_amp_dist, componentC_amp_dist,
                noise,
                patterns, covariate_dists)
        end
    end
end
