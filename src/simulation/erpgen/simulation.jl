# Clone noise settings while overriding noiselevel when supported.
function noise_with_level(noise, noiselevel::Real)
    return maybe_diag(:noise_with_level) do
        if hasproperty(noise, :noiselevel)
            fields = fieldnames(typeof(noise))
            kwargs = (; (f => getfield(noise, f) for f in fields if f != :noiselevel)..., noiselevel = noiselevel)
            return typeof(noise)(; kwargs...)
        end
        return noise
    end
end

# Simulate ERP patterns and return data, events, and parameter metadata.
function simulate_pattern_trials(rng::AbstractRNG, mu::Real, sigma::Real,
        n_trials::Int, sampling_rate::Int, epoch_duration_s::Real,
        p100_width_dist::Distribution,
        p100_window_offset_dist::Distribution,
        p100_n170_gap_dist::Distribution,
        n170_p300_gap_dist::Distribution,
        n170_width_dist::Distribution,
        p300_width_dist::Distribution,
        p1_beta_dist::Distribution,
        p3_beta_dist::Distribution,
        n1_beta1_dist::Distribution,
        n1_beta2_dist::Distribution,
        n1_beta3_dist::Distribution,
        componentA_amp_dist::Distribution,
        componentB_amp_dist::Distribution,
        componentC_amp_dist::Distribution,
        tilted_bar_hanning_length_dist::Distribution,
        one_sided_fan_duration_divisor_dist::Distribution,
        one_sided_fan_log_mu_offset_dist::Distribution,
        one_sided_fan_log_sigma_dist::Distribution,
        one_sided_fan_support_max_dist::Distribution,
        noise::NoiseConfig,
        loaded_patterns::AbstractVector{Symbol},
        covariate_dists::AbstractDict{Symbol, <:Distribution},
        diverging_bar_levels::AbstractVector{String} = ["car", "face"])
    return maybe_diag(:simulate_pattern_trials) do
        simulate_rng = fresh_rng(rng)
        loaded_patterns = unique(filter(!=(:no_class), collect(loaded_patterns)))
        covariates = covariates_for_patterns(loaded_patterns, covariate_dists)
        conditions = Dict{Symbol, Any}()
        if :diverging_bar in loaded_patterns
            conditions[PATTERN_CATEGORICALS[:diverging_bar]] = collect(diverging_bar_levels)
        end

        base_design = isempty(conditions) ? nothing : SingleSubjectDesign(;
            conditions = conditions,
            event_order_function = (_, x) -> time_seeded_shuffle(x),
        )

        n_trials = max(100, n_trials)
        n_trials += isodd(n_trials) ? 1 : 0
        design = CovariateDesign(; design = base_design, n_trials = n_trials, covariates = covariates)

        sr = max(1, sampling_rate)
        epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))
        signal_len = max(2, Int(round(epoch_duration_s * sr)) + 1)

        min_sample = 1 / sr
        min_width = (2 * min_sample) + eps(Float64(min_sample))

        sample_peak_offset(center, width) = begin
            peak_min = center - width / 2 + min_sample
            peak_max = center + width / 2 - min_sample
            # UnfoldSim.hanning requires offset > round((width + 1) / 2) in sample space.
            # We enforce this lower bound and then sample a relative peak position
            # via percent roll (5%-95%) inside the valid interval.
            required_min = (width / 2) + (2 * min_sample)
            lo = max(peak_min, required_min)
            hi = peak_max
            if lo < hi
                pct = time_seeded_rand(Uniform(0.05, 0.95))
                return lo + pct * (hi - lo)
            elseif isapprox(lo, hi; atol = eps(Float64))
                return lo
            end
            # Degenerate edge-case fallback: keep offset in the hanning window.
            return clamp(max(required_min, center), peak_min, peak_max)
        end

        p100_width = max(min_width, time_seeded_rand(p100_width_dist))
        p100_window_center = time_seeded_rand(p100_window_offset_dist)
        min_p100_window_center = (p100_width / 2) + min_sample
        p100_window_center = max(p100_window_center, min_p100_window_center)
        p100_offset = sample_peak_offset(p100_window_center, p100_width)

        # Draw component gaps from distributions instead of using hard-coded means.
        n170_width = max(min_width, time_seeded_rand(n170_width_dist))
        p100_n170_gap = max(min_sample, time_seeded_rand(p100_n170_gap_dist))
        n170_window_center = p100_window_center + p100_n170_gap
        min_n170_window_center = (n170_width / 2) + min_sample
        n170_window_center = max(n170_window_center, min_n170_window_center)
        n170_offset = sample_peak_offset(n170_window_center, n170_width)

        p300_width = max(min_width, time_seeded_rand(p300_width_dist))
        n170_p300_gap = max(min_sample, time_seeded_rand(n170_p300_gap_dist))
        p300_window_center = n170_window_center + n170_p300_gap
        min_p300_window_center = (p300_width / 2) + min_sample
        p300_window_center = max(p300_window_center, min_p300_window_center)
        p300_offset = sample_peak_offset(p300_window_center, p300_width)

        p1_beta = time_seeded_rand(p1_beta_dist)
        p1_basis = UnfoldSim.hanning(p100_width, p100_offset, sr)
        p1 = LinearModelComponent(; basis = p1_basis, formula = @formula(0 ~ 1), β = [p1_beta])

        has_diverging = :diverging_bar in loaded_patterns
        has_hourglass = :hourglass in loaded_patterns

        n1_beta1 = time_seeded_rand(n1_beta1_dist)
        n1_beta2 = has_diverging ? time_seeded_rand(n1_beta2_dist) : NaN
        n1_beta3 = has_hourglass ? time_seeded_rand(n1_beta3_dist) : NaN
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
        p3_beta = time_seeded_rand(p3_beta_dist)
        p3 = LinearModelComponent(; basis = p3_basis, formula = @formula(0 ~ 1), β = [p3_beta])

        components_vec = AbstractComponent[p1, n1, p3]
        componentA_amp = missing
        componentB_amp = missing
        componentC_amp = missing
        tilted_bar_hanning_length = missing
        one_sided_fan_duration_divisor = missing
        one_sided_fan_log_mu_offset = missing
        one_sided_fan_log_sigma = missing
        one_sided_fan_support_max = missing

        if :one_sided_fan in loaded_patterns
            componentA_amp = Float64(time_seeded_rand(componentA_amp_dist))
            one_sided_fan_duration_divisor = max(sqrt(eps(Float64)),
                Float64(time_seeded_rand(one_sided_fan_duration_divisor_dist)))
            one_sided_fan_log_mu_offset = Float64(time_seeded_rand(one_sided_fan_log_mu_offset_dist))
            one_sided_fan_log_sigma = max(sqrt(eps(Float64)),
                Float64(time_seeded_rand(one_sided_fan_log_sigma_dist)))
            one_sided_fan_support_max = max(sqrt(eps(Float64)),
                Float64(time_seeded_rand(one_sided_fan_support_max_dist)))
            componentA = TimeVaryingComponent(
                (evts, maxlength) -> basis_one_sided_fan(evts, maxlength;
                    duration_divisor = one_sided_fan_duration_divisor,
                    log_mu_offset = one_sided_fan_log_mu_offset,
                    log_sigma = one_sided_fan_log_sigma,
                    support_max = one_sided_fan_support_max),
                signal_len,
                componentA_amp,
            )
            push!(components_vec, componentA)
        end

        if :two_sided_fan in loaded_patterns
            componentB_amp = Float64(time_seeded_rand(componentB_amp_dist))
            componentB = TimeVaryingComponent(basis_two_sided_fan, signal_len, componentB_amp)
            push!(components_vec, componentB)
        end

        if :tilted_bar in loaded_patterns
            componentC_amp = Float64(time_seeded_rand(componentC_amp_dist))
            tilted_bar_hanning_length = max(2,
                Int(round(time_seeded_rand(tilted_bar_hanning_length_dist))))
            componentC = TimeVaryingComponent(
                evts -> basis_tilted_bar(evts; window_length = tilted_bar_hanning_length),
                signal_len,
                componentC_amp,
            )
            push!(components_vec, componentC)
        end

        data, simulated_events = maybe_diag(:simulate_unfoldsim) do
            simulate(
                simulate_rng,
                design,
                components_vec,
                LogNormalOnset(; μ = Float64(mu), σ = Float64(sigma)),
                NoNoise(),
                return_epoched = true,
            )
        end

        # Enforce deterministic epoch length from sampling_rate * epoch_duration_s.
        # UnfoldSim's returned epoched length can vary with component setup; for
        # downstream ERP image comparability we pad/truncate to `signal_len`.
        current_len = size(data, 1)
        if current_len != signal_len
            ntr = size(data, 2)
            if current_len < signal_len
                padded = zeros(eltype(data), signal_len, ntr)
                padded[1:current_len, :] .= data
                data = padded
            else
                data = data[1:signal_len, :]
            end
        end

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
            noiselevel = max(0.0, time_seeded_rand(dist))
            noiselevels[Symbol(nameof(noise_type))] = noiselevel
            noise_with = noise_with_level(noise_inst, noiselevel)
            for trial in axes(data, 2)
                data[:, trial] .+= UnfoldSim.simulate_noise(fresh_rng(), noise_with, size(data, 1))
            end
        end

        simulated_events[!, DELTA_LATENCY] = vcat(diff(simulated_events.latency), 0)

        hanning_params = (
            p100_width = p100_width,
            p100_window_center = p100_window_center,
            p100_offset = p100_offset,
            p100_n170_gap = p100_n170_gap,
            n170_width = n170_width,
            n170_window_center = n170_window_center,
            n170_offset = n170_offset,
            n170_p300_gap = n170_p300_gap,
            p300_width = p300_width,
            p300_window_center = p300_window_center,
            p300_offset = p300_offset,
        )

        n1_beta_map = (
            baseline = Float64(n1_beta1),
            diverging_bar = has_diverging ? Float64(n1_beta2) : missing,
            hourglass = has_hourglass ? Float64(n1_beta3) : missing,
        )
        component_amps = (
            componentA_amp = componentA_amp,
            componentB_amp = componentB_amp,
            componentC_amp = componentC_amp,
        )
        basis_shape_params = (
            tilted_bar_hanning_length = tilted_bar_hanning_length,
            one_sided_fan_duration_divisor = one_sided_fan_duration_divisor,
            one_sided_fan_log_mu_offset = one_sided_fan_log_mu_offset,
            one_sided_fan_log_sigma = one_sided_fan_log_sigma,
            one_sided_fan_support_max = one_sided_fan_support_max,
        )
        return (
            data = data,
            events = simulated_events,
            noiselevels = noiselevels,
            p1_beta = Float64(p1_beta),
            p3_beta = Float64(p3_beta),
            n1_betas = n1_beta_map,
            hanning_params = hanning_params,
            component_amps = component_amps,
            basis_shape_params = basis_shape_params,
        )
    end
end

# Simulate ERP data (time x trials) with diagnostics.
function simulate_erp_trials(rng::AbstractRNG, mu::Real, sigma::Real,
        n_trials::Int, sampling_rate::Int, epoch_duration_s::Real,
        p100_width_dist::Distribution,
        p100_window_offset_dist::Distribution,
        p100_n170_gap_dist::Distribution,
        n170_p300_gap_dist::Distribution,
        n170_width_dist::Distribution,
        p300_width_dist::Distribution,
        p1_beta_dist::Distribution,
        p3_beta_dist::Distribution,
        n1_beta1_dist::Distribution,
        n1_beta2_dist::Distribution,
        n1_beta3_dist::Distribution,
        componentA_amp_dist::Distribution,
        componentB_amp_dist::Distribution,
        componentC_amp_dist::Distribution,
        tilted_bar_hanning_length_dist::Distribution,
        one_sided_fan_duration_divisor_dist::Distribution,
        one_sided_fan_log_mu_offset_dist::Distribution,
        one_sided_fan_log_sigma_dist::Distribution,
        one_sided_fan_support_max_dist::Distribution,
        noise::NoiseConfig,
        loaded_patterns::AbstractVector{Symbol},
        covariate_dists::AbstractDict{Symbol, <:Distribution},
        diverging_bar_levels::AbstractVector{String} = ["car", "face"])
    return maybe_diag(:simulate_erp_trials) do
        with_logger(NullLogger()) do
            return simulate_pattern_trials(rng, mu, sigma, n_trials, sampling_rate, epoch_duration_s,
                p100_width_dist, p100_window_offset_dist, p100_n170_gap_dist, n170_p300_gap_dist,
                n170_width_dist, p300_width_dist, p1_beta_dist, p3_beta_dist,
                n1_beta1_dist, n1_beta2_dist, n1_beta3_dist,
                componentA_amp_dist, componentB_amp_dist, componentC_amp_dist,
                tilted_bar_hanning_length_dist,
                one_sided_fan_duration_divisor_dist,
                one_sided_fan_log_mu_offset_dist,
                one_sided_fan_log_sigma_dist,
                one_sided_fan_support_max_dist,
                noise,
                loaded_patterns, covariate_dists, diverging_bar_levels)
        end
    end
end
