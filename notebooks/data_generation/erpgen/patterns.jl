# Clone noise settings while overriding noiselevel when supported.
function _with_noiselevel(noise, noiselevel::Real)
    if hasproperty(noise, :noiselevel)
        fields = fieldnames(typeof(noise))
        kwargs = (; (f => getfield(noise, f) for f in fields if f != :noiselevel)..., noiselevel = noiselevel)
        return typeof(noise)(; kwargs...)
    end
    return noise
end

# Simulate six ERP patterns and return raw time x trials data.
function simulate_6patterns(mu = 3.2, sigma = 0.5; epoch_duration_s::Real = 1.0, sampling_rate::Real = 100,
        condition_levels::Int = 8, noise_pool = DEFAULT_NOISE_POOL,
        noiselevel_dists = DEFAULT_NOISELEVEL_DISTS,
        p100_width_dist::Distribution = Normal(0.1, 0.015),
        p100_offset_dist::Distribution = Normal(0.1, 0.015),
        p300_width_dist::Distribution = Normal(0.3, 0.045),
        p300_offset_dist::Distribution = Normal(0.3, 0.045),
        n170_width_dist::Distribution = Normal(0.15, 0.0225),
        n170_offset_dist::Distribution = Normal(0.17, 0.0255),
        p1_beta_dist::Distribution = Normal(5.0, 1.0),
        p3_beta_dist::Distribution = Normal(5.0, 0.75),
        n1_beta1_dist::Distribution = Normal(5.0, 0.75),
        n1_beta2_dist::Distribution = Normal(3.0, 0.45),
        n1_beta3_dist::Distribution = Normal(2.0, 0.3),
        componentA_amp_dist::Distribution = Normal(5.0, 1.0),
        componentB_amp_dist::Distribution = Normal(-10.0, 1.0),
        componentC_amp_dist::Distribution = Normal(5.0, 1.0),
        one_sided_fan_duration_start_dist::Distribution = Normal(20.0, 3.0),
        one_sided_fan_duration_end_dist::Distribution = Normal(100.0, 15.0),
        two_sided_fan_duration_start_dist::Distribution = Normal(10.0, 1.5),
        two_sided_fan_duration_end_dist::Distribution = Normal(30.0, 4.5),
        tilted_bar_duration_start_dist::Distribution = Normal(5.0, 0.75),
        tilted_bar_duration_end_dist::Distribution = Normal(40.0, 6.0),
        hourglass_continuous_start_dist::Distribution = Normal(-2.0, 0.3),
        hourglass_continuous_end_dist::Distribution = Normal(2.0, 0.3),
        rng = MersenneTwister(time_ns()))

    ensure_latest_unfoldsim!(propagate = false)

    sr = max(1, sampling_rate)
    epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))
    signal_len = max(2, Int(round(epoch_duration_s * sr)) + 1)

    min_sample = 1 / sr

    p100_width = max(min_sample, rand(Random.Xoshiro(time_ns()), p100_width_dist))
    p100_offset = rand(Random.Xoshiro(time_ns()), p100_offset_dist)
    min_p100_offset = (p100_width / 2) + min_sample
    p100_offset = max(p100_offset, min_p100_offset)

    p300_width = max(min_sample, rand(Random.Xoshiro(time_ns()), p300_width_dist))
    p300_offset = rand(Random.Xoshiro(time_ns()), p300_offset_dist)
    min_p300_offset = (p300_width / 2) + min_sample
    p300_offset = max(p300_offset, min_p300_offset)

    n170_width = max(min_sample, rand(Random.Xoshiro(time_ns()), n170_width_dist))
    n170_offset = rand(Random.Xoshiro(time_ns()), n170_offset_dist)
    min_n170_offset = (n170_width / 2) + min_sample
    n170_offset = max(n170_offset, min_n170_offset)

    one_sided_fan_duration_start = rand(Random.Xoshiro(time_ns()), one_sided_fan_duration_start_dist)
    one_sided_fan_duration_end = rand(Random.Xoshiro(time_ns()), one_sided_fan_duration_end_dist)

    two_sided_fan_duration_start = rand(Random.Xoshiro(time_ns()), two_sided_fan_duration_start_dist)
    two_sided_fan_duration_end = rand(Random.Xoshiro(time_ns()), two_sided_fan_duration_end_dist)

    tilted_bar_duration_start = rand(Random.Xoshiro(time_ns()), tilted_bar_duration_start_dist)
    tilted_bar_duration_end = rand(Random.Xoshiro(time_ns()), tilted_bar_duration_end_dist)

    hourglass_continuous_start = rand(Random.Xoshiro(time_ns()), hourglass_continuous_start_dist)
    hourglass_continuous_end = rand(Random.Xoshiro(time_ns()), hourglass_continuous_end_dist)

    one_sided_fan_duration_vals = range(one_sided_fan_duration_start, one_sided_fan_duration_end,
        length = condition_levels)
    two_sided_fan_duration_vals = range(two_sided_fan_duration_start, two_sided_fan_duration_end,
        length = condition_levels)
    tilted_bar_duration_vals = range(tilted_bar_duration_start, tilted_bar_duration_end,
        length = condition_levels)
    hourglass_continuous_vals = range(hourglass_continuous_start, hourglass_continuous_end,
        length = condition_levels)

    design = SingleSubjectDesign(; 
        conditions = Dict(
            :diverging_bar_condition => ["car", "face"],
            :hourglass_continuous => hourglass_continuous_vals,
            :one_sided_fan_duration => one_sided_fan_duration_vals,
            :two_sided_fan_duration => two_sided_fan_duration_vals,
            :tilted_bar_duration => tilted_bar_duration_vals,
        ),
        event_order_function = (rng, x) -> shuffle(rng, x),
    )

    p1_beta = rand(Random.Xoshiro(time_ns()), p1_beta_dist)
    p1_basis = UnfoldSim.hanning(p100_width, p100_offset, sr)
    p1 = LinearModelComponent(; basis = p1_basis, formula = @formula(0 ~ 1), β = [p1_beta])

    n1_beta1 = rand(Random.Xoshiro(time_ns()), n1_beta1_dist)
    n1_beta2 = rand(Random.Xoshiro(time_ns()), n1_beta2_dist)
    n1_beta3 = rand(Random.Xoshiro(time_ns()), n1_beta3_dist)
    n1_basis = -UnfoldSim.hanning(n170_width, n170_offset, sr)
    n1 = LinearModelComponent(; 
        basis = n1_basis,
        formula = @formula(0 ~ 1 + diverging_bar_condition + hourglass_continuous),
        β = [n1_beta1, n1_beta2, n1_beta3],
    )

    p3_basis = UnfoldSim.hanning(p300_width, p300_offset, sr)
    p3_beta = rand(Random.Xoshiro(time_ns()), p3_beta_dist)
    p3 = LinearModelComponent(; basis = p3_basis, formula = @formula(0 ~ 1), β = [p3_beta])

    componentA_amp = rand(Random.Xoshiro(time_ns()), componentA_amp_dist)
    componentB_amp = rand(Random.Xoshiro(time_ns()), componentB_amp_dist)
    componentC_amp = rand(Random.Xoshiro(time_ns()), componentC_amp_dist)
    componentA = TimeVaryingComponent(basis_lognormal, signal_len, componentA_amp)
    componentB = TimeVaryingComponent(basis_hanning, signal_len, componentB_amp)
    componentC = TimeVaryingComponent(basis_linear, signal_len, componentC_amp)

    rng = Random.Xoshiro(time_ns())
    data, simulated_events = simulate(
        rng,
        design,
        [p1, n1, p3, componentA, componentB, componentC],
        LogNormalOnset(; μ = mu, σ = sigma),
        NoNoise(),
        return_epoched = true,
    )

    noiselevels = Dict{Symbol, Float64}()
    for noise in noise_pool
        noise_type = typeof(noise)
        if !haskey(noiselevel_dists, noise_type)
            throw(ArgumentError("Missing noiselevel_dists entry for $(noise_type)."))
        end
        dist = noiselevel_dists[noise_type]
        noiselevel = max(0.0, rand(Random.Xoshiro(time_ns()), dist))
        noiselevels[Symbol(nameof(typeof(noise)))] = noiselevel
        noise_inst = _with_noiselevel(noise, noiselevel)
        for trial in axes(data, 2)
            data[:, trial] .+= UnfoldSim.simulate_noise(Random.Xoshiro(time_ns()), noise_inst, size(data, 1))
        end
    end

    simulated_events[!, DELTA_LATENCY] = vcat(diff(simulated_events.latency), 0)
    sim_6patterns = data
    return sim_6patterns, simulated_events, noiselevels, p1_beta, p3_beta, (n1_beta1, n1_beta2, n1_beta3),
           (p100_width = p100_width, p100_offset = p100_offset,
            p300_width = p300_width, p300_offset = p300_offset,
            n170_width = n170_width, n170_offset = n170_offset)
end

const PATTERN_NAMES = [:sigmoid, :one_sided_fan, :two_sided_fan, :diverging_bar, :hourglass, :tilted_bar, :no_class]
const VARIANT_SPECS = (
    (name = :normal, trial_order = :normal, inverted = false),
    (name = :reversed, trial_order = :reversed, inverted = false),
    (name = :inverted, trial_order = :normal, inverted = true),
    (name = :reversed_inverted, trial_order = :reversed, inverted = true),
)
const VARIANT_NAMES = ntuple(i -> VARIANT_SPECS[i].name, length(VARIANT_SPECS))
const VARIANT_COUNT = length(VARIANT_SPECS)

const SORTERS = Dict{Symbol, Function}(
    :sigmoid => evts -> collect(zip(evts[!, DELTA_LATENCY], evts.latency)),
    :one_sided_fan => evts -> evts.one_sided_fan_duration,
    :two_sided_fan => evts -> evts.two_sided_fan_duration,
    :diverging_bar => evts -> evts.diverging_bar_condition .== "car",
    :hourglass => evts -> evts.hourglass_continuous,
    :tilted_bar => evts -> evts.tilted_bar_duration,
    :no_class => _ -> nothing,
)

@inline function sortvalues_for_pattern(pname::Symbol, evts, rng::AbstractRNG)
    if pname === :no_class
        # Randomize no_class trial order explicitly at the source.
        return rand(Random.Xoshiro(time_ns()), size(evts, 1))
    end
    return SORTERS[pname](evts)
end
