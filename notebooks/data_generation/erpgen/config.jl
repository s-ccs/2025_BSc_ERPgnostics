const DEFAULT_NOISELEVEL_DIST = Normal(15, 2)
const DEFAULT_NOISELEVEL_DISTS = Dict(
    PinkNoise => DEFAULT_NOISELEVEL_DIST,
    WhiteNoise => DEFAULT_NOISELEVEL_DIST,
    RedNoise => DEFAULT_NOISELEVEL_DIST,
    ExponentialNoise => DEFAULT_NOISELEVEL_DIST,
)
const DEFAULT_CROP_START_DIST = Normal(10, 25)
const DEFAULT_CROP_END_DIST = Normal(10, 25)
const DEFAULT_DROPOUT_RATE_DIST = Normal(10, 250)
const DEFAULT_N_TRIALS_DIST = Normal(2500, 250)
const DEFAULT_NOISE_POOL = [PinkNoise(), WhiteNoise(), RedNoise(), ExponentialNoise(τ = 350)]
const FILTER_BORDER = "reflect"

const VARIANT_SPECS = (
    (name = :normal, trial_order = :normal, inverted = false),
    (name = :reversed, trial_order = :reversed, inverted = false),
    (name = :inverted, trial_order = :normal, inverted = true),
    (name = :reversed_inverted, trial_order = :reversed, inverted = true),
)
const VARIANT_NAMES = ntuple(i -> VARIANT_SPECS[i].name, length(VARIANT_SPECS))
const VARIANT_COUNT = length(VARIANT_SPECS)

const RESIZE_METHOD_SPECS = [
    (name = :nearest,
        method = Interpolations.Constant(),
        params = "none",
        notes = "Nearest-neighbor (piecewise constant). Equivalent to BSpline(Constant())."),
    (name = :linear,
        method = Interpolations.Linear(),
        params = "none",
        notes = "Linear/bilinear interpolation. Equivalent to BSpline(Linear())."),
    (name = :quadratic_line_ongrid,
        method = Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid())),
        params = "bc::BoundaryCondition (Flat/Line/Free/Reflect/Periodic/Throw) + gridstyle (OnGrid/OnCell)",
        notes = "Quadratic B-spline with boundary condition and gridstyle."),
    (name = :cubic_line_ongrid,
        method = Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())),
        params = "bc::BoundaryCondition (Flat/Line/Free/Reflect/Periodic/Throw) + gridstyle (OnGrid/OnCell)",
        notes = "Cubic B-spline with boundary condition and gridstyle."),
    (name = :lanczos4_opencv,
        method = Lanczos4OpenCV(),
        params = "none",
        notes = "Lanczos 4 windowed-sinc interpolation (OpenCV compatible)."),
]
const DEFAULT_RESIZE_METHODS = map(spec -> spec.method, RESIZE_METHOD_SPECS)

Base.@kwdef struct SimulationConfig
    mu_dist::Distribution = Normal(3.2, 0.3)
    sigma_dist::Distribution = Normal(0.5, 0.1)
    epoch_duration_dist::Distribution = Normal(1.0, 0.25)
    sampling_rate_dist::Distribution = Normal(350, 5)
    n_trials_dist::Distribution = DEFAULT_N_TRIALS_DIST
end

Base.@kwdef struct ComponentConfig
    p100_width_dist::Distribution = Normal(0.1, 0.015)
    p100_offset_dist::Distribution = Normal(0.1, 0.015)
    p100_n170_gap_dist::Distribution = Normal(0.07, 0.015)
    n170_p300_gap_dist::Distribution = Normal(0.13, 0.02)
    p300_width_dist::Distribution = Normal(0.3, 0.045)
    n170_width_dist::Distribution = Normal(0.15, 0.0225)
    p1_beta_dist::Distribution = Normal(5.0, 1.0)
    p3_beta_dist::Distribution = Normal(5.0, 0.75)
    n1_beta1_dist::Distribution = Normal(5.0, 0.75)
    n1_beta2_dist::Distribution = Normal(3.0, 0.45)
    n1_beta3_dist::Distribution = Normal(2.0, 0.3)
    componentA_amp_dist::Distribution = Normal(5.0, 1.0)
    componentB_amp_dist::Distribution = Normal(-10.0, 1.0)
    componentC_amp_dist::Distribution = Normal(5.0, 1.0)
end

function default_pattern_covariates()
    return maybe_diag(:default_pattern_covariates) do
        # ERPgnostics mapping: duration -> one_sided_fan_duration, durationB -> two_sided_fan_duration,
        # duration_linear -> tilted_bar_duration, continuous -> hourglass_continuous.
        return Dict{Symbol, Distribution}(
            :one_sided_fan_duration => Uniform(20.0, 100.0),
            :two_sided_fan_duration => Uniform(10.0, 30.0),
            :tilted_bar_duration => Uniform(5.0, 40.0),
            :hourglass_continuous => Uniform(-2.0, 2.0),
        )
    end
end

const DEFAULT_PATTERN_LIST = [:sigmoid, :one_sided_fan, :two_sided_fan, :diverging_bar, :hourglass, :tilted_bar]

Base.@kwdef struct PatternConfig
    patterns::Vector{Symbol} = DEFAULT_PATTERN_LIST
    covariate_dists::Dict{Symbol, Distribution} = default_pattern_covariates()
    diverging_bar_levels::Vector{String} = ["car", "face"]
end

Base.@kwdef struct NoiseConfig
    noise_pool::AbstractVector{<:UnfoldSim.AbstractNoise} = DEFAULT_NOISE_POOL
    noiselevel_dists::AbstractDict{<:DataType, <:Distribution} = DEFAULT_NOISELEVEL_DISTS
end

Base.@kwdef struct ProcessingConfig
    dropout_trials_rate_dist::Distribution = DEFAULT_DROPOUT_RATE_DIST
    crop_start_dist = DEFAULT_CROP_START_DIST
    crop_end_dist = DEFAULT_CROP_END_DIST
    zscore_timepoints::Bool = true
    resize_antialias::Bool = true
    low_pass_factor::Real = 0.75
    resize_method = Interpolations.Linear()
    target_height::Int = 64
    target_width::Int = 64
end

Base.@kwdef struct RuntimeConfig
    threaded::Bool = false
    show_progress::Bool = true
    blas_threads::Int = 1
    progress_every::Int = 10
end

Base.@kwdef struct GenerationConfig
    sim::SimulationConfig = SimulationConfig()
    components::ComponentConfig = ComponentConfig()
    patterns::PatternConfig = PatternConfig()
    noise::NoiseConfig = NoiseConfig()
    processing::ProcessingConfig = ProcessingConfig()
    runtime::RuntimeConfig = RuntimeConfig()
end

UnfoldSim.@with_kw mutable struct CovariateDesign <: UnfoldSim.AbstractDesign
    design = nothing
    n_trials::Int
    covariates::Dict{Symbol, Any}
    events_cache = nothing
end

function UnfoldSim.size(design::CovariateDesign)
    return maybe_diag(:CovariateDesign_size) do
        return design.n_trials
    end
end

function UnfoldSim.generate_events(rng::UnfoldSim.AbstractRNG, design::CovariateDesign)
    return maybe_diag(:generate_events) do
        if design.events_cache !== nothing
            return design.events_cache
        end

        all_evts = Pair{Symbol, Any}[]
        for (covariate, dist) in design.covariates
            push!(all_evts, covariate => rand(rng, dist, design.n_trials))
        end

        if design.design === nothing
            if isempty(all_evts)
                design.events_cache = UnfoldSim.DataFrame(trial_index = collect(1:design.n_trials))
                return design.events_cache
            end
            design.events_cache = UnfoldSim.DataFrame(all_evts)
            return design.events_cache
        end

        base_size = size(design.design)[1]
        n_rep = design.n_trials / base_size
        if n_rep != floor(n_rep)
            error("design.n_trials need to be divisible by size(design.design)")
        end

        categorical_events = generate_events(deepcopy(rng), RepeatDesign(design.design, Int(n_rep)))
        if isempty(all_evts)
            design.events_cache = categorical_events
            return design.events_cache
        end
        covariate_events = UnfoldSim.DataFrame(all_evts)
        design.events_cache = hcat(categorical_events, covariate_events)
        return design.events_cache
    end
end
