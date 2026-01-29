const PATTERN_NAMES = [:sigmoid, :one_sided_fan, :two_sided_fan, :diverging_bar, :hourglass, :tilted_bar, :no_class]
const VARIANT_SPECS = (
    (name = :normal, trial_order = :normal, inverted = false),
    (name = :reversed, trial_order = :reversed, inverted = false),
    (name = :inverted, trial_order = :normal, inverted = true),
    (name = :reversed_inverted, trial_order = :reversed, inverted = true),
)
const VARIANT_NAMES = ntuple(i -> VARIANT_SPECS[i].name, length(VARIANT_SPECS))
const VARIANT_COUNT = length(VARIANT_SPECS)

const PATTERN_COVARIATES = Dict{Symbol, Symbol}(
    :one_sided_fan => :one_sided_fan_duration,
    :two_sided_fan => :two_sided_fan_duration,
    :tilted_bar => :tilted_bar_duration,
    :hourglass => :hourglass_continuous,
)

const PATTERN_CATEGORICALS = Dict{Symbol, Symbol}(
    :diverging_bar => :diverging_bar_condition,
)

const SORTERS = Dict{Symbol, Function}(
    :sigmoid => evts -> collect(zip(evts[!, DELTA_LATENCY], evts.latency)),
    :one_sided_fan => evts -> evts.one_sided_fan_duration,
    :two_sided_fan => evts -> evts.two_sided_fan_duration,
    :diverging_bar => evts -> evts.diverging_bar_condition .== "car",
    :hourglass => evts -> evts.hourglass_continuous,
    :tilted_bar => evts -> evts.tilted_bar_duration,
    :no_class => _ -> nothing,
)

function covariates_for_patterns(patterns::AbstractVector{Symbol},
        covariate_dists::AbstractDict{Symbol, <:Distribution})
    covariates = Dict{Symbol, Distribution}()
    for pname in patterns
        cov_name = get(PATTERN_COVARIATES, pname, nothing)
        cov_name === nothing && continue
        if !haskey(covariate_dists, cov_name)
            throw(ArgumentError("Missing covariate distribution for $(cov_name) required by pattern $(pname)."))
        end
        covariates[cov_name] = covariate_dists[cov_name]
    end
    return covariates
end

@inline function pattern_sort_values(pname::Symbol, evts, rng::AbstractRNG)
    if pname === :no_class
        # Randomize no_class trial order explicitly at the source.
        return rand(rng, size(evts, 1))
    end
    return SORTERS[pname](evts)
end
