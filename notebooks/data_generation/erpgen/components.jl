# Time-varying component wrapper for UnfoldSim.
struct TimeVaryingComponent <: AbstractComponent
    basisfunction::Any
    maxlength::Any
    beta::Any
end

# Use maxlength as component length.
function Base.length(c::TimeVaryingComponent)
    return maybe_diag(:TimeVaryingComponent_length) do
        return c.maxlength
    end
end

# Simulate a time-varying component with explicit RNG.
function UnfoldSim.simulate_component(rng, c::TimeVaryingComponent, design::AbstractDesign)
    return maybe_diag(:simulate_component_rng) do
        evts = generate_events(fresh_rng(rng), design)
        data = if applicable(c.basisfunction, evts, c.maxlength)
            c.beta .* c.basisfunction(evts, c.maxlength)
        else
            c.beta .* c.basisfunction(evts)
        end
        return normalize_basis_length(data, c.maxlength)
    end
end

# Simulate a time-varying component with default RNG.
function UnfoldSim.simulate_component(c::TimeVaryingComponent, design::AbstractDesign; rng = fresh_rng())
    return maybe_diag(:simulate_component_default) do
        return UnfoldSim.simulate_component(rng, c, design)
    end
end

# Generate a linear ERP basis (tilted bar).
function basis_tilted_bar(evts)
    return maybe_diag(:basis_tilted_bar) do
        shifts = -round.(Int, evts.tilted_bar_duration)
        basis = pad_array.(Ref(UnfoldSim.DSP.hanning(50)), shifts, 0)
        return basis
    end
end

# Generate a lognormal ERP basis (one-sided fan).
function basis_one_sided_fan(evts, maxlength)
    return maybe_diag(:basis_one_sided_fan) do
        basis = pdf.(LogNormal.(evts.one_sided_fan_duration ./ 40 .- 0.2, 1),
            Ref(range(0, 10, length = maxlength)))
        basis = basis ./ maximum.(basis)
        return basis
    end
end

# Generate a hanning ERP basis (two-sided fan).
function basis_two_sided_fan(evts)
    return maybe_diag(:basis_two_sided_fan) do
        maxdur = maximum(evts.two_sided_fan_duration)

        basis = UnfoldSim.DSP.hanning.(Int.(round.(evts.two_sided_fan_duration)))
        shifts = Int.(.-round.(maxdur .- evts.two_sided_fan_duration) .÷ 2)
        basis = pad_array.(basis, shifts, 0)
        return basis
    end
end

# Ensure all basis functions share the same length.
function normalize_basis_length(basis, maxlength)
    return maybe_diag(:normalize_basis_length) do
        difftomax = maxlength .- length.(basis)
        if any(difftomax .< 0)
            @warn "Basis longer than maxlength in at least one case. Either increase maxlength or redefine function. Attempt to truncate the basis"
            basis[difftomax .> 0] = pad_array.(basis[difftomax .> 0], difftomax[difftomax .> 0], 0)
            basis = [b[1:maxlength] for b in basis]
        else
            basis = pad_array.(basis, difftomax, 0)
        end
        return reduce(hcat, basis)
    end
end

const PATTERN_NAMES = [:sigmoid, :one_sided_fan, :two_sided_fan, :diverging_bar, :hourglass, :tilted_bar, :no_class]

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
    # Use latency as a secondary key so the sigmoid order stays stable instead of
    # collapsing into near-random stripes when many Δlatency values are similar.
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
    return maybe_diag(:covariates_for_patterns) do
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
end

@inline function pattern_sort_values(pname::Symbol, evts, rng::AbstractRNG)
    return maybe_diag(:pattern_sort_values) do
        if pname === :no_class
            # Randomize no_class trial order explicitly at the source.
            return rand(fresh_rng(rng), size(evts, 1))
        end
        return SORTERS[pname](evts)
    end
end
