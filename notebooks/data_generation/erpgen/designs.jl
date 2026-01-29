UnfoldSim.@with_kw mutable struct CovariateDesign <: UnfoldSim.AbstractDesign
    design = nothing
    n_trials::Int
    covariates::Dict{Symbol, Any}
    events_cache = nothing
end

UnfoldSim.size(design::CovariateDesign) = design.n_trials

function UnfoldSim.generate_events(rng::UnfoldSim.AbstractRNG, design::CovariateDesign)
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
