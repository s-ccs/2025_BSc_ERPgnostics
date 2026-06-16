# =============================================================================
# Search strategies
#
# Three single-pass strategies propose simulator parameter candidates over the
# 48-dimensional space:
#   - broad_random   : independent uniform draw per dimension (domain randomisation).
#   - monte_carlo     : the same uniform draw, kept as a separate baseline.
#   - latin_hypercube : a stratified design that spreads candidates more evenly.
# Each strategy starts from the same base parameter vector and overwrites the
# searched dimensions; the baseline configuration is evaluated unchanged.
# =============================================================================

"""
    CandidateRecord

One simulator candidate proposed by a search strategy (or the baseline).

# Fields
- `search_method::String`: strategy name, or `"baseline"`.
- `candidate_index::Int`: 1-based index within the strategy (`0` for baseline).
- `cfg::ERPGen.GenerationConfig`: the simulator configuration to evaluate.
- `params::Vector{Float64}`: the 48-element parameter vector behind `cfg`.
- `parameter_subset_size::Int`: number of searched dimensions (`0` for baseline).
"""
struct CandidateRecord
    search_method::String
    candidate_index::Int
    cfg::ERPGen.GenerationConfig
    params::Vector{Float64}
    parameter_subset_size::Int
end

"""
    parameter_subset_indices(parameter_subset, param_symbols)

Resolve which parameter dimensions a strategy searches. `:all` selects every
dimension; a vector of symbols selects (and validates) named dimensions.

# Arguments
- `parameter_subset`: `:all` or a `Vector{Symbol}` of parameter symbols.
- `param_symbols::Vector{Symbol}`: all 48 parameter symbols in order.

# Returns
- `Vector{Int}`: indices into the flat parameter vector.
"""
function parameter_subset_indices(parameter_subset, param_symbols::Vector{Symbol})
    parameter_subset === :all && return collect(eachindex(param_symbols))
    if parameter_subset isa AbstractVector{Symbol}
        index_by_symbol = Dict(sym => i for (i, sym) in enumerate(param_symbols))
        unknown = [sym for sym in parameter_subset if !haskey(index_by_symbol, sym)]
        isempty(unknown) || error("Unknown parameter_subset symbol(s): $(unknown)")
        length(unique(parameter_subset)) == length(parameter_subset) || error("parameter_subset contains duplicate symbols.")
        return [index_by_symbol[sym] for sym in parameter_subset]
    end
    error("parameter_subset must be :all or Vector{Symbol}, got $(parameter_subset).")
end

"""
    candidate_params(method, candidate_index, rng, subset_indices, lhs_design, base_params, param_ranges)

Build the parameter vector for one candidate. Latin hypercube reads the
candidate's row from a precomputed design; the other strategies draw uniformly.
Unsearched dimensions keep their base value.

# Arguments
- `method::Symbol`: search strategy.
- `candidate_index::Int`: candidate row index.
- `rng::Random.AbstractRNG`: random source for the uniform strategies.
- `subset_indices::Vector{Int}`: searched dimensions.
- `lhs_design`: Latin hypercube matrix, or `nothing` for the other strategies.
- `base_params::Vector{Float64}`: starting parameter vector.
- `param_ranges`: the 48 `(low, high)` bounds.

# Returns
- `Vector{Float64}`: the candidate's parameter vector.
"""
function candidate_params(method::Symbol, candidate_index::Int, rng::Random.AbstractRNG,
        subset_indices::Vector{Int}, lhs_design, base_params::Vector{Float64}, param_ranges)
    params = copy(base_params)
    isempty(subset_indices) && return params

    if method == :latin_hypercube
        # Map the design's [0, 1) entries onto each searched dimension's range.
        for (local_idx, global_idx) in enumerate(subset_indices)
            lo, hi = param_ranges[global_idx]
            params[global_idx] = lo + lhs_design[candidate_index, local_idx] * (hi - lo)
        end
    else
        sampled = sample_param_vector(rng, param_ranges[subset_indices])
        for (local_idx, global_idx) in enumerate(subset_indices)
            params[global_idx] = sampled[local_idx]
        end
    end
    return params
end

"""
    make_candidate_records(method, n_candidates, base_cfg, base_params, param_ranges, param_symbols)

Generate all candidate records for one search strategy. The strategy's sampling
RNG is seeded from a fresh `new_seed()`, so each run proposes new candidates.

# Arguments
- `method::Symbol`: search strategy.
- `n_candidates::Int`: number of candidates to propose.
- `base_cfg::ERPGen.GenerationConfig`: configuration to copy unset fields from.
- `base_params::Vector{Float64}`: starting parameter vector.
- `param_ranges`: the 48 `(low, high)` bounds.
- `param_symbols::Vector{Symbol}`: the 48 parameter symbols.

# Returns
- `Vector{CandidateRecord}`: the proposed candidates, in index order.
"""
function make_candidate_records(method::Symbol, n_candidates::Int,
        base_cfg::ERPGen.GenerationConfig, base_params::Vector{Float64}, param_ranges, param_symbols::Vector{Symbol})
    subset_indices = parameter_subset_indices(:all, param_symbols)
    rng = Random.Xoshiro(new_seed())
    lhs_design = method == :latin_hypercube ? latin_hypercube(n_candidates, length(subset_indices), rng) : nothing

    records = CandidateRecord[]
    for candidate_index in 1:n_candidates
        params = candidate_params(method, candidate_index, rng, subset_indices, lhs_design, base_params, param_ranges)
        push!(records, CandidateRecord(String(method), candidate_index, build_cfg_from_params(base_cfg, params), Float64.(params), length(subset_indices)))
    end
    return records
end

"""
    baseline_record(base_cfg, base_params) -> CandidateRecord

Build the record for the hand-crafted baseline configuration (the "starting
parameters"), which is evaluated without any search.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: the baseline configuration.
- `base_params::Vector{Float64}`: its parameter vector.

# Returns
- `CandidateRecord`: the baseline record (`candidate_index = 0`).
"""
function baseline_record(base_cfg::ERPGen.GenerationConfig, base_params::Vector{Float64})
    return CandidateRecord("baseline", 0, base_cfg, Float64.(base_params), 0)
end
