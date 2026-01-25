# Time-varying component wrapper for UnfoldSim.
struct TimeVaryingComponent <: AbstractComponent
    basisfunction::Any
    maxlength::Any
    beta::Any
end

# Use maxlength as component length.
Base.length(c::TimeVaryingComponent) = c.maxlength

# Simulate a time-varying component with explicit RNG.
function UnfoldSim.simulate_component(rng, c::TimeVaryingComponent, design::AbstractDesign)
    evts = generate_events(deepcopy(rng), design)
    data = c.beta .* c.basisfunction(evts, c.maxlength)
    return truncate_basisfunction(data, c.maxlength)
end

# Simulate a time-varying component with default RNG.
function UnfoldSim.simulate_component(c::TimeVaryingComponent, design::AbstractDesign; rng = MersenneTwister(time_ns()))
    return UnfoldSim.simulate_component(rng, c, design)
end

# Generate a linear ERP basis (tilted bar).
function basis_linear(evts, maxlength)
    shifts = -round.(Int, evts.tilted_bar_duration)
    basis = pad_array.(Ref(UnfoldSim.DSP.hanning(50)), shifts, 0)
    return basis
end

# Generate a lognormal ERP basis (one-sided fan).
function basis_lognormal(evts, maxlength)
    basis = pdf.(LogNormal.(evts.one_sided_fan_duration ./ 40 .- 0.2, 1),
        Ref(range(0, 10, length = maxlength)))
    basis = basis ./ maximum.(basis)
    return basis
end

# Generate a hanning ERP basis (two-sided fan).
function basis_hanning(evts, maxlength)
    maxdur = maximum(evts.two_sided_fan_duration)

    basis = UnfoldSim.DSP.hanning.(Int.(round.(evts.two_sided_fan_duration)))
    shifts = Int.(.-round.(maxdur .- evts.two_sided_fan_duration) .÷ 2)
    basis = pad_array.(basis, shifts, 0)
    return basis
end

# Ensure all basis functions share the same length.
function truncate_basisfunction(basis, maxlength)
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
