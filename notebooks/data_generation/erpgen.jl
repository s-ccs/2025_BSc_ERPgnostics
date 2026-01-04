module ERPGen

using Distributed
using Distributions
using Random
using Statistics: mean, std
using ImageFiltering: KernelFactors, imfilter
using Images: imresize
using Normalization
using Dates
using JLD2
using LinearAlgebra
using Logging
using StatsModels: @formula
using UnfoldSim
using DataFrames

import UnfoldSim.simulate_component
import Base: length

const MODULE_PATH = abspath(@__FILE__)
const PROJECT_DIR = abspath(@__DIR__)
const DELTA_LATENCY = Symbol("\u0394latency")

mutable struct Diagnostics
    enabled::Bool
    counts::Dict{Symbol, Int}
    times::Dict{Symbol, Float64}
    started_at::Float64
end

const DIAG = Diagnostics(false, Dict{Symbol, Int}(), Dict{Symbol, Float64}(), time())
const DIAG_LOCK = ReentrantLock()

function enable_diagnostics!(flag::Bool = true; propagate::Bool = true)
    DIAG.enabled = flag
    if flag
        DIAG.started_at = time()
    end
    if propagate && nworkers() > 0
        for p in workers()
            try
                Distributed.remotecall_fetch(ERPGen.enable_diagnostics!, p, flag; propagate = false)
            catch
            end
        end
    end
    return DIAG.enabled
end

function reset_diagnostics!()
    lock(DIAG_LOCK)
    try
        empty!(DIAG.counts)
        empty!(DIAG.times)
        DIAG.started_at = time()
    finally
        unlock(DIAG_LOCK)
    end
    return nothing
end

@inline function _diag_update!(name::Symbol, dt::Float64)
    lock(DIAG_LOCK)
    try
        DIAG.counts[name] = get(DIAG.counts, name, 0) + 1
        DIAG.times[name] = get(DIAG.times, name, 0.0) + dt
    finally
        unlock(DIAG_LOCK)
    end
    return nothing
end

@inline function diag_call(name::Symbol, f::Function)
    if !DIAG.enabled
        return f()
    end
    t0 = time_ns()
    try
        return f()
    finally
        dt = (time_ns() - t0) / 1e9
        _diag_update!(name, dt)
    end
end

@inline function diag_call(f::Function, name::Symbol)
    return diag_call(name, f)
end

function diagnostics_snapshot()
    lock(DIAG_LOCK)
    try
        return (
            counts = copy(DIAG.counts),
            times = copy(DIAG.times),
            elapsed = time() - DIAG.started_at,
        )
    finally
        unlock(DIAG_LOCK)
    end
end

function _merge_diag!(counts, times, snap)
    for (k, v) in snap.counts
        counts[k] = get(counts, k, 0) + v
    end
    for (k, v) in snap.times
        times[k] = get(times, k, 0.0) + v
    end
    return nothing
end

function _print_diag(prefix::AbstractString, snap)
    println(prefix, " elapsed=", round(snap.elapsed; digits = 1), "s")
    for k in sort(collect(keys(snap.counts)))
        c = snap.counts[k]
        t = get(snap.times, k, 0.0)
        println("  ", k, ": count=", c, ", time=", round(t; digits = 3), "s")
    end
    return nothing
end

function print_diagnostics(; by_worker::Bool = true)
    if !DIAG.enabled
        println("Diagnostics disabled. Call enable_diagnostics!(true) first.")
        return nothing
    end

    total_counts = Dict{Symbol, Int}()
    total_times = Dict{Symbol, Float64}()

    local_snap = diagnostics_snapshot()
    _merge_diag!(total_counts, total_times, local_snap)

    if by_worker && nworkers() > 0
        _print_diag("pid $(myid())", local_snap)
        for w in workers()
            snap = try
                Distributed.remotecall_fetch(ERPGen.diagnostics_snapshot, w)
            catch
                continue
            end
            _merge_diag!(total_counts, total_times, snap)
            _print_diag("pid $(w)", snap)
        end
    end

    total_snap = (counts = total_counts, times = total_times, elapsed = local_snap.elapsed)
    _print_diag("total", total_snap)
    return nothing
end

function monitor_workers(; interval::Real = 10, cycles::Int = 0, by_worker::Bool = true)
    i = 0
    while cycles <= 0 || i < cycles
        sleep(interval)
        print_diagnostics(by_worker = by_worker)
        i += 1
    end
    return nothing
end

function start_monitor(; interval::Real = 10, by_worker::Bool = true)
    return Timer(_ -> print_diagnostics(by_worker = by_worker), interval; interval = interval)
end

function stop_monitor!(timer::Timer)
    close(timer)
    return nothing
end

struct TimeVaryingComponent <: AbstractComponent
    basisfunction::Any
    maxlength::Any
    beta::Any
end

Base.length(c::TimeVaryingComponent) = c.maxlength

function UnfoldSim.simulate_component(rng, c::TimeVaryingComponent, design::AbstractDesign)
    evts = generate_events(deepcopy(rng), design)
    data = c.beta .* c.basisfunction(evts, c.maxlength)
    return truncate_basisfunction(data, c.maxlength)
end

function UnfoldSim.simulate_component(c::TimeVaryingComponent, design::AbstractDesign; rng = MersenneTwister(time_ns()))
    return UnfoldSim.simulate_component(rng, c, design)
end

function basis_linear(evts, maxlength)
    shifts = -round.(Int, evts.duration_linear)
    basis = pad_array.(Ref(UnfoldSim.DSP.hanning(50)), shifts, 0)
    return basis
end

function basis_lognormal(evts, maxlength)
    basis = pdf.(LogNormal.(evts.duration ./ 40 .- 0.2, 1), Ref(range(0, 10, length = maxlength)))
    basis = basis ./ maximum.(basis)
    return basis
end

function basis_hanning(evts, maxlength)
    fn = "duration"
    if "durationB" in names(evts)
        fn = "durationB"
    end
    maxdur = maximum(evts[:, fn])

    basis = UnfoldSim.DSP.hanning.(Int.(round.(evts[:, fn])))
    shifts = Int.(.-round.(maxdur .- evts[:, fn]) .÷ 2)
    basis = pad_array.(basis, shifts, 0)
    return basis
end

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

function _with_noiselevel(noise, noiselevel::Int)
    if hasproperty(noise, :noiselevel)
        fields = fieldnames(typeof(noise))
        kwargs = (; (f => getfield(noise, f) for f in fields if f != :noiselevel)..., noiselevel = noiselevel)
        return typeof(noise)(; kwargs...)
    end
    return noise
end

function simulate_6patterns(mu = 3.2, sigma = 0.5; maxlength::Int = 100, n_levels::Int = 8,
        noise = PinkNoise(), noiselevel_dist = Geometric(0.25), rng = MersenneTwister(time_ns()))

    design = SingleSubjectDesign(; 
        conditions = Dict(
            :condition => ["car", "face"],
            :continuous => range(-2, 2, length = n_levels),
            :duration => range(20, 100, length = n_levels),
            :durationB => range(10, 30, length = n_levels),
            :duration_linear => range(5, 40, length = n_levels),
        ),
        event_order_function = (r, x) -> shuffle(r, x),
    )

    p1 = LinearModelComponent(; basis = p100(), formula = @formula(0 ~ 1), β = [5])

    n1 = LinearModelComponent(; 
        basis = n170(),
        formula = @formula(0 ~ 1 + condition + continuous),
        β = [5, 3, 2],
    )

    p3 = LinearModelComponent(; basis = p300(), formula = @formula(0 ~ 1), β = [5])
    componentA = TimeVaryingComponent(basis_lognormal, maxlength, 5)
    componentB = TimeVaryingComponent(basis_hanning, maxlength, -10)
    componentC = TimeVaryingComponent(basis_linear, maxlength, 5)

    noiselevel = max(1, Int(1 + rand(rng, noiselevel_dist)))
    noise = _with_noiselevel(noise, noiselevel)

    use_exp_noise = noise isa ExponentialNoise
    noise_for_sim = use_exp_noise ? NoNoise() : noise

    data, sim_evts = simulate(
        rng,
        design,
        [p1, n1, p3, componentA, componentB, componentC],
        LogNormalOnset(; μ = mu, σ = sigma),
        noise_for_sim,
        return_epoched = true,
    )

    if use_exp_noise
        for trial in axes(data, 2)
            data[:, trial] .+= UnfoldSim.simulate_noise(rng, noise, size(data, 1))
        end
    end

    sim_evts[!, DELTA_LATENCY] = vcat(diff(sim_evts.latency), 0)
    sim_6patterns = data .- mean(data, dims = 2)
    return sim_6patterns, sim_evts, noiselevel
end

const PATTERN_NAMES = [:sigmoid, :one_sided_fan, :two_sided_fan, :diverging_bar, :hourglass, :tilted_bar, :noclass]

const SORTERS = Dict(
    :sigmoid => evts -> evts[!, DELTA_LATENCY],
    :one_sided_fan => evts -> evts.duration,
    :two_sided_fan => evts -> evts.durationB,
    :diverging_bar => evts -> evts.condition .== "car",
    :hourglass => evts -> evts.continuous,
    :tilted_bar => evts -> evts.duration_linear,
    :noclass => _ -> nothing,
)

const DEFAULT_NOISE_POOL = [PinkNoise(), WhiteNoise(), RedNoise(), ExponentialNoise()]
const FILTER_BORDER = "reflect"

function erpimage_sorted(data_all::AbstractMatrix, sortvalues;
        zscore_trials::Bool = true,
        gauss_sigma::Real = 0.0,
        gauss_smooth::Bool = true,
        gauss_kernel_len::Int = 20,
        trial_blur::Real = 0.0)

    if sortvalues === nothing
        data_sorted = data_all
    else
        idx = sortperm(sortvalues)
        data_sorted = data_all[:, idx]
    end

    if trial_blur > 0
        kernel = KernelFactors.gaussian((0, trial_blur))
        data_sorted = imfilter(data_sorted, kernel, FILTER_BORDER)
    end

    img = Float32.(permutedims(data_sorted, (2, 1)))

    if zscore_trials
        img = Float32.(Normalization.normalize(img, ZScore; dims = 2))
    end

    if gauss_smooth && gauss_sigma > 0
        kernel_len = isodd(gauss_kernel_len) ? gauss_kernel_len : gauss_kernel_len + 1
        kern = KernelFactors.gaussian((gauss_sigma, gauss_sigma), (kernel_len, kernel_len))
        img = Float32.(imfilter(img, kern, FILTER_BORDER))
    end

    return img
end

function resize_img(img::AbstractMatrix, target_h::Int, target_w::Int;
        resize_antialias::Bool = true,
        antialias_factor::Real = 0.75)

    if target_h <= 0 || target_w <= 0 || size(img) == (target_h, target_w)
        return Float32.(img)
    end

    out = Float32.(img)
    if resize_antialias && antialias_factor > 0
        sigma = (antialias_factor * size(out, 1) / target_h,
            antialias_factor * size(out, 2) / target_w)
        out = Float32.(imfilter(out, KernelFactors.gaussian(sigma), FILTER_BORDER))
    end

    return Float32.(imresize(out, (target_h, target_w)))
end

function ensure_threading!(threaded::Bool)
    threaded || return
    n = Threads.nthreads()
    required = 16
    if n < required
        error("threaded=true requires $(required) Julia threads; restart the kernel with JULIA_NUM_THREADS=$(required).")
    end
    return
end

function start_progress_logger!(reps_done::Threads.Atomic{Int}, n_per_pattern::Int, progress_every::Int)
    progress_every <= 0 && return nothing
    return @async begin
        last = 0
        while true
            done = reps_done[]
            if done >= n_per_pattern
                println("Progress: ", n_per_pattern, "/", n_per_pattern,
                    " reps (per class=", n_per_pattern, ", total images=", n_per_pattern * length(PATTERN_NAMES), ")")
                break
            elseif done - last >= progress_every
                println("Progress: ", done, "/", n_per_pattern,
                    " reps (per class=", done, ", total images=", done * length(PATTERN_NAMES), ")")
                last = done
            end
            sleep(0.25)
        end
    end
end

function start_progress_logger_processes!(progress_chan, n_per_pattern::Int, progress_every::Int)
    progress_every <= 0 && return nothing
    return @async begin
        done = 0
        last = 0
        while done < n_per_pattern
            done += take!(progress_chan)
            if done - last >= progress_every || done >= n_per_pattern
                if done > n_per_pattern
                    done = n_per_pattern
                end
                println("Progress: ", done, "/", n_per_pattern,
                    " reps (per class=", done, ", total images=", done * length(PATTERN_NAMES), ")")
                last = done
            end
        end
    end
end

function apply_gauss!(img::Matrix{Float32}, kernel2d)
    return Float32.(imfilter(img, kernel2d, FILTER_BORDER))
end

mutable struct AntialiasCache
    size::Tuple{Int, Int}
    kernel::Any
end

AntialiasCache() = AntialiasCache((0, 0), nothing)

function apply_filters!(img::Matrix{Float32}, gauss_cfg, resize_opts, cache::AntialiasCache)
    out = img

    if gauss_cfg.enabled
        out = apply_gauss!(out, gauss_cfg.kernel2d)
    end

    if resize_opts.apply_antialias && resize_opts.antialias_factor > 0
        needs_resize = resize_opts.height > 0 && resize_opts.width > 0 &&
                       size(out) != (resize_opts.height, resize_opts.width)
        if needs_resize
            if cache.size != size(out)
                sigma = (resize_opts.antialias_factor * size(out, 1) / resize_opts.height,
                    resize_opts.antialias_factor * size(out, 2) / resize_opts.width)
                cache.kernel = KernelFactors.gaussian(sigma)
                cache.size = size(out)
            end
            out = Float32.(imfilter(out, cache.kernel, FILTER_BORDER))
        end
    end

    return out
end

function make_gauss_resize_cfg(gauss_sigma::Real, gauss_smooth::Bool, gauss_kernel_len::Int,
        resize_antialias::Bool, antialias_factor::Real,
        target_height::Int, target_width::Int)

    gauss_enabled = gauss_smooth && gauss_sigma > 0
    kernel_len = isodd(gauss_kernel_len) ? gauss_kernel_len : gauss_kernel_len + 1
    kern2d = gauss_enabled ? KernelFactors.gaussian((gauss_sigma, gauss_sigma), (kernel_len, kernel_len)) : nothing
    antialias_enabled = resize_antialias && antialias_factor > 0
    gauss_cfg = (; enabled = gauss_enabled, kernel2d = kern2d)
    resize_opts = (; height = target_height,
        width = target_width,
        apply_antialias = antialias_enabled,
        antialias_factor = antialias_factor,
    )
    return gauss_cfg, resize_opts
end

function sample_sim_params(rng, mu_dist, sigma_dist, maxlength_dist, maxlength_min::Int, n_levels_dist, noise_pool)
    mu = max(1, rand(rng, mu_dist))
    sigma = max(0.01, rand(rng, sigma_dist))
    maxlength = max(maxlength_min, Int(round(rand(rng, maxlength_dist))))
    n_levels = max(4, Int(round(rand(rng, n_levels_dist))))
    noise = rand(rng, noise_pool)
    return mu, sigma, maxlength, n_levels, noise
end

function simulate_patterns(rng, mu, sigma; maxlength::Int, n_levels::Int, noise, noiselevel_dist)
    return diag_call(:simulate_6patterns) do
        with_logger(NullLogger()) do
            simulate_6patterns(mu, sigma;
                maxlength = maxlength,
                n_levels = n_levels,
                noise = noise,
                noiselevel_dist = noiselevel_dist,
                rng = rng,
            )
        end
    end
end

function render_patterns!(images, labels, metadata, base::Int, data, evts, mu, sigma, maxlength, n_levels, noise, noiselevel;
        zscore_trials::Bool, gauss_cfg, resize_opts, gauss_kernel_len::Int, trial_blur::Real,
        target_height::Int, target_width::Int, antialias_factor::Real, cache::AntialiasCache)

    for (pidx, pname) in enumerate(PATTERN_NAMES)
        sortvalues = SORTERS[pname](evts)
        img = diag_call(:erpimage_sorted) do
            erpimage_sorted(data, sortvalues;
                zscore_trials = zscore_trials,
                gauss_sigma = 0.0,
                gauss_smooth = false,
                gauss_kernel_len = gauss_kernel_len,
                trial_blur = trial_blur,
            )
        end

        filtered = diag_call(:apply_filters) do
            apply_filters!(img, gauss_cfg, resize_opts, cache)
        end
        resized = diag_call(:resize_img) do
            resize_img(filtered, target_height, target_width;
                resize_antialias = false,
                antialias_factor = antialias_factor,
            )
        end

        idx = base + pidx
        images[idx] = resized
        labels[idx] = pname
        metadata[idx] = (
            pattern = pname,
            mu = mu,
            sigma = sigma,
            maxlength = maxlength,
            n_levels = n_levels,
            noise = string(typeof(noise)),
            noiselevel = noiselevel,
        )
    end
    return nothing
end

function setup_workers!(n_workers::Int)
    n_workers <= 0 && return
    needed = n_workers - nworkers()
    if needed > 0
        addprocs(needed; exeflags = "--threads=1 --project=$(PROJECT_DIR)")
    end
    for p in workers()
        Distributed.remotecall_eval(Main, p, :(include($(MODULE_PATH))))
        Distributed.remotecall_eval(Main, p, :(using .ERPGen))
    end
    if DIAG.enabled
        for p in workers()
            try
                Distributed.remotecall_fetch(ERPGen.enable_diagnostics!, p, true; propagate = false)
            catch
            end
        end
    end
    return
end

function build_chunk_data(rep_start::Int, rep_end::Int, seed::UInt,
        n_classes::Int, mu_dist, sigma_dist, maxlength_dist, maxlength_min::Int, n_levels_dist,
        noise_pool, noiselevel_dist, zscore_trials::Bool, gauss_cfg, resize_opts,
        target_height::Int, target_width::Int, antialias_factor::Real,
        gauss_kernel_len::Int, trial_blur::Real, blas_threads::Int, progress_chan, progress_every::Int)

    if blas_threads > 0
        BLAS.set_num_threads(blas_threads)
    end

    local_rng = Random.Xoshiro(seed)
    cache = AntialiasCache()

    n_reps = rep_end - rep_start + 1
    local_total = n_reps * n_classes
    local_imgs = Vector{Matrix{Float32}}(undef, local_total)
    local_labels = Vector{Symbol}(undef, local_total)
    local_meta = Vector{NamedTuple}(undef, local_total)

    local_done = 0

    for offset in 1:n_reps
        mu, sigma, maxlength, n_levels, noise = sample_sim_params(
            local_rng, mu_dist, sigma_dist, maxlength_dist, maxlength_min, n_levels_dist, noise_pool)
        data, evts, noiselevel = simulate_patterns(local_rng, mu, sigma;
            maxlength = maxlength,
            n_levels = n_levels,
            noise = noise,
            noiselevel_dist = noiselevel_dist,
        )

        base = (offset - 1) * n_classes

        render_patterns!(
            local_imgs, local_labels, local_meta, base,
            data, evts, mu, sigma, maxlength, n_levels, noise, noiselevel;
            zscore_trials = zscore_trials,
            gauss_cfg = gauss_cfg,
            resize_opts = resize_opts,
            gauss_kernel_len = gauss_kernel_len,
            trial_blur = trial_blur,
            target_height = target_height,
            target_width = target_width,
            antialias_factor = antialias_factor,
            cache = cache,
        )

        if progress_chan !== nothing && progress_every > 0
            local_done += 1
            if local_done >= progress_every
                put!(progress_chan, local_done)
                local_done = 0
            end
        end
    end

    if progress_chan !== nothing && local_done > 0
        put!(progress_chan, local_done)
    end

    return (rep_start, rep_end, local_imgs, local_labels, local_meta)
end

function generate_erp_images_threads(; n_per_pattern::Int = 10,
        mu_mean::Real = 3.2,
        mu_sd::Real = 0.3,
        sigma_mean::Real = 0.5,
        sigma_sd::Real = 0.1,
        maxlength_mean::Real = 100,
        maxlength_sd::Real = 10,
        maxlength_min::Int = 100,
        n_levels_mean::Real = 8,
        n_levels_sd::Real = 2,
        target_height::Int = 64,
        target_width::Int = 64,
        zscore_trials::Bool = true,
        gauss_sigma::Real = 1.0,
        gauss_smooth::Bool = true,
        gauss_kernel_len::Int = 20,
        trial_blur::Real = 0.0,
        resize_antialias::Bool = true,
        antialias_factor::Real = 0.75,
        noise_pool = DEFAULT_NOISE_POOL,
        noiselevel_dist = Geometric(0.25),
        threaded::Bool = false,
        blas_threads::Int = 1,
        progress_every::Int = 10,
        rng::AbstractRNG = MersenneTwister(time_ns()))

    n_classes = length(PATTERN_NAMES)
    total = n_classes * n_per_pattern

    images = Vector{Matrix{Float32}}(undef, total)
    labels = Vector{Symbol}(undef, total)
    metadata = Vector{NamedTuple}(undef, total)

    mu_dist = Normal(mu_mean, mu_sd)
    sigma_dist = Normal(sigma_mean, sigma_sd)
    maxlength_dist = Normal(maxlength_mean, maxlength_sd)
    n_levels_dist = Normal(n_levels_mean, n_levels_sd)

    ensure_threading!(threaded)
    if threaded && blas_threads > 0
        BLAS.set_num_threads(blas_threads)
    end

    nthreads = threaded ? Threads.nthreads() : 1
    rngs = [Random.Xoshiro(rand(rng, UInt)) for _ in 1:nthreads]
    reps_done = progress_every > 0 ? Threads.Atomic{Int}(0) : nothing
    progress_task = progress_every > 0 ? start_progress_logger!(reps_done, n_per_pattern, progress_every) : nothing
    progress_stride = progress_every > 0 ? max(1, progress_every) : 0

    gauss_cfg, resize_opts = make_gauss_resize_cfg(
        gauss_sigma, gauss_smooth, gauss_kernel_len,
        resize_antialias, antialias_factor,
        target_height, target_width,
    )

    chunk = cld(n_per_pattern, nthreads)

    function build_chunk!(chunk_id::Int)
        rep_start = (chunk_id - 1) * chunk + 1
        rep_end = min(n_per_pattern, chunk_id * chunk)
        rep_start > rep_end && return

        local_rng = rngs[chunk_id]
        cache = AntialiasCache()
        local_done = 0

        for rep in rep_start:rep_end
            mu, sigma, maxlength, n_levels, noise = sample_sim_params(
                local_rng, mu_dist, sigma_dist, maxlength_dist, maxlength_min, n_levels_dist, noise_pool)
            data, evts, noiselevel = simulate_patterns(local_rng, mu, sigma;
                maxlength = maxlength,
                n_levels = n_levels,
                noise = noise,
                noiselevel_dist = noiselevel_dist,
            )

            base = (rep - 1) * n_classes

            render_patterns!(
                images, labels, metadata, base,
                data, evts, mu, sigma, maxlength, n_levels, noise, noiselevel;
                zscore_trials = zscore_trials,
                gauss_cfg = gauss_cfg,
                resize_opts = resize_opts,
                gauss_kernel_len = gauss_kernel_len,
                trial_blur = trial_blur,
                target_height = target_height,
                target_width = target_width,
                antialias_factor = antialias_factor,
                cache = cache,
            )

            if reps_done !== nothing
                local_done += 1
                if local_done >= progress_stride
                    Threads.atomic_add!(reps_done, local_done)
                    local_done = 0
                end
            end
        end

        if reps_done !== nothing && local_done > 0
            Threads.atomic_add!(reps_done, local_done)
        end
    end

    if threaded
        Threads.@threads :static for chunk_id in 1:nthreads
            build_chunk!(chunk_id)
        end
    else
        build_chunk!(1)
    end

    progress_task !== nothing && wait(progress_task)

    return images, labels, metadata
end

function generate_erp_images_processes(; n_per_pattern::Int = 10,
        mu_mean::Real = 3.2,
        mu_sd::Real = 0.3,
        sigma_mean::Real = 0.5,
        sigma_sd::Real = 0.1,
        maxlength_mean::Real = 100,
        maxlength_sd::Real = 10,
        maxlength_min::Int = 100,
        n_levels_mean::Real = 8,
        n_levels_sd::Real = 2,
        target_height::Int = 64,
        target_width::Int = 64,
        zscore_trials::Bool = true,
        gauss_sigma::Real = 1.0,
        gauss_smooth::Bool = true,
        gauss_kernel_len::Int = 20,
        trial_blur::Real = 0.0,
        resize_antialias::Bool = true,
        antialias_factor::Real = 0.75,
        noise_pool = DEFAULT_NOISE_POOL,
        noiselevel_dist = Geometric(0.25),
        n_workers::Int = 16,
        blas_threads::Int = 1,
        progress_every::Int = 10,
        rng::AbstractRNG = MersenneTwister(time_ns()))

    n_classes = length(PATTERN_NAMES)
    total = n_classes * n_per_pattern

    mu_dist = Normal(mu_mean, mu_sd)
    sigma_dist = Normal(sigma_mean, sigma_sd)
    maxlength_dist = Normal(maxlength_mean, maxlength_sd)
    n_levels_dist = Normal(n_levels_mean, n_levels_sd)

    gauss_cfg, resize_opts = make_gauss_resize_cfg(
        gauss_sigma, gauss_smooth, gauss_kernel_len,
        resize_antialias, antialias_factor,
        target_height, target_width,
    )

    setup_workers!(n_workers)
    worker_ids = workers()
    n_workers_actual = min(n_workers, length(worker_ids))
    worker_ids = worker_ids[1:n_workers_actual]

    chunk = cld(n_per_pattern, n_workers_actual)
    seeds = [rand(rng, UInt) for _ in 1:n_workers_actual]

    progress_chan = progress_every > 0 ? RemoteChannel(() -> Channel{Int}(n_workers_actual * 2)) : nothing
    progress_task = progress_every > 0 ? start_progress_logger_processes!(progress_chan, n_per_pattern, progress_every) : nothing

    futures = Vector{Future}(undef, n_workers_actual)
    for (i, w) in enumerate(worker_ids)
        rep_start = (i - 1) * chunk + 1
        rep_end = min(n_per_pattern, i * chunk)
        if rep_start > rep_end
            futures[i] = @spawnat w nothing
            continue
        end
        futures[i] = @spawnat w ERPGen.build_chunk_data(
            rep_start, rep_end, seeds[i],
            n_classes, mu_dist, sigma_dist, maxlength_dist, maxlength_min, n_levels_dist,
            noise_pool, noiselevel_dist, zscore_trials, gauss_cfg, resize_opts,
            target_height, target_width, antialias_factor, gauss_kernel_len, trial_blur,
            blas_threads, progress_chan, progress_every,
        )
    end

    images = Vector{Matrix{Float32}}(undef, total)
    labels = Vector{Symbol}(undef, total)
    metadata = Vector{NamedTuple}(undef, total)

    for fut in futures
        chunk_data = fetch(fut)
        chunk_data === nothing && continue
        rep_start, rep_end, local_imgs, local_labels, local_meta = chunk_data
        base = (rep_start - 1) * n_classes
        local_total = length(local_imgs)
        images[base + 1:base + local_total] = local_imgs
        labels[base + 1:base + local_total] = local_labels
        metadata[base + 1:base + local_total] = local_meta
    end

    progress_task !== nothing && wait(progress_task)

    return images, labels, metadata
end

function generate_erp_images(; parallel_mode::Symbol = :threads, n_workers::Int = 16,
        threaded::Bool = false, kwargs...)

    if threaded && parallel_mode == :processes
        return generate_erp_images_processes(; n_workers = n_workers, kwargs...)
    end

    return generate_erp_images_threads(; threaded = threaded, kwargs...)
end

function save_erp_dataset(images, labels, metadata;
        dataset_dir::AbstractString = joinpath(pwd(), "datasets"),
        prefix::AbstractString = "erp_dataset",
        settings = NamedTuple(),
        environment = NamedTuple())

    mkpath(dataset_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "$(prefix)_$(timestamp).jld2"
    path = joinpath(dataset_dir, filename)
    env = merge((generated_at = timestamp,), environment)

    jldsave(path;
        images = images,
        labels = labels,
        metadata = metadata,
        settings = settings,
        environment = env,
    )

    return path
end

export PATTERN_NAMES, DEFAULT_NOISE_POOL
export generate_erp_images, save_erp_dataset
export setup_workers!
export enable_diagnostics!, reset_diagnostics!, diagnostics_snapshot
export print_diagnostics, monitor_workers, start_monitor, stop_monitor!

end
