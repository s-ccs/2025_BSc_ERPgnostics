mutable struct Diagnostics
    enabled::Bool
    counts::Dict{Symbol, Int}
    times::Dict{Symbol, Float64}
    started_at::Float64
end

const DIAG = Diagnostics(false, Dict{Symbol, Int}(), Dict{Symbol, Float64}(), time())
const DIAG_LOCK = ReentrantLock()

# Enable or disable diagnostics and optionally propagate to workers.
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

# Reset in-memory diagnostics counters and timer.
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

# Update diagnostics counters with a timing delta.
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

# Time a function call while collecting diagnostics.
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

# Convenience overload for diag_call with swapped args.
@inline function diag_call(f::Function, name::Symbol)
    return diag_call(name, f)
end

# Snapshot diagnostics counters and timers.
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

# Merge diagnostics counters from a snapshot.
function _merge_diag!(counts, times, snap)
    for (k, v) in snap.counts
        counts[k] = get(counts, k, 0) + v
    end
    for (k, v) in snap.times
        times[k] = get(times, k, 0.0) + v
    end
    return nothing
end

# Print diagnostics for a single snapshot.
function _print_diag(prefix::AbstractString, snap)
    println(prefix, " elapsed=", round(snap.elapsed; digits = 1), "s")
    for k in sort(collect(keys(snap.counts)))
        c = snap.counts[k]
        t = get(snap.times, k, 0.0)
        println("  ", k, ": count=", c, ", time=", round(t; digits = 3), "s")
    end
    return nothing
end

# Print diagnostics totals, optionally per worker.
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

# Periodically print diagnostics in a loop.
function monitor_workers(; interval::Real = 10, cycles::Int = 0, by_worker::Bool = true)
    i = 0
    while cycles <= 0 || i < cycles
        sleep(interval)
        print_diagnostics(by_worker = by_worker)
        i += 1
    end
    return nothing
end

# Start a timer that prints diagnostics.
function start_monitor(; interval::Real = 10, by_worker::Bool = true)
    return Timer(_ -> print_diagnostics(by_worker = by_worker), interval; interval = interval)
end

# Stop a diagnostics timer.
function stop_monitor!(timer::Timer)
    close(timer)
    return nothing
end
