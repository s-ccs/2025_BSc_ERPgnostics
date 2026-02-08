mutable struct Diagnostics
    enabled::Bool
    counts::Dict{Symbol, Int}
    times::Dict{Symbol, Float64}
    edges::Dict{Tuple{Symbol, Symbol}, Tuple{Int, Float64}}
    started_at::Float64
end

const DIAG = Diagnostics(false, Dict{Symbol, Int}(), Dict{Symbol, Float64}(),
    Dict{Tuple{Symbol, Symbol}, Tuple{Int, Float64}}(), time())
const DIAGNOSTICS_ENABLED = Ref(false)
const DIAG_LOCK = ReentrantLock()

# Enable or disable diagnostics and optionally propagate to workers.
function enable_diagnostics!(flag::Bool = true; propagate::Bool = true)
    DIAGNOSTICS_ENABLED[] = flag
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

# Conditionally wrap function execution with diagnostics.
function maybe_diag(name::Symbol, f::Function)
    if DIAG.enabled
        return diag_call(name, f)
    end
    return f()
end

# Support do-block style: maybe_diag(:name) do ... end
function maybe_diag(f::Function, name::Symbol)
    return maybe_diag(name, f)
end

# Reset in-memory diagnostics counters and timer.
function reset_diagnostics!()
    lock(DIAG_LOCK)
    try
        empty!(DIAG.counts)
        empty!(DIAG.times)
        empty!(DIAG.edges)
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

# Update diagnostics edges (parent -> child) with a timing delta.
@inline function _diag_update_edge!(parent::Symbol, child::Symbol, dt::Float64)
    key = (parent, child)
    lock(DIAG_LOCK)
    try
        prev = get(DIAG.edges, key, (0, 0.0))
        DIAG.edges[key] = (prev[1] + 1, prev[2] + dt)
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
    stack = get!(task_local_storage(), :diag_stack, Symbol[])
    parent = isempty(stack) ? :root : stack[end]
    push!(stack, name)
    t0 = time_ns()
    try
        return f()
    finally
        dt = (time_ns() - t0) / 1e9
        pop!(stack)
        _diag_update!(name, dt)
        _diag_update_edge!(parent, name, dt)
    end
end

# Convenience overload for diag_call with swapped args.
@inline function diag_call(f::Function, name::Symbol)
    return diag_call(name, f)
end

function _format_diag_time(t::Real)
    if t < 1
        return string(round(t * 1000; digits = 1), "ms")
    end
    return string(round(t; digits = 2), "s")
end

function _format_diag_percent(p::Real)
    return string(round(p; digits = 1), "%")
end

function format_diagnostics_table(rows::Vector{NamedTuple};
        title::AbstractString = "ERPGen Diagnostics Report",
        max_width::Int = displaysize(stdout)[2])
    isempty(rows) && return "Diagnostics enabled but no data collected."

    headers = ("Function", "Calls", "Total", "Avg", "% Time")
    name_col = [row.name for row in rows]
    calls_col = [string(row.calls) for row in rows]
    total_col = [row.total_str for row in rows]
    avg_col = [row.avg_str for row in rows]
    pct_col = [row.pct_str for row in rows]

    widths = [
        max(length(headers[1]), maximum(length.(name_col))),
        max(length(headers[2]), maximum(length.(calls_col))),
        max(length(headers[3]), maximum(length.(total_col))),
        max(length(headers[4]), maximum(length.(avg_col))),
        max(length(headers[5]), maximum(length.(pct_col))),
    ]

    function _table_width(ws)
        return 2 + sum(ws) + 3 * 5 + 1
    end

    if max_width > 0 && _table_width(widths) > max_width
        min_name_width = max(length(headers[1]), 12)
        available = max_width - (_table_width(widths) - widths[1])
        widths[1] = clamp(available, min_name_width, widths[1])
    end

    function _row(cols)
        return "│ " *
               rpad(cols[1], widths[1]) * " │ " *
               lpad(cols[2], widths[2]) * " │ " *
               lpad(cols[3], widths[3]) * " │ " *
               lpad(cols[4], widths[4]) * " │ " *
               lpad(cols[5], widths[5]) * " │"
    end

    header_row = _row(headers)
    total_width = length(header_row) - 2

    top = "┌" * repeat("─", total_width) * "┐"
    title_line = "│ " * rpad(title, total_width - 2) * " │"
    sep_title = "├" * repeat("─", total_width) * "┤"
    header_sep = "├" * join([repeat("─", widths[i] + 2) for i in 1:5], "┼") * "┤"
    bottom = "└" * join([repeat("─", widths[i] + 2) for i in 1:5], "┴") * "┘"

    lines = String[ top, title_line, sep_title, header_row, header_sep ]
    function _wrap_name(name::AbstractString, width::Int)
        if length(name) <= width
            return [name]
        end
        prefix = ""
        rest = name
        m = match(r"^(.*?[├└]─\s*)", name)
        if m !== nothing
            prefix = m.captures[1]
            prefix_end = lastindex(prefix)
            rest_start = prefix_end < lastindex(name) ? nextind(name, prefix_end) : (lastindex(name) + 1)
            rest = rest_start <= lastindex(name) ? name[rest_start:end] : ""
        end
        avail = max(width - length(prefix), 4)
        parts = String[]
        remaining = rest
        while length(remaining) > avail
            window = remaining[1:avail]
            cut = findlast(c -> c == '_' || c == ' ' || c == '-', window)
            cut = cut === nothing ? avail : cut
            push!(parts, prefix * remaining[1:cut])
            remaining = lstrip(remaining[cut+1:end])
        end
        push!(parts, prefix * remaining)
        return parts
    end

    for row in rows
        parts = _wrap_name(row.name, widths[1])
        for (idx, part) in enumerate(parts)
            if idx == 1
                push!(lines, _row((part, string(row.calls), row.total_str, row.avg_str, row.pct_str)))
            else
                push!(lines, _row((part, "", "", "", "")))
            end
        end
    end
    push!(lines, bottom)
    return join(lines, "\n")
end

# Snapshot diagnostics counters and timers.
function diagnostics_snapshot()
    lock(DIAG_LOCK)
    try
        return (
            counts = copy(DIAG.counts),
            times = copy(DIAG.times),
            edges = copy(DIAG.edges),
            elapsed = time() - DIAG.started_at,
        )
    finally
        unlock(DIAG_LOCK)
    end
end

# Merge diagnostics across workers and return totals.
function diagnostics_totals(; by_worker::Bool = true)
    if !DIAG.enabled
        return (
            counts = Dict{Symbol, Int}(),
            times = Dict{Symbol, Float64}(),
            edges = Dict{Tuple{Symbol, Symbol}, Tuple{Int, Float64}}(),
            elapsed = 0.0,
        )
    end

    total_counts = Dict{Symbol, Int}()
    total_times = Dict{Symbol, Float64}()
    total_edges = Dict{Tuple{Symbol, Symbol}, Tuple{Int, Float64}}()

    local_snap = diagnostics_snapshot()
    _merge_diag!(total_counts, total_times, local_snap)
    _merge_edges!(total_edges, local_snap)

    if by_worker && nworkers() > 0
        for w in workers()
            w == myid() && continue
            snap = try
                Distributed.remotecall_fetch(ERPGen.diagnostics_snapshot, w)
            catch
                continue
            end
            _merge_diag!(total_counts, total_times, snap)
            _merge_edges!(total_edges, snap)
        end
    end

    return (counts = total_counts, times = total_times, edges = total_edges, elapsed = local_snap.elapsed)
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

function _merge_edges!(edges, snap)
    for (k, v) in snap.edges
        prev = get(edges, k, (0, 0.0))
        edges[k] = (prev[1] + v[1], prev[2] + v[2])
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
            w == myid() && continue
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

# Print a sorted diagnostics overview.
function print_diagnostics_sorted(; by::Symbol = :time, desc::Bool = true, by_worker::Bool = true)
    if !DIAG.enabled
        println("Diagnostics disabled. Call enable_diagnostics!(true) first.")
        return nothing
    end

    snap = diagnostics_totals(by_worker = by_worker)
    keys_list = collect(keys(snap.counts))
    rows = map(keys_list) do k
        c = snap.counts[k]
        t = get(snap.times, k, 0.0)
        avg = c > 0 ? t / c : 0.0
        (name = k, count = c, time = t, avg = avg)
    end

    sort_key = by === :count ? r -> r.count :
               by === :avg   ? r -> r.avg :
               by === :name  ? r -> string(r.name) :
               r -> r.time
    sort!(rows, by = sort_key, rev = desc)

    println("Diagnostics total elapsed=", round(snap.elapsed; digits = 3), "s (sorted by ", by, ")")
    for r in rows
        println("  ", r.name,
            ": count=", r.count,
            ", total=", round(r.time; digits = 3), "s",
            ", avg=", round(r.avg; digits = 6), "s")
    end
    return nothing
end

# Print a hierarchical diagnostics tree.
function print_diagnostics_tree(; by::Symbol = :time, desc::Bool = true, by_worker::Bool = true)
    if !DIAG.enabled
        println("Diagnostics disabled. Call enable_diagnostics!(true) first.")
        return nothing
    end

    snap = diagnostics_totals(by_worker = by_worker)
    counts = snap.counts
    times = snap.times
    edges = snap.edges

    if isempty(edges)
        if isempty(counts)
            println("Diagnostics enabled but no data collected.")
            return nothing
        end
        root_children = collect(keys(counts))
        sort!(root_children, by = sort_key, rev = desc)
        root_total = sum(get(times, child, 0.0) for child in root_children)
        root_total = root_total > 0 ? root_total : 1.0
        rows = NamedTuple[]
        for child in root_children
            calls = get(counts, child, 0)
            total = get(times, child, 0.0)
            avg = calls > 0 ? total / calls : 0.0
            pct = (total / root_total) * 100
            push!(rows, (
                name = string(child),
                calls = calls,
                total_str = _format_diag_time(total),
                avg_str = _format_diag_time(avg),
                pct_str = _format_diag_percent(pct),
            ))
        end
        println(format_diagnostics_table(rows))
        return nothing
    end

    children = Dict{Symbol, Vector{Symbol}}()
    for ((parent, child), _) in edges
        push!(get!(children, parent, Symbol[]), child)
    end

    sort_key = by === :count ? n -> get(counts, n, 0) :
               by === :avg   ? n -> begin
                   c = get(counts, n, 0)
                   t = get(times, n, 0.0)
                   c > 0 ? t / c : 0.0
               end :
               by === :name  ? n -> string(n) :
               n -> get(times, n, 0.0)
    root_children = get(children, :root, Symbol[])
    sort!(root_children, by = sort_key, rev = desc)

    if isempty(root_children)
        println("Diagnostics enabled but no data collected.")
        return nothing
    end

    root_total = sum((get(times, child, 0.0) for child in root_children); init = 0.0)
    root_total = root_total > 0 ? root_total : 1.0

    rows = NamedTuple[]

    function _add_node(name::Symbol, prefix::AbstractString, is_last::Bool, parent_time::Real, is_root::Bool)
        calls = get(counts, name, 0)
        total = get(times, name, 0.0)
        avg = calls > 0 ? total / calls : 0.0
        pct = parent_time > 0 ? (total / parent_time) * 100 : 0.0

        display_name = is_root ? string(name) :
            string(prefix, is_last ? "└─ " : "├─ ", name)

        push!(rows, (
            name = display_name,
            calls = calls,
            total_str = _format_diag_time(total),
            avg_str = _format_diag_time(avg),
            pct_str = _format_diag_percent(pct),
        ))

        kids = get(children, name, Symbol[])
        if !isempty(kids)
            sort!(kids, by = sort_key, rev = desc)
            next_prefix = is_root ? "" : string(prefix, is_last ? "   " : "│  ")
            for (idx, child) in enumerate(kids)
                _add_node(child, next_prefix, idx == length(kids), total, false)
            end
        end
    end

    for (idx, child) in enumerate(root_children)
        _add_node(child, "", idx == length(root_children), root_total, true)
    end

    println(format_diagnostics_table(rows))
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
