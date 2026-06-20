# Native GLMakie ERPgnostics-style explorer for the real-data training scores.
#
# Run from the repository root:
#
#     julia --project=src/real_data_training src/real_data_training/erpgnostics_topoplot_explorer.jl
#
# Optional environment variables:
#     REAL_DATA_TRAINING_PARENT_SCORES   CSV path, defaults to src/real_data_training/lean_parent_scores.csv
#     REAL_DATA_TRAINING_DATASETS_ROOT   dataset root, defaults to datasets/
#     REAL_DATA_TRAINING_START_DATASET   initial dataset key
#     REAL_DATA_TRAINING_START_SORT      initial sorting variable
#     REAL_DATA_TRAINING_START_CHANNEL   initial channel name
#     REAL_DATA_TRAINING_EXPLORER_SMOKE  true/false, load one detail without opening a window

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        pwd(),
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
        joinpath(pwd(), ".."),
        joinpath(pwd(), "..", ".."),
    ]))
    for candidate in candidates
        if isdir(joinpath(candidate, "datasets")) && isdir(joinpath(candidate, "src"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir), pwd=$(pwd()).")
end

const REPO_ROOT = find_repo_root()
const REAL_DATA_TRAINING_DIR = @__DIR__

Pkg.activate(REAL_DATA_TRAINING_DIR; io = devnull)

using CSV
using DataFrames
using ImageFiltering: KernelFactors, imfilter
using JLD2
using Printf: @sprintf
using Statistics
using GLMakie

GLMakie.activate!()

env_config(name::AbstractString, default::AbstractString) = get(ENV, name, default)

const DATASETS_ROOT = env_config("REAL_DATA_TRAINING_DATASETS_ROOT", joinpath(REPO_ROOT, "datasets"))
const DEFAULT_PARENT_SCORES_PATH = get(
    ENV,
    "REAL_DATA_TRAINING_PARENT_SCORES",
    joinpath(REAL_DATA_TRAINING_DIR, "lean_parent_scores.csv"),
)

const REFERENCE_DATASET_KEY = "fixations_dataset"
# Load the channel positions from their original location instead of a local copy.
const REFERENCE_POSITIONS_PATH = env_config(
    "REAL_DATA_TRAINING_REFERENCE_POSITIONS",
    joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY, "positions_128.jld2"),
)
const DETAIL_LOWPASS_SIGMA = 75.0f0
const DETAIL_LOWPASS_KERNEL_SIZE = (21, 21)
const DETAIL_FILTER_BORDER = "reflect"
const TOPOPLOT_CENTER = (0.5, 0.5)
const TOPOPLOT_SAFE_RADIUS = 0.47
const TOPOPLOT_MIN_DISTANCE = 0.055

cellstr(x) = (ismissing(x) || x === nothing) ? "" : string(x)

dataset_dir(dataset_key::AbstractString) = joinpath(DATASETS_ROOT, dataset_key)
events_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "events.jld2")
signals_dir(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "signals")
signal_path(dataset_key::AbstractString, channel_name::AbstractString) =
    joinpath(signals_dir(dataset_key), string(channel_name, ".jld2"))

function channel_index_from_name(channel_name::AbstractString)
    for pattern in (r"^ch0*(\d+)$"i, r"^E0*(\d+)$"i)
        m = match(pattern, String(channel_name))
        m === nothing || return parse(Int, m.captures[1])
    end
    return 0
end

function signal_metadata(dataset_key::AbstractString, channel_name::AbstractString)
    path = signal_path(dataset_key, channel_name)
    isfile(path) || return Dict{String, Any}()
    return JLD2.load(path, "metadata")
end

function channel_index(dataset_key::AbstractString, channel_name::AbstractString)
    metadata = signal_metadata(dataset_key, channel_name)
    return Int(get(metadata, "channel_idx", channel_index_from_name(channel_name)))
end

function dataset_channels(dataset_key::AbstractString)
    dir = signals_dir(dataset_key)
    isdir(dir) || error("Missing signals directory: $(dir)")
    rows = NamedTuple[]
    for file in filter(path -> endswith(path, ".jld2"), readdir(dir; join = false))
        channel_name = replace(file, ".jld2" => "")
        push!(rows, (
            dataset_key = String(dataset_key),
            channel_name = channel_name,
            channel_idx = channel_index(dataset_key, channel_name),
        ))
    end
    channels = DataFrame(rows)
    sort!(channels, [:channel_idx, :channel_name])
    return channels
end

function load_events_file(dataset_key::AbstractString)
    path = events_path(dataset_key)
    isfile(path) || error("Missing events file: $(path)")
    return (
        events = JLD2.load(path, "events"),
        metadata = JLD2.load(path, "metadata"),
    )
end

function load_signal_file(dataset_key::AbstractString, channel_name::AbstractString)
    path = signal_path(dataset_key, channel_name)
    isfile(path) || error("Missing signal file: $(path)")
    data = Matrix{Float32}(JLD2.load(path, "data_time_trials"))
    metadata = JLD2.load(path, "metadata")
    return (
        data_time_trials = data,
        metadata = metadata,
        channel_idx = Int(get(metadata, "channel_idx", channel_index_from_name(channel_name))),
    )
end

const EVENTS_CACHE = Dict{String, Any}()
const SIGNAL_CACHE = Dict{Tuple{String, String}, Any}()
const DETAIL_CACHE = Dict{Tuple{String, String, String}, Any}()

events_for_dataset(dataset_key::AbstractString) =
    get!(() -> load_events_file(dataset_key), EVENTS_CACHE, String(dataset_key))

signal_for_channel(dataset_key::AbstractString, channel_name::AbstractString) =
    get!(() -> load_signal_file(dataset_key, channel_name), SIGNAL_CACHE, (String(dataset_key), String(channel_name)))

function score_column(df::DataFrame, candidates::Vector{Symbol}, label::AbstractString)
    for name in candidates
        name in propertynames(df) && return name
    end
    error("Score CSV is missing a $(label) column. Tried: $(join(string.(candidates), ", ")).")
end

function normalize_parent_scores(raw::DataFrame)
    dataset_col = score_column(raw, [:dataset_key, :dataset], "dataset")
    sort_col = score_column(raw, [:sort_variable, :sorting_variable], "sort variable")
    channel_col = score_column(raw, [:channel_name, :channel], "channel")
    score_col = score_column(raw, [:score_class, :score], "score")

    manual_col = :manual_label in propertynames(raw) ? :manual_label :
        (:true_erp_class in propertynames(raw) ? :true_erp_class : nothing)
    has_manual_col = :has_manual_label in propertynames(raw) ? :has_manual_label : nothing
    channel_idx_col = :channel_idx in propertynames(raw) ? :channel_idx : nothing

    rows = NamedTuple[]
    for row in eachrow(raw)
        dataset_key = cellstr(row[dataset_col])
        sort_variable = cellstr(row[sort_col])
        channel_name = cellstr(row[channel_col])
        manual_label = manual_col === nothing ? "unlabeled" : cellstr(row[manual_col])
        isempty(manual_label) && (manual_label = "unlabeled")
        manual_label == "unlabelled" && (manual_label = "unlabeled")
        has_manual = has_manual_col === nothing ?
            !(manual_label in ("", "unlabeled", "unlabelled")) :
            (row[has_manual_col] === true || lowercase(cellstr(row[has_manual_col])) in ("true", "1", "yes"))
        idx = channel_idx_col === nothing ?
            channel_index(dataset_key, channel_name) :
            Int(row[channel_idx_col])
        binary = manual_label in ("", "unlabeled", "unlabelled", "no_class") ? 0 : 1

        push!(rows, (
            dataset_key = dataset_key,
            sort_variable = sort_variable,
            channel_name = channel_name,
            channel_idx = idx,
            score_class = Float32(row[score_col]),
            true_erp_class = manual_label,
            has_manual_label = Bool(has_manual),
            true_binary_label = binary,
        ))
    end

    scores = DataFrame(rows)
    sort!(scores, [:dataset_key, :sort_variable, :channel_idx, :channel_name])
    return scores
end

function load_parent_scores(path::AbstractString = DEFAULT_PARENT_SCORES_PATH)
    isfile(path) || error("Missing parent score CSV: $(path)")
    return normalize_parent_scores(CSV.read(path, DataFrame))
end

function score_dataset_keys(score_df::DataFrame)
    return sort(unique(String.(score_df.dataset_key)))
end

function score_sort_variables(score_df::DataFrame, dataset_key::AbstractString)
    sub = score_df[score_df.dataset_key .== String(dataset_key), :]
    vars = sort(unique(String.(sub.sort_variable)))
    return vars
end

function scored_channel_names(score_df::DataFrame, dataset_key::AbstractString, sort_variable::AbstractString)
    sub = score_df[
        (score_df.dataset_key .== String(dataset_key)) .&
        (score_df.sort_variable .== String(sort_variable)),
        :,
    ]
    sort!(sub, [:channel_idx, :channel_name])
    return String.(sub.channel_name)
end

function first_valid_or(default_value::AbstractString, options::Vector{String})
    isempty(options) && error("No options available.")
    candidate = String(default_value)
    return candidate in options ? candidate : first(options)
end

function initial_dataset_key(score_df::DataFrame)
    keys = score_dataset_keys(score_df)
    isempty(keys) && error("Score CSV does not contain any datasets.")
    preferred = env_config("REAL_DATA_TRAINING_START_DATASET", "")
    !isempty(preferred) && return first_valid_or(preferred, keys)
    REFERENCE_DATASET_KEY in keys && return REFERENCE_DATASET_KEY
    return first(keys)
end

function initial_sort_variable(score_df::DataFrame, dataset_key::AbstractString)
    vars = score_sort_variables(score_df, dataset_key)
    isempty(vars) && error("No scored sort variables for $(dataset_key).")
    preferred = env_config("REAL_DATA_TRAINING_START_SORT", "")
    !isempty(preferred) && return first_valid_or(preferred, vars)
    for candidate in ("duration", "fixation_duration", "fixation_duration_ms", "latency", "epoch_index")
        candidate in vars && return candidate
    end
    return first(vars)
end

function initial_channel_name(score_df::DataFrame, dataset_key::AbstractString, sort_variable::AbstractString)
    channels = scored_channel_names(score_df, dataset_key, sort_variable)
    isempty(channels) && error("No scored channels for $(dataset_key), $(sort_variable).")
    preferred = env_config("REAL_DATA_TRAINING_START_CHANNEL", "")
    !isempty(preferred) && return first_valid_or(preferred, channels)

    sub = score_df[
        (score_df.dataset_key .== String(dataset_key)) .&
        (score_df.sort_variable .== String(sort_variable)),
        :,
    ]
    positives = sub[Int.(sub.true_binary_label) .== 1, :]
    if !isempty(positives)
        sort!(positives, [:channel_idx, :channel_name])
        return cellstr(positives.channel_name[1])
    end
    return first(channels)
end

function load_reference_positions()
    isfile(REFERENCE_POSITIONS_PATH) || error("Missing reference positions file: $(REFERENCE_POSITIONS_PATH)")
    positions = JLD2.load(REFERENCE_POSITIONS_PATH, "single_stored_object")
    rows = NamedTuple[]
    for (idx, point) in enumerate(positions)
        push!(rows, (
            dataset_key = REFERENCE_DATASET_KEY,
            channel_name = @sprintf("ch%03d", idx),
            channel_idx = idx,
            x = Float64(point[1]),
            y = Float64(point[2]),
            position_source = "positions_128",
        ))
    end
    return DataFrame(rows)
end

function project_to_topoplot_circle(x::Real, y::Real; radius::Float64 = TOPOPLOT_SAFE_RADIUS)
    cx, cy = TOPOPLOT_CENTER
    xf, yf = Float64(x), Float64(y)
    (isfinite(xf) && isfinite(yf)) || return cx, cy

    dx, dy = xf - cx, yf - cy
    r = hypot(dx, dy)
    (r <= radius || r == 0.0) && return xf, yf
    scale = radius / r
    return cx + dx * scale, cy + dy * scale
end

function clamp_topoplot_positions!(xs::Vector{Float64}, ys::Vector{Float64};
        radius::Float64 = TOPOPLOT_SAFE_RADIUS)
    for i in eachindex(xs)
        xs[i], ys[i] = project_to_topoplot_circle(xs[i], ys[i]; radius = radius)
    end
    return xs, ys
end

function relax_topoplot_positions!(xs::Vector{Float64}, ys::Vector{Float64};
        min_distance::Float64 = TOPOPLOT_MIN_DISTANCE,
        radius::Float64 = TOPOPLOT_SAFE_RADIUS,
        iterations::Int = 120)
    n = length(xs)
    n <= 1 && return xs, ys

    for _ in 1:iterations
        moved = false
        for i in 1:(n - 1), j in (i + 1):n
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            d = hypot(dx, dy)
            d >= min_distance && continue

            if d < 1e-9
                theta = 2pi * (0.61803398875 * i + 0.41421356237 * j)
                ux, uy = cos(theta), sin(theta)
                d = 0.0
            else
                ux, uy = dx / d, dy / d
            end

            shift = 0.5 * (min_distance - d)
            xs[i] += ux * shift
            ys[i] += uy * shift
            xs[j] -= ux * shift
            ys[j] -= uy * shift
            moved = true
        end
        clamp_topoplot_positions!(xs, ys; radius = radius)
        moved || break
    end
    return xs, ys
end

function topoplot_layout_positions(positions::DataFrame)
    out = copy(positions)
    xs = Float64.(out.x)
    ys = Float64.(out.y)
    clamp_topoplot_positions!(xs, ys)
    relax_topoplot_positions!(xs, ys)
    out.x = xs
    out.y = ys
    return out
end

function side_x(suffix::AbstractString)
    suffix == "Z" && return 0.5
    n = tryparse(Int, suffix)
    n === nothing && return nothing
    if isodd(n)
        return get(Dict(1 => 0.42, 3 => 0.34, 5 => 0.24, 7 => 0.14, 9 => 0.04), n, 0.08)
    end
    return get(Dict(2 => 0.58, 4 => 0.66, 6 => 0.76, 8 => 0.86, 10 => 0.96), n, 0.92)
end

function standard_channel_position(channel_name::AbstractString)
    name = uppercase(strip(String(channel_name)))
    compact = replace(name, r"[^A-Z0-9]" => "")

    occursin("HEOG", compact) && return (1.08, 0.64)
    occursin("VEOG", compact) && return (0.50, 1.10)
    compact in ("LO1", "LOC1") && return (0.10, 0.86)
    compact in ("LO2", "LOC2") && return (0.90, 0.86)
    compact in ("IO1", "IOG1") && return (0.38, 1.06)
    compact in ("IO2", "IOG2") && return (0.62, 1.06)
    compact == "A1" && return (0.00, 0.50)
    compact == "A2" && return (1.00, 0.50)

    m = match(r"^(FP|AF|FT|FC|TP|CP|PO|F|C|T|P|O|I)(Z|[0-9]+)$", compact)
    m === nothing && return nothing
    prefix, suffix = m.captures
    y = get(Dict(
        "FP" => 0.95,
        "AF" => 0.86,
        "F" => 0.76,
        "FT" => 0.64,
        "FC" => 0.62,
        "T" => 0.50,
        "C" => 0.50,
        "TP" => 0.36,
        "CP" => 0.36,
        "P" => 0.24,
        "PO" => 0.13,
        "O" => 0.04,
        "I" => -0.03,
    ), prefix, nothing)
    y === nothing && return nothing
    x = side_x(suffix)
    x === nothing && return nothing

    # Frontal-polar and occipital rows are narrower on the projected head.
    if prefix == "FP" && suffix != "Z"
        x = x < 0.5 ? 0.35 : 0.65
    elseif prefix == "O" && suffix != "Z"
        x = x < 0.5 ? 0.40 : 0.60
    elseif prefix == "I" && suffix != "Z"
        x = x < 0.5 ? 0.42 : 0.58
    end
    return (Float64(x), Float64(y))
end

function synthetic_channel_positions(channels::DataFrame)
    n = nrow(channels)
    rows = NamedTuple[]
    golden_angle = pi * (3 - sqrt(5))
    for (rank, row) in enumerate(eachrow(channels))
        if n == 1
            x, y = 0.5, 0.5
        else
            radius = 0.48 * sqrt((rank - 0.5) / n)
            theta = rank * golden_angle
            x = 0.5 + radius * cos(theta)
            y = 0.5 + radius * sin(theta)
        end
        push!(rows, (
            dataset_key = cellstr(row.dataset_key),
            channel_name = cellstr(row.channel_name),
            channel_idx = Int(row.channel_idx),
            x = Float64(x),
            y = Float64(y),
            position_source = "synthetic",
        ))
    end
    return DataFrame(rows)
end

function standard_or_synthetic_channel_positions(channels::DataFrame)
    synth = synthetic_channel_positions(channels)
    positions = Vector{Union{Nothing, Tuple{Float64, Float64}}}(undef, nrow(channels))
    n_known = 0
    for (i, row) in enumerate(eachrow(channels))
        pos = standard_channel_position(row.channel_name)
        positions[i] = pos
        pos === nothing || (n_known += 1)
    end
    use_standard = n_known >= min(nrow(channels), max(3, ceil(Int, 0.35 * nrow(channels))))
    use_standard || return synth

    rows = NamedTuple[]
    for i in 1:nrow(channels)
        row = channels[i, :]
        pos = positions[i]
        if pos === nothing
            x, y = synth.x[i], synth.y[i]
            source = "synthetic"
        else
            x, y = pos
            source = "standard_1020_approx"
        end
        push!(rows, (
            dataset_key = cellstr(row.dataset_key),
            channel_name = cellstr(row.channel_name),
            channel_idx = Int(row.channel_idx),
            x = Float64(x),
            y = Float64(y),
            position_source = source,
        ))
    end
    return DataFrame(rows)
end

function load_dataset_positions(dataset_key::AbstractString)
    positions = if String(dataset_key) == REFERENCE_DATASET_KEY && isfile(REFERENCE_POSITIONS_PATH)
        load_reference_positions()
    else
        standard_or_synthetic_channel_positions(dataset_channels(dataset_key))
    end
    return topoplot_layout_positions(positions)
end

function draw_head_outline!(ax)
    theta = range(0, 2pi; length = 241)
    lines!(ax, 0.5 .+ 0.52 .* cos.(theta), 0.5 .+ 0.52 .* sin.(theta); color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [0.46, 0.50, 0.54], [1.01, 1.10, 1.01]; color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [-0.04, -0.10, -0.04], [0.55, 0.50, 0.45]; color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [1.04, 1.10, 1.04], [0.55, 0.50, 0.45]; color = :gray35, linewidth = 2, inspectable = false)
    return ax
end

function closest_channel_index(mouse_pos, positions::DataFrame)
    best_idx = 1
    best_dist_sq = Inf
    for i in 1:nrow(positions)
        dx = Float64(positions.x[i]) - Float64(mouse_pos[1])
        dy = Float64(positions.y[i]) - Float64(mouse_pos[2])
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq
            best_dist_sq = dist_sq
            best_idx = i
        end
    end
    return best_idx
end

function score_positions(score_df::DataFrame, dataset_key::AbstractString, sort_variable::AbstractString)
    positions = load_dataset_positions(dataset_key)
    sub = score_df[
        (score_df.dataset_key .== String(dataset_key)) .&
        (score_df.sort_variable .== String(sort_variable)),
        :,
    ]
    out = leftjoin(positions, sub; on = [:dataset_key, :channel_name, :channel_idx])
    out.score_class = coalesce.(out.score_class, 0.0f0)
    out.true_binary_label = coalesce.(out.true_binary_label, 0)
    out.true_erp_class = coalesce.(out.true_erp_class, "unlabeled")
    out.has_manual_label = coalesce.(out.has_manual_label, false)
    sort!(out, [:channel_idx, :channel_name])
    return out
end

function score_row(score_df::DataFrame, dataset_key::AbstractString, sort_variable::AbstractString, channel_name::AbstractString)
    idx = findfirst(
        i -> score_df.dataset_key[i] == String(dataset_key) &&
            score_df.sort_variable[i] == String(sort_variable) &&
            score_df.channel_name[i] == String(channel_name),
        1:nrow(score_df),
    )
    idx === nothing && return nothing
    return score_df[idx, :]
end

function display_class_label(label)
    value = cellstr(label)
    (isempty(value) || value == "unlabelled") && (value = "unlabeled")
    return replace(value, "_" => " ")
end

function is_valid_sort_value(v)
    (ismissing(v) || v === nothing) && return false
    v isa Real && return isfinite(Float64(v))
    return !isempty(strip(string(v)))
end

function valid_sort_mask(events::DataFrame, sort_col::Symbol)
    sort_col in propertynames(events) || error("Sort column $(sort_col) missing.")
    return [is_valid_sort_value(v) for v in events[!, sort_col]]
end

function sortvalues_from(events::DataFrame, sort_col::Symbol)
    values = events[!, sort_col]
    finite_values = collect(skipmissing(values))
    if all(v -> v isa Real, finite_values)
        return Float64.(values)
    end
    return string.(values)
end

function sorted_order_for_variant(events::DataFrame, sort_col::Symbol; inverse_sort::Bool = false)
    order = sortperm(sortvalues_from(events, sort_col))
    inverse_sort && reverse!(order)
    return order
end

function zscore_timepoints_local(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    mu = mean(x; dims = 2)
    sigma = std(x; dims = 2, corrected = true)
    sigma_safe = map(sigma) do value
        value32 = Float32(value)
        isfinite(value32) && value32 > 0f0 ? value32 : 1f0
    end
    return Float32.((x .- Float32.(mu)) ./ sigma_safe)
end

function gaussian_kernel_for_target(sigma_factor, in_size, target_size, kernel_size)
    kernel_size = Tuple(Int.(kernel_size))
    target_size = Tuple(Int.(target_size))
    all(isodd, kernel_size) || throw(ArgumentError("kernel_size must be odd in both dimensions, got $(kernel_size)."))

    sigma_trials = max(Float32(sigma_factor) * Float32(in_size[1]) / Float32(target_size[1]), 1f-3)
    sigma_time = max(Float32(sigma_factor) * Float32(in_size[2]) / Float32(target_size[2]), 1f-3)
    return KernelFactors.gaussian((sigma_trials, sigma_time), kernel_size)
end

function smooth_detail_image(img_trials_time::AbstractMatrix)
    img = Float32.(img_trials_time)
    min(size(img)...) <= 1 && return Matrix{Float32}(img)
    kernel = gaussian_kernel_for_target(
        DETAIL_LOWPASS_SIGMA,
        size(img),
        size(img),
        DETAIL_LOWPASS_KERNEL_SIZE,
    )
    return Matrix{Float32}(imfilter(img, kernel, DETAIL_FILTER_BORDER))
end

function numeric_sort_values(values)
    out = Float64[]
    ok = true
    for value in values
        if ismissing(value) || value === nothing
            push!(out, NaN)
            continue
        end
        parsed = value isa Real ? Float64(value) : tryparse(Float64, string(value))
        if parsed === nothing
            ok = false
            break
        end
        push!(out, parsed)
    end
    ok && return out, String("")

    levels = Dict{String, Int}()
    encoded = Float64[]
    for value in string.(values)
        if !haskey(levels, value)
            levels[value] = length(levels) + 1
        end
        push!(encoded, levels[value])
    end
    return encoded, "category code"
end

function filtered_origin(dataset_key::AbstractString, sort_variable::AbstractString, channel_name::AbstractString)
    events_bundle = events_for_dataset(dataset_key)
    signal_bundle = signal_for_channel(dataset_key, channel_name)
    n = min(nrow(events_bundle.events), size(signal_bundle.data_time_trials, 2))
    events = events_bundle.events[1:n, :]
    data = signal_bundle.data_time_trials[:, 1:n]
    sort_col = Symbol(sort_variable)
    keep = valid_sort_mask(events, sort_col)
    any(keep) || error("No valid sort values for $(dataset_key), $(sort_variable).")
    return (
        events = events[keep, :],
        metadata = events_bundle.metadata,
        data_time_trials = data[:, keep],
        channel_idx = signal_bundle.channel_idx,
    )
end

function dataset_detail_interactive(dataset_key::AbstractString, sort_variable::AbstractString, channel_name::AbstractString)
    key = (String(dataset_key), String(sort_variable), String(channel_name))
    return get!(DETAIL_CACHE, key) do
        origin = filtered_origin(dataset_key, sort_variable, channel_name)
        sort_col = Symbol(sort_variable)
        order = sorted_order_for_variant(origin.events, sort_col)
        data_sorted = origin.data_time_trials[:, order]
        sort_values_full = sortvalues_from(origin.events, sort_col)[order]

        z = zscore_timepoints_local(data_sorted)
        img = smooth_detail_image(Float32.(permutedims(z, (2, 1))))

        time_start_s = Float64(get(origin.metadata, "time_start_s", 0.0))
        time_end_s = Float64(get(origin.metadata, "time_end_s", 1.0))
        times = collect(range(time_start_s, time_end_s; length = size(data_sorted, 1)))
        trials = collect(1:size(data_sorted, 2))
        sort_values = Any[value for value in sort_values_full]
        mean_wave_full = vec(mean(data_sorted; dims = 2))
        mean_wave = Float32.(mean_wave_full)

        return (
            image = img,
            times = times,
            trials = trials,
            sort_values = sort_values,
            mean_wave = mean_wave,
            channel_idx = origin.channel_idx,
            full_n_trials = size(data_sorted, 2),
            full_n_timepoints = size(data_sorted, 1),
        )
    end
end

function finite_float_values(values)
    out = Float64[]
    for value in values
        v = Float64(value)
        isfinite(v) && push!(out, v)
    end
    return out
end

function erp_image_color_stats(img::AbstractMatrix; q_low::Float64 = 0.02, q_high::Float64 = 0.98)
    vals = finite_float_values(img)
    if isempty(vals)
        vmax = 1.0
    else
        lo = quantile(vals, q_low)
        hi = quantile(vals, q_high)
        vmax = max(abs(lo), abs(hi), 1e-6)
    end
    clipped = clamp.(Float32.(img), Float32(-vmax), Float32(vmax))
    tick_vals = Float32[-vmax, 0.0f0, vmax]
    tick_labels = [@sprintf("%.2g", -vmax), "0", @sprintf("%.2g", vmax)]
    return (
        image = Matrix{Float32}(clipped),
        colorrange = (Float32(-vmax), Float32(vmax)),
        ticks = (tick_vals, tick_labels),
        colormap = :RdBu,
    )
end

function option_at(options, idx)
    idx isa Integer || return nothing
    isempty(options) && return nothing
    return options[clamp(Int(idx), 1, length(options))]
end

function real_data_training_erpgnostics_explorer(;
        score_path::AbstractString = DEFAULT_PARENT_SCORES_PATH,
        start_dataset_key::AbstractString = env_config("REAL_DATA_TRAINING_START_DATASET", ""),
        start_sort_variable::AbstractString = env_config("REAL_DATA_TRAINING_START_SORT", ""),
        start_channel::AbstractString = env_config("REAL_DATA_TRAINING_START_CHANNEL", ""))

    score_df = load_parent_scores(score_path)
    dataset_keys = score_dataset_keys(score_df)
    isempty(dataset_keys) && error("No datasets in score file: $(score_path)")

    initial_dataset = isempty(start_dataset_key) ?
        initial_dataset_key(score_df) :
        first_valid_or(start_dataset_key, dataset_keys)

    initial_sorts = score_sort_variables(score_df, initial_dataset)
    initial_sort = isempty(start_sort_variable) ?
        initial_sort_variable(score_df, initial_dataset) :
        first_valid_or(start_sort_variable, initial_sorts)

    initial_channels = scored_channel_names(score_df, initial_dataset, initial_sort)
    initial_channel = isempty(start_channel) ?
        initial_channel_name(score_df, initial_dataset, initial_sort) :
        first_valid_or(start_channel, initial_channels)

    selected_dataset = Observable(initial_dataset)
    selected_sort = Observable(initial_sort)
    selected_channel = Observable(initial_channel)

    function valid_sort(dataset_key, sort_variable)
        return first_valid_or(String(sort_variable), score_sort_variables(score_df, dataset_key))
    end

    function valid_channel(dataset_key, sort_variable, channel_name)
        return first_valid_or(String(channel_name), scored_channel_names(score_df, dataset_key, sort_variable))
    end

    valid_sort_obs = lift(selected_dataset, selected_sort) do ds, sv
        valid_sort(ds, sv)
    end

    valid_channel_obs = lift(selected_dataset, valid_sort_obs, selected_channel) do ds, sv, ch
        valid_channel(ds, sv, ch)
    end

    topo_df_obs = lift(selected_dataset, valid_sort_obs) do ds, sv
        score_positions(score_df, ds, sv)
    end
    topo_x = lift(df -> Float64.(df.x), topo_df_obs)
    topo_y = lift(df -> Float64.(df.y), topo_df_obs)
    topo_score = lift(df -> Float32.(df.score_class), topo_df_obs)

    detail_obs = lift(selected_dataset, valid_sort_obs, valid_channel_obs) do ds, sv, ch
        dataset_detail_interactive(ds, sv, ch)
    end

    visual_obs = lift(d -> erp_image_color_stats(d.image), detail_obs)
    time_obs = lift(d -> d.times, detail_obs)
    trial_obs = lift(d -> d.trials, detail_obs)
    image_obs = lift(v -> v.image, visual_obs)
    image_colorrange_obs = lift(v -> v.colorrange, visual_obs)
    image_ticks_obs = lift(v -> v.ticks, visual_obs)
    image_colormap_obs = lift(v -> v.colormap, visual_obs)
    sort_curve_obs = lift(d -> first(numeric_sort_values(d.sort_values)), detail_obs)
    mean_obs = lift(d -> d.mean_wave, detail_obs)

    title_obs = lift(selected_dataset, valid_sort_obs, valid_channel_obs) do ds, sv, ch
        row = score_row(score_df, ds, sv, ch)
        row === nothing && return "unlabeled | score missing"
        return @sprintf("%s | model score %.3f", display_class_label(row.true_erp_class), Float64(row.score_class))
    end

    sort_xlabel_obs = lift(valid_sort_obs, detail_obs) do sv, d
        _, suffix = numeric_sort_values(d.sort_values)
        isempty(suffix) ? String(sv) : "$(sv) ($(suffix))"
    end

    fig = Figure(size = (1540, 920), figure_padding = (82, 30, 28, 18), backgroundcolor = :white)
    Label(fig[1, 1:4], title_obs; fontsize = 24, tellwidth = false)

    dataset_menu = Menu(fig[2, 1], options = dataset_keys, default = initial_dataset, width = 380)
    sort_menu = Menu(fig[3, 1], options = initial_sorts, default = initial_sort, width = 380)
    channel_menu = Menu(fig[4, 1], options = initial_channels, default = initial_channel, width = 380)
    updating_menus = Ref(false)

    function with_menu_update!(f)
        updating_menus[] = true
        try
            return f()
        finally
            updating_menus[] = false
        end
    end

    function set_dataset!(dataset_key::AbstractString)
        dataset_key = String(dataset_key)
        dataset_key in dataset_keys || return nothing
        sort_options = score_sort_variables(score_df, dataset_key)
        isempty(sort_options) && return nothing
        sort_variable = first(sort_options)
        channel_options = scored_channel_names(score_df, dataset_key, sort_variable)
        isempty(channel_options) && return nothing

        with_menu_update!() do
            selected_dataset[] = dataset_key
            selected_sort[] = sort_variable
            selected_channel[] = first(channel_options)

            sort_menu.options[] = sort_options
            sort_menu.i_selected[] = 1
            sort_menu.selection[] = sort_variable
            channel_menu.options[] = channel_options
            channel_menu.i_selected[] = 1
            channel_menu.selection[] = first(channel_options)
        end
        return nothing
    end

    function set_sort!(sort_variable::AbstractString)
        sort_variable = String(sort_variable)
        sort_options = score_sort_variables(score_df, selected_dataset[])
        sort_variable in sort_options || return nothing

        channel_options = scored_channel_names(score_df, selected_dataset[], sort_variable)
        isempty(channel_options) && return nothing
        channel_name = valid_channel(selected_dataset[], sort_variable, selected_channel[])
        channel_idx = findfirst(==(channel_name), channel_options)
        channel_idx === nothing && (channel_idx = 1)

        with_menu_update!() do
            selected_sort[] = sort_variable
            selected_channel[] = channel_options[Int(channel_idx)]
            channel_menu.options[] = channel_options
            channel_menu.i_selected[] = Int(channel_idx)
            channel_menu.selection[] = channel_options[Int(channel_idx)]
        end
        return nothing
    end

    function set_channel!(channel_name::AbstractString)
        channel_name = String(channel_name)
        options = scored_channel_names(score_df, selected_dataset[], valid_sort_obs[])
        channel_name in options || return nothing
        selected_channel[] = channel_name
        return nothing
    end

    on(dataset_menu.selection) do ds
        updating_menus[] && return nothing
        ds === nothing && return nothing
        set_dataset!(String(ds))
    end
    on(dataset_menu.i_selected) do idx
        updating_menus[] && return nothing
        ds = option_at(dataset_keys, idx)
        ds === nothing || set_dataset!(ds)
    end

    on(sort_menu.selection) do sv
        updating_menus[] && return nothing
        sv === nothing && return nothing
        set_sort!(String(sv))
    end
    on(sort_menu.i_selected) do idx
        updating_menus[] && return nothing
        sv = option_at(score_sort_variables(score_df, selected_dataset[]), idx)
        sv === nothing || set_sort!(sv)
    end

    on(channel_menu.selection) do ch
        updating_menus[] && return nothing
        ch === nothing && return nothing
        set_channel!(String(ch))
    end
    on(channel_menu.i_selected) do idx
        updating_menus[] && return nothing
        ch = option_at(scored_channel_names(score_df, selected_dataset[], valid_sort_obs[]), idx)
        ch === nothing || set_channel!(ch)
    end

    ax_topo = Axis(fig[5:8, 1]; title = "Click a channel", aspect = DataAspect(), backgroundcolor = :white)
    hidedecorations!(ax_topo)
    hidespines!(ax_topo)
    xlims!(ax_topo, -0.32, 1.32)
    ylims!(ax_topo, -0.18, 1.22)
    draw_head_outline!(ax_topo)

    topo_scatter = scatter!(
        ax_topo,
        topo_x,
        topo_y;
        color = topo_score,
        colormap = :viridis,
        colorrange = (0.0f0, 1.0f0),
        markersize = 16,
        strokewidth = 0,
        inspectable = false,
    )
    Colorbar(fig[9, 1], topo_scatter; label = "P(pattern)", vertical = false, width = GLMakie.Relative(0.90))

    on(events(ax_topo.scene).mousebutton, priority = 20) do event
        if event.button == Mouse.left && event.action == Mouse.press && is_mouseinside(ax_topo.scene)
            positions = topo_df_obs[]
            idx = closest_channel_index(mouseposition(ax_topo.scene), positions)
            channel_name = cellstr(positions.channel_name[idx])
            set_channel!(channel_name)
            channel_options = scored_channel_names(score_df, selected_dataset[], valid_sort_obs[])
            menu_idx = findfirst(==(channel_name), channel_options)
            if menu_idx !== nothing
                channel_menu.i_selected[] = Int(menu_idx)
                channel_menu.selection[] = channel_name
            end
            return Consume(true)
        end
        return Consume(false)
    end

    ax_img = Axis(fig[2:8, 2]; xlabel = "time after onset (s)", ylabel = "sorted trials", backgroundcolor = :white)
    hm = heatmap!(
        ax_img,
        time_obs,
        trial_obs,
        lift(img -> permutedims(img, (2, 1)), image_obs);
        colormap = image_colormap_obs,
        colorrange = image_colorrange_obs,
        inspectable = false,
    )
    Colorbar(fig[2:8, 3], hm; label = "z-scored voltage", ticks = image_ticks_obs, width = 18)

    ax_sort = Axis(fig[2:8, 4]; xlabel = sort_xlabel_obs, ylabel = "sorted trials", backgroundcolor = :white)
    lines!(ax_sort, sort_curve_obs, trial_obs; color = :gray20, linewidth = 2, inspectable = false)

    ax_mean = Axis(fig[9, 2]; xlabel = "time after onset (s)", ylabel = "mean ERP", backgroundcolor = :white)
    lines!(ax_mean, time_obs, mean_obs; color = :black, linewidth = 2.2, inspectable = false)
    linkxaxes!(ax_img, ax_mean)

    on(detail_obs) do _
        autolimits!(ax_img)
        autolimits!(ax_sort)
        autolimits!(ax_mean)
    end

    colsize!(fig.layout, 1, 500)
    colsize!(fig.layout, 2, 560)
    colsize!(fig.layout, 3, 36)
    colsize!(fig.layout, 4, 470)
    rowsize!(fig.layout, 9, 120)
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 8)

    return (
        figure = fig,
        score_df = score_df,
        selected_dataset = selected_dataset,
        selected_sort = selected_sort,
        selected_channel = selected_channel,
    )
end

function smoke_test()
    score_df = load_parent_scores(DEFAULT_PARENT_SCORES_PATH)
    ds = initial_dataset_key(score_df)
    sv = initial_sort_variable(score_df, ds)
    ch = initial_channel_name(score_df, ds, sv)
    detail = dataset_detail_interactive(ds, sv, ch)
    pos = score_positions(score_df, ds, sv)
    println("Loaded ", nrow(score_df), " scores from ", DEFAULT_PARENT_SCORES_PATH)
    println("Initial selection: ", ds, " / ", sv, " / ", ch)
    println("Detail image size: ", size(detail.image), ", full trials/timepoints: ", detail.full_n_trials, "/", detail.full_n_timepoints)
    println("Topoplot channels: ", nrow(pos), ", position source(s): ", join(sort(unique(String.(pos.position_source))), ", "))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if lowercase(env_config("REAL_DATA_TRAINING_EXPLORER_SMOKE", "false")) in ("true", "1", "yes")
        smoke_test()
    else
        app = real_data_training_erpgnostics_explorer()
        screen = display(app.figure)
        println("Real-data ERPgnostics explorer is running.")
        println("Score file: ", DEFAULT_PARENT_SCORES_PATH)
        println("Datasets root: ", DATASETS_ROOT)
        println("Close the GLMakie window to exit.")
        try
            wait(screen)
        catch
            readline()
        end
    end
end
