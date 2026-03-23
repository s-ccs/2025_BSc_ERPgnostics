module Week15Topoplots

using CairoMakie
using DataFrames
using HDF5
using JLD2
using LinearAlgebra
using Printf: @sprintf
using Statistics

using Main.Week15TryNewData
const TD = Main.Week15TryNewData

export activate_topoplot_backend!
export build_fixation_duration_topoplot_context
export plot_fixation_duration_topoplot

const FIXATION_POSITIONS_PATH = joinpath(TD.FIXATIONS_DATASET_DIR, "positions_128.jld2")
const DUMMY_FEATURE_NAMES = [
    :peak_trend_abs,
    :peak_span,
    :diag_advantage,
    :late_energy_shift,
    :row_energy_flatness,
]
const DUMMY_FEATURE_LABELS = Dict(
    :peak_trend_abs => "abs(cor(row_rank, peak_time))",
    :peak_span => "peak-time span",
    :diag_advantage => "main minus anti diagonal energy",
    :late_energy_shift => "late minus early abs-energy",
    :row_energy_flatness => "row-energy std",
)
const DUMMY_SIGMOID_WEIGHTS = Dict(
    :peak_trend_abs => 1.15,
    :peak_span => 0.90,
    :diag_advantage => 0.80,
    :late_energy_shift => 0.55,
    :row_energy_flatness => -0.45,
)
const DUMMY_SIGMOID_BIAS = -0.10

sigmoid_scalar(x::Real) = 1.0 / (1.0 + exp(-Float64(x)))

function safe_std(x::AbstractVector{<:Real})
    sigma = std(Float64.(x))
    return isfinite(sigma) && sigma > 0 ? sigma : 1.0
end

function safe_zscore(x::AbstractVector{<:Real})
    mu = mean(Float64.(x))
    sigma = safe_std(x)
    return (Float64.(x) .- mu) ./ sigma
end

function safe_cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x_f = Float64.(x)
    y_f = Float64.(y)
    if length(x_f) <= 1 || std(x_f) <= eps(Float64) || std(y_f) <= eps(Float64)
        return 0.0
    end
    return cor(x_f, y_f)
end

function activate_topoplot_backend!()
    try
        @eval using WGLMakie
        WGLMakie.activate!()
        return :WGLMakie
    catch err
        CairoMakie.activate!()
        @warn "WGLMakie is unavailable. Falling back to CairoMakie. The figure renders, but click interaction needs WGLMakie or GLMakie." exception = (err, catch_backtrace())
        return :CairoMakie
    end
end

function load_normalized_fixation_positions()
    @assert isfile(FIXATION_POSITIONS_PATH) "File not found: $FIXATION_POSITIONS_PATH"
    raw_positions = Point2f.(JLD2.load_object(FIXATION_POSITIONS_PATH))
    xs = Float64[first(p) for p in raw_positions]
    ys = Float64[last(p) for p in raw_positions]

    center_x = mean(xs)
    center_y = mean(ys)
    shifted = Point2f[(p[1] - center_x, p[2] - center_y) for p in raw_positions]
    radius = maximum(norm, shifted)
    radius <= 0 && (radius = 1.0)

    return Point2f[(p[1] / radius, p[2] / radius) for p in shifted]
end

function local_band_abs_mean(img::AbstractMatrix, row::Int, col::Int; band_radius::Int = 3)
    lo = max(1, col - band_radius)
    hi = min(size(img, 2), col + band_radius)
    return mean(abs, @view img[row, lo:hi])
end

function dummy_channel_features(img::AbstractMatrix)
    n_rows, n_cols = size(img)
    row_axis = collect(range(-1.0, 1.0; length = n_rows))
    peak_cols = Vector{Float64}(undef, n_rows)
    row_energy = Vector{Float64}(undef, n_rows)
    diag_vals = Vector{Float64}(undef, n_rows)
    anti_vals = Vector{Float64}(undef, n_rows)

    span_den = max(n_cols - 1, 1)
    for row in 1:n_rows
        row_view = @view img[row, :]
        _, peak_idx = findmax(abs.(row_view))
        peak_cols[row] = Float64(peak_idx)
        row_energy[row] = mean(abs, row_view)

        frac = n_rows == 1 ? 0.0 : (row - 1) / (n_rows - 1)
        main_col = Int(round(1 + frac * (n_cols - 1)))
        anti_col = Int(round(n_cols - frac * (n_cols - 1)))
        diag_vals[row] = local_band_abs_mean(img, row, main_col)
        anti_vals[row] = local_band_abs_mean(img, row, anti_col)
    end

    peak_cols_norm = (peak_cols .- 1.0) ./ span_den
    late_start = max(1, Int(round(0.62 * n_cols)))
    early_stop = min(n_cols, Int(round(0.38 * n_cols)))

    return (
        peak_trend_abs = abs(safe_cor(row_axis, peak_cols_norm)),
        peak_span = quantile(peak_cols_norm, 0.90) - quantile(peak_cols_norm, 0.10),
        diag_advantage = mean(diag_vals) - mean(anti_vals),
        late_energy_shift = mean(abs, @view img[:, late_start:end]) - mean(abs, @view img[:, 1:early_stop]),
        row_energy_flatness = std(row_energy),
    )
end

function interpolate_topomap(positions::AbstractVector, values::AbstractVector{<:Real};
        grid_size::Int = 180,
        power::Float64 = 3.0,
        head_radius::Float64 = 1.0)
    xs = collect(range(-1.08, 1.08; length = grid_size))
    ys = collect(range(-1.08, 1.08; length = grid_size))
    grid = Matrix{Float32}(undef, length(ys), length(xs))

    px = Float64[first(p) for p in positions]
    py = Float64[last(p) for p in positions]
    vv = Float64.(values)
    eps2 = 1e-6

    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        if hypot(x, y) > head_radius
            grid[j, i] = NaN32
            continue
        end

        dist2 = (px .- x) .^ 2 .+ (py .- y) .^ 2
        nearest = argmin(dist2)
        if dist2[nearest] < eps2
            grid[j, i] = Float32(vv[nearest])
            continue
        end

        weights = 1.0 ./ (dist2 .+ eps2) .^ (power / 2)
        grid[j, i] = Float32(sum(weights .* vv) / sum(weights))
    end

    return (xs = xs, ys = ys, values = grid)
end

function head_outline_points(; n::Int = 361, radius::Float64 = 1.0)
    angles = range(0, 2pi; length = n)
    return Point2f[(radius * cos(a), radius * sin(a)) for a in angles]
end

function nose_outline_points()
    return Point2f[
        (-0.11f0, 0.98f0),
        (0.00f0, 1.11f0),
        (0.11f0, 0.98f0),
    ]
end

function ear_outline_points(side::Symbol)
    x_sign = side === :left ? -1.0f0 : 1.0f0
    return Point2f[
        (x_sign * 0.97f0, 0.18f0),
        (x_sign * 1.08f0, 0.10f0),
        (x_sign * 1.10f0, -0.06f0),
        (x_sign * 1.02f0, -0.18f0),
    ]
end

function duration_image_from_dataset(erps, duration_order::AbstractVector{<:Integer}, channel::Int;
        start_idx::Int = TD.FIXATION_TIME_ZERO_IDX,
        n_trials::Int,
        target_size::Tuple{Int, Int},
        lowpass::Bool)
    data_time_trials = Float32.(erps[channel, start_idx:end, 1:n_trials])
    data_sorted = data_time_trials[:, duration_order]
    base_img = Float32.(permutedims(TD.zscore_timepoints(data_sorted), (2, 1)))
    proc_img = TD.process_erp_image(base_img, target_size; lowpass = lowpass)
    return proc_img
end

function top_contribution_text(score_df::DataFrame, row_idx::Int; top_k::Int = 3)
    pairs = Tuple{String, Float64}[]
    for feature_name in DUMMY_FEATURE_NAMES
        contrib_col = Symbol(string(feature_name), "_contrib")
        push!(pairs, (DUMMY_FEATURE_LABELS[feature_name], Float64(score_df[row_idx, contrib_col])))
    end
    sort!(pairs; by = x -> abs(x[2]), rev = true)
    return join([
        @sprintf("%s = %+0.2f", pairs[idx][1], pairs[idx][2])
        for idx in 1:min(top_k, length(pairs))
    ], " | ")
end

function build_fixation_duration_topoplot_context(;
        target_size::Tuple{Int, Int} = TD.REAL_TARGET_SIZE,
        lowpass::Bool = true,
        channels::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    @assert isfile(TD.FIXATION_H5_PATH) "File not found: $(TD.FIXATION_H5_PATH)"
    @assert isfile(TD.FIXATION_EVENTS_CSV_PATH) "File not found: $(TD.FIXATION_EVENTS_CSV_PATH)"

    events = TD.CSV.read(TD.FIXATION_EVENTS_CSV_PATH, DataFrame)
    n_trials = nrow(events)
    duration_order = TD.trial_sort_order(events, :duration)

    positions_all = load_normalized_fixation_positions()
    rows = NamedTuple[]
    images = Matrix{Float32}[]
    n_timepoints_post = 0

    TD.with_erps_dataset(TD.FIXATION_H5_PATH) do erps
        all_channels = collect(1:size(erps, 1))
        selected_channels = channels === nothing ? all_channels : Int.(collect(channels))
        @assert all(1 .<= selected_channels .<= size(erps, 1)) "Channel selection contains out-of-range indices."
        n_timepoints_post = size(erps, 2) - TD.FIXATION_TIME_ZERO_IDX + 1

        for channel in selected_channels
            img = duration_image_from_dataset(
                erps,
                duration_order,
                channel;
                n_trials = n_trials,
                target_size = target_size,
                lowpass = lowpass,
            )
            feats = dummy_channel_features(img)
            push!(rows, (
                channel = channel,
                channel_name = @sprintf("ch%03d", channel),
                peak_trend_abs = feats.peak_trend_abs,
                peak_span = feats.peak_span,
                diag_advantage = feats.diag_advantage,
                late_energy_shift = feats.late_energy_shift,
                row_energy_flatness = feats.row_energy_flatness,
            ))
            push!(images, img)
        end
    end

    score_df = DataFrame(rows)
    for feature_name in DUMMY_FEATURE_NAMES
        raw_values = Float64.(score_df[!, feature_name])
        score_df[!, Symbol(string(feature_name), "_z")] = safe_zscore(raw_values)
    end

    dummy_logit = fill(DUMMY_SIGMOID_BIAS, nrow(score_df))
    for feature_name in DUMMY_FEATURE_NAMES
        z_col = Symbol(string(feature_name), "_z")
        contrib_col = Symbol(string(feature_name), "_contrib")
        score_df[!, contrib_col] = DUMMY_SIGMOID_WEIGHTS[feature_name] .* Float64.(score_df[!, z_col])
        dummy_logit .+= Float64.(score_df[!, contrib_col])
    end

    score_df[!, :dummy_sigmoid_logit] = dummy_logit
    score_df[!, :p_sigmoid] = sigmoid_scalar.(dummy_logit)
    score_df[!, :dummy_label] = ifelse.(score_df.p_sigmoid .>= 0.5, "sigmoid", "no_class")
    score_df[!, :p_no_class] = 1 .- score_df.p_sigmoid

    image_stats = [TD.image_color_stats(img) for img in images]
    selected_positions = positions_all[Int.(score_df.channel)]
    topo_grid = interpolate_topomap(selected_positions, Float64.(score_df.p_sigmoid))

    return (
        sort_var = :duration,
        lowpass = lowpass,
        target_size = target_size,
        n_trials = n_trials,
        n_timepoints_post = n_timepoints_post,
        positions = selected_positions,
        topo_grid = topo_grid,
        score_df = score_df,
        images = images,
        image_stats = image_stats,
        resized_size = size(first(images)),
    )
end

function plot_fixation_duration_topoplot(ctx; initial_channel::Union{Nothing, Int} = nothing)
    @assert nrow(ctx.score_df) > 0 "No channels available for plotting."

    initial_row = if initial_channel === nothing
        argmax(Float64.(ctx.score_df.p_sigmoid))
    else
        found = findfirst(==(initial_channel), Int.(ctx.score_df.channel))
        found === nothing && error("Initial channel $(initial_channel) is not part of the current context.")
        found
    end

    selected_row = Observable(initial_row)
    selected_channel = @lift(Int(ctx.score_df.channel[$selected_row]))
    selected_name = @lift(String(ctx.score_df.channel_name[$selected_row]))
    selected_prob = @lift(Float64(ctx.score_df.p_sigmoid[$selected_row]))
    selected_logit = @lift(Float64(ctx.score_df.dummy_sigmoid_logit[$selected_row]))
    selected_label = @lift(String(ctx.score_df.dummy_label[$selected_row]))

    image_obs = @lift(ctx.image_stats[$selected_row].clipped)
    cmap_obs = @lift(ctx.image_stats[$selected_row].cmap)
    colorrange_obs = @lift(ctx.image_stats[$selected_row].colorrange)
    ticks_obs = @lift((ctx.image_stats[$selected_row].tick_vals, ctx.image_stats[$selected_row].tick_labels))
    info_text_obs = @lift(@sprintf(
        "%s | label = %s | p(sigmoid) = %.3f | logit = %+0.3f\nlargest dummy contributions: %s",
        ctx.score_df.channel_name[$selected_row],
        ctx.score_df.dummy_label[$selected_row],
        ctx.score_df.p_sigmoid[$selected_row],
        ctx.score_df.dummy_sigmoid_logit[$selected_row],
        top_contribution_text(ctx.score_df, $selected_row),
    ))

    topomap_cmap = cgrad([:gray88, :khaki1, :goldenrod1, :darkorange3])

    fig = Figure(size = (1460, 760), figure_padding = 20)
    Label(
        fig[0, 1:2],
        "Fixation duration topoplot prototype | dummy sigmoid output layer | click a sensor to open the ERP image";
        fontsize = 21,
        tellwidth = false,
    )

    left = GridLayout(fig[1, 1])
    ax_topo = Axis(
        left[1, 1];
        title = "duration only | p(dummy sigmoid) across channels",
        aspect = DataAspect(),
        xlabel = "",
        ylabel = "",
        xgridvisible = false,
        ygridvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
        backgroundcolor = :white,
    )
    hidedecorations!(ax_topo)

    hm_topo = heatmap!(
        ax_topo,
        ctx.topo_grid.xs,
        ctx.topo_grid.ys,
        ctx.topo_grid.values;
        colormap = topomap_cmap,
        colorrange = (0.0, 1.0),
        interpolate = true,
        inspectable = false,
        nan_color = RGBAf(0, 0, 0, 0),
    )

    lines!(ax_topo, head_outline_points(); color = :black, linewidth = 2.0)
    lines!(ax_topo, nose_outline_points(); color = :black, linewidth = 2.0)
    lines!(ax_topo, ear_outline_points(:left); color = :black, linewidth = 2.0)
    lines!(ax_topo, ear_outline_points(:right); color = :black, linewidth = 2.0)

    marker_sizes = 11 .+ 12 .* Float32.(ctx.score_df.p_sigmoid)
    scatter!(
        ax_topo,
        first.(ctx.positions),
        last.(ctx.positions);
        color = Float32.(ctx.score_df.p_sigmoid),
        colormap = topomap_cmap,
        colorrange = (0.0, 1.0),
        markersize = marker_sizes,
        strokecolor = :white,
        strokewidth = 1.2,
    )
    picker_plot = scatter!(
        ax_topo,
        first.(ctx.positions),
        last.(ctx.positions);
        color = RGBAf(0, 0, 0, 0),
        strokecolor = RGBAf(0, 0, 0, 0),
        markersize = max.(marker_sizes, 26),
    )

    selected_pos_obs = @lift([ctx.positions[$selected_row]])
    scatter!(
        ax_topo,
        selected_pos_obs;
        color = :transparent,
        markersize = 28,
        strokecolor = :black,
        strokewidth = 3.0,
    )

    xlims!(ax_topo, -1.18, 1.18)
    ylims!(ax_topo, -1.12, 1.20)
    Colorbar(
        left[1, 2],
        hm_topo;
        label = "p(dummy sigmoid)",
        ticks = (0.0:0.25:1.0, ["0.00", "0.25", "0.50", "0.75", "1.00"]),
        width = 14,
    )

    right = GridLayout(fig[1, 2])
    ticks = TD.fixation_axis_ticks((
        resized_size = ctx.resized_size,
        n_timepoints_post = ctx.n_timepoints_post,
        n_trials = ctx.n_trials,
    ))

    ax_img = Axis(
        right[1, 1];
        title = @lift(@sprintf(
            "%s | duration ERP image | %s",
            $selected_name,
            $selected_prob >= 0.5 ? "higher dummy sigmoid probability" : "lower dummy sigmoid probability",
        )),
        xlabel = "post-stimulus timepoints\nelapsed time after stimulus",
        ylabel = "trial rank",
        xticks = ticks.xticks,
        yticks = ticks.yticks,
        titlesize = 16,
        xlabelsize = 12,
        ylabelsize = 12,
        xticklabelsize = 10,
        yticklabelsize = 10,
    )

    hm_img = heatmap!(
        ax_img,
        1:ctx.resized_size[2],
        1:ctx.resized_size[1],
        @lift(permutedims($image_obs, (2, 1)));
        colormap = cmap_obs,
        colorrange = colorrange_obs,
    )
    Colorbar(
        right[1, 2],
        hm_img;
        label = "duration-sorted ERP image",
        ticks = ticks_obs,
        ticklabelsize = 9,
        width = 14,
    )

    Label(
        right[2, 1:2],
        info_text_obs;
        tellwidth = false,
        justification = :left,
        halign = :left,
        valign = :top,
        fontsize = 12,
    )

    on(events(ax_topo.scene).mousebutton, priority = 20) do event
        if event.button == Mouse.left && event.action == Mouse.press
            picked_plot, idx = pick(ax_topo.scene, events(ax_topo.scene).mouseposition[], 12)
            if picked_plot === picker_plot && idx > 0
                selected_row[] = idx
                return Consume(true)
            end
        end
        return Consume(false)
    end

    return (
        figure = fig,
        selected_row = selected_row,
        selected_channel = selected_channel,
        selected_name = selected_name,
        selected_prob = selected_prob,
        selected_logit = selected_logit,
        selected_label = selected_label,
    )
end

end
