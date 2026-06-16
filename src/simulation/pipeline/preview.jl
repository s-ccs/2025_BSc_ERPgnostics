# =============================================================================
# Sim-vs-real preview figures
#
# For a given simulator configuration this renders three simulated and three
# real sigmoid ERP images side by side, in the square style of
# notebooks/week_24/erp_pattern_examples_square.ipynb. One figure is produced for
# the default (baseline) parameters and one per search strategy using its best
# candidate; existing files are overwritten so the output folder stays small.
#
# The images use the notebook's exact Gaussian smoothing: sigma factor 75 applied
# at the full recording resolution (no resize to 64x64), so the preview is a
# crisp, publication-style visualisation rather than the 64x64 model input.
# =============================================================================

"""
    diverging_cmap_zero_anchored(vmin, vmax; n_steps = 64)

Build a Blue→White→Red colormap whose white midpoint lands exactly at zero
within `[vmin, vmax]`, so positive and negative amplitudes stay visually
distinct even for asymmetric ranges.

# Arguments
- `vmin`, `vmax`: amplitude range; `vmax` is nudged above `vmin` if equal.
- `n_steps::Int`: colour resolution per side (keyword).

# Returns
- A CairoMakie `cgrad` colormap.
"""
function diverging_cmap_zero_anchored(vmin::Real, vmax::Real; n_steps::Int = 64)
    vmin_f = Float64(vmin)
    vmax_f = Float64(vmax)
    vmax_f <= vmin_f && (vmax_f = vmin_f + 1.0e-6)

    src = cgrad(:RdBu, rev = true)
    zero_pos = clamp((0.0 - vmin_f) / (vmax_f - vmin_f), 0.0, 1.0)
    n_steps = max(2, n_steps)

    # Degenerate ranges (all one sign) use a single half of the source colormap.
    if zero_pos <= 0.0
        colors = [src[0.5 + 0.5 * i / n_steps] for i in 0:n_steps]
        return cgrad(colors, collect(range(0.0, 1.0; length = n_steps + 1)))
    elseif zero_pos >= 1.0
        colors = [src[0.0 + 0.5 * i / n_steps] for i in 0:n_steps]
        return cgrad(colors, collect(range(0.0, 1.0; length = n_steps + 1)))
    end

    # Stretch the negative and positive halves so the white point sits at zero.
    colors = Vector{Any}(undef, 2 * n_steps + 1)
    positions = Vector{Float64}(undef, 2 * n_steps + 1)
    for i in 0:n_steps
        colors[i + 1] = src[0.5 * i / n_steps]
        positions[i + 1] = zero_pos * i / n_steps
    end
    for i in 1:n_steps
        colors[n_steps + 1 + i] = src[0.5 + 0.5 * i / n_steps]
        positions[n_steps + 1 + i] = zero_pos + (1.0 - zero_pos) * i / n_steps
    end
    return cgrad(colors, positions)
end

"""
    clipped_color_stats(data; q_low = 0.01, q_high = 0.99)

Compute amplitude clipping bounds, a zero-anchored colormap, and three colorbar
ticks (low quantile, zero, high quantile) for an ERP image.

# Arguments
- `data::AbstractMatrix`: the ERP image.
- `q_low`, `q_high`: lower and upper amplitude quantiles for clipping (keywords).

# Returns
- `Tuple`: `(clipped, colorrange, tick_values, tick_labels, colormap)`.
"""
function clipped_color_stats(data::AbstractMatrix; q_low::Float64 = 0.01, q_high::Float64 = 0.99)
    vals = Float32[]
    for v in data
        fv = Float32(v)
        isfinite(fv) && push!(vals, fv)
    end
    isempty(vals) && push!(vals, 0.0f0)

    ql = Float32(quantile(vals, q_low))
    qh = Float32(quantile(vals, q_high))
    qh < ql && (qh = ql)

    # Keep zero inside the range so the diverging colormap stays anchored.
    vmin = min(ql, 0.0f0)
    vmax = max(qh, 0.0f0)
    if all(x -> x >= 0.0f0, vals)
        vmin, vmax = 0.0f0, max(qh, 1.0f-6)
    elseif all(x -> x <= 0.0f0, vals)
        vmin, vmax = min(ql, -1.0f-6), 0.0f0
    end
    if vmax <= vmin
        delta = max(abs(vmin), abs(vmax), 1.0f0) * 1.0f-6
        vmin -= delta
        vmax += delta
    end

    clipped = clamp.(Float32.(data), vmin, vmax)
    cmap = diverging_cmap_zero_anchored(vmin, vmax)

    pairs = collect(zip(
        Float32[ql, 0.0f0, qh],
        [@sprintf("%.3f", ql), @sprintf("%.3f", 0.0f0), @sprintf("%.3f", qh)],
    ))
    sort!(pairs; by = first)
    tick_vals = Float32[pairs[1][1], pairs[2][1], pairs[3][1]]
    tick_text = [pairs[1][2], pairs[2][2], pairs[3][2]]
    for i in 2:3
        tick_vals[i] <= tick_vals[i - 1] && (tick_vals[i] = nextfloat(tick_vals[i - 1]))
    end
    return clipped, (vmin, vmax), tick_vals, tick_text, cmap
end

"""
    axis_triplet(n) -> Vector{Int}

Return the tick positions `[1, middle, n]` for an axis of length `n`, collapsed
to fewer ticks when `n` is small.
"""
function axis_triplet(n::Int)
    n <= 1 && return [1]
    return unique([1, Int(round((n + 1) / 2)), n])
end

"""
    notebook_smooth_image(data_time_trials, events, sort_col, config)

Build a full-resolution ERP image exactly like the week_24 notebook: sort the
trials, z-score each timepoint, then Gaussian-smooth at the **original**
resolution (`out_size == in_size`, sigma factor `config.low_pass_factor`, kernel
`config.lowpass_kernel_size`, reflective borders). No resize is applied.

# Arguments
- `data_time_trials::AbstractMatrix`: EEG values in timepoints-by-trials layout.
- `events::DataFrame`: per-trial event table containing `sort_col`.
- `sort_col::Symbol`: column that defines the trial order.
- `config::RunConfig`: supplies the Gaussian smoothing settings.

# Returns
- `Matrix{Float32}`: the smoothed full-resolution image (trials x time).
"""
function notebook_smooth_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol, config::RunConfig)
    img = sorted_zscore_image(data_time_trials, events, sort_col)
    # out_size == in_size keeps the full resolution and gives the notebook's sigma.
    kernel = gaussian_lowpass_kernel(config.low_pass_factor, size(img), size(img), config.lowpass_kernel_size)
    return Float32.(imfilter(Float32.(img), kernel, config.filter_border))
end

"""
    simulate_preview_sigmoids(cfg, config, real; n = 3)

Simulate `n` full-resolution sigmoid ERP images from `cfg` for the preview, using
the notebook smoothing. No dropout or cropping is applied, matching the rest of
the pipeline.

# Returns
- `Vector{Matrix{Float32}}`: the simulated sigmoid images.
"""
function simulate_preview_sigmoids(cfg::ERPGen.GenerationConfig, config::RunConfig, real::RealValidationData; n::Int = 3)
    imgs = Matrix{Float32}[]
    for _ in 1:n
        raw = ERPGen.simulate_raw_erp(cfg, Random.Xoshiro(new_seed()))
        data = Float32.(raw.data)
        events = copy(raw.events)
        size(data) == (real.n_timepoints, real.n_trials) ||
            error("Simulated dimensions $(size(data)) do not match the fixations dataset $((real.n_timepoints, real.n_trials)).")
        events[!, :strategy_sort_key] = simulated_sort_column(events, :sigmoid, Random.Xoshiro(new_seed()))
        push!(imgs, notebook_smooth_image(data, events, :strategy_sort_key, config))
    end
    return imgs
end

"""
    real_preview_sigmoids(real, config; n = 3)

Build `n` full-resolution real sigmoid panels (image plus a source title) by
reloading the corresponding channel signals and applying the notebook smoothing.

# Returns
- `Vector{Tuple{Matrix{Float32},String}}`: `(image, title)` per panel.
"""
function real_preview_sigmoids(real::RealValidationData, config::RunConfig; n::Int = 3)
    rows = real.eval_df[real.eval_df.binary_label .== 1, :]
    nrow(rows) >= n || error("Need at least $(n) real sigmoid channels for the preview, found $(nrow(rows)).")

    cache = Dict{String, Matrix{Float32}}()
    panels = Tuple{Matrix{Float32}, String}[]
    for i in 1:n
        row = rows[i, :]
        data = load_channel_signal(config, row.channel_name, cache)
        img = notebook_smooth_image(data, real.events, Symbol(row.sort_variable), config)
        title = "real sigmoid\n$(row.dataset_key)\nch=$(row.channel_name) | sort=$(row.sort_variable)"
        push!(panels, (img, title))
    end
    return panels
end

"""
    add_square_panel!(fig, row, col, img; title, show_ylabel)

Draw one square ERP-image panel (heatmap + colorbar) into the figure grid, in
the week_24 style: square aspect, `sorted trials` on the y-axis, time-sample
indices on the x-axis, and an unlabelled colorbar.

# Arguments
- `fig`: the CairoMakie figure.
- `row::Int`, `col::Int`: 1-based grid position (column maps to two layout cells).
- `img::AbstractMatrix`: the ERP image (trials x time).
- `title::AbstractString`: panel title (keyword).
- `show_ylabel::Bool`: whether to show the `sorted trials` y-label (keyword).

# Returns
- the created axis.
"""
function add_square_panel!(fig, row::Int, col::Int, img::AbstractMatrix; title::AbstractString, show_ylabel::Bool)
    clipped, colorrange, tick_vals, tick_text, cmap = clipped_color_stats(img)
    n_trials, n_time = size(clipped)
    time_ticks = axis_triplet(n_time)
    trial_ticks = axis_triplet(n_trials)

    ax = Axis(fig[row, 2 * col - 1];
        title = title,
        titlesize = 15,
        xlabel = "time-sample index",
        ylabel = show_ylabel ? "sorted trials" : "",
        xticks = (time_ticks, string.(time_ticks)),
        yticks = (trial_ticks, string.(trial_ticks)),
        aspect = AxisAspect(1),
    )
    hm = heatmap!(ax, 1:n_time, 1:n_trials, permutedims(clipped, (2, 1));
        colormap = cmap, colorrange = colorrange, rasterize = true)
    Colorbar(fig[row, 2 * col], hm; ticks = (tick_vals, tick_text), width = 12)
    return ax
end

"""
    render_preview(cfg, real, config; name, heading) -> String

Render and save one preview figure (three simulated sigmoid panels over three
real sigmoid panels) for the simulator configuration `cfg`. The file
`preview_<name>.png` in `config.output_dir` is overwritten if it exists.

# Arguments
- `cfg::ERPGen.GenerationConfig`: configuration to simulate the top row from.
- `real::RealValidationData`: source of the real sigmoid panels.
- `config::RunConfig`: supplies the smoothing settings and output directory.
- `name::AbstractString`: file name suffix, e.g. `"default"` or a strategy name.
- `heading::AbstractString`: figure heading describing the setting.

# Returns
- `String`: the path of the saved PNG.
"""
function render_preview(cfg::ERPGen.GenerationConfig, real::RealValidationData, config::RunConfig; name::AbstractString, heading::AbstractString)
    sim_imgs = simulate_preview_sigmoids(cfg, config, real; n = 3)
    real_panels = real_preview_sigmoids(real, config; n = 3)

    fig = Figure(size = (1500, 1080), figure_padding = (18, 28, 16, 14))
    Label(fig[0, 1:6], heading; fontsize = 18, font = :bold, tellwidth = false, padding = (0, 0, 0, 8))
    for col in 1:3
        add_square_panel!(fig, 1, col, sim_imgs[col]; title = "simulated sigmoid $(col)", show_ylabel = col == 1)
    end
    for col in 1:3
        img, title = real_panels[col]
        add_square_panel!(fig, 2, col, img; title = title, show_ylabel = col == 1)
    end
    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 18)
    resize_to_layout!(fig)

    output_path = joinpath(config.output_dir, "preview_$(name).png")
    mkpath(dirname(output_path))
    save(output_path, fig)
    return output_path
end
