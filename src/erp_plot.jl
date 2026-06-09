if !isdefined(@__MODULE__, :default_data_root)
    include(joinpath(@__DIR__, "erp_data.jl"))
end

if !isdefined(@__MODULE__, :resize_image)
    include(joinpath(@__DIR__, "erp_processing.jl"))
end

using CairoMakie
using Printf: @sprintf
using StatsBase: quantile

"""
    metadata_value(metadata, key, default)

Read a metadata value from a dictionary-like or property-accessible object.

# Arguments
- `metadata`: Metadata object returned from a dataset file.
- `key`: String key to read.
- `default`: Fallback value when the key is absent.

# Returns
- Metadata value for `key`, or `default`.
"""
function metadata_value(metadata, key, default)
    if metadata isa AbstractDict
        return get(metadata, key, default)
    end
    symbol_key = Symbol(key)
    return symbol_key in propertynames(metadata) ? getproperty(metadata, symbol_key) : default
end

"""
    axis_triplet(n)

Build compact axis tick indices for an ERP image dimension.

# Arguments
- `n`: Number of samples on one axis.

# Returns
- `Vector{Int}`: First, middle, and last index, deduplicated for short axes.
"""
function axis_triplet(n)
    n <= 1 && return [1]
    middle = Int(round((n + 1) / 2))
    return unique([1, middle, n])
end

"""
    tick_labels(values)

Format numeric axis tick labels compactly.

# Arguments
- `values`: Numeric tick values.

# Returns
- `Vector{String}`: Rounded labels for display.
"""
function tick_labels(values)
    return [abs(value) >= 100 ? string(round(value; digits = 1)) : string(round(value; digits = 3)) for value in values]
end

"""
    finite_values(image)

Collect finite image values for color statistics.

# Arguments
- `image`: Numeric image matrix.

# Returns
- `Vector{Float32}`: Finite values from `image`, or `[0f0]` for an empty finite set.
"""
function finite_values(image)
    values = Float32[]
    for value in image
        value32 = Float32(value)
        isfinite(value32) && push!(values, value32)
    end
    isempty(values) && push!(values, 0f0)
    return values
end

"""
    make_diverging_cmap_zero_centered(vmin, vmax; n_steps=64)

Create an `RdBu` colormap whose white center is placed at value zero.

# Arguments
- `vmin`: Lower color range value.
- `vmax`: Upper color range value.
- `n_steps`: Number of color samples per side.

# Returns
- CairoMakie color gradient for diverging ERP amplitudes.
"""
function make_diverging_cmap_zero_centered(vmin, vmax; n_steps = 64)
    vmin_value = Float64(vmin)
    vmax_value = Float64(vmax)
    vmax_value <= vmin_value && (vmax_value = vmin_value + 1e-6)

    source = cgrad(:RdBu, rev = true)
    zero_position = clamp((0.0 - vmin_value) / (vmax_value - vmin_value), 0.02, 0.98)
    n_steps = max(2, Int(n_steps))

    colors = Vector{Any}(undef, 2 * n_steps + 1)
    positions = Vector{Float64}(undef, 2 * n_steps + 1)

    for index in 0:n_steps
        source_position = 0.5 * index / n_steps
        destination_position = zero_position * index / n_steps
        colors[index + 1] = source[source_position]
        positions[index + 1] = destination_position
    end

    for index in 1:n_steps
        source_position = 0.5 + 0.5 * index / n_steps
        destination_position = zero_position + (1.0 - zero_position) * index / n_steps
        colors[n_steps + 1 + index] = source[source_position]
        positions[n_steps + 1 + index] = destination_position
    end

    return cgrad(colors, positions)
end

"""
    clipped_color_stats_quantile_zero_ticks(image; q_low=0.01, q_high=0.99)

Compute quantile-clipped ERP image color statistics and zero-centered ticks.

# Arguments
- `image`: ERP image matrix.
- `q_low`: Lower quantile used for clipping.
- `q_high`: Upper quantile used for clipping.

# Returns
- `NamedTuple`: `(clipped, colorrange, tick_values, tick_labels, colormap)`.
"""
function clipped_color_stats_quantile_zero_ticks(image; q_low = 0.01, q_high = 0.99)
    values = finite_values(image)
    low = Float32(quantile(values, q_low))
    high = Float32(quantile(values, q_high))
    high < low && (high = low)

    all_nonnegative = all(value -> value >= 0f0, values)
    all_nonpositive = all(value -> value <= 0f0, values)

    vmin = low
    vmax = high
    if all_nonnegative
        vmin = 0f0
        vmax = max(high, 1f-6)
    elseif all_nonpositive
        vmin = min(low, -1f-6)
        vmax = 0f0
    else
        vmin = min(low, 0f0)
        vmax = max(high, 0f0)
    end

    if vmax <= vmin
        delta = max(abs(vmin), abs(vmax), 1f0) * 1f-6
        vmin -= delta
        vmax += delta
    end

    clipped = Matrix{Float32}(clamp.(Float32.(image), vmin, vmax))
    colorrange = (vmin, vmax)
    colormap = make_diverging_cmap_zero_centered(vmin, vmax)

    tick_pairs = collect(zip(
        Float32[low, 0f0, high],
        [@sprintf("%.3f", low), @sprintf("%.3f", 0f0), @sprintf("%.3f", high)],
    ))
    sort!(tick_pairs; by = first)
    tick_values = Float32[first(tick_pairs[1]), first(tick_pairs[2]), first(tick_pairs[3])]
    tick_labels_text = [last(tick_pairs[1]), last(tick_pairs[2]), last(tick_pairs[3])]

    for index in 2:3
        if tick_values[index] <= tick_values[index - 1]
            tick_values[index] = nextfloat(tick_values[index - 1])
        end
    end

    return (
        clipped = clipped,
        colorrange = colorrange,
        tick_values = tick_values,
        tick_labels = tick_labels_text,
        colormap = colormap,
    )
end

"""
    image_time_axis(metadata, n_timepoints)

Build x-axis coordinates for plotting an ERP image.

# Arguments
- `metadata`: Dataset metadata, ideally containing `time_start_s` and `time_end_s`.
- `n_timepoints`: Number of image columns to plot.

# Returns
- `Tuple`: `(values, label)` where `values` is a vector of x-axis coordinates and
  `label` is the axis label.
"""
function image_time_axis(metadata, n_timepoints)
    has_start = metadata isa AbstractDict && haskey(metadata, "time_start_s")
    has_end = metadata isa AbstractDict && haskey(metadata, "time_end_s")
    if has_start && has_end
        start_s = Float64(metadata["time_start_s"])
        end_s = Float64(metadata["time_end_s"])
        return collect(range(start_s, end_s; length = n_timepoints)), "time [s]"
    end
    return collect(1:n_timepoints), "timepoint"
end

"""
    class_label(dataset_key, channel_name, sort_variable, data_root)

Read the ERP class label used in the plot title.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel signal file name without `.jld2`.
- `sort_variable`: Event column used for sorting.
- `data_root`: Root folder containing dataset folders.

# Returns
- `String`: ERP class name, or `"unlabeled"` when no matching label row exists.
"""
function class_label(dataset_key, channel_name, sort_variable, data_root)
    label_rows = labels_for(dataset_key, channel_name, sort_variable; data_root = data_root)
    isempty(label_rows) && return "unlabeled"
    return cellstring(label_rows.erp_class[1])
end

"""
    plot_erp_image(dataset_key, channel_name, sort_variable; data_root=default_data_root(), smooth=true, resize=false)

Load, process, and plot one ERP image as a CairoMakie heatmap.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel signal file name without `.jld2`.
- `sort_variable`: Event column used to sort trials.
- `data_root`: Root folder containing dataset folders.
- `smooth`: Apply Gaussian smoothing before plotting when `true`.
- `resize`: Resize the plotted image to `(64, 64)` when `true`.
- `figure_kwargs`: Keyword arguments forwarded to `CairoMakie.Figure`.
- `colormap_quantile`: Upper quantile used for the Week-25 style colorbar.
- `time_unit`: `:seconds` or `:milliseconds` for the bottom x-axis.

# Returns
- `CairoMakie.Figure`: Figure containing the ERP heatmap and quantile-clipped
  colorbar.
"""
function plot_erp_image(
        dataset_key,
        channel_name,
        sort_variable;
        data_root = default_data_root(),
        smooth = true,
        resize = false,
        figure_kwargs = (size = (980, 680),),
        colormap_quantile = 0.99,
        time_unit = :seconds)

    events_bundle = load_events(dataset_key; data_root = data_root)
    signal_bundle = load_signal(dataset_key, channel_name; data_root = data_root)
    size(signal_bundle.data_time_trials, 2) == nrow(events_bundle.events) || throw(ArgumentError(
        "Signal trial count does not match events row count for $(dataset_key), channel $(channel_name).",
    ))

    order = trial_sort_order(events_bundle.events, sort_variable)
    sorted_trials = sort_trials(signal_bundle.data_time_trials, order)
    image = sorted_trials |>
        zscore_timepoints |>
        trials_time_image

    if smooth
        image = smooth_image(image)
    end
    if resize
        image = resize_image(image)
    end

    n_trials, n_timepoints = size(image)
    time_values, time_label = image_time_axis(events_bundle.metadata, n_timepoints)
    if Symbol(time_unit) == :milliseconds && time_label == "time [s]"
        time_values = time_values .* 1000.0
        time_label = "time [ms]"
    elseif !(Symbol(time_unit) in (:seconds, :milliseconds))
        throw(ArgumentError("time_unit must be :seconds or :milliseconds."))
    end
    dataset_label = metadata_value(events_bundle.metadata, "dataset_label", String(dataset_key))
    erp_class = class_label(dataset_key, channel_name, sort_variable, data_root)
    color_stats = clipped_color_stats_quantile_zero_ticks(
        image;
        q_low = Float64(1 - colormap_quantile),
        q_high = Float64(colormap_quantile),
    )
    time_tick_indices = axis_triplet(n_timepoints)
    time_tick_positions = time_values[time_tick_indices]
    time_tick_labels = tick_labels(time_tick_positions)
    timepoint_tick_labels = string.(time_tick_indices)
    y_tick_values = axis_triplet(n_trials)

    fig_kwargs = merge((size = (980, 680), figure_padding = (18, 28, 16, 14)), NamedTuple(figure_kwargs))
    fig = Figure(; fig_kwargs...)
    title = string(
        dataset_label,
        " | channel ",
        channel_name,
        "\n",
        "sort by ",
        sort_variable,
        " | class: ",
        erp_class,
    )
    Label(fig[0, 1:2], title;
        fontsize = 17,
        font = :bold,
        tellwidth = false,
        padding = (0, 0, 0, 8),
    )

    ax = Axis(
        fig[1, 1];
        xlabel = time_label,
        ylabel = "trial (sorted by $(sort_variable))",
        xticks = (time_tick_positions, time_tick_labels),
        yticks = y_tick_values,
        xlabelpadding = 8,
        ylabelpadding = 8,
        xticklabelpad = 4,
        yticklabelpad = 4,
    )
    xlims!(ax, first(time_values), last(time_values))
    ylims!(ax, 1, n_trials)

    heatmap_plot = heatmap!(
        ax,
        time_values,
        1:n_trials,
        permutedims(color_stats.clipped, (2, 1));
        colormap = color_stats.colormap,
        colorrange = color_stats.colorrange,
        rasterize = true,
    )

    top_axis = Axis(fig[1, 1];
        xaxisposition = :top,
        yaxisposition = :right,
        xlabel = "timepoint",
        xticks = (time_tick_positions, timepoint_tick_labels),
        backgroundcolor = (:white, 0.0),
        xlabelpadding = 8,
        xticklabelpad = 4,
    )
    linkxaxes!(ax, top_axis)
    xlims!(top_axis, first(time_values), last(time_values))
    hideydecorations!(top_axis)
    hidespines!(top_axis, :l, :r, :b)
    top_axis.ygridvisible = false
    top_axis.xgridvisible = false

    Colorbar(fig[1, 2], heatmap_plot;
        label = "amplitude (z)",
        ticks = (color_stats.tick_values, color_stats.tick_labels),
        width = 14,
    )
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    return fig
end
