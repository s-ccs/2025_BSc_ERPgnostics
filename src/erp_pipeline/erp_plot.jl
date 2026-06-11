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
    display_text(value)

Format identifier-like values for plot titles.

# Arguments
- `value`: Value to display.

# Returns
- `String`: Human-readable text with underscores replaced by spaces.
"""
function display_text(value)
    return replace(String(value), "_" => " ")
end

"""
    display_class(value)

Format ERP class names for plot titles.

# Arguments
- `value`: ERP class label.

# Returns
- `String`: Title-cased class label.
"""
function display_class(value)
    return titlecase(display_text(value))
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
    values = vec(Float32.(image))
    # Color statistics should ignore NaN and Inf values.
    filter!(isfinite, values)
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
    # Anchor the diverging colormap around zero even for asymmetric ranges.
    zero_position = clamp((0.0 - vmin_value) / (vmax_value - vmin_value), 0.02, 0.98)
    n_steps = max(2, Int(n_steps))

    left_source_positions = range(0.0, 0.5; length = n_steps + 1)
    right_source_positions = range(0.5, 1.0; length = n_steps + 1)[2:end]
    left_target_positions = range(0.0, zero_position; length = n_steps + 1)
    right_target_positions = range(zero_position, 1.0; length = n_steps + 1)[2:end]

    colors = [source[position] for position in [left_source_positions; right_source_positions]]
    positions = collect([left_target_positions; right_target_positions])
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
    # Keep zero visible whenever the data crosses or touches it.
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

    # Expand degenerate ranges so Makie receives a valid color interval.
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
    erp_plot_title(dataset_label, channel_name, sort_variable, erp_class)

Build the compact Week-23-style ERP image title.

# Arguments
- `dataset_label`: Readable dataset label.
- `channel_name`: Channel signal file name without `.jld2`.
- `sort_variable`: Event column used for sorting.
- `erp_class`: ERP class label or `"unlabeled"`.

# Returns
- `String`: Multi-line plot title.
"""
function erp_plot_title(dataset_label, channel_name, sort_variable, erp_class)
    return @sprintf(
        "%s\n%s\nch=%s | sort=%s",
        display_class(erp_class),
        String(dataset_label),
        String(channel_name),
        display_text(sort_variable),
    )
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
        figure_kwargs = (size = (900, 650),),
        colormap_quantile = 0.99)

    events_bundle = load_events(dataset_key; data_root = data_root)
    signal_bundle = load_signal(dataset_key, channel_name; data_root = data_root)
    size(signal_bundle.data_time_trials, 2) == nrow(events_bundle.events) || throw(ArgumentError(
        "Signal trial count does not match events row count for $(dataset_key), channel $(channel_name).",
    ))

    # Keep plot data preparation in the same explicit order as the processing module.
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
    timepoint_values = collect(1:n_timepoints)
    dataset_label = metadata_value(events_bundle.metadata, "dataset_label", String(dataset_key))
    erp_class = class_label(dataset_key, channel_name, sort_variable, data_root)

    # Clip colors by quantile so outliers do not dominate the heatmap.
    color_stats = clipped_color_stats_quantile_zero_ticks(
        image;
        q_low = Float64(1 - colormap_quantile),
        q_high = Float64(colormap_quantile),
    )
    timepoint_tick_values = axis_triplet(n_timepoints)
    x_ticks = (timepoint_tick_values, string.(timepoint_tick_values))
    y_tick_values = axis_triplet(n_trials)
    y_ticks = (y_tick_values, string.(y_tick_values))

    fig_kwargs = merge((size = (900, 650), figure_padding = 24), NamedTuple(figure_kwargs))
    fig = Figure(; fig_kwargs...)
    ax = Axis(
        fig[1, 1];
        title = erp_plot_title(dataset_label, channel_name, sort_variable, erp_class),
        titlesize = 26,
        xlabel = "timepoint",
        ylabel = "sorted trials",
        xticks = x_ticks,
        yticks = y_ticks,
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 14,
        yticklabelsize = 14,
    )
    xlims!(ax, first(timepoint_values), last(timepoint_values))
    ylims!(ax, 1, n_trials)

    heatmap_plot = heatmap!(
        ax,
        timepoint_values,
        1:n_trials,
        permutedims(color_stats.clipped, (2, 1));
        colormap = color_stats.colormap,
        colorrange = color_stats.colorrange,
        rasterize = true,
    )

    Colorbar(fig[1, 2], heatmap_plot;
        ticks = (color_stats.tick_values, color_stats.tick_labels),
        ticklabelsize = 14,
        width = 18,
    )
    colgap!(fig.layout, 14)
    resize_to_layout!(fig)
    return fig
end
