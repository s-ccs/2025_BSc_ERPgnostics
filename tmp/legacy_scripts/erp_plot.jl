if !isdefined(@__MODULE__, :ERPDataIO)
    include(joinpath(@__DIR__, "erp_io.jl"))
end

if !isdefined(@__MODULE__, :ERPImageProcessing)
    include(joinpath(@__DIR__, "erp_image_processing.jl"))
end

module ERPDataPlot

using CairoMakie
using ..ERPDataIO: load_erp
using ..ERPImageProcessing: clipped_color_stats_quantile_zero_ticks
using ..ERPImageProcessing: gaussian_smooth_image
using ..ERPImageProcessing: make_diverging_cmap_zero_anchored
using ..ERPImageProcessing: trials_time_image
using ..ERPImageProcessing: zscore_timepoints

export plot_erp_image
export processed_erp_image_trials_time

const DEFAULT_SMOOTHING_SIGMA_FACTOR = 75f0

function axis_triplet(n::Int)
    n <= 1 && return [1]
    mid = Int(round((n + 1) / 2))
    return unique([1, mid, n])
end

function time_axis_values(metadata::Dict{String, Any}, time_unit::Symbol)
    n_timepoints = Int(metadata["n_timepoints_post"])
    start_s = Float64(metadata["time_start_s"])
    end_s = Float64(metadata["time_end_s"])
    values_s = collect(range(start_s, end_s; length = n_timepoints))
    if time_unit == :seconds
        return values_s, "time [s]"
    elseif time_unit == :milliseconds
        return values_s .* 1000.0, "time [ms]"
    end
    error("time_unit must be :seconds or :milliseconds.")
end

function tick_labels(values)
    return [abs(v) >= 100 ? string(round(v; digits = 1)) : string(round(v; digits = 3)) for v in values]
end

function processed_erp_image_trials_time(data_time_trials::AbstractMatrix;
        smooth::Bool = true,
        smoothing_sigma_factor::Float32 = DEFAULT_SMOOTHING_SIGMA_FACTOR)
    data_z = zscore_timepoints(data_time_trials)
    img_trials_time = trials_time_image(data_z)
    if smooth && min(size(img_trials_time)...) > 1
        img_trials_time = gaussian_smooth_image(
            img_trials_time;
            sigma_factor = smoothing_sigma_factor,
        )
    end
    return img_trials_time
end

function plot_erp_image(dataset_key, channel_name, sort_variable;
        figure_kwargs = (size = (980, 680),),
        colormap_quantile = 0.98,
        time_unit = :seconds,
        smooth = true,
        smoothing_sigma_factor = DEFAULT_SMOOTHING_SIGMA_FACTOR)
    erp = load_erp(String(dataset_key), String(channel_name), String(sort_variable))
    data_time_trials = erp.data_time_trials
    metadata = erp.metadata
    n_timepoints, n_trials = size(data_time_trials)
    img_trials_time = processed_erp_image_trials_time(
        data_time_trials;
        smooth = Bool(smooth),
        smoothing_sigma_factor = Float32(smoothing_sigma_factor),
    )

    clipped, colorrange, tick_vals, tick_labels_text, _ = clipped_color_stats_quantile_zero_ticks(
        img_trials_time;
        q_low = Float64(1 - colormap_quantile),
        q_high = Float64(colormap_quantile),
    )
    vmin, vmax = colorrange
    cmap = make_diverging_cmap_zero_anchored(vmin, vmax)

    time_values, time_label = time_axis_values(metadata, Symbol(time_unit))
    time_tick_indices = axis_triplet(n_timepoints)
    time_tick_positions = time_values[time_tick_indices]
    timepoint_tick_labels = string.(time_tick_indices)
    time_tick_labels = tick_labels(time_tick_positions)
    y_tick_values = axis_triplet(n_trials)

    title = string(
        metadata["dataset_label"],
        " | channel ",
        channel_name,
        "\n",
        "sort by ",
        sort_variable,
        " | class: ",
        erp.erp_class,
    )

    fig_kwargs = merge((size = (980, 680), figure_padding = (18, 28, 16, 14)), NamedTuple(figure_kwargs))
    fig = Figure(; fig_kwargs...)
    Label(fig[0, 1:2], title;
        fontsize = 17,
        font = :bold,
        tellwidth = false,
        padding = (0, 0, 0, 8),
    )
    ax = Axis(fig[1, 1];
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

    hm = heatmap!(
        ax,
        time_values,
        1:n_trials,
        Matrix{Float32}(permutedims(clipped, (2, 1)));
        colormap = cmap,
        colorrange = colorrange,
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

    colorbar_label = "amplitude (z)"
    Colorbar(fig[1, 2], hm;
        label = colorbar_label,
        ticks = (tick_vals, tick_labels_text),
    )
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    return fig
end

end

using .ERPDataPlot: plot_erp_image
using .ERPDataPlot: processed_erp_image_trials_time
