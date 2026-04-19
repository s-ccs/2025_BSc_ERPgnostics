module Week15SortVariableOverview

using CairoMakie
using DataFrames
using HDF5
using Random

include(joinpath(@__DIR__, "labelstudio_export_helpers.jl"))
using .Week15LabelStudioExport: load_8bit_export_data, build_image_from_record

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: clipped_color_stats_quantile_zero_ticks

export OPENNEURO_REPO_DIR
export OPENNEURO_DERIVED_DIR
export DEFAULT_GAME_TRIAL_TYPES
export DEFAULT_RANDOM_CHANNEL_COUNT
export DEFAULT_OVERVIEW_SORT_COLUMNS
export DEFAULT_SORT_COLUMN_CANDIDATES
export load_8bit_overview_data
export summarize_sort_columns
export plottable_sort_columns
export overview_sort_order_audit_df
export build_sortvar_preview
export plot_sort_variable_figure
export plot_all_sort_variable_figures

const OPENNEURO_REPO_DIR = Week15LabelStudioExport.OPENNEURO_REPO_DIR
const OPENNEURO_DERIVED_DIR = Week15LabelStudioExport.OPENNEURO_DERIVED_DIR
const DEFAULT_GAME_TRIAL_TYPES = Week15LabelStudioExport.DEFAULT_GAME_TRIAL_TYPES

const DEFAULT_RANDOM_CHANNEL_COUNT = 4
const DEFAULT_OVERVIEW_SORT_COLUMNS = [:onset_s, :sample_index, :event_rank_within_type, :epoch_index]
const DEFAULT_SORT_COLUMN_CANDIDATES = [
    (original_name = "onset", derived_name = :onset_s),
    (original_name = "duration", derived_name = :duration_s),
    (original_name = "sample", derived_name = :sample_index),
    (original_name = "trial_type", derived_name = :trial_type),
    (original_name = "response_time", derived_name = :response_time_s),
    (original_name = "stim_file", derived_name = :stim_file),
    (original_name = "value", derived_name = :trigger_value),
    (original_name = "event_rank_within_type", derived_name = :event_rank_within_type),
    (original_name = "epoch_index", derived_name = :epoch_index),
    (original_name = "run", derived_name = :run),
    (original_name = "source_file", derived_name = :source_file),
    (original_name = "trial_type_slug", derived_name = :trial_type_slug),
]

const PANEL_PX = 245
const ROW_PX = 210
const CB_PX = 72

load_8bit_overview_data(; trial_types = collect(DEFAULT_GAME_TRIAL_TYPES)) =
    load_8bit_export_data(; trial_types = trial_types)

function game_events_subset(data_bundle)
    allowed = Set(String.(data_bundle.trial_types))
    mask = [trial_type in allowed for trial_type in data_bundle.trial_type_str]
    return copy(data_bundle.events[mask, :]), String.(data_bundle.trial_type_str[mask])
end

function constant_within_trial_types(events_subset::DataFrame, trial_type_subset::Vector{String}, col::Symbol)
    for trial_type in unique(trial_type_subset)
        mask = trial_type_subset .== trial_type
        vals = events_subset[mask, col]
        uniq = unique(collect(skipmissing(vals)))
        if length(uniq) > 1
            return false
        end
    end
    return true
end

function semantic_note(col::Symbol)
    if col == :onset_s
        return "useful: onset in seconds"
    elseif col == :sample_index
        return "useful: same ordering as onset_s, but in samples"
    elseif col == :event_rank_within_type
        return "useful: running index within one trial_type"
    elseif col == :epoch_index
        return "useful: global epoch order"
    elseif col == :trigger_value
        return "effectively constant within one trial_type image"
    elseif col == :trial_type || col == :trial_type_slug
        return "grouping variable only"
    elseif col == :run || col == :source_file
        return "constant for the selected game events"
    elseif col == :duration_s || col == :response_time_s || col == :stim_file
        return "empty for the selected game events"
    end
    return ""
end

function summarize_sort_columns(data_bundle; candidates = DEFAULT_SORT_COLUMN_CANDIDATES)
    events_subset, trial_type_subset = game_events_subset(data_bundle)
    rows = NamedTuple[]

    for candidate in candidates
        derived_name = candidate.derived_name

        if !(derived_name in propertynames(events_subset))
            push!(rows, (
                original_name = candidate.original_name,
                derived_name = String(derived_name),
                nonmissing = 0,
                unique_nonmissing = 0,
                should_plot = false,
                note = "not present in events.csv",
            ))
            continue
        end

        vals = events_subset[!, derived_name]
        present_vals = collect(skipmissing(vals))
        nonmissing = length(present_vals)
        unique_nonmissing = length(unique(present_vals))
        within_trial_constant = nonmissing == 0 ? false : constant_within_trial_types(events_subset, trial_type_subset, derived_name)

        should_plot = nonmissing > 0 && unique_nonmissing > 1 && !within_trial_constant
        note = if nonmissing == 0
            semantic_note(derived_name)
        elseif unique_nonmissing <= 1
            "constant"
        elseif within_trial_constant
            "constant within each trial_type image"
        else
            semantic_note(derived_name)
        end

        push!(rows, (
            original_name = candidate.original_name,
            derived_name = String(derived_name),
            nonmissing = nonmissing,
            unique_nonmissing = unique_nonmissing,
            should_plot = should_plot,
            note = note,
        ))
    end

    return DataFrame(rows)
end

function plottable_sort_columns(data_bundle; candidates = DEFAULT_SORT_COLUMN_CANDIDATES)
    summary = summarize_sort_columns(data_bundle; candidates = candidates)
    return Symbol.(summary.derived_name[summary.should_plot])
end

function overview_sort_order_audit_df(data_bundle; sort_columns = plottable_sort_columns(data_bundle))
    events_subset, trial_type_subset = game_events_subset(data_bundle)
    rows = NamedTuple[]
    for sort_col in sort_columns
        sort_col in propertynames(events_subset) || continue
        for trial_type in unique(trial_type_subset)
            mask = trial_type_subset .== trial_type
            events_trials = copy(events_subset[mask, :])
            nrow(events_trials) == 0 && continue
            sort_cols = Week15LabelStudioExport.effective_sort_columns(events_trials, sort_col)
            sort_cols_with_row = vcat(sort_cols, [:__row_idx__])
            order = Week15LabelStudioExport.trial_sort_order(events_trials, sort_col)

            order_df = DataFrame()
            order_df[!, :__row_idx__] = collect(1:nrow(events_trials))
            for col in sort_cols
                order_df[!, col] = copy(events_trials[!, col])
            end
            sort!(order_df, sort_cols_with_row)
            expected_order = Int.(order_df[!, :__row_idx__])

            push!(rows, (
                trial_type = String(trial_type),
                sort_col = String(sort_col),
                effective_sort_columns = join(string.(sort_cols_with_row), ", "),
                n_trials = nrow(events_trials),
                unique_values = length(unique(collect(skipmissing(events_trials[!, sort_col])))),
                source_guard = length(sort_cols) > 1 && first(sort_cols) != sort_col,
                status = sort(order) == collect(1:nrow(events_trials)) && order == expected_order ? "ok" : "mismatch",
            ))
        end
    end
    return DataFrame(rows)
end

function select_random_channels(channel_names::Vector{String}, sort_col::Symbol; n_channels::Int = DEFAULT_RANDOM_CHANNEL_COUNT, seed::Int = 15)
    n_total = length(channel_names)
    n_pick = min(n_channels, n_total)
    sort_seed = seed + sum(Int(c) for c in String(sort_col))
    rng = MersenneTwister(sort_seed)
    idxs = sort(randperm(rng, n_total)[1:n_pick])
    return [(channel = idx, channel_name = channel_names[idx]) for idx in idxs]
end

function build_sortvar_preview(data_bundle;
        sort_col::Symbol,
        n_channels::Int = DEFAULT_RANDOM_CHANNEL_COUNT,
        seed::Int = 15,
        trial_types = nothing,
        low_pass_factor::Real = Week15LabelStudioExport.LOWPASS_SIGMA)
    trial_types === nothing && (trial_types = collect(data_bundle.trial_types))
    sort_col in propertynames(data_bundle.events) || error("Sort column not found in events.csv: $sort_col")

    selected_channels = select_random_channels(String.(data_bundle.channel_names), sort_col;
        n_channels = n_channels,
        seed = seed,
    )

    images = Matrix{Float32}[]
    metadata = NamedTuple[]

    h5open(data_bundle.h5_path, "r") do file
        erps = file["epochs"]
        for trial_type in trial_types
            for ch in selected_channels
                record = (
                    channel = Int(ch.channel),
                    channel_name = String(ch.channel_name),
                    trial_type = String(trial_type),
                    sort_col = sort_col,
                )
                img = build_image_from_record(erps, data_bundle, record; low_pass_factor = low_pass_factor)
                push!(images, img)
                push!(metadata, (
                    channel = Int(ch.channel),
                    channel_name = String(ch.channel_name),
                    trial_type = String(trial_type),
                    sort_variable = String(sort_col),
                    n_trials = size(img, 1),
                    n_timepoints = size(img, 2),
                ))
            end
        end
    end

    return (
        sort_col = sort_col,
        trial_types = String.(trial_types),
        channel_selection = selected_channels,
        images = images,
        metadata = metadata,
    )
end

function shared_color_stats(images)
    vals = Float32[]
    sizehint!(vals, sum(length, images))
    for img in images
        append!(vals, vec(Float32.(img)))
    end
    _, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(reshape(vals, :, 1))
    return (
        colorrange = colorrange,
        tick_vals = tick_vals,
        tick_labels = tick_labels,
        cmap = cmap,
    )
end

function axis_ticks(img::AbstractMatrix)
    n_trials, n_time = size(img)
    x_mid = clamp(Int(round((n_time + 1) / 2)), 1, n_time)
    y_mid = clamp(Int(round((n_trials + 1) / 2)), 1, n_trials)
    return (
        xticks = ([1, x_mid, n_time], [string(1), string(x_mid), string(n_time)]),
        yticks = ([1, y_mid, n_trials], [string(1), string(y_mid), string(n_trials)]),
    )
end

function plot_sort_variable_figure(preview)
    n_rows = length(preview.trial_types)
    n_cols = length(preview.channel_selection)
    stats = shared_color_stats(preview.images)

    fig = Figure(
        size = (PANEL_PX * n_cols + CB_PX + 120, ROW_PX * n_rows + 100),
        figure_padding = 18,
    )
    Label(fig[0, 1:n_cols], "Sort variable: $(preview.sort_col)"; fontsize = 24, tellwidth = false)

    hm_ref = nothing
    for row in 1:n_rows
        for col in 1:n_cols
            idx = (row - 1) * n_cols + col
            img = clamp.(Float32.(preview.images[idx]), stats.colorrange[1], stats.colorrange[2])
            meta = preview.metadata[idx]
            ticks = axis_ticks(img)

            ax = Axis(fig[row, col];
                title = row == 1 ? "ch$(meta.channel) $(meta.channel_name)" : "",
                xlabel = row == n_rows ? "time" : "",
                ylabel = col == 1 ? "$(meta.trial_type)\ntrials" : "",
                titlesize = 15,
                xlabelsize = 13,
                ylabelsize = 13,
                xticklabelsize = 10,
                yticklabelsize = 10,
            )
            ax.xticks = ticks.xticks
            ax.yticks = ticks.yticks

            hm = heatmap!(
                ax,
                1:size(img, 2),
                1:size(img, 1),
                permutedims(img, (2, 1));
                colormap = stats.cmap,
                colorrange = stats.colorrange,
            )
            hm_ref === nothing && (hm_ref = hm)

            if row < n_rows
                hidexdecorations!(ax; label = false, ticklabels = false, ticks = false, grid = false)
            end
            if col > 1
                hideydecorations!(ax; label = false, ticklabels = false, ticks = false, grid = false)
            end
        end
    end

    Colorbar(fig[1:n_rows, n_cols + 1], hm_ref;
        width = 20,
        ticklabelsize = 12,
        ticks = (stats.tick_vals, stats.tick_labels),
    )

    for row in 1:n_rows
        rowsize!(fig.layout, row, Fixed(ROW_PX))
    end
    for col in 1:n_cols
        colsize!(fig.layout, col, Fixed(PANEL_PX))
    end
    colsize!(fig.layout, n_cols + 1, Fixed(CB_PX))
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)

    return fig
end

function plot_all_sort_variable_figures(data_bundle;
        sort_columns = plottable_sort_columns(data_bundle),
        n_channels::Int = 2,
        seed::Int = 15,
        trial_types = nothing,
        low_pass_factor::Real = Week15LabelStudioExport.LOWPASS_SIGMA)
    rows = NamedTuple[]
    for sort_col in sort_columns
        preview = build_sortvar_preview(data_bundle;
            sort_col = sort_col,
            n_channels = n_channels,
            seed = seed,
            trial_types = trial_types,
            low_pass_factor = low_pass_factor,
        )
        display(plot_sort_variable_figure(preview))
        push!(rows, (
            sort_col = String(sort_col),
            n_images = length(preview.images),
            channels = join([item.channel_name for item in preview.channel_selection], ", "),
            trial_types = join(preview.trial_types, ", "),
        ))
    end
    return DataFrame(rows)
end

end
