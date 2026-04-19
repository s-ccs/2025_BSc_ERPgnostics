module Week15LabelStudioExport

using CSV
using CairoMakie
using DataFrames
using HDF5
using ImageFiltering: imfilter
using JSON3
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: gaussian_kernel, zscore_timepoints, clipped_color_stats_quantile_zero_ticks

export NOTEBOOK_DIR
export OPENNEURO_REPO_DIR
export OPENNEURO_DERIVED_DIR
export OPENNEURO_H5_PATH
export OPENNEURO_EVENTS_PATH
export DEFAULT_EXPORT_ROOT
export DEFAULT_SORT_COLUMN
export DEFAULT_GAME_TRIAL_TYPES
export load_8bit_export_data
export build_candidate_records
export select_balanced_records
export sort_distribution_df
export build_preview_images
export plot_erp_grid
export export_labelstudio_images

const NOTEBOOK_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OPENNEURO_REPO_DIR = joinpath(REPO_ROOT, "notebooks", "datasets", "ds003517")
const OPENNEURO_DERIVED_DIR = joinpath(REPO_ROOT, "notebooks", "datasets", "ds003517_sub001_derived")
const OPENNEURO_H5_PATH = joinpath(OPENNEURO_DERIVED_DIR, "epochs.hdf5")
const OPENNEURO_EVENTS_PATH = joinpath(OPENNEURO_DERIVED_DIR, "events.csv")
const OPENNEURO_PREPARE_SCRIPT = joinpath(REPO_ROOT, "scripts", "prepare_openneuro_8bit_dataset.py")
const OPENNEURO_PREPARE_PYTHON = let venv_python = joinpath(REPO_ROOT, ".venv_8bit", "bin", "python")
    isfile(venv_python) ? venv_python : "python"
end
const DEFAULT_EXPORT_ROOT = joinpath(NOTEBOOK_DIR, "label_studio_data_unlabelled_week15_8bit_game_500")

const FILTER_BORDER = "reflect"
const LOWPASS_SIGMA = 75.0f0
const LOWPASS_KERNEL_SIZE = (21, 21)

const DEFAULT_SORT_COLUMN = :onset_s
const DEFAULT_GAME_TRIAL_TYPES = [
    "SHOOT_BUTTON",
    "COLLECT_STAR",
    "MISSILE_HIT_ENEMY",
    "PLAYER_CRASH_WALL",
    "PLAYER_CRASH_ENEMY",
    "COLLECT_AMMO",
]
const SORT_TIEBREAKER_COLUMNS = [
    :run,
    :source_file,
    :epoch_index,
    :sample_index,
    :event_rank_within_type,
    :flash_index_within_run,
    :flash_index_within_trial,
    :onset_s,
    :stimulus_onset_s,
]
const LOCAL_WITHIN_RUN_SORT_COLUMNS = Set([
    :onset_s,
    :sample_index,
    :event_rank_within_type,
])
const RUN_SORT_COLUMN_CANDIDATES = [
    :run,
    :source_file,
]

const PANEL_PX = 320
const CB_PX = 62
const ROW_PX = 400
const CB_HEIGHT_REL = PANEL_PX / ROW_PX

sanitize_slug(x) = replace(string(x), r"[^A-Za-z0-9_-]" => "_")

function ensure_openneuro_8bit_dataset!()
    if isfile(OPENNEURO_H5_PATH) && isfile(OPENNEURO_EVENTS_PATH)
        return nothing
    end

    @assert isdir(OPENNEURO_REPO_DIR) "Expected cloned OpenNeuro dataset not found: $OPENNEURO_REPO_DIR"
    @assert isfile(OPENNEURO_PREPARE_SCRIPT) "Preprocessing script not found: $OPENNEURO_PREPARE_SCRIPT"
    cmd = Cmd([
        OPENNEURO_PREPARE_PYTHON,
        OPENNEURO_PREPARE_SCRIPT,
        "--source-root", OPENNEURO_REPO_DIR,
        "--output-dir", OPENNEURO_DERIVED_DIR,
    ])
    run(cmd)

    @assert isfile(OPENNEURO_H5_PATH) "8bit HDF5 not found after preprocessing: $OPENNEURO_H5_PATH"
    @assert isfile(OPENNEURO_EVENTS_PATH) "8bit events CSV not found after preprocessing: $OPENNEURO_EVENTS_PATH"
    return nothing
end

function infer_sampling_rate(times_s::AbstractVector{<:Real})
    length(times_s) < 2 && return 0
    dt = Float64(times_s[2] - times_s[1])
    dt <= 0 && return 0
    return round(Int, 1 / dt)
end

function load_h5_metadata(path::AbstractString)
    return h5open(path, "r") do file
        dataset = file["epochs"]
        times_s = haskey(file, "times_s") ? Float32.(read(file["times_s"])) : Float32[]
        n_channels = size(dataset, 1)
        channel_names = haskey(file, "channel_names") ? String.(read(file["channel_names"])) : [@sprintf("ch%03d", i) for i in 1:n_channels]
        return (
            times_s = times_s,
            channel_names = channel_names,
            sampling_rate = infer_sampling_rate(times_s),
            full_shape = size(dataset),
        )
    end
end

function eightbit_time_zero_index(times_s::AbstractVector{<:Real})
    idx = findfirst(t -> t >= 0, times_s)
    idx === nothing && error("No non-negative timepoint found in 8bit dataset.")
    return Int(idx)
end

function load_8bit_export_data(; trial_types::Vector{String} = collect(DEFAULT_GAME_TRIAL_TYPES))
    ensure_openneuro_8bit_dataset!()

    events = CSV.read(OPENNEURO_EVENTS_PATH, DataFrame)
    meta = load_h5_metadata(OPENNEURO_H5_PATH)
    time_zero_idx = eightbit_time_zero_index(meta.times_s)
    trial_type_str = String.(events.trial_type)

    available = Set(trial_type_str)
    missing_trial_types = [tt for tt in trial_types if !(tt in available)]
    isempty(missing_trial_types) || error("Requested trial types not found in 8bit events.csv: $(missing_trial_types)")

    return (
        events = events,
        trial_type_str = trial_type_str,
        times_s = meta.times_s,
        channel_names = meta.channel_names,
        sampling_rate = meta.sampling_rate,
        full_shape = meta.full_shape,
        time_zero_idx = time_zero_idx,
        h5_path = OPENNEURO_H5_PATH,
        source_file = basename(OPENNEURO_H5_PATH),
        trial_types = collect(trial_types),
    )
end

function sortvalues_from(df::DataFrame, col::Symbol)
    values = df[!, col]
    if eltype(values) <: Number
        return Float64.(values)
    end
    return collect(values)
end

function run_sort_column(df::DataFrame)
    present = propertynames(df)
    for col in RUN_SORT_COLUMN_CANDIDATES
        col in present || continue
        length(unique(collect(skipmissing(df[!, col])))) > 1 || continue
        return col
    end
    return nothing
end

function effective_sort_columns(df::DataFrame, sort_col::Symbol)
    sort_cols = Symbol[sort_col]
    if sort_col in LOCAL_WITHIN_RUN_SORT_COLUMNS
        run_col = run_sort_column(df)
        if run_col !== nothing && run_col != sort_col
            sort_cols = Symbol[run_col, sort_col]
        end
    end
    for col in SORT_TIEBREAKER_COLUMNS
        col == sort_col && continue
        col in propertynames(df) || continue
        col in sort_cols && continue
        push!(sort_cols, col)
    end
    return sort_cols
end

function trial_sort_order(df::DataFrame, sort_col::Symbol)
    row_col = :__row_idx__
    sort_cols = effective_sort_columns(df, sort_col)
    sort_cols_with_row = vcat(sort_cols, [row_col])

    order_df = DataFrame()
    order_df[!, row_col] = collect(1:nrow(df))
    for col in sort_cols
        order_df[!, col] = copy(df[!, col])
    end

    sort!(order_df, sort_cols_with_row)
    return Int.(order_df[!, row_col])
end

function extract_8bit_channel_trials(erps, events::DataFrame, channel::Int, time_zero_idx::Int; post_stim_only::Bool = true)
    @assert 1 <= channel <= size(erps, 1) "Channel out of range: $channel"
    start_idx = post_stim_only ? time_zero_idx : 1
    data = Float32.(erps[channel, start_idx:end, :])
    n = min(size(data, 2), nrow(events))
    return data[:, 1:n], copy(events[1:n, :])
end

function build_base_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    @assert size(data_time_trials, 2) == nrow(events_trials) "Trial count mismatch between matrix and events."
    @assert sort_col in propertynames(events_trials) "Sort column not found: $sort_col"

    order = trial_sort_order(events_trials, sort_col)
    data_sorted = Float32.(data_time_trials[:, order])
    data_z = zscore_timepoints(data_sorted)
    return Float32.(permutedims(data_z, (2, 1)))
end

function apply_lowpass(img_trials_time::AbstractMatrix; low_pass_factor::Real = LOWPASS_SIGMA)
    out = Float32.(img_trials_time)
    if low_pass_factor > 0 && min(size(out)...) > 1
        kernel = gaussian_kernel(low_pass_factor, size(out), size(out), LOWPASS_KERNEL_SIZE)
        out = Float32.(imfilter(out, kernel, FILTER_BORDER))
    end
    return out
end

function build_candidate_records(data_bundle;
        trial_types::Vector{String} = collect(DEFAULT_GAME_TRIAL_TYPES),
        sort_col::Symbol = DEFAULT_SORT_COLUMN)
    records = NamedTuple[]

    for trial_type in trial_types
        mask = data_bundle.trial_type_str .== trial_type
        any(mask) || continue

        events_subset = copy(data_bundle.events[mask, :])
        n_trials_total = nrow(events_subset)
        n_trials_total == 0 && continue

        for (channel, channel_name) in enumerate(data_bundle.channel_names)
            push!(records, (
                channel = Int(channel),
                channel_name = String(channel_name),
                trial_type = String(trial_type),
                trial_type_slug = sanitize_slug(trial_type),
                n_trials = Int(n_trials_total),
                sort_col = sort_col,
            ))
        end
    end

    sort!(records; by = r -> (r.trial_type, r.channel))
    return records
end

function evenly_pick_indices(n::Int, need::Int)
    need = clamp(need, 0, n)
    need == 0 && return Int[]
    need >= n && return collect(1:n)

    chosen = Int[]
    seen = falses(n)
    for idx in round.(Int, range(1, n, length = need))
        if !seen[idx]
            push!(chosen, idx)
            seen[idx] = true
        end
    end

    if length(chosen) < need
        for idx in 1:n
            if !seen[idx]
                push!(chosen, idx)
                seen[idx] = true
                length(chosen) == need && break
            end
        end
    end

    return sort(chosen)
end

function select_balanced_records(records; target_count::Int = 500)
    n_total = length(records)
    target = min(target_count, n_total)
    target == n_total && return collect(records)

    trial_types = unique(String.(getfield.(records, :trial_type)))
    selected = NamedTuple[]
    leftovers = NamedTuple[]

    base = fld(target, length(trial_types))
    remainder = target - base * length(trial_types)

    for (i, trial_type) in enumerate(trial_types)
        group = [record for record in records if record.trial_type == trial_type]
        need = min(length(group), base + (i <= remainder ? 1 : 0))
        idxs = evenly_pick_indices(length(group), need)
        idxset = Set(idxs)
        append!(selected, group[idxs])
        append!(leftovers, [group[j] for j in eachindex(group) if !(j in idxset)])
    end

    if length(selected) < target
        extra_idxs = evenly_pick_indices(length(leftovers), target - length(selected))
        append!(selected, leftovers[extra_idxs])
    end

    sort!(selected; by = r -> (r.trial_type, r.channel))
    return selected
end

function sort_distribution_df(records)
    df = DataFrame(trial_type = String.(getfield.(records, :trial_type)))
    out = combine(groupby(df, :trial_type), nrow => :count)
    sort!(out, :trial_type)
    return out
end

function build_image_from_record(erps, data_bundle, record; low_pass_factor::Real = LOWPASS_SIGMA)
    data_full, events_full = extract_8bit_channel_trials(erps, data_bundle.events, record.channel, data_bundle.time_zero_idx; post_stim_only = true)
    mask = data_bundle.trial_type_str[1:nrow(events_full)] .== record.trial_type
    events_subset = copy(events_full[mask, :])
    data_subset = data_full[:, mask]
    img = build_base_image(data_subset, events_subset, record.sort_col)
    return apply_lowpass(img; low_pass_factor = low_pass_factor)
end

function build_preview_images(data_bundle, selections; indices = [1, 125, 250, 375, 500])
    n = length(selections)
    chosen = Int[]
    for raw_idx in indices
        idx = clamp(Int(raw_idx), 1, n)
        idx in chosen || push!(chosen, idx)
    end

    imgs = Matrix{Float32}[]
    meta = NamedTuple[]

    h5open(data_bundle.h5_path, "r") do file
        erps = file["epochs"]
        for (preview_id, idx) in enumerate(chosen)
            sel = selections[idx]
            img = build_image_from_record(erps, data_bundle, sel)
            push!(imgs, img)
            push!(meta, (
                id = preview_id,
                selection_index = idx,
                channel = Int(sel.channel),
                channel_name = String(sel.channel_name),
                trial_type = String(sel.trial_type),
                n_trials = size(img, 1),
                n_timepoints = size(img, 2),
                sampling_rate = Int(data_bundle.sampling_rate),
                source_file = String(data_bundle.source_file),
                sort_variable = String(sel.sort_col),
            ))
        end
    end

    return (images = imgs, metadata = meta, indices = chosen)
end

function plot_erp_grid(images, metadata; n_cols::Int = 2)
    n_images = length(images)
    n_rows = cld(n_images, n_cols)
    fig = Figure(size = ((PANEL_PX + CB_PX + 34) * n_cols, ROW_PX * n_rows), figure_padding = 24)

    for (idx, (img, meta)) in enumerate(zip(images, metadata))
        row = cld(idx, n_cols)
        col = mod1(idx, n_cols)

        clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(img)
        n_trials, n_time = size(clipped)
        x_first, x_mid, x_last = 1, Int(round((n_time + 1) / 2)), n_time
        y_first, y_mid, y_last = 1, Int(round((n_trials + 1) / 2)), n_trials

        img_col = 2 * col - 1
        cb_col = 2 * col

        title = "idx $(meta.selection_index) | ch$(meta.channel) $(meta.channel_name) | $(meta.trial_type)"

        ax = Axis(fig[row, img_col];
            title = title,
            titlesize = 20,
            xlabel = "time",
            ylabel = "trials",
            xlabelsize = 22,
            ylabelsize = 22,
            xticklabelsize = 18,
            yticklabelsize = 18,
            aspect = AxisAspect(1),
        )
        ax.xticks = ([x_first, x_mid, x_last], [string(x_first), string(x_mid), string(x_last)])
        ax.yticks = ([y_first, y_mid, y_last], [string(y_first), string(y_mid), string(y_last)])

        hm = heatmap!(
            ax,
            1:n_time,
            1:n_trials,
            permutedims(Float32.(clipped), (2, 1));
            colormap = cmap,
            colorrange = colorrange,
        )

        Colorbar(fig[row, cb_col], hm;
            width = 20,
            height = Relative(CB_HEIGHT_REL),
            valign = :center,
            ticklabelsize = 18,
            ticks = (tick_vals, tick_labels),
        )

        rowsize!(fig.layout, row, Fixed(ROW_PX))
        colsize!(fig.layout, img_col, Fixed(PANEL_PX))
        colsize!(fig.layout, cb_col, Fixed(CB_PX))
    end

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 16)
    resize_to_layout!(fig)
    return fig
end

function export_labelstudio_images(;
        target_count::Int = 500,
        export_root::AbstractString = DEFAULT_EXPORT_ROOT,
        trial_types::Vector{String} = collect(DEFAULT_GAME_TRIAL_TYPES),
        sort_col::Symbol = DEFAULT_SORT_COLUMN)
    data_bundle = load_8bit_export_data(; trial_types = trial_types)
    candidates = build_candidate_records(data_bundle;
        trial_types = trial_types,
        sort_col = sort_col,
    )
    selections = select_balanced_records(candidates; target_count = target_count)

    mkpath(export_root)
    images_dir = joinpath(export_root, "images")
    rm(images_dir; recursive = true, force = true)
    mkpath(images_dir)
    for old_manifest in filter(name -> startswith(name, "tasks_unlabelled_week15_8bit_game_"), readdir(export_root; join = true))
        rm(old_manifest; force = true)
    end

    export_dir_name = basename(export_root)
    manifest_rows = NamedTuple[]

    h5open(data_bundle.h5_path, "r") do file
        erps = file["epochs"]
        for (i, sel) in enumerate(selections)
            img = build_image_from_record(erps, data_bundle, sel)

            clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(img)
            n_trials, n_time = size(clipped)
            x_first, x_mid, x_last = 1, Int(round((n_time + 1) / 2)), n_time
            y_first, y_mid, y_last = 1, Int(round((n_trials + 1) / 2)), n_trials

            filename = @sprintf(
                "erp_week15_8bit_%03d_ch%03d_%s.png",
                i,
                Int(sel.channel),
                sel.trial_type_slug,
            )
            output_path = joinpath(images_dir, filename)

            fig = Figure(size = (PANEL_PX + CB_PX + 34, ROW_PX), figure_padding = 24)
            ax = Axis(fig[1, 1];
                title = "ch$(sel.channel) $(sel.channel_name) | $(sel.trial_type)",
                titlesize = 20,
                xlabel = "time",
                ylabel = "trials",
                xlabelsize = 22,
                ylabelsize = 22,
                xticklabelsize = 18,
                yticklabelsize = 18,
                aspect = AxisAspect(1),
            )
            ax.xticks = ([x_first, x_mid, x_last], [string(x_first), string(x_mid), string(x_last)])
            ax.yticks = ([y_first, y_mid, y_last], [string(y_first), string(y_mid), string(y_last)])

            hm = heatmap!(
                ax,
                1:n_time,
                1:n_trials,
                permutedims(Float32.(clipped), (2, 1));
                colormap = cmap,
                colorrange = colorrange,
            )

            Colorbar(fig[1, 2], hm;
                width = 20,
                height = Relative(CB_HEIGHT_REL),
                valign = :center,
                ticklabelsize = 18,
                ticks = (tick_vals, tick_labels),
            )

            colsize!(fig.layout, 1, Fixed(PANEL_PX))
            colsize!(fig.layout, 2, Fixed(CB_PX))
            rowsize!(fig.layout, 1, Fixed(ROW_PX))
            colgap!(fig.layout, 12)
            resize_to_layout!(fig)
            CairoMakie.save(output_path, fig)

            local_file_rel = joinpath(export_dir_name, "images", filename)
            local_file_rel = replace(local_file_rel, '\\' => "/")

            push!(manifest_rows, (
                id = i,
                image = "/data/local-files/?d=$(local_file_rel)",
                channel = Int(sel.channel),
                channel_name = String(sel.channel_name),
                trial_type = String(sel.trial_type),
                sort_variable = String(sel.sort_col),
                n_trials = n_trials,
                n_timepoints = n_time,
                sampling_rate = Int(data_bundle.sampling_rate),
                source_file = String(data_bundle.source_file),
                image_file = filename,
            ))
        end
    end

    manifest_df = DataFrame(manifest_rows)
    export_count = nrow(manifest_df)
    csv_manifest_path = joinpath(export_root, @sprintf("tasks_unlabelled_week15_8bit_game_%03d.csv", export_count))
    json_manifest_path = joinpath(export_root, @sprintf("tasks_unlabelled_week15_8bit_game_%03d.json", export_count))

    CSV.write(csv_manifest_path, manifest_df)
    tasks = [
        Dict(
            "id" => Int(r.id),
            "data" => Dict(
                "image" => String(r.image),
                "channel" => Int(r.channel),
                "channel_name" => String(r.channel_name),
                "trial_type" => String(r.trial_type),
                "sort_variable" => String(r.sort_variable),
                "n_trials" => Int(r.n_trials),
                "n_timepoints" => Int(r.n_timepoints),
                "sampling_rate" => Int(r.sampling_rate),
                "source_file" => String(r.source_file),
                "image_file" => String(r.image_file),
            ),
        ) for r in eachrow(manifest_df)
    ]
    open(json_manifest_path, "w") do io
        JSON3.pretty(io, tasks)
    end

    return (
        export_root = export_root,
        images_dir = images_dir,
        csv_manifest_path = csv_manifest_path,
        json_manifest_path = json_manifest_path,
        available_count = length(candidates),
        exported_count = export_count,
        selections = selections,
        distribution_df = sort_distribution_df(selections),
        manifest_df = manifest_df,
    )
end

end
