#!/usr/bin/env julia
# Build Week 21 Label Studio summary tables and plots.
#
# This is the Julia rewrite of week21_labeling_summary_plots.py. It reads the
# tracking CSV produced by update_labelstudio_annotation_tracking.py and writes
# CSV summary tables, PNG plots, and a summary.json into
# notebooks/week_21/outputs/week21_labeling_summary.

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
    ]))
    for candidate in candidates
        if isdir(joinpath(candidate, "notebooks")) && isdir(joinpath(candidate, "scripts"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir).")
end

const REPO_ROOT = find_repo_root()
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
Pkg.activate(MODEL_ENV_DIR)

using CSV
using DataFrames
using JSON3
using CairoMakie
using Printf: @sprintf

const PATTERN_CLASSES = [
    "sigmoid",
    "one_sided_fan",
    "two_sided_fan",
    "diverging_bar",
    "hourglass",
    "tilted_bar",
]

const CLASS_COLORS = Dict(
    "no_class"      => "#7a869a",
    "sigmoid"       => "#2166ac",
    "one_sided_fan" => "#67a9cf",
    "two_sided_fan" => "#1b9e77",
    "diverging_bar" => "#d73027",
    "hourglass"     => "#fdae61",
    "tilted_bar"    => "#984ea3",
)

const PATTERN_BAR_COLOR = "#d95f02"
const POSITIVE_RATE_COLOR = "#1b9e77"
const EXCLUDED_TRAINING_DATASETS = Set(["02_new_eeget_rsod"])

cellstr(x) = ismissing(x) ? "" : string(x)

function truthy(x)
    return lowercase(strip(cellstr(x))) in ("true", "1", "yes")
end

function safe_float(x, default::Float64 = 0.0)
    ismissing(x) && return default
    x isa Number && return Float64(x)
    s = strip(string(x))
    isempty(s) && return default
    parsed = tryparse(Float64, s)
    return parsed === nothing ? default : parsed
end

function short_label(s::AbstractString, max_len::Int = 34)
    return length(s) <= max_len ? String(s) : String(s[1:max_len-1]) * "..."
end

ratio(num, den) = den == 0 ? 0.0 : num / den

function _sorted_json_dict(d::AbstractDict)
    pairs = sort(collect(d); by = first)
    parts = ["$(JSON3.write(string(k))):$(JSON3.write(v))" for (k, v) in pairs]
    return "{" * join(parts, ",") * "}"
end

function read_classified_annotations(path::AbstractString)
    df = CSV.read(path, DataFrame; types = String, missingstring = "")
    return df[cellstr.(df.label_status) .== "classified", :]
end

function load_source_references(week21::AbstractString)
    refs = Dict{String, Dict{String, Any}}()
    for entry in readdir(week21; join = true)
        isdir(entry) || continue
        startswith(basename(entry), "labelstudio_export_") || continue
        for sub in readdir(entry; join = true)
            isdir(sub) || continue
            json_path = joinpath(sub, "source_reference.json")
            isfile(json_path) || continue
            data = try
                JSON3.read(read(json_path, String), Dict{String, Any})
            catch
                continue
            end
            dataset_key = string(get(data, "dataset_key", basename(sub)))
            current = get!(refs, dataset_key, Dict{String, Any}())
            for (k, v) in data
                current[String(k)] = v
            end
            files = get!(current, "reference_files", String[])
            push!(files, json_path)
        end
    end
    return refs
end

function summarize_by_dataset(rows_df::DataFrame)
    out = NamedTuple[]
    for sdf in groupby(rows_df, :dataset_key)
        items = DataFrame(sdf)
        total = nrow(items)
        pattern = sum(truthy.(items.is_pattern_class))
        no_class = total - pattern
        class_counts = Dict{String, Int}()
        for cls in items.erp_class
            key = cellstr(cls)
            class_counts[key] = get(class_counts, key, 0) + 1
        end
        export_batches = sort(unique(cellstr.(items.export_batch)))
        dataset_key = cellstr(items.dataset_key[1])
        push!(out, (
            dataset_key = dataset_key,
            dataset_label = cellstr(items.dataset_label[1]),
            total_labeled = total,
            pattern_labeled = pattern,
            no_class_labeled = no_class,
            positive_rate = round(ratio(pattern, total); digits = 4),
            n_channels = length(unique(cellstr.(items.channel_name))),
            n_sort_variables = length(unique(cellstr.(items.sort_variable))),
            n_export_batches = length(export_batches),
            export_batches = join(export_batches, ";"),
            pattern_classes = join([cls for cls in PATTERN_CLASSES if get(class_counts, cls, 0) > 0], ";"),
            excluded_from_training = lowercase(string(dataset_key in EXCLUDED_TRAINING_DATASETS)),
        ))
    end
    df = DataFrame(out)
    sort!(df, [:total_labeled, :dataset_key]; rev = [true, false])
    return df
end

function summarize_by_export_batch(rows_df::DataFrame)
    out = NamedTuple[]
    for sdf in groupby(rows_df, :export_batch)
        items = DataFrame(sdf)
        total = nrow(items)
        pattern = sum(truthy.(items.is_pattern_class))
        push!(out, (
            export_batch = cellstr(items.export_batch[1]),
            total_labeled = total,
            pattern_labeled = pattern,
            no_class_labeled = total - pattern,
            positive_rate = round(ratio(pattern, total); digits = 4),
            n_datasets = length(unique(cellstr.(items.dataset_key))),
        ))
    end
    df = DataFrame(out)
    sort!(df, [:total_labeled, :export_batch]; rev = [true, false])
    return df
end

function summarize_dataset_classes(rows_df::DataFrame)
    counts = Dict{Tuple{String, String, String}, Int}()
    for r in eachrow(rows_df)
        key = (cellstr(r.dataset_key), cellstr(r.dataset_label), cellstr(r.erp_class))
        counts[key] = get(counts, key, 0) + 1
    end
    out = NamedTuple[]
    for ((dk, dl, cls), n) in counts
        push!(out, (
            dataset_key = dk,
            dataset_label = dl,
            erp_class = cls,
            count = n,
            is_pattern_class = lowercase(string(cls != "no_class")),
        ))
    end
    df = DataFrame(out)
    sort!(df, [:dataset_key, :erp_class])
    return df
end

function summarize_sort_variables(rows_df::DataFrame)
    all_out = NamedTuple[]
    pos_out = NamedTuple[]
    for sdf in groupby(rows_df, [:dataset_key, :sort_variable])
        items = DataFrame(sdf)
        total = nrow(items)
        pos_mask = truthy.(items.is_pattern_class)
        positive = items[pos_mask, :]

        class_counts = Dict{String, Int}()
        for cls in items.erp_class
            key = cellstr(cls)
            class_counts[key] = get(class_counts, key, 0) + 1
        end
        positive_class_counts = Dict{String, Int}(
            cls => class_counts[cls] for cls in PATTERN_CLASSES if get(class_counts, cls, 0) > 0
        )

        row = (
            dataset_key = cellstr(items.dataset_key[1]),
            dataset_label = cellstr(items.dataset_label[1]),
            sort_variable = cellstr(items.sort_variable[1]),
            export_batches = join(sort(unique(cellstr.(items.export_batch))), ";"),
            total_labeled = total,
            pattern_labeled = nrow(positive),
            no_class_labeled = get(class_counts, "no_class", 0),
            positive_rate = round(ratio(nrow(positive), total); digits = 4),
            n_channels_total = length(unique(cellstr.(items.channel_name))),
            n_channels_with_positive = length(unique(cellstr.(positive.channel_name))),
            channels_with_positive = join(sort(unique(cellstr.(positive.channel_name))), ";"),
            pattern_classes = join([c for c in PATTERN_CLASSES if get(class_counts, c, 0) > 0], ";"),
            class_counts_json = _sorted_json_dict(class_counts),
            positive_class_counts_json = _sorted_json_dict(positive_class_counts),
        )
        push!(all_out, row)
        nrow(positive) > 0 && push!(pos_out, row)
    end

    all_df = DataFrame(all_out)
    pos_df = DataFrame(pos_out)
    isempty(all_df) || sort!(all_df, [:dataset_key, :sort_variable])
    isempty(pos_df) || sort!(pos_df, [:pattern_labeled, :dataset_key, :sort_variable]; rev = [true, false, false])
    return all_df, pos_df
end

function positive_instances(rows_df::DataFrame)
    pos = rows_df[truthy.(rows_df.is_pattern_class), :]
    out_cols = [:dataset_key, :dataset_label, :sort_variable, :channel_name, :channel_idx,
                :erp_class, :export_batch, :tracking_key, :label_studio_project_id,
                :label_studio_task_id, :image_file]
    df = select(pos, out_cols...)
    isempty(df) || sort!(df, [:dataset_key, :sort_variable, :channel_name])
    return df
end

function _ref_str_list(ref::AbstractDict, key::AbstractString)
    raw = get(ref, key, nothing)
    raw === nothing && return String[]
    return [string(x) for x in raw]
end

function source_reference_rows(dataset_summary::DataFrame, refs::Dict{String, Dict{String, Any}})
    out = NamedTuple[]
    for r in eachrow(dataset_summary)
        dataset_key = cellstr(r.dataset_key)
        ref = get(refs, dataset_key, Dict{String, Any}())
        component = get(ref, "source_component", nothing)
        component = component === nothing ? get(ref, "component", "") : component
        push!(out, (
            dataset_key = dataset_key,
            dataset_label = cellstr(r.dataset_label),
            week19_notebook_path = string(get(ref, "week19_notebook_path", "")),
            source_component = string(component),
            reader_docs = string(get(ref, "reader_docs", "")),
            h5_path = string(get(ref, "h5_path", "")),
            events_path = string(get(ref, "events_path", "")),
            selected_sort_columns = join(_ref_str_list(ref, "selected_sort_columns"), ";"),
            recommended_sort_columns = join(_ref_str_list(ref, "recommended_sort_columns"), ";"),
            reference_files = join(_ref_str_list(ref, "reference_files"), ";"),
        ))
    end
    return DataFrame(out)
end

# --- Plot helpers ----------------------------------------------------------

function _bar_figure(n_rows::Int; width::Int = 1400, base::Float64 = 200.0, per_row::Float64 = 38.0)
    height = round(Int, max(520.0, per_row * n_rows + base))
    return Figure(size = (width, height))
end

function plot_stacked_no_class_pattern(dataset_summary::DataFrame, path::AbstractString)
    rows = dataset_summary[end:-1:1, :]
    isempty(rows) && return path
    labels = [short_label(cellstr(k)) for k in rows.dataset_key]
    no_class = Float64.(rows.no_class_labeled)
    pattern = Float64.(rows.pattern_labeled)
    y = collect(1.0:Float64(nrow(rows)))

    fig = _bar_figure(nrow(rows))
    ax = Axis(fig[1, 1];
        title = "Week 21 labeled images by data source",
        xlabel = "labeled ERP images",
        yticks = (y, labels),
    )
    barplot!(ax, y, no_class; direction = :x,
        color = CLASS_COLORS["no_class"], label = "no_class")
    barplot!(ax, y, pattern; direction = :x, offset = no_class,
        color = PATTERN_BAR_COLOR, label = "pattern class")
    for (i, r) in enumerate(eachrow(rows))
        total = Int(r.total_labeled)
        gap = max(3.0, total * 0.01)
        text!(ax, Float64(total) + gap, Float64(i);
            text = "$(Int(r.pattern_labeled))/$(total)",
            align = (:left, :center), fontsize = 11)
    end
    axislegend(ax; position = :rb)
    save(path, fig)
    return path
end

function plot_positive_rate(dataset_summary::DataFrame, path::AbstractString)
    rows = sort(dataset_summary, :positive_rate)
    isempty(rows) && return path
    labels = [short_label(cellstr(k)) for k in rows.dataset_key]
    values = Float64.(rows.positive_rate)
    y = collect(1.0:Float64(nrow(rows)))

    fig = _bar_figure(nrow(rows); width = 1300)
    ax = Axis(fig[1, 1];
        title = "Pattern-class rate by data source",
        xlabel = "pattern-class rate",
        yticks = (y, labels),
    )
    barplot!(ax, y, values; direction = :x, color = POSITIVE_RATE_COLOR)
    xmax = max(0.25, min(1.0, maximum(values; init = 0.0) + 0.12))
    xlims!(ax, 0, xmax)
    for (i, v) in enumerate(values)
        text!(ax, min(v + 0.012, 0.98), Float64(i);
            text = @sprintf("%.1f%%", 100 * v),
            align = (:left, :center), fontsize = 11)
    end
    save(path, fig)
    return path
end

function plot_classes_by_dataset(dataset_class_rows::DataFrame, dataset_summary::DataFrame, path::AbstractString)
    keep = dataset_summary[dataset_summary.pattern_labeled .> 0, :]
    datasets = String.(reverse(cellstr.(keep.dataset_key)))
    isempty(datasets) && return path
    counts = Dict{Tuple{String, String}, Int}()
    for r in eachrow(dataset_class_rows)
        counts[(cellstr(r.dataset_key), cellstr(r.erp_class))] = Int(r.count)
    end
    y = collect(1.0:Float64(length(datasets)))

    fig = _bar_figure(length(datasets); width = 1500, per_row = 42.0, base = 220.0)
    ax = Axis(fig[1, 1];
        title = "Manual pattern classes found by data source",
        xlabel = "pattern-class instances",
        yticks = (y, [short_label(d) for d in datasets]),
    )
    left = zeros(Float64, length(datasets))
    for cls in PATTERN_CLASSES
        vals = Float64[get(counts, (d, cls), 0) for d in datasets]
        barplot!(ax, y, vals; direction = :x, offset = copy(left),
            color = CLASS_COLORS[cls], label = cls)
        left .+= vals
    end
    Legend(fig[1, 2], ax; framevisible = false)
    save(path, fig)
    return path
end

function plot_export_batches(batch_summary::DataFrame, path::AbstractString)
    rows = batch_summary[end:-1:1, :]
    isempty(rows) && return path
    labels = String.(cellstr.(rows.export_batch))
    no_class = Float64.(rows.no_class_labeled)
    pattern = Float64.(rows.pattern_labeled)
    y = collect(1.0:Float64(nrow(rows)))

    height = round(Int, max(450.0, 60.0 * nrow(rows) + 200.0))
    fig = Figure(size = (1400, height))
    ax = Axis(fig[1, 1];
        title = "Labeling volume by export batch",
        xlabel = "labeled ERP images",
        yticks = (y, labels),
    )
    barplot!(ax, y, no_class; direction = :x,
        color = CLASS_COLORS["no_class"], label = "no_class")
    barplot!(ax, y, pattern; direction = :x, offset = no_class,
        color = PATTERN_BAR_COLOR, label = "pattern class")
    axislegend(ax; position = :rb)
    save(path, fig)
    return path
end

function plot_top_positive_sort_variables(positive_sort_rows::DataFrame, path::AbstractString; top_n::Int = 25)
    n = min(top_n, nrow(positive_sort_rows))
    n == 0 && return path
    rows = positive_sort_rows[1:n, :]
    labels = [short_label("$(cellstr(r.dataset_key)) | $(cellstr(r.sort_variable))", 54) for r in eachrow(rows)]
    y = collect(1.0:Float64(n))

    height = round(Int, max(700.0, 36.0 * n + 200.0))
    fig = Figure(size = (1600, height))
    ax = Axis(fig[1, 1];
        title = "Top $(n) sort variables with pattern instances",
        xlabel = "pattern-class instances",
        yticks = (y, labels),
        yreversed = true,
    )
    left = zeros(Float64, n)
    for cls in PATTERN_CLASSES
        vals = Float64[
            get(JSON3.read(cellstr(r.class_counts_json), Dict{String, Int}), cls, 0)
            for r in eachrow(rows)
        ]
        barplot!(ax, y, vals; direction = :x, offset = copy(left),
            color = CLASS_COLORS[cls], label = cls)
        left .+= vals
    end
    Legend(fig[1, 2], ax; framevisible = false)
    save(path, fig)
    return path
end

function plot_sort_variable_heatmap(positive_sort_rows::DataFrame, path::AbstractString)
    isempty(positive_sort_rows) && return path
    datasets = sort(unique(cellstr.(positive_sort_rows.dataset_key)))
    sv_counts = Dict{String, Int}()
    for sv in positive_sort_rows.sort_variable
        key = cellstr(sv)
        sv_counts[key] = get(sv_counts, key, 0) + 1
    end
    sort_variables = first.(sort(collect(sv_counts); by = p -> -p.second))
    sort_variables = sort_variables[1:min(28, length(sort_variables))]
    (isempty(datasets) || isempty(sort_variables)) && return path

    value = Dict{Tuple{String, String}, Int}()
    for r in eachrow(positive_sort_rows)
        k = (cellstr(r.dataset_key), cellstr(r.sort_variable))
        value[k] = get(value, k, 0) + Int(r.pattern_labeled)
    end

    matrix = zeros(Int, length(sort_variables), length(datasets))
    for (i, sv) in enumerate(sort_variables), (j, ds) in enumerate(datasets)
        matrix[i, j] = get(value, (ds, sv), 0)
    end

    width = round(Int, max(1100.0, 46.0 * length(sort_variables) + 360.0))
    height = round(Int, max(620.0, 44.0 * length(datasets) + 240.0))
    fig = Figure(size = (width, height))
    ax = Axis(fig[1, 1];
        title = "Pattern-class count by data source and sort variable",
        xlabel = "sort variable",
        ylabel = "data source",
        xticks = (1:length(sort_variables), sort_variables),
        yticks = (1:length(datasets), [short_label(d, 32) for d in datasets]),
        xticklabelrotation = pi / 3,
        xticklabelalign = (:right, :top),
        xticklabelsize = 10,
        yticklabelsize = 10,
    )
    hm = heatmap!(ax, 1:length(sort_variables), 1:length(datasets), matrix; colormap = :YlOrRd)
    for j in 1:length(datasets), i in 1:length(sort_variables)
        v = matrix[i, j]
        v > 0 && text!(ax, Float64(i), Float64(j); text = string(v),
            align = (:center, :center), fontsize = 9)
    end
    Colorbar(fig[1, 2], hm; label = "pattern-class count")
    save(path, fig)
    return path
end

function plot_annotation_lead_time(rows_df::DataFrame, path::AbstractString)
    batch_values = Dict{String, Vector{Float64}}()
    for r in eachrow(rows_df)
        lt = safe_float(r.annotation_lead_time)
        if isfinite(lt) && lt > 0
            push!(get!(batch_values, cellstr(r.export_batch), Float64[]), min(lt, 60.0))
        end
    end
    batches = sort(collect(keys(batch_values)); by = b -> -length(batch_values[b]))
    isempty(batches) && return path

    positions = Float64[]
    values = Float64[]
    for (i, b) in enumerate(batches)
        for v in batch_values[b]
            push!(positions, Float64(i))
            push!(values, v)
        end
    end

    height = round(Int, max(450.0, 60.0 * length(batches) + 200.0))
    fig = Figure(size = (1400, height))
    ax = Axis(fig[1, 1];
        title = "Label Studio annotation time by export batch",
        xlabel = "annotation lead time, clipped at 60s",
        yticks = (collect(1.0:Float64(length(batches))), batches),
    )
    boxplot!(ax, positions, values; orientation = :horizontal, show_outliers = false)
    save(path, fig)
    return path
end

# --- Summary entrypoint ----------------------------------------------------

function write_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

_row_to_dict(r) = Dict{String, Any}(string(k) => v for (k, v) in pairs(r))

function build_summary(; repo_root::AbstractString = REPO_ROOT)
    week21 = joinpath(repo_root, "notebooks", "week_21")
    output_dir = joinpath(week21, "outputs", "week21_labeling_summary")
    plots_dir = joinpath(output_dir, "plots")
    tables_dir = joinpath(output_dir, "tables")
    mkpath(plots_dir)
    mkpath(tables_dir)

    annotations_csv = joinpath(week21, "labelstudio_annotations_all.csv")
    rows_df = read_classified_annotations(annotations_csv)

    dataset_summary = summarize_by_dataset(rows_df)
    batch_summary = summarize_by_export_batch(rows_df)
    dataset_class_rows = summarize_dataset_classes(rows_df)
    sort_variable_rows, positive_sort_rows = summarize_sort_variables(rows_df)
    positive_instance_rows = positive_instances(rows_df)
    references = source_reference_rows(dataset_summary, load_source_references(week21))

    CSV.write(joinpath(tables_dir, "used_data_sources_summary.csv"), dataset_summary)
    CSV.write(joinpath(tables_dir, "export_batch_summary.csv"), batch_summary)
    CSV.write(joinpath(tables_dir, "dataset_class_summary.csv"), dataset_class_rows)
    CSV.write(joinpath(tables_dir, "sort_variable_summary.csv"), sort_variable_rows)
    CSV.write(joinpath(tables_dir, "positive_sort_variables_summary.csv"), positive_sort_rows)
    CSV.write(joinpath(tables_dir, "positive_instances.csv"), positive_instance_rows)
    CSV.write(joinpath(tables_dir, "source_references_summary.csv"), references)

    plot_paths = Dict(
        "labeled_images_by_dataset" => joinpath(plots_dir, "labeled_images_by_dataset.png"),
        "positive_rate_by_dataset" => joinpath(plots_dir, "positive_rate_by_dataset.png"),
        "pattern_classes_by_dataset" => joinpath(plots_dir, "pattern_classes_by_dataset.png"),
        "labels_by_export_batch" => joinpath(plots_dir, "labels_by_export_batch.png"),
        "top_positive_sort_variables" => joinpath(plots_dir, "top_positive_sort_variables.png"),
        "positive_sort_variable_heatmap" => joinpath(plots_dir, "positive_sort_variable_heatmap.png"),
        "annotation_lead_time_by_batch" => joinpath(plots_dir, "annotation_lead_time_by_batch.png"),
    )

    plot_stacked_no_class_pattern(dataset_summary, plot_paths["labeled_images_by_dataset"])
    plot_positive_rate(dataset_summary, plot_paths["positive_rate_by_dataset"])
    plot_classes_by_dataset(dataset_class_rows, dataset_summary, plot_paths["pattern_classes_by_dataset"])
    plot_export_batches(batch_summary, plot_paths["labels_by_export_batch"])
    plot_top_positive_sort_variables(positive_sort_rows, plot_paths["top_positive_sort_variables"])
    plot_sort_variable_heatmap(positive_sort_rows, plot_paths["positive_sort_variable_heatmap"])
    plot_annotation_lead_time(rows_df, plot_paths["annotation_lead_time_by_batch"])

    totals = Dict{String, Any}(
        "classified_annotations" => nrow(rows_df),
        "data_sources" => length(unique(cellstr.(rows_df.dataset_key))),
        "export_batches" => length(unique(cellstr.(rows_df.export_batch))),
        "pattern_instances" => nrow(positive_instance_rows),
        "no_class_instances" => nrow(rows_df) - nrow(positive_instance_rows),
        "sort_variables_with_patterns" => nrow(positive_sort_rows),
        "datasets_with_patterns" => count(>(0), dataset_summary.pattern_labeled),
        "excluded_training_datasets" => sort(collect(EXCLUDED_TRAINING_DATASETS)),
    )

    top_n = min(12, nrow(positive_sort_rows))
    summary = Dict{String, Any}(
        "output_dir" => output_dir,
        "tables_dir" => tables_dir,
        "plots_dir" => plots_dir,
        "annotations_csv" => annotations_csv,
        "totals" => totals,
        "top_positive_sort_variables" => [_row_to_dict(r) for r in eachrow(positive_sort_rows[1:top_n, :])],
        "dataset_summary" => [_row_to_dict(r) for r in eachrow(dataset_summary)],
        "plot_paths" => Dict(k => v for (k, v) in plot_paths if isfile(v)),
    )
    write_json(joinpath(output_dir, "summary.json"), summary)
    return summary
end

function main()
    summary = build_summary()
    println(JSON3.write(summary["totals"]))
    println("Output: ", summary["output_dir"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
