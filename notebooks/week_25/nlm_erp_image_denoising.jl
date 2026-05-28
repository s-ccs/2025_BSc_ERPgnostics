# %% [markdown]
# # ERP images with nonlocal-means denoising
#
# This notebook follows the Week-24 visualisation style, but keeps the preprocessing fixed as:
#
# 1. sort trials by the labelled sort variable,
# 2. z-score per time point across trials,
# 3. compare three image variants without resizing:
#    - NL-means denoising,
#    - NL-means denoising followed by the project Gaussian smoothing,
#    - project Gaussian smoothing only.
#
# The denoising step follows Strauss, Teuber, Steidl, and Corona-Strauss (2013), PMID 23060344,
# DOI 10.1109/TNSRE.2012.2220568. The paper applies 2D nonlocal means to ERP images using
# patch similarity to exploit self-similarity across trials. It reports a 10x10 patch,
# patch-kernel sigma `(0.5, 5.0)`, and lambda `1000` for their amplitude scale. Because this
# notebook applies NL-means after z-scoring, the lambda default is exposed separately below.

# %%
# ============================================================================
# Imports, paths, and constants
# ============================================================================

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        pwd(),
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(pwd(), ".."),
        joinpath(pwd(), "..", ".."),
    ]))
    for candidate in candidates
        if isdir(joinpath(candidate, "notebooks")) && isdir(joinpath(candidate, "datasets"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir), pwd=$(pwd()).")
end

REPO_ROOT = find_repo_root()
NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_25")
MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
Pkg.activate(MODEL_ENV_DIR; io = devnull)

using CairoMakie
using CSV
using DataFrames
using Dates
using ImageFiltering: imfilter
using JLD2
using JSON3
using Printf: @sprintf
using Statistics

CairoMakie.activate!(type = "svg")

function quiet_include(path::AbstractString)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            include(path)
        end
    end
end

if !isdefined(Main, :ERPCNNExperimentUtils)
    quiet_include(joinpath(REPO_ROOT, "notebooks", "utils", "erp_cnn_experiment_utils.jl"))
end
if !isdefined(Main, :ERPDataIO)
    quiet_include(joinpath(REPO_ROOT, "scripts", "erp_io.jl"))
end

ERPImageUtils = Main.ERPCNNExperimentUtils.ERPImageUtils
DataIO = Main.ERPDataIO

DATASETS_ROOT = joinpath(REPO_ROOT, "datasets")
OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "nlm_erp_image_denoising")
MANIFEST_PATH = joinpath(OUTPUT_DIR, "selected_instances.csv")
METADATA_PATH = joinpath(OUTPUT_DIR, "metadata.json")

CLASS_ID = Dict(
    "no_class" => 0,
    "sigmoid" => 1,
    "one_sided_fan" => 2,
    "two_sided_fan" => 3,
    "diverging_bar" => 4,
    "hourglass" => 5,
    "tilted_bar" => 6,
)

PATTERN_CLASSES = [
    "sigmoid",
    "one_sided_fan",
    "two_sided_fan",
    "diverging_bar",
    "hourglass",
    "tilted_bar",
]

MAX_INSTANCES_PER_CLASS = parse(Int, get(ENV, "WEEK25_MAX_INSTANCES_PER_CLASS", "10"))

# Paper-scale NL-means parameters: 10x10 patch, sigma=(0.5, 5.0), lambda=1000.
# The implementation uses an odd centered patch radius, so (5, 5) gives an 11x11
# patch at the same scale. Lambda is adapted for z-scored images by default.
NLM_PATCH_RADIUS = (5, 5)
NLM_PATCH_SIGMA = (0.5f0, 5.0f0)
NLM_PAPER_LAMBDA = 1000.0f0
NLM_LAMBDA = parse(Float32, get(ENV, "WEEK25_NLM_LAMBDA", "1.0"))
NLM_SEARCH_RADIUS = (
    parse(Int, get(ENV, "WEEK25_NLM_SEARCH_RADIUS_TRIALS", "3")),
    parse(Int, get(ENV, "WEEK25_NLM_SEARCH_RADIUS_TIME", "6")),
)

# Same smoothing convention as the Week-24 pipeline visualisations.
LOW_PASS_FACTOR = 75.0f0
LOWPASS_KERNEL_SIZE = (21, 21)
FILTER_BORDER = "reflect"

mkpath(OUTPUT_DIR)
println("REPO_ROOT = ", REPO_ROOT)
println("Output directory = ", OUTPUT_DIR)
println("Max instances per class = ", MAX_INSTANCES_PER_CLASS)
println("NL-means lambda for z-scored images = ", NLM_LAMBDA)

# %%
# ============================================================================
# Label loading and deterministic class-balanced selection
# ============================================================================

cellstr(x) = (ismissing(x) || x === nothing) ? "" : string(x)

dataset_dir(dataset_key::AbstractString) = joinpath(DATASETS_ROOT, dataset_key)
events_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "events.jld2")
labels_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "labels.jld2")
signals_dir(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "signals")
signal_path(dataset_key::AbstractString, channel_name::AbstractString) = joinpath(signals_dir(dataset_key), string(channel_name, ".jld2"))

function channel_index_from_name(channel_name::AbstractString)
    m = match(r"^ch(\d+)$", String(channel_name))
    m === nothing && return 0
    return parse(Int, m.captures[1])
end

function discover_real_dataset_keys()
    keys = String[]
    for dataset_key in sort(readdir(DATASETS_ROOT))
        dataset_key == "simulated" && continue
        if isdir(dataset_dir(dataset_key)) &&
                isfile(events_path(dataset_key)) &&
                isfile(labels_path(dataset_key)) &&
                isdir(signals_dir(dataset_key))
            push!(keys, dataset_key)
        end
    end
    isempty(keys) && error("No real built datasets found in $(DATASETS_ROOT).")
    return keys
end

function load_dataset_labels(dataset_key::AbstractString)
    raw = JLD2.load(labels_path(dataset_key), "labels")
    events = JLD2.load(events_path(dataset_key), "events")
    events_metadata = JLD2.load(events_path(dataset_key), "metadata")
    dataset_label = String(get(events_metadata, "dataset_label", dataset_key))

    rows = NamedTuple[]
    for row in eachrow(raw)
        channel_name = cellstr(row.channel_name)
        sort_variable = cellstr(row.sort_variable)
        erp_class = cellstr(row.erp_class)
        haskey(CLASS_ID, erp_class) || continue
        Symbol(sort_variable) in propertynames(events) || continue
        isfile(signal_path(dataset_key, channel_name)) || continue

        sig_meta = JLD2.load(signal_path(dataset_key, channel_name), "metadata")
        push!(rows, (
            dataset_key = String(dataset_key),
            dataset_label = dataset_label,
            channel_name = channel_name,
            channel_idx = Int(get(sig_meta, "channel_idx", channel_index_from_name(channel_name))),
            sort_variable = sort_variable,
            erp_class = erp_class,
            erp_class_id = CLASS_ID[erp_class],
            binary_label = erp_class == "no_class" ? 0 : 1,
        ))
    end
    return DataFrame(rows)
end

function load_all_real_labels()
    parts = DataFrame[]
    for dataset_key in discover_real_dataset_keys()
        labels = load_dataset_labels(dataset_key)
        isempty(labels) || push!(parts, labels)
    end
    labels = isempty(parts) ? DataFrame() : vcat(parts...; cols = :union)
    isempty(labels) && error("No labelled real ERP rows were found in $(DATASETS_ROOT).")
    sort!(labels, [:dataset_key, :sort_variable, :channel_name])
    labels.source_row_id = collect(1:nrow(labels))
    return labels
end

function select_pattern_examples(labels::DataFrame; max_per_class::Int = MAX_INSTANCES_PER_CLASS)
    selected_parts = DataFrame[]
    positive = labels[in.(labels.erp_class, Ref(PATTERN_CLASSES)), :]
    for class_name in PATTERN_CLASSES
        class_df = positive[positive.erp_class .== class_name, :]
        sort!(class_df, [:dataset_key, :sort_variable, :channel_name])
        isempty(class_df) && continue
        push!(selected_parts, first(class_df, min(max_per_class, nrow(class_df))))
    end
    isempty(selected_parts) && error("No positive pattern examples found.")
    return vcat(selected_parts...; cols = :union)
end

labels_all = load_all_real_labels()
selected_examples = select_pattern_examples(labels_all)
selection_summary = combine(groupby(selected_examples, :erp_class), nrow => :n_selected)

CSV.write(MANIFEST_PATH, selected_examples)
println(selection_summary)
println("Saved ", MANIFEST_PATH)

# %%
# ============================================================================
# Fixed preprocessing: sort trials, then z-score per time point
# ============================================================================

function is_valid_sort_value(v)
    (ismissing(v) || v === nothing) && return false
    if v isa Real
        return isfinite(Float64(v))
    end
    return !isempty(strip(string(v)))
end

function valid_sort_mask(events::DataFrame, sort_col::Symbol)
    sort_col in propertynames(events) || error("Sort column $(sort_col) missing.")
    return [is_valid_sort_value(v) for v in events[!, sort_col]]
end

function load_sorted_zscored_image(row)
    dataset_key = cellstr(row.dataset_key)
    channel_name = cellstr(row.channel_name)
    sort_variable = cellstr(row.sort_variable)
    sort_col = Symbol(sort_variable)

    events = JLD2.load(events_path(dataset_key), "events")
    events_metadata = JLD2.load(events_path(dataset_key), "metadata")
    data_time_trials = Matrix{Float32}(JLD2.load(signal_path(dataset_key, channel_name), "data_time_trials"))
    n = min(nrow(events), size(data_time_trials, 2))
    events = events[1:n, :]
    data_time_trials = data_time_trials[:, 1:n]

    keep = valid_sort_mask(events, sort_col)
    any(keep) || error("No valid sort values for $(dataset_key) $(channel_name) $(sort_variable).")
    events_valid = events[keep, :]
    data_valid = data_time_trials[:, keep]
    sort_order = DataIO.trial_sort_order(events_valid, sort_col)

    sorted_time_trials = data_valid[:, sort_order]
    z_time_trials = ERPImageUtils.zscore_timepoints(sorted_time_trials)
    img_trials_time = Float32.(permutedims(z_time_trials, (2, 1)))

    return (
        dataset_key = dataset_key,
        dataset_label = cellstr(row.dataset_label),
        channel_name = channel_name,
        channel_idx = Int(row.channel_idx),
        sort_variable = sort_variable,
        erp_class = cellstr(row.erp_class),
        erp_class_id = Int(row.erp_class_id),
        source_row_id = Int(row.source_row_id),
        image = img_trials_time,
        sort_order = sort_order,
        n_trials_original = Int(n),
        n_trials_valid = Int(size(img_trials_time, 1)),
        n_trials_filtered_out = Int(length(keep) - count(keep)),
        n_timepoints = Int(size(img_trials_time, 2)),
        events_metadata = events_metadata,
    )
end

# %%
# ============================================================================
# Nonlocal means for ERP images
# ============================================================================

function reflect_index(i::Int, n::Int)
    while i < 1 || i > n
        i = i < 1 ? 2 - i : 2 * n - i
    end
    return i
end

function shifted_reflect(img::AbstractMatrix{Float32}, dr::Int, dc::Int)
    n_trials, n_timepoints = size(img)
    out = similar(img)
    @inbounds for c in 1:n_timepoints, r in 1:n_trials
        out[r, c] = img[reflect_index(r + dr, n_trials), reflect_index(c + dc, n_timepoints)]
    end
    return out
end

function nlm_patch_kernel(patch_radius::Tuple{Int, Int}, patch_sigma::Tuple{<:Real, <:Real})
    radius_trials, radius_time = patch_radius
    sigma_trials, sigma_time = Float32.(patch_sigma)
    kernel = Matrix{Float32}(undef, 2 * radius_trials + 1, 2 * radius_time + 1)
    @inbounds for dt in -radius_time:radius_time, dr in -radius_trials:radius_trials
        kernel[dr + radius_trials + 1, dt + radius_time + 1] =
            exp(-0.5f0 * ((Float32(dr) / sigma_trials)^2 + (Float32(dt) / sigma_time)^2))
    end
    return kernel ./ sum(kernel)
end

const NLM_PATCH_KERNEL = nlm_patch_kernel(NLM_PATCH_RADIUS, NLM_PATCH_SIGMA)

function nonlocal_means_erp(img::AbstractMatrix;
        search_radius::Tuple{Int, Int} = NLM_SEARCH_RADIUS,
        lambda::Real = NLM_LAMBDA,
        patch_kernel::AbstractMatrix{Float32} = NLM_PATCH_KERNEL)

    x = Float32.(img)
    acc = zeros(Float32, size(x))
    normalizer = zeros(Float32, size(x))
    lambda_f = Float32(lambda)

    for dc in -search_radius[2]:search_radius[2], dr in -search_radius[1]:search_radius[1]
        shifted = (dr == 0 && dc == 0) ? x : shifted_reflect(x, dr, dc)
        patch_distance = Float32.(imfilter((x .- shifted).^2, patch_kernel, "reflect"))
        weights = exp.(-patch_distance ./ lambda_f)
        acc .+= weights .* shifted
        normalizer .+= weights
    end

    return Float32.(acc ./ max.(normalizer, eps(Float32)))
end

function gaussian_smooth_same_size(img::AbstractMatrix)
    kernel = ERPImageUtils.gaussian_kernel(
        LOW_PASS_FACTOR,
        size(img),
        size(img),
        LOWPASS_KERNEL_SIZE,
    )
    return Float32.(imfilter(Float32.(img), kernel, FILTER_BORDER))
end

function build_comparison_images(row)
    prepared = load_sorted_zscored_image(row)
    nlm_img = nonlocal_means_erp(prepared.image)
    nlm_smoothed_img = gaussian_smooth_same_size(nlm_img)
    smoothed_img = gaussian_smooth_same_size(prepared.image)

    return merge(prepared, (
        nlm = nlm_img,
        nlm_smoothed = nlm_smoothed_img,
        smoothed = smoothed_img,
    ))
end

# %%
# ============================================================================
# Plot helpers
# ============================================================================

VARIANT_SPECS = [
    (key = :nlm, label = "NL-means"),
    (key = :nlm_smoothed, label = "NL-means + smoothing"),
    (key = :smoothed, label = "Smoothing only"),
]

TILE_WIDTH = 250
TILE_HEIGHT = 230
COLORBAR_WIDTH = 10
COLORBAR_TICK_FONT_SIZE = 9
HEADER_FONT_SIZE = 24
INSTANCE_TITLE_FONT_SIZE = 13
ROW_LABEL_FONT_SIZE = 18

function color_stats_for_image(img::AbstractMatrix)
    _, colorrange, tick_vals, tick_labels, _ = ERPImageUtils.clipped_color_stats_quantile_zero_ticks(Float32.(img))
    vmin, vmax = colorrange
    cmap = ERPImageUtils.make_diverging_cmap_zero_centered(vmin, vmax)
    return colorrange, tick_vals, tick_labels, cmap
end

function compact_dataset_label(dataset_key::AbstractString)
    label = replace(dataset_key, "02_new_" => "")
    label = replace(label, "_" => " ")
    return label
end

function instance_title(comp)
    return "$(compact_dataset_label(comp.dataset_key))\n$(comp.channel_name) | $(comp.sort_variable)\n$(comp.n_trials_valid)x$(comp.n_timepoints)"
end

function add_erp_thumbnail!(fig, row_idx::Int, axis_col::Int, colorbar_col::Int, img::AbstractMatrix;
        title::AbstractString)

    colorrange, tick_vals, tick_labels, cmap = color_stats_for_image(img)
    vmin, vmax = colorrange
    clipped = clamp.(Float32.(img), Float32(vmin), Float32(vmax))
    n_trials, n_timepoints = size(clipped)

    ax = Axis(fig[row_idx, axis_col];
        title = title,
        titlesize = INSTANCE_TITLE_FONT_SIZE,
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        xgridvisible = false,
        ygridvisible = false,
        xlabel = "",
        ylabel = "",
        aspect = AxisAspect(1),
    )
    hidespines!(ax)
    hm = heatmap!(
        ax,
        1:n_timepoints,
        1:n_trials,
        permutedims(clipped, (2, 1));
        colormap = cmap,
        colorrange = colorrange,
        rasterize = true,
    )
    Colorbar(fig[row_idx, colorbar_col], hm;
        ticks = (tick_vals, tick_labels),
        ticklabelsize = COLORBAR_TICK_FONT_SIZE,
        width = COLORBAR_WIDTH,
    )
    return ax, hm
end

function safe_slug(s::AbstractString)
    slug = lowercase(replace(s, r"[^A-Za-z0-9]+" => "_"))
    return strip(slug, '_')
end

function plot_class_examples(class_name::AbstractString, examples::DataFrame)
    n_examples = nrow(examples)
    n_examples == 0 && error("No examples passed for $(class_name).")

    comparisons = NamedTuple[]
    for (idx, row) in enumerate(eachrow(examples))
        println(@sprintf(
            "[%s %02d/%02d] %s | %s | %s",
            class_name,
            idx,
            n_examples,
            cellstr(row.dataset_key),
            cellstr(row.channel_name),
            cellstr(row.sort_variable),
        ))
        push!(comparisons, build_comparison_images(row))
    end

    fig_width = max(900, 140 + (TILE_WIDTH + 45) * n_examples)
    fig_height = 120 + TILE_HEIGHT * length(VARIANT_SPECS)
    fig = Figure(size = (fig_width, fig_height), figure_padding = 24)

    Label(fig[1, 1:(2 * n_examples)], replace(class_name, "_" => " ");
        fontsize = HEADER_FONT_SIZE,
        font = :bold,
        tellwidth = false,
        padding = (0, 0, 0, 8),
    )

    for (variant_idx, variant) in enumerate(VARIANT_SPECS)
        fig_row = variant_idx + 1
        Label(fig[fig_row, 0], variant.label;
            rotation = pi / 2,
            fontsize = ROW_LABEL_FONT_SIZE,
            font = :bold,
            tellheight = false,
            padding = (0, 8, 0, 0),
        )
        for (example_idx, comp) in enumerate(comparisons)
            add_erp_thumbnail!(
                fig,
                fig_row,
                2 * example_idx - 1,
                2 * example_idx,
                getproperty(comp, variant.key);
                title = variant_idx == 1 ? instance_title(comp) : "",
            )
        end
    end

    for example_idx in 1:n_examples
        colsize!(fig.layout, 2 * example_idx - 1, Fixed(TILE_WIDTH))
        colsize!(fig.layout, 2 * example_idx, Fixed(42))
    end
    for gap_idx in 1:2:(2 * n_examples - 1)
        colgap!(fig.layout, gap_idx, 8)
    end
    for gap_idx in 2:2:(2 * n_examples - 2)
        colgap!(fig.layout, gap_idx, 16)
    end
    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    return fig, comparisons
end

# %%
# ============================================================================
# Build and export class figures
# ============================================================================

function write_json(path::AbstractString, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

figures = Dict{String, Any}()
processed_manifest_rows = NamedTuple[]

for class_name in PATTERN_CLASSES
    examples = selected_examples[selected_examples.erp_class .== class_name, :]
    isempty(examples) && continue

    fig, comparisons = plot_class_examples(class_name, examples)
    png_path = joinpath(OUTPUT_DIR, "$(safe_slug(class_name))_nlm_comparison.png")
    svg_path = joinpath(OUTPUT_DIR, "$(safe_slug(class_name))_nlm_comparison.svg")
    save(png_path, fig; px_per_unit = 2)
    save(svg_path, fig)
    figures[class_name] = fig

    for comp in comparisons
        push!(processed_manifest_rows, (
            dataset_key = comp.dataset_key,
            dataset_label = comp.dataset_label,
            channel_name = comp.channel_name,
            channel_idx = comp.channel_idx,
            sort_variable = comp.sort_variable,
            erp_class = comp.erp_class,
            erp_class_id = comp.erp_class_id,
            source_row_id = comp.source_row_id,
            n_trials_original = comp.n_trials_original,
            n_trials_valid = comp.n_trials_valid,
            n_trials_filtered_out = comp.n_trials_filtered_out,
            n_timepoints = comp.n_timepoints,
            figure_png = png_path,
            figure_svg = svg_path,
        ))
    end

    println("Saved ", png_path)
    println("Saved ", svg_path)
end

processed_manifest = DataFrame(processed_manifest_rows)
CSV.write(MANIFEST_PATH, processed_manifest)

metadata = Dict{String, Any}(
    "task" => "ERP image NL-means denoising visual comparison",
    "classes" => PATTERN_CLASSES,
    "max_instances_per_class" => MAX_INSTANCES_PER_CLASS,
    "selection_policy" => "deterministic first rows per class after sorting by dataset_key, sort_variable, channel_name",
    "preprocessing" => "sort trials by labelled sort variable -> zscore_timepoints -> no resize",
    "variants" => [String(v.label) for v in VARIANT_SPECS],
    "nlm_reference" => "Strauss et al. 2013, PMID 23060344, DOI 10.1109/TNSRE.2012.2220568",
    "nlm_patch_radius" => collect(NLM_PATCH_RADIUS),
    "nlm_patch_size" => [2 * NLM_PATCH_RADIUS[1] + 1, 2 * NLM_PATCH_RADIUS[2] + 1],
    "nlm_patch_sigma" => [Float64(NLM_PATCH_SIGMA[1]), Float64(NLM_PATCH_SIGMA[2])],
    "nlm_paper_lambda" => Float64(NLM_PAPER_LAMBDA),
    "nlm_lambda_zscored_images" => Float64(NLM_LAMBDA),
    "nlm_search_radius" => collect(NLM_SEARCH_RADIUS),
    "gaussian_low_pass_factor" => Float64(LOW_PASS_FACTOR),
    "gaussian_kernel_size" => collect(LOWPASS_KERNEL_SIZE),
    "filter_border" => FILTER_BORDER,
    "resize_applied" => false,
    "manifest_path" => MANIFEST_PATH,
    "output_dir" => OUTPUT_DIR,
    "timestamp" => string(now()),
)
write_json(METADATA_PATH, metadata)

println("Saved ", MANIFEST_PATH)
println("Saved ", METADATA_PATH)
processed_manifest

# %%
# Final visual sanity-check preview in the notebook output.
for class_name in PATTERN_CLASSES
    haskey(figures, class_name) && display(figures[class_name])
end
