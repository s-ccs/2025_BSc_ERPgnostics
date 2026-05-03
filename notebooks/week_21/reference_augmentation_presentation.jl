#!/usr/bin/env julia
# Build presentation plots for one labeled class ERP image from the reference
# fixation dataset and the fixed-trial augmentation variants used for training.

module Week21ReferenceAugmentationPresentation

include(joinpath(@__DIR__, "resnet18_labeled_erp_cv.jl"))

using CairoMakie
using CSV
using DataFrames
using Dates
using JSON3
using Printf: @sprintf

const PRESENTATION_OUTPUT_DIR = joinpath(
    NOTEBOOK_DIR,
    "outputs",
    "reference_augmentation_presentation",
)

const DEFAULT_REFERENCE_TRACKING_KEY = get(
    ENV,
    "WEEK21_REFERENCE_AUG_TRACKING_KEY",
    "fixations_dataset||ch002||duration",
)

const PRESENTATION_TARGET_TRIALS = parse(
    Int,
    get(ENV, "WEEK21_REFERENCE_AUG_TARGET_TRIALS", string(TARGET_TRIALS)),
)

function classified_reference_patterns(labels::DataFrame)
    mask = (labels.dataset_key .== REFERENCE_DATASET_KEY) .&
        (labels.binary_label .== 1)
    out = labels[mask, :]
    isempty(out) && error("No positive class rows found for $(REFERENCE_DATASET_KEY).")
    return out
end

function select_reference_row(labels::DataFrame; tracking_key::AbstractString)
    candidates = classified_reference_patterns(labels)
    if !isempty(strip(tracking_key))
        idx = findfirst(candidates.tracking_key .== tracking_key)
        if idx !== nothing
            return candidates[idx, :]
        end
        @warn "Requested tracking key was not found among positive reference rows; using the first positive reference row instead." tracking_key
    end
    return candidates[1, :]
end

function smoothed_sorted_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    img_trials_time = CNNUtils.preprocess_sorted_zscore_image(
        data_time_trials,
        events_trials,
        sort_col,
    )
    return CNNUtils.apply_gaussian_pre_resize(
        img_trials_time;
        target_size = size(img_trials_time),
        low_pass_sigma = Generalization.LOWPASS_SIGMA,
        lowpass_kernel_size = Generalization.LOWPASS_KERNEL_SIZE,
        filter_border = Generalization.FILTER_BORDER,
    )
end

function global_plot_stats(images::Vector{Matrix{Float32}})
    vals = Float32[]
    for img in images
        for v in img
            isfinite(v) && push!(vals, Float32(v))
        end
    end
    isempty(vals) && push!(vals, 0f0)
    _, colorrange, tick_vals, tick_labels, cmap =
        CNNUtils.clipped_color_stats_quantile_zero_ticks(reshape(vals, :, 1))
    return (
        colorrange = colorrange,
        tick_vals = tick_vals,
        tick_labels = tick_labels,
        cmap = cmap,
    )
end

clip_to_range(img::AbstractMatrix, colorrange) =
    clamp.(Float32.(img), Float32(colorrange[1]), Float32(colorrange[2]))

function erp_heatmap!(
        ax::Axis,
        img::AbstractMatrix;
        colorrange,
        cmap,
        show_ylabel::Bool = true,
        show_xlabel::Bool = true)

    n_trials, n_time = size(img)
    clipped = clip_to_range(img, colorrange)
    hm = heatmap!(
        ax,
        1:n_time,
        1:n_trials,
        permutedims(clipped, (2, 1));
        colormap = cmap,
        colorrange = colorrange,
    )
    ax.xticks = ([1, max(1, cld(n_time, 2)), n_time], string.([1, max(1, cld(n_time, 2)), n_time]))
    ax.yticks = ([1, max(1, cld(n_trials, 2)), n_trials], string.([1, max(1, cld(n_trials, 2)), n_trials]))
    ax.xlabel = show_xlabel ? "post-stimulus time samples" : ""
    ax.ylabel = show_ylabel ? "sorted trials" : ""
    return hm
end

function chunk_title(chunk)
    if chunk.chunk_role == "distributed_remainder_filled"
        return @sprintf(
            "aug %02d: remainder %d + fill %d",
            chunk.chunk_index,
            chunk.remainder_trials,
            chunk.reused_fill_count,
        )
    end
    return @sprintf("aug %02d: mod split", chunk.chunk_index)
end

function plot_original_image(path::AbstractString, original_img::Matrix{Float32}, row, stats)
    fig = Figure(size = (1500, 1050), fontsize = 24, figure_padding = 42)
    title = @sprintf(
        "Original reference ERP image | %s | %s | %s",
        cellstr(row.erp_class),
        cellstr(row.channel_name),
        cellstr(row.sort_variable),
    )
    Label(fig[1, 1:2], title; fontsize = 32, font = :bold, halign = :left)
    ax = Axis(fig[2, 1]; titlesize = 24)
    hm = erp_heatmap!(
        ax,
        original_img;
        colorrange = stats.colorrange,
        cmap = stats.cmap,
    )
    Colorbar(
        fig[2, 2],
        hm;
        ticks = (stats.tick_vals, stats.tick_labels),
        label = "z-score",
        width = 24,
        ticklabelsize = 20,
        labelsize = 22,
    )
    colsize!(fig.layout, 2, Fixed(90))
    save(path, fig)
    return path
end

function plot_all_augmented_chunks(
        path::AbstractString,
        augmented_images::Vector{Matrix{Float32}},
        chunks;
        stats,
        row)

    n = length(augmented_images)
    ncols = 4
    nrows = cld(n, ncols)
    fig = Figure(size = (2200, 440 + 410 * nrows), fontsize = 18, figure_padding = 42)
    Label(
        fig[1, 1:ncols],
        @sprintf(
            "All fixed-trial augmented images | %s | %s | sort=%s | target trials=%d",
            cellstr(row.erp_class),
            cellstr(row.channel_name),
            cellstr(row.sort_variable),
            PRESENTATION_TARGET_TRIALS,
        );
        fontsize = 30,
        font = :bold,
        halign = :left,
    )

    first_hm = nothing
    for (i, img) in enumerate(augmented_images)
        r = div(i - 1, ncols) + 2
        c = mod(i - 1, ncols) + 1
        ax = Axis(
            fig[r, c];
            title = chunk_title(chunks[i]),
            titlesize = 18,
            xlabelsize = 16,
            ylabelsize = 16,
            xticklabelsize = 13,
            yticklabelsize = 13,
        )
        hm = erp_heatmap!(
            ax,
            img;
            colorrange = stats.colorrange,
            cmap = stats.cmap,
            show_ylabel = c == 1,
            show_xlabel = r == nrows + 1,
        )
        first_hm === nothing && (first_hm = hm)
    end

    Colorbar(
        fig[2:(nrows + 1), ncols + 1],
        first_hm;
        ticks = (stats.tick_vals, stats.tick_labels),
        label = "z-score",
        width = 24,
        ticklabelsize = 16,
        labelsize = 18,
    )
    colsize!(fig.layout, ncols + 1, Fixed(90))
    rowgap!(fig.layout, 24)
    colgap!(fig.layout, 18)
    save(path, fig)
    return path
end

function selected_chunk_indices(chunks)
    base = collect(1:min(4, length(chunks)))
    remainder_idx = findfirst(c -> c.chunk_role == "distributed_remainder_filled", chunks)
    if remainder_idx !== nothing
        push!(base, remainder_idx)
    elseif length(chunks) > 4
        push!(base, length(chunks))
    end
    return unique(base)
end

function plot_slide_figure(
        path::AbstractString,
        original_img::Matrix{Float32},
        augmented_images::Vector{Matrix{Float32}},
        chunks;
        stats,
        row,
        origin)

    chosen = selected_chunk_indices(chunks)
    fig = Figure(size = (2400, 1350), fontsize = 22, figure_padding = 48)
    title = @sprintf(
        "Reference class image augmentation | %s | %s | sort=%s",
        cellstr(row.erp_class),
        cellstr(row.channel_name),
        cellstr(row.sort_variable),
    )
    subtitle = @sprintf(
        "original n=%d trials -> %d augmented images with %d trials each",
        origin.n_trials,
        length(chunks),
        PRESENTATION_TARGET_TRIALS,
    )
    Label(fig[1, 1:5], title; fontsize = 36, font = :bold, halign = :left)
    Label(fig[2, 1:5], subtitle; fontsize = 26, halign = :left)

    ax_original = Axis(
        fig[3:6, 1:2];
        title = "original labeled ERP image",
        titlesize = 26,
        xlabelsize = 20,
        ylabelsize = 20,
        xticklabelsize = 16,
        yticklabelsize = 16,
    )
    first_hm = erp_heatmap!(
        ax_original,
        original_img;
        colorrange = stats.colorrange,
        cmap = stats.cmap,
    )

    for (plot_i, chunk_idx) in enumerate(chosen)
        r = div(plot_i - 1, 2) + 3
        c = mod(plot_i - 1, 2) + 3
        ax = Axis(
            fig[r, c];
            title = chunk_title(chunks[chunk_idx]),
            titlesize = 19,
            xlabelsize = 15,
            ylabelsize = 15,
            xticklabelsize = 13,
            yticklabelsize = 13,
        )
        erp_heatmap!(
            ax,
            augmented_images[chunk_idx];
            colorrange = stats.colorrange,
            cmap = stats.cmap,
            show_ylabel = c == 3,
            show_xlabel = r >= 5,
        )
    end

    Colorbar(
        fig[3:6, 5],
        first_hm;
        ticks = (stats.tick_vals, stats.tick_labels),
        label = "z-score",
        width = 24,
        ticklabelsize = 18,
        labelsize = 20,
    )

    colsize!(fig.layout, 1, Relative(0.26))
    colsize!(fig.layout, 2, Relative(0.26))
    colsize!(fig.layout, 5, Fixed(90))
    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 18)
    save(path, fig)
    return path
end

function chunk_plan_dataframe(chunks)
    rows = NamedTuple[]
    for c in chunks
        preview = join(c.trial_indices[1:min(12, length(c.trial_indices))], ";")
        filler_preview = isempty(c.filler_indices) ? "" :
            join(c.filler_indices[1:min(12, length(c.filler_indices))], ";")
        push!(rows, (
            chunk_index = c.chunk_index,
            chunk_role = c.chunk_role,
            n_trials = length(c.trial_indices),
            full_mod_split_k = c.full_mod_split_k,
            remainder_trials = c.remainder_trials,
            unique_trial_count_before_fill = c.unique_trial_count_before_fill,
            reused_fill_count = c.reused_fill_count,
            trial_indices_preview = preview,
            filler_indices_preview = filler_preview,
        ))
    end
    return DataFrame(rows)
end

function selected_instance_dataframe(row, origin, chunks)
    return DataFrame([(
        tracking_key = cellstr(row.tracking_key),
        dataset_key = cellstr(row.dataset_key),
        dataset_label = cellstr(row.dataset_label),
        channel_name = cellstr(row.channel_name),
        channel_idx = Int(row.channel_idx_int),
        sort_variable = cellstr(row.sort_variable),
        erp_class = cellstr(row.erp_class),
        source_row_id = Int(row.source_row_id),
        origin_trials = Int(origin.n_trials),
        origin_timepoints_post = Int(origin.n_timepoints_post),
        target_trials = PRESENTATION_TARGET_TRIALS,
        augmented_images = length(chunks),
        full_mod_split_chunks = chunks[1].full_mod_split_k,
        remainder_trials = chunks[1].remainder_trials,
    )])
end

function build_reference_augmentation_presentation(;
        tracking_key::AbstractString = DEFAULT_REFERENCE_TRACKING_KEY,
        output_dir::AbstractString = PRESENTATION_OUTPUT_DIR,
        target_trials::Int = PRESENTATION_TARGET_TRIALS)

    mkpath(output_dir)
    println("Loading labeled annotations.")
    labels = load_labeled_annotations()
    row = select_reference_row(labels; tracking_key = tracking_key)

    println("Loading reference ERP origin for ", cellstr(row.tracking_key), ".")
    source_status_df = Screening.discover_week19_data_sources()
    ctx = build_data_context(source_status_df)
    origin = origin_for_label(row, ctx)
    sort_col = Symbol(cellstr(row.sort_variable))
    sort_col in propertynames(origin.events) || error("Missing sort column $(sort_col).")

    order = sorted_order(origin.events, sort_col)
    chunks = target_trial_mod_chunks(
        order,
        target_trials;
        source_row_id = Int(row.source_row_id),
        sort_variable = cellstr(row.sort_variable),
    )

    println("Building original smoothed ERP image.")
    original_img = smoothed_sorted_image(origin.data_time_trials, origin.events, sort_col)

    println("Building ", length(chunks), " augmented smoothed ERP images.")
    augmented_images = Matrix{Float32}[]
    for c in chunks
        idxs = c.trial_indices
        push!(
            augmented_images,
            smoothed_sorted_image(
                origin.data_time_trials[:, idxs],
                origin.events[idxs, :],
                sort_col,
            ),
        )
    end

    stats = global_plot_stats(vcat([original_img], augmented_images))

    original_path = joinpath(output_dir, "reference_original_full_trials.png")
    all_aug_path = joinpath(output_dir, "reference_augmented_all_chunks.png")
    slide_path = joinpath(output_dir, "reference_augmentation_presentation_slide.png")
    chunk_plan_path = joinpath(output_dir, "chunk_plan.csv")
    selected_instance_path = joinpath(output_dir, "selected_instance.csv")
    summary_path = joinpath(output_dir, "summary.json")

    println("Writing plots to ", output_dir)
    plot_original_image(original_path, original_img, row, stats)
    plot_all_augmented_chunks(all_aug_path, augmented_images, chunks; stats = stats, row = row)
    plot_slide_figure(slide_path, original_img, augmented_images, chunks; stats = stats, row = row, origin = origin)

    CSV.write(chunk_plan_path, chunk_plan_dataframe(chunks))
    CSV.write(selected_instance_path, selected_instance_dataframe(row, origin, chunks))

    summary = (
        created_at = string(now()),
        output_dir = output_dir,
        tracking_key = cellstr(row.tracking_key),
        dataset_key = cellstr(row.dataset_key),
        dataset_label = cellstr(row.dataset_label),
        channel_name = cellstr(row.channel_name),
        channel_idx = Int(row.channel_idx_int),
        sort_variable = cellstr(row.sort_variable),
        erp_class = cellstr(row.erp_class),
        source_row_id = Int(row.source_row_id),
        origin_trials = Int(origin.n_trials),
        origin_timepoints_post = Int(origin.n_timepoints_post),
        target_trials = target_trials,
        augmented_images = length(chunks),
        full_mod_split_chunks = chunks[1].full_mod_split_k,
        remainder_trials = chunks[1].remainder_trials,
        remainder_filled = any(c -> c.chunk_role == "distributed_remainder_filled", chunks),
        lowpass_sigma = Generalization.LOWPASS_SIGMA,
        lowpass_kernel_size = Generalization.LOWPASS_KERNEL_SIZE,
        filter_border = Generalization.FILTER_BORDER,
        preprocessing = "sort -> zscore_timepoints -> Gaussian smoothing without final resize for presentation plots",
        original_plot = original_path,
        all_augmented_plot = all_aug_path,
        slide_plot = slide_path,
        chunk_plan_csv = chunk_plan_path,
        selected_instance_csv = selected_instance_path,
        summary_json = summary_path,
    )
    write_json(summary_path, summary)

    println("Selected instance: ", summary.tracking_key, " | class=", summary.erp_class)
    println("Origin trials: ", summary.origin_trials, " | augmented images: ", summary.augmented_images)
    println("Slide plot: ", slide_path)
    return summary
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Week21ReferenceAugmentationPresentation.build_reference_augmentation_presentation()
end
