module Week20ResNet18DataSourceScreening

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

include(joinpath(@__DIR__, "resnet_fixation_generalization_experiment.jl"))
using .Week20ResNetFixationGeneralization

const Generalization = Week20ResNetFixationGeneralization
const Week15 = Week20ResNetFixationGeneralization.Week15TryNewData
const CNNUtils = Week20ResNetFixationGeneralization.ERPCNNExperimentUtils

const REPO_ROOT = Generalization.REPO_ROOT
const NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_20")
const MODEL_ENV_DIR = Generalization.MODEL_ENV_DIR
const WEEK19_DATA_SOURCE_NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_19", "data_sources")
const DATASETS_ROOT = Generalization.DATASETS_ROOT
const MODEL_NAME = "resnet18_pretrained_week19_source_screening"
const TOP_N_CANDIDATES = 8

Pkg.activate(MODEL_ENV_DIR)

using CairoMakie
using CUDA
using DataFrames
using Dates
using Flux
using HDF5
using JSON3
using Random
using Statistics: mean, std

export TOP_N_CANDIDATES
export discover_week19_data_sources
export materialize_all_source_images
export train_resnet18_screening_model
export predict_source_images
export summarize_by_dataset
export summarize_by_dataset_sort
export top_class_candidates
export attach_candidate_images
export plot_dataset_candidate_grid
export plot_screening_overview
export run_resnet18_data_source_screening

function stable_slug(x::AbstractString)
    y = lowercase(x)
    y = replace(y, r"[^a-z0-9]+" => "_")
    y = replace(y, r"(^_+|_+$)" => "")
    return isempty(y) ? "item" : y
end

function notebook_source_text(path::AbstractString)
    nb = JSON3.read(read(path, String))
    io = IOBuffer()
    for cell in nb.cells
        String(cell.cell_type) == "code" || continue
        src = cell.source
        if src isa AbstractString
            print(io, src)
        else
            for line in src
                print(io, String(line))
            end
        end
        print(io, "\n")
    end
    return String(take!(io))
end

function extract_dataset_key(source::AbstractString)
    m = match(r"const\s+DATASET_KEY\s*=\s*\"([^\"]+)\"", source)
    return m === nothing ? missing : String(m.captures[1])
end

function extract_baseline_correct(source::AbstractString)
    occursin(r"baseline_correct\s*=\s*false", source) && return false
    occursin(r"baseline_correct\s*=\s*true", source) && return true
    return true
end

function bundle_files(dataset_key::AbstractString)
    dir = joinpath(DATASETS_ROOT, dataset_key)
    return (
        dir = dir,
        h5 = joinpath(dir, "epochs.hdf5"),
        events = joinpath(dir, "events.csv"),
        metadata = joinpath(dir, "metadata.json"),
    )
end

function standard_bundle_ready(dataset_key::AbstractString)
    files = bundle_files(dataset_key)
    all(isfile, [files.h5, files.events, files.metadata]) || return false
    return h5open(files.h5, "r") do f
        if haskey(f, "subjects")
            return length(keys(f["subjects"])) > 0
        end
        return true
    end
end

function discover_week19_data_sources(;
        data_source_notebook_dir::AbstractString = WEEK19_DATA_SOURCE_NOTEBOOK_DIR)

    paths = sort(filter(p -> endswith(p, ".ipynb"), readdir(data_source_notebook_dir; join = true)))
    rows = NamedTuple[]

    for path in paths
        notebook_file = basename(path)
        notebook_file == "00_index.ipynb" && continue

        source = notebook_source_text(path)
        dataset_key = extract_dataset_key(source)
        baseline_correct = extract_baseline_correct(source)

        ready = dataset_key !== missing && standard_bundle_ready(dataset_key)
        component = missing
        sort_columns = String[]
        n_channels = missing
        n_subjects = missing
        n_trials = missing
        skip_reason = ""

        if dataset_key === missing
            skip_reason = "no DATASET_KEY in notebook"
        elseif !ready
            files = bundle_files(dataset_key)
            skip_reason = "standard bundle missing or incomplete at $(files.dir)"
        else
            bundle = Week15.load_clean_dataset_bundle(dataset_key)
            component = String(bundle.metadata.component)
            sort_columns = string.(Week15.available_sort_columns(bundle))
            n_channels = Int(bundle.n_channels)
            n_subjects = length(bundle.subject_labels)
            n_trials = nrow(bundle.events)
            if isempty(sort_columns)
                ready = false
                skip_reason = "no varying sort columns found"
            end
        end

        push!(rows, (
            notebook_file = notebook_file,
            notebook_path = path,
            dataset_key = dataset_key,
            component = component,
            ready = ready,
            skip_reason = skip_reason,
            baseline_correct = baseline_correct,
            n_subjects = n_subjects,
            n_channels = n_channels,
            n_trials = n_trials,
            n_sort_columns = length(sort_columns),
            sort_columns = join(sort_columns, ", "),
        ))
    end

    df = DataFrame(rows)
    sort!(df, [:ready, :notebook_file], rev = [true, false])
    return df
end

function ready_source_rows(source_status_df::DataFrame; dataset_keys = nothing)
    df = source_status_df[source_status_df.ready .== true, :]
    if dataset_keys !== nothing
        wanted = Set(String.(dataset_keys))
        df = df[[String(k) in wanted for k in df.dataset_key], :]
    end
    return df
end

function post_stim_indices_from_times(times_s::AbstractVector{<:Real})
    return Week15.post_stim_indices(
        times_s;
        time_window_s = Week15.REAL_PREVIEW_TIME_WINDOW_S,
    )
end

function load_subject_cache(bundle)
    caches = NamedTuple[]
    for subject_label in bundle.subject_labels
        subj = Week15.load_subject_data(bundle.h5_path, subject_label)
        events_subset = Week15.select_subject_events(bundle, subject_label)
        epoch_indices = Int.(events_subset.epoch_index)
        post_idx = post_stim_indices_from_times(subj.times_s)
        post_times_s = subj.times_s[post_idx]
        push!(caches, (
            subject_label = String(subject_label),
            epochs = subj.epochs,
            times_s = subj.times_s,
            channel_names = String.(subj.channel_names),
            n_timepoints = Int(subj.n_timepoints),
            sfreq_hz = Float64(subj.sfreq_hz),
            events = events_subset,
            epoch_indices = epoch_indices,
            post_idx = post_idx,
            post_time_start_s = Float32(first(post_times_s)),
            post_time_end_s = Float32(last(post_times_s)),
        ))
    end
    return caches
end

function merged_channel_trials_from_cache(bundle, subject_caches, channel_name::AbstractString; baseline_correct::Bool)
    data_parts = Matrix{Float32}[]
    event_parts = DataFrame[]
    subject_labels = String[]
    channel_indices = Int[]
    post_len = nothing
    sfreq_hz = nothing
    time_start_s = nothing
    time_end_s = nothing

    for cache in subject_caches
        channel_idx = findfirst(==(channel_name), cache.channel_names)
        channel_idx === nothing && continue

        post_len === nothing || post_len == length(cache.post_idx) ||
            error("Cannot merge $(bundle.dataset_key): post-stimulus length differs across subjects.")
        sfreq_hz === nothing || sfreq_hz == cache.sfreq_hz ||
            error("Cannot merge $(bundle.dataset_key): sampling rate differs across subjects.")
        time_start_s === nothing || time_start_s == cache.post_time_start_s ||
            error("Cannot merge $(bundle.dataset_key): preview start time differs across subjects.")
        time_end_s === nothing || time_end_s == cache.post_time_end_s ||
            error("Cannot merge $(bundle.dataset_key): preview end time differs across subjects.")

        post_len = length(cache.post_idx)
        sfreq_hz = cache.sfreq_hz
        time_start_s = cache.post_time_start_s
        time_end_s = cache.post_time_end_s

        data_full_time_trials = reshape(
            Float32.(cache.epochs[channel_idx, :, cache.epoch_indices]),
            cache.n_timepoints,
            length(cache.epoch_indices),
        )
        if baseline_correct
            data_full_time_trials = Week15.baseline_correct_time_trials(data_full_time_trials, cache.times_s)
        end

        data_time_trials = reshape(
            Float32.(data_full_time_trials[cache.post_idx, :]),
            length(cache.post_idx),
            length(cache.epoch_indices),
        )

        push!(data_parts, data_time_trials)
        push!(event_parts, cache.events)
        push!(subject_labels, cache.subject_label)
        push!(channel_indices, Int(channel_idx))
    end

    isempty(data_parts) && error("Channel $(channel_name) not found in dataset $(bundle.dataset_key).")
    data_time_trials = hcat(data_parts...)
    events_merged = vcat(event_parts...; cols = :union)
    @assert size(data_time_trials, 2) == nrow(events_merged) "Trial count mismatch after merging subjects."

    return (
        data_time_trials = data_time_trials,
        events = events_merged,
        subject_label = length(subject_labels) == 1 ? first(subject_labels) : "merged_experiment",
        channel_idx = first(channel_indices),
        n_trials = nrow(events_merged),
        n_timepoints_post = post_len,
        time_start_s = Float32(time_start_s),
        time_end_s = Float32(time_end_s),
        sampling_rate_hz = Float64(sfreq_hz),
    )
end

function preprocess_source_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    img_trials_time = Week15.build_base_image(data_time_trials, events_trials, sort_col)
    return CNNUtils.apply_pipeline_to_image(
        img_trials_time;
        pipeline_name = :gaussian_reference,
        target_size = Generalization.TARGET_SIZE,
        low_pass_sigma = Generalization.LOWPASS_SIGMA,
        lowpass_kernel_size = Generalization.LOWPASS_KERNEL_SIZE,
        filter_border = Generalization.FILTER_BORDER,
    )
end

function sort_column_rank_map(sort_columns::Vector{Symbol})
    return Dict(col => idx for (idx, col) in enumerate(sort_columns))
end

function materialize_source_images(row::DataFrameRow; max_channels::Union{Nothing, Int} = nothing)
    dataset_key = String(row.dataset_key)
    bundle = Week15.load_clean_dataset_bundle(dataset_key)
    sort_columns = Week15.available_sort_columns(bundle)
    ranks = sort_column_rank_map(sort_columns)
    subject_caches = load_subject_cache(bundle)
    baseline_correct = Bool(row.baseline_correct)

    channel_names = String.(bundle.channel_names)
    if max_channels !== nothing
        channel_names = channel_names[1:min(max_channels, length(channel_names))]
    end

    rows = NamedTuple[]
    images = Matrix{Float32}[]

    for sort_col in sort_columns
        for channel_name in channel_names
            origin = merged_channel_trials_from_cache(
                bundle,
                subject_caches,
                channel_name;
                baseline_correct = baseline_correct,
            )
            img = preprocess_source_image(origin.data_time_trials, origin.events, sort_col)
            origin_id = join([dataset_key, origin.subject_label, channel_name, String(sort_col)], "::")
            image_id = origin_id * "::full"

            push!(rows, (
                dataset_key = dataset_key,
                dataset_label = String(bundle.metadata.component),
                source_notebook = String(row.notebook_file),
                component = String(bundle.metadata.component),
                subject_label = String(origin.subject_label),
                channel_name = String(channel_name),
                channel_idx = Int(origin.channel_idx),
                sort_col = String(sort_col),
                sort_col_rank = Int(ranks[sort_col]),
                sort_unique_values = Int(Week15.unique_nonmissing_count(origin.events[!, sort_col])),
                variant = "full",
                origin_id = origin_id,
                image_id = image_id,
                n_trials = Int(origin.n_trials),
                n_timepoints_post = Int(origin.n_timepoints_post),
                time_start_s = Float32(origin.time_start_s),
                time_end_s = Float32(origin.time_end_s),
                sampling_rate_hz = Float64(origin.sampling_rate_hz),
                baseline_correct = baseline_correct,
                source_h5 = String(bundle.h5_path),
                source_events = String(bundle.events_path),
            ))
            push!(images, img)
        end
    end

    df = DataFrame(rows)
    df.processed_img = images
    return df
end

function materialize_all_source_images(source_status_df::DataFrame = discover_week19_data_sources();
        dataset_keys = nothing,
        max_channels::Union{Nothing, Int} = nothing)

    dfs = DataFrame[]
    ready_df = ready_source_rows(source_status_df; dataset_keys = dataset_keys)

    for row in eachrow(ready_df)
        println("Materializing ", row.dataset_key, " | sort columns: ", row.n_sort_columns,
            " | channels: ", row.n_channels, " | baseline_correct=", row.baseline_correct)
        df = materialize_source_images(row; max_channels = max_channels)
        println("  images: ", nrow(df))
        push!(dfs, df)
        GC.gc(false)
    end

    isempty(dfs) && return DataFrame()
    return vcat(dfs...; cols = :union)
end

function train_resnet18_screening_model(;
        nepochs::Int = Generalization.TRAIN_EPOCHS,
        lr::Float32 = Generalization.TRAIN_LR,
        seed::Int = Generalization.TRAINING_SEED_BASE + 18)

    device, use_cuda = Generalization.setup_device()
    batchsize = use_cuda ? Generalization.TRAIN_BATCHSIZE_GPU : Generalization.TRAIN_BATCHSIZE_CPU

    println("Preparing labeled fixation training dataset.")
    train_ctx = Generalization.prepare_training_dataset()

    Random.seed!(seed)
    model, pretrained_params_loaded = Generalization.build_resnet_single_channel_pretrained(18)
    model = device(model)

    println("Training ", MODEL_NAME, " for ", nepochs, " epochs.")
    model, history_df, train_time_s = Generalization.train_full_model!(
        model,
        train_ctx.X,
        train_ctx.y;
        model_name = MODEL_NAME,
        nepochs = nepochs,
        lr = lr,
        batchsize = batchsize,
        seed = seed,
        device = device,
    )

    train_metrics = Generalization.evaluate_training_fit(
        model,
        train_ctx.X,
        train_ctx.y;
        batchsize = Generalization.PREDICT_BATCHSIZE,
        device = device,
    )

    train_metrics_df = DataFrame([(
        model_name = MODEL_NAME,
        depth = 18,
        nepochs = nepochs,
        lr = Float32(lr),
        batchsize = batchsize,
        train_time_s = train_time_s,
        pretrained_params_loaded = pretrained_params_loaded,
        accuracy = train_metrics.accuracy,
        balanced_accuracy = train_metrics.balanced_accuracy,
        macro_f1 = train_metrics.macro_f1,
        precision = train_metrics.precision,
        recall = train_metrics.recall,
        prob_class_mean = train_metrics.prob_class_mean,
        confidence_mean = train_metrics.confidence_mean,
    )])

    return (
        model = model,
        train_ctx = train_ctx,
        history_df = history_df,
        train_metrics_df = train_metrics_df,
        device = device,
        use_cuda = use_cuda,
        batchsize = batchsize,
        seed = seed,
    )
end

function predict_source_images(model, target_df::DataFrame;
        device::Function,
        batchsize::Int = Generalization.PREDICT_BATCHSIZE)

    X = CNNUtils.images_to_tensor(target_df.processed_img)
    logits, probs = Generalization.predict_logits_probs(model, X; batchsize = batchsize, device = device)
    rows = NamedTuple[]

    for i in 1:nrow(target_df)
        prob_no = Float32(probs[1, i])
        prob_class = Float32(probs[2, i])
        pred_label = prob_class >= prob_no ? 1 : 0
        confidence = max(prob_no, prob_class)
        entropy = -sum(Float64[p * log(max(p, eps(Float32))) for p in (prob_no, prob_class)])
        r = target_df[i, :]

        push!(rows, (
            model_name = MODEL_NAME,
            dataset_key = String(r.dataset_key),
            dataset_label = String(r.dataset_label),
            source_notebook = String(r.source_notebook),
            component = String(r.component),
            image_id = String(r.image_id),
            origin_id = String(r.origin_id),
            subject_label = String(r.subject_label),
            channel_name = String(r.channel_name),
            channel_idx = Int(r.channel_idx),
            sort_col = String(r.sort_col),
            sort_col_rank = Int(r.sort_col_rank),
            sort_unique_values = Int(r.sort_unique_values),
            variant = String(r.variant),
            n_trials = Int(r.n_trials),
            n_timepoints_post = Int(r.n_timepoints_post),
            time_start_s = Float32(r.time_start_s),
            time_end_s = Float32(r.time_end_s),
            sampling_rate_hz = Float64(r.sampling_rate_hz),
            baseline_correct = Bool(r.baseline_correct),
            logit_no_class = Float32(logits[1, i]),
            logit_class = Float32(logits[2, i]),
            prob_no_class = prob_no,
            prob_class = prob_class,
            predicted_label = pred_label,
            predicted_class = Generalization.class_name(pred_label),
            confidence = Float32(confidence),
            class_margin = Float32(prob_class - prob_no),
            entropy = Float32(entropy),
        ))
    end

    return DataFrame(rows)
end

function summarize_by_dataset_sort(pred_df::DataFrame)
    rows = NamedTuple[]
    isempty(pred_df) && return DataFrame(rows)

    for sdf in groupby(pred_df, [:dataset_key, :dataset_label, :sort_col, :sort_col_rank])
        probs = Float64.(sdf.prob_class)
        conf = Float64.(sdf.confidence)
        pred_class_n = count(==(1), sdf.predicted_label)
        top_idx = argmax(probs)
        push!(rows, (
            dataset_key = String(sdf.dataset_key[1]),
            dataset_label = String(sdf.dataset_label[1]),
            sort_col = String(sdf.sort_col[1]),
            sort_col_rank = Int(sdf.sort_col_rank[1]),
            n_images = nrow(sdf),
            n_channels = length(unique(sdf.channel_name)),
            predicted_class_n = pred_class_n,
            predicted_class_rate = pred_class_n / nrow(sdf),
            prob_class_mean = mean(probs),
            prob_class_std = nrow(sdf) > 1 ? std(probs) : 0.0,
            prob_class_max = maximum(probs),
            confidence_mean = mean(conf),
            top_channel = String(sdf.channel_name[top_idx]),
            top_image_id = String(sdf.image_id[top_idx]),
            top_predicted_class = String(sdf.predicted_class[top_idx]),
        ))
    end

    out = DataFrame(rows)
    sort!(out, [:prob_class_max, :predicted_class_rate], rev = [true, true])
    return out
end

function summarize_by_dataset(pred_df::DataFrame)
    rows = NamedTuple[]
    isempty(pred_df) && return DataFrame(rows)

    sort_summary = summarize_by_dataset_sort(pred_df)

    for sdf in groupby(pred_df, [:dataset_key, :dataset_label])
        probs = Float64.(sdf.prob_class)
        pred_class_n = count(==(1), sdf.predicted_label)
        top_idx = argmax(probs)
        sort_sdf = sort_summary[String.(sort_summary.dataset_key) .== String(sdf.dataset_key[1]), :]
        high_sort_n = count(sort_sdf.prob_class_max .>= 0.75)
        push!(rows, (
            dataset_key = String(sdf.dataset_key[1]),
            dataset_label = String(sdf.dataset_label[1]),
            n_images = nrow(sdf),
            n_sort_columns = length(unique(sdf.sort_col)),
            n_channels = length(unique(sdf.channel_name)),
            predicted_class_n = pred_class_n,
            predicted_class_rate = pred_class_n / nrow(sdf),
            prob_class_mean = mean(probs),
            prob_class_max = maximum(probs),
            n_sort_columns_with_prob_class_ge_075 = high_sort_n,
            top_sort_col = String(sdf.sort_col[top_idx]),
            top_channel = String(sdf.channel_name[top_idx]),
            top_image_id = String(sdf.image_id[top_idx]),
            top_predicted_class = String(sdf.predicted_class[top_idx]),
        ))
    end

    out = DataFrame(rows)
    sort!(out, [:prob_class_max, :predicted_class_rate], rev = [true, true])
    return out
end

function top_class_candidates(pred_df::DataFrame; n::Int = TOP_N_CANDIDATES)
    rows = NamedTuple[]
    isempty(pred_df) && return DataFrame(rows)

    for sdf0 in groupby(pred_df, [:dataset_key, :dataset_label, :sort_col, :sort_col_rank])
        sdf = DataFrame(sdf0)
        sort!(sdf, [:prob_class, :class_margin, :confidence], rev = [true, true, true])
        keep_n = min(n, nrow(sdf))
        for rank in 1:keep_n
            r = sdf[rank, :]
            push!(rows, (
                dataset_key = String(r.dataset_key),
                dataset_label = String(r.dataset_label),
                sort_col = String(r.sort_col),
                sort_col_rank = Int(r.sort_col_rank),
                candidate_rank = rank,
                image_id = String(r.image_id),
                origin_id = String(r.origin_id),
                channel_name = String(r.channel_name),
                channel_idx = Int(r.channel_idx),
                predicted_class = String(r.predicted_class),
                predicted_label = Int(r.predicted_label),
                prob_class = Float32(r.prob_class),
                prob_no_class = Float32(r.prob_no_class),
                confidence = Float32(r.confidence),
                class_margin = Float32(r.class_margin),
                n_trials = Int(r.n_trials),
                sort_unique_values = Int(r.sort_unique_values),
                baseline_correct = Bool(r.baseline_correct),
            ))
        end
    end

    out = DataFrame(rows)
    sort!(out, [:dataset_key, :sort_col_rank, :candidate_rank])
    return out
end

function image_lookup_dict(target_df::DataFrame)
    return Dict(String(row.image_id) => row.processed_img for row in eachrow(target_df))
end

function attach_candidate_images(candidates_df::DataFrame, target_df::DataFrame)
    lookup = image_lookup_dict(target_df)
    out = copy(candidates_df)
    out.processed_img = [lookup[String(image_id)] for image_id in out.image_id]
    return out
end

function plot_dataset_candidate_grid(candidate_images_df::DataFrame, dataset_key::AbstractString;
        n::Int = TOP_N_CANDIDATES,
        title_prefix::AbstractString = "Top ResNet18 class candidates")

    sdf = candidate_images_df[String.(candidate_images_df.dataset_key) .== String(dataset_key), :]
    isempty(sdf) && error("No candidate images found for dataset_key=$(dataset_key).")
    sort!(sdf, [:sort_col_rank, :candidate_rank])
    sort_cols = unique(String.(sdf.sort_col))
    dataset_label = String(sdf.dataset_label[1])
    n_rows = length(sort_cols)
    n_cols = n

    fig = Figure(size = (250 * n_cols + 160, 205 * n_rows + 80), figure_padding = 14)
    Label(fig[0, 1:n_cols], "$(title_prefix): $(dataset_label) ($(dataset_key))",
        fontsize = 20,
        tellwidth = false)

    for (row_idx, sort_col) in enumerate(sort_cols)
        Label(fig[row_idx, 0], sort_col, rotation = pi / 2, tellheight = false, fontsize = 14)
        sort_df = sdf[String.(sdf.sort_col) .== sort_col, :]
        sort!(sort_df, :candidate_rank)

        for col_idx in 1:n_cols
            cell = GridLayout(fig[row_idx, col_idx])
            if col_idx > nrow(sort_df)
                ax = Axis(cell[1, 1]; title = "not available")
                hidedecorations!(ax)
                hidespines!(ax)
                text!(ax, 0.5, 0.5; text = "no candidate", space = :relative, align = (:center, :center), fontsize = 11)
                continue
            end

            r = sort_df[col_idx, :]
            img = Float32.(r.processed_img)
            clipped, colorrange, _, _, cmap = CNNUtils.clipped_color_stats_quantile_zero_ticks(img)
            title = "#$(r.candidate_rank) $(r.channel_name)\nP(class)=" *
                string(round(Float64(r.prob_class); digits = 3)) *
                " pred=$(r.predicted_class)"
            ax = Axis(cell[1, 1];
                title = title,
                titlesize = 10,
                xlabel = "time",
                ylabel = "trials",
                xlabelsize = 9,
                ylabelsize = 9,
                xticklabelsize = 7,
                yticklabelsize = 7,
            )
            heatmap!(
                ax,
                1:size(clipped, 2),
                1:size(clipped, 1),
                permutedims(clipped, (2, 1));
                colormap = cmap,
                colorrange = colorrange,
            )
        end
    end

    rowgap!(fig.layout, 14)
    colgap!(fig.layout, 8)
    return fig
end

function plot_screening_overview(dataset_summary_df::DataFrame)
    sdf = copy(dataset_summary_df)
    sort!(sdf, :prob_class_max, rev = true)
    x = collect(1:nrow(sdf))
    labels = String.(sdf.dataset_key)

    fig = Figure(size = (1500, 640), figure_padding = 18)
    ax = Axis(
        fig[1, 1];
        title = "ResNet18 screening signal by data source",
        xlabel = "data source",
        ylabel = "max P(class)",
        xticks = (x, labels),
        xticklabelrotation = pi / 5,
        titlesize = 22,
        xlabelsize = 15,
        ylabelsize = 15,
        xticklabelsize = 10,
        yticklabelsize = 11,
    )
    barplot!(ax, x, Float64.(sdf.prob_class_max); color = Makie.wong_colors()[2])
    hlines!(ax, [0.5]; color = :gray45, linestyle = :dash, linewidth = 1.5)
    hlines!(ax, [0.75]; color = :gray25, linestyle = :dot, linewidth = 1.5)
    ylims!(ax, 0, 1)
    return fig
end

function run_resnet18_data_source_screening(;
        source_status_df::Union{Nothing, DataFrame} = nothing,
        dataset_keys = nothing,
        max_channels::Union{Nothing, Int} = nothing,
        nepochs::Int = Generalization.TRAIN_EPOCHS,
        lr::Float32 = Generalization.TRAIN_LR,
        seed::Int = Generalization.TRAINING_SEED_BASE + 18,
        top_n::Int = TOP_N_CANDIDATES)

    status_df = source_status_df === nothing ? discover_week19_data_sources() : source_status_df
    training = train_resnet18_screening_model(; nepochs = nepochs, lr = lr, seed = seed)

    println("Materializing Week-19 source images.")
    target_df = materialize_all_source_images(status_df; dataset_keys = dataset_keys, max_channels = max_channels)

    println("Classifying ", nrow(target_df), " source images.")
    prediction_df = predict_source_images(
        training.model,
        target_df;
        device = training.device,
        batchsize = Generalization.PREDICT_BATCHSIZE,
    )

    dataset_sort_summary_df = summarize_by_dataset_sort(prediction_df)
    dataset_summary_df = summarize_by_dataset(prediction_df)
    candidates_df = top_class_candidates(prediction_df; n = top_n)
    candidate_images_df = attach_candidate_images(candidates_df, target_df)

    return (
        source_status_df = status_df,
        training = training,
        target_df = target_df,
        prediction_df = prediction_df,
        dataset_sort_summary_df = dataset_sort_summary_df,
        dataset_summary_df = dataset_summary_df,
        candidates_df = candidates_df,
        candidate_images_df = candidate_images_df,
        run_info_df = DataFrame(
            key = [
                "created_at",
                "repo_root",
                "week19_data_source_notebook_dir",
                "model_name",
                "top_n_candidates",
                "target_size",
                "lowpass_sigma",
                "train_epochs",
                "train_lr",
                "seed",
                "images_exported",
            ],
            value = string.([
                now(),
                REPO_ROOT,
                WEEK19_DATA_SOURCE_NOTEBOOK_DIR,
                MODEL_NAME,
                top_n,
                Generalization.TARGET_SIZE,
                Generalization.LOWPASS_SIGMA,
                nepochs,
                lr,
                seed,
                false,
            ]),
        ),
    )
end

end
