# This file is generated from resnet18_erpgnostics_reference_explorer.jl and contains
# shared setup, data loading, augmentation, training, model-artifact, scoring,
# and plotting helpers for the split train/export and import/explorer notebooks.

# %% [markdown]
# # ResNet18 ERPgnostics-style reference explorer
#
# This notebook trains the same single-channel ResNet18 family used in
# `notebooks/week_23/augmentation_inverse_sort_polarity.ipynb`, but it reads the
# curated real JLD2 datasets from `datasets/` and ignores `datasets/simulated/`.
#
# The final section scores whole fixation-reference ERP parent images. Each
# parent score is the mean class probability across the four sort/polarity
# augmentations of that parent image. The score is shown as an interactive
# topoplot. Clicking a channel updates a large ERP-image detail view modelled
# after `pipeline_visualisations.ipynb` and `erp_pattern_examples_square.ipynb`.

# %%
import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        pwd(),
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
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
NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_24")
MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
Pkg.activate(MODEL_ENV_DIR; io = devnull)

using CairoMakie
import WGLMakie
using CUDA
using CSV
using DataFrames
using Dates
using Flux
using ImageFiltering: imfilter
using Images: imresize
using JLD2
using JSON3
using Printf: @sprintf
using Random
using Statistics

function quiet_include(path::AbstractString)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            include(path)
        end
    end
end

if !isdefined(Main, :Week20ResNetFixationGeneralization)
    quiet_include(joinpath(REPO_ROOT, "notebooks", "week_20", "resnet_fixation_generalization_experiment.jl"))
end

using .Week20ResNetFixationGeneralization

Generalization = Week20ResNetFixationGeneralization
CNNUtils = Generalization.ERPCNNExperimentUtils
ERPImageUtils = CNNUtils.ERPImageUtils

CairoMakie.activate!(type = "svg")

println("REPO_ROOT = ", REPO_ROOT)
println("DATASETS_ROOT = ", joinpath(REPO_ROOT, "datasets"))

# %% [markdown]
# ## Configuration
#
# The defaults match the Week-23 training setup closely: 200 trials per training
# chunk, four inverse-sort/polarity variants, five folds, and the project-wide
# Gaussian reference preprocessing. Override the expensive parts with
# environment variables before running the notebook, for example:
#
# ```julia
# ENV["WEEK24_RESNET18_EPOCHS"] = "2"
# ENV["WEEK24_RUN_CV"] = "false"
# ```

# %%
DATASETS_ROOT = joinpath(REPO_ROOT, "datasets")
OUTPUT_DIR = joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_reference_explorer")
MODEL_NAME = "resnet18_real_jld2_inverse_sort_polarity_binary"

K_FOLDS = parse(Int, get(ENV, "WEEK24_RESNET18_FOLDS", "5"))
TRAIN_EPOCHS = parse(Int, get(ENV, "WEEK24_RESNET18_EPOCHS", string(Generalization.TRAIN_EPOCHS)))
TRAIN_LR = parse(Float32, get(ENV, "WEEK24_RESNET18_LR", string(Generalization.TRAIN_LR)))
GLOBAL_SEED = parse(Int, get(ENV, "WEEK24_RESNET18_SEED", "20260525"))
TARGET_TRIALS = parse(Int, get(ENV, "WEEK24_TARGET_TRIALS", "200"))
NO_CLASS_CHUNKS_PER_ORIGIN = parse(Int, get(ENV, "WEEK24_NO_CLASS_CHUNKS_PER_ORIGIN", "1"))
RUN_CV = lowercase(get(ENV, "WEEK24_RUN_CV", "true")) in ("true", "1", "yes")

TARGET_SIZE = Generalization.TARGET_SIZE
LOWPASS_SIGMA = Generalization.LOWPASS_SIGMA
LOWPASS_KERNEL_SIZE = Generalization.LOWPASS_KERNEL_SIZE
FILTER_BORDER = Generalization.FILTER_BORDER

CLASS_ID = Dict(
    "no_class" => 0,
    "sigmoid" => 1,
    "one_sided_fan" => 2,
    "two_sided_fan" => 3,
    "diverging_bar" => 4,
    "hourglass" => 5,
    "tilted_bar" => 6,
)

AUGMENTATION_VARIANTS = [
    (name = "reference", label = "normal sort, normal polarity", inverse_sort = false, inverse_polarity = false),
    (name = "inverse_sort", label = "inverse sort, normal polarity", inverse_sort = true, inverse_polarity = false),
    (name = "inverse_polarity", label = "normal sort, inverse polarity", inverse_sort = false, inverse_polarity = true),
    (name = "inverse_sort_inverse_polarity", label = "inverse sort, inverse polarity", inverse_sort = true, inverse_polarity = true),
]

REFERENCE_DATASET_KEY = "fixations_dataset"
REFERENCE_POSITIONS_PATH = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY, "positions_128.jld2")
REFERENCE_SORT_PRIORITY = ["duration", "sac_amplitude", "fix_avgpos_x", "fix_avgpos_y", "fix_avgpupilsize", "rt_ms"]
REFERENCE_INITIAL_CHANNEL = "ch096"

mkpath(OUTPUT_DIR)

# %% [markdown]
# ## Dataset loading
#
# This section reads the real built datasets from `datasets/<dataset>/events.jld2`,
# `labels.jld2`, and `signals/<channel>.jld2`. Signal files are loaded lazily and
# cached so repeated channel/sort combinations do not reload the same matrices.

# %%
cellstr(x) = (ismissing(x) || x === nothing) ? "" : string(x)

function write_json(path::AbstractString, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

function clean_output_dir!(output_dir::AbstractString)
    mkpath(output_dir)
    for child in readdir(output_dir; join = true)
        if isfile(child) && (endswith(child, ".csv") || endswith(child, ".json") || endswith(child, ".png") || endswith(child, ".svg"))
            rm(child; force = true)
        end
    end
    return output_dir
end

dataset_dir(dataset_key::AbstractString) = joinpath(DATASETS_ROOT, dataset_key)
events_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "events.jld2")
labels_path(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "labels.jld2")
signals_dir(dataset_key::AbstractString) = joinpath(dataset_dir(dataset_key), "signals")
signal_path(dataset_key::AbstractString, channel_name::AbstractString) = joinpath(signals_dir(dataset_key), string(channel_name, ".jld2"))

function discover_real_dataset_keys()
    keys = String[]
    for dataset_key in sort(readdir(DATASETS_ROOT))
        dataset_key == "simulated" && continue
        dir = dataset_dir(dataset_key)
        if isdir(dir) && isfile(events_path(dataset_key)) && isfile(labels_path(dataset_key)) && isdir(signals_dir(dataset_key))
            push!(keys, dataset_key)
        end
    end
    isempty(keys) && error("No real built datasets found in $(DATASETS_ROOT).")
    return keys
end

function channel_index_from_name(channel_name::AbstractString)
    m = match(r"^ch(\d+)$", String(channel_name))
    m === nothing && return 0
    return parse(Int, m.captures[1])
end

function load_events_file(dataset_key::AbstractString)
    path = events_path(dataset_key)
    isfile(path) || error("Missing events file: $(path)")
    return (
        events = JLD2.load(path, "events"),
        metadata = JLD2.load(path, "metadata"),
    )
end

function load_signal_file(dataset_key::AbstractString, channel_name::AbstractString)
    path = signal_path(dataset_key, channel_name)
    isfile(path) || error("Missing signal file: $(path)")
    data = Matrix{Float32}(JLD2.load(path, "data_time_trials"))
    metadata = JLD2.load(path, "metadata")
    return (
        data_time_trials = data,
        metadata = metadata,
        channel_idx = Int(get(metadata, "channel_idx", channel_index_from_name(channel_name))),
    )
end

function build_data_context()
    return (
        events_cache = Dict{String, Any}(),
        signal_cache = Dict{Tuple{String, String}, Any}(),
    )
end

const SHARED_DATA_CONTEXT = Ref{Any}(nothing)

function shared_data_context()
    if SHARED_DATA_CONTEXT[] === nothing
        SHARED_DATA_CONTEXT[] = build_data_context()
    end
    return SHARED_DATA_CONTEXT[]
end

function events_for_dataset(ctx, dataset_key::AbstractString)
    return get!(ctx.events_cache, String(dataset_key)) do
        load_events_file(dataset_key)
    end
end

function signal_for_channel(ctx, dataset_key::AbstractString, channel_name::AbstractString)
    key = (String(dataset_key), String(channel_name))
    return get!(ctx.signal_cache, key) do
        load_signal_file(dataset_key, channel_name)
    end
end

function load_dataset_labels(dataset_key::AbstractString)
    raw = JLD2.load(labels_path(dataset_key), "labels")
    events_bundle = load_events_file(dataset_key)
    dataset_label = String(get(events_bundle.metadata, "dataset_label", dataset_key))

    rows = NamedTuple[]
    for row in eachrow(raw)
        channel_name = cellstr(row.channel_name)
        sort_variable = cellstr(row.sort_variable)
        erp_class = cellstr(row.erp_class)
        haskey(CLASS_ID, erp_class) || continue
        Symbol(sort_variable) in propertynames(events_bundle.events) || continue
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

labels_without_images(sample_df::DataFrame) = select(sample_df, Not(:processed_img))

# %% [markdown]
# ## Augmentation and model-input materialization
#
# Each labelled parent origin is converted into one or more fixed-trial parent
# chunks. Positive labels keep all chunks; `no_class` labels keep a deterministic
# subset to avoid overwhelming the class examples. Every kept parent chunk gets
# the four sort/polarity augmentations used in Week 23.

# %%
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

function sortvalues_from(events::DataFrame, sort_col::Symbol)
    values = events[!, sort_col]
    finite_values = collect(skipmissing(values))
    if all(v -> v isa Real, finite_values)
        return Float64.(values)
    end
    return string.(values)
end

function sorted_order_for_variant(events::DataFrame, sort_col::Symbol; inverse_sort::Bool = false)
    values = sortvalues_from(events, sort_col)
    order = sortperm(values)
    inverse_sort && reverse!(order)
    return order
end

function stable_seed(parts...)
    seed_u64 = hash(parts, UInt(GLOBAL_SEED))
    seed = Int(seed_u64 & UInt(typemax(Int)))
    return seed > 0 ? seed : 1
end

function fill_remainder_indices(order::Vector{Int}, remainder_idxs::Vector{Int}, target_trials::Int,
        source_row_id::Int, sort_variable::AbstractString)

    needed = target_trials - length(remainder_idxs)
    needed <= 0 && return Int[]

    used = Set(remainder_idxs)
    candidates = [idx for idx in order if !(idx in used)]
    if length(candidates) < needed
        candidates = copy(order)
    end

    rng = MersenneTwister(stable_seed(source_row_id, sort_variable, target_trials, length(order)))
    return shuffle(rng, candidates)[1:needed]
end

function target_trial_mod_chunks(order::Vector{Int}, target_trials::Int;
        source_row_id::Int,
        sort_variable::AbstractString)

    n = length(order)
    target_trials <= 0 && error("target_trials must be positive.")
    n < target_trials && error("Cannot make fixed-size chunks of $(target_trials) from only $(n) trials.")

    full_chunk_count = div(n, target_trials)
    remainder_count = rem(n, target_trials)
    full_chunk_count >= 1 || error("Cannot build mod chunks for n=$(n), target_trials=$(target_trials).")

    full_bins = [Int[] for _ in 1:full_chunk_count]
    remainder = Int[]
    rank = 1

    while rank <= n
        progressed = false
        for bin in full_bins
            if length(bin) < target_trials && rank <= n
                push!(bin, order[rank])
                rank += 1
                progressed = true
            end
        end
        if remainder_count > 0 && length(remainder) < remainder_count && rank <= n
            push!(remainder, order[rank])
            rank += 1
            progressed = true
        end
        progressed || break
    end

    chunks = NamedTuple[]
    for (chunk_index, idxs) in enumerate(full_bins)
        push!(chunks, (
            chunk_index = Int(chunk_index),
            chunk_count = 0,
            chunk_role = "full_mod_split",
            full_mod_split_k = Int(full_chunk_count),
            remainder_trials = Int(remainder_count),
            unique_trial_count_before_fill = Int(length(unique(idxs))),
            reused_fill_count = 0,
            filler_indices = Int[],
            trial_indices = copy(idxs),
        ))
    end

    if remainder_count > 0
        filler = fill_remainder_indices(order, remainder, target_trials, source_row_id, sort_variable)
        idxs = vcat(remainder, filler)
        push!(chunks, (
            chunk_index = Int(length(chunks) + 1),
            chunk_count = 0,
            chunk_role = "distributed_remainder_filled",
            full_mod_split_k = Int(full_chunk_count),
            remainder_trials = Int(remainder_count),
            unique_trial_count_before_fill = Int(length(unique(remainder))),
            reused_fill_count = Int(length(filler)),
            filler_indices = filler,
            trial_indices = idxs,
        ))
    end

    chunk_count = length(chunks)
    return [merge(c, (chunk_count = chunk_count,)) for c in chunks]
end

function no_class_chunk_indices(chunks, source_row_id::Int)
    isempty(chunks) && return Int[]
    keep_n = min(NO_CLASS_CHUNKS_PER_ORIGIN, length(chunks))
    start = mod(source_row_id - 1, length(chunks)) + 1
    return [mod(start + j - 2, length(chunks)) + 1 for j in 1:keep_n]
end

function zscore_timepoints_local(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    mu = mean(x; dims = 2)
    sigma = std(x; dims = 2, corrected = true)
    sigma_safe = ifelse.(Float32.(sigma) .== 0f0, 1f0, Float32.(sigma))
    return Float32.((x .- Float32.(mu)) ./ sigma_safe)
end

function pre_resize_augmented_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol;
        inverse_sort::Bool,
        inverse_polarity::Bool)

    size(data_time_trials, 2) == nrow(events) || error("Trial count mismatch between signal and events.")
    order = sorted_order_for_variant(events, sort_col; inverse_sort = inverse_sort)
    data_ordered = Float32.(data_time_trials[:, order])
    inverse_polarity && (data_ordered .*= -1f0)
    data_z = zscore_timepoints_local(data_ordered)
    return Float32.(permutedims(data_z, (2, 1)))
end

function preprocess_model_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol, augmentation)
    img_trials_time = pre_resize_augmented_image(
        data_time_trials,
        events,
        sort_col;
        inverse_sort = Bool(augmentation.inverse_sort),
        inverse_polarity = Bool(augmentation.inverse_polarity),
    )

    return CNNUtils.apply_pipeline_to_image(
        img_trials_time;
        pipeline_name = :gaussian_reference,
        target_size = TARGET_SIZE,
        low_pass_sigma = LOWPASS_SIGMA,
        lowpass_kernel_size = LOWPASS_KERNEL_SIZE,
        filter_border = FILTER_BORDER,
    )
end

function origin_for_label(row, ctx)
    dataset_key = cellstr(row.dataset_key)
    channel_name = cellstr(row.channel_name)
    events_bundle = events_for_dataset(ctx, dataset_key)
    signal_bundle = signal_for_channel(ctx, dataset_key, channel_name)
    n = min(nrow(events_bundle.events), size(signal_bundle.data_time_trials, 2))
    return (
        events = events_bundle.events[1:n, :],
        metadata = events_bundle.metadata,
        data_time_trials = signal_bundle.data_time_trials[:, 1:n],
        channel_idx = signal_bundle.channel_idx,
        n_trials = n,
        n_timepoints = size(signal_bundle.data_time_trials, 1),
    )
end

function filtered_origin_for_sort(origin, sort_col::Symbol)
    keep = valid_sort_mask(origin.events, sort_col)
    any(keep) || error("No valid sort values for $(sort_col).")
    return (
        events = origin.events[keep, :],
        metadata = origin.metadata,
        data_time_trials = origin.data_time_trials[:, keep],
        channel_idx = origin.channel_idx,
        n_trials = count(keep),
        n_timepoints = size(origin.data_time_trials, 1),
        n_filtered_out = length(keep) - count(keep),
    )
end

function materialize_augmented_samples(labels::DataFrame, ctx; target_trials::Int = TARGET_TRIALS)
    rows = NamedTuple[]
    images = Matrix{Float32}[]
    skipped = NamedTuple[]
    base_sample_id = 0

    for row in eachrow(labels)
        sort_col = Symbol(cellstr(row.sort_variable))
        origin_raw = origin_for_label(row, ctx)
        origin = try
            filtered_origin_for_sort(origin_raw, sort_col)
        catch err
            push!(skipped, (
                source_row_id = Int(row.source_row_id),
                dataset_key = cellstr(row.dataset_key),
                channel_name = cellstr(row.channel_name),
                sort_variable = cellstr(row.sort_variable),
                reason = sprint(showerror, err),
            ))
            continue
        end

        if origin.n_trials < target_trials
            push!(skipped, (
                source_row_id = Int(row.source_row_id),
                dataset_key = cellstr(row.dataset_key),
                channel_name = cellstr(row.channel_name),
                sort_variable = cellstr(row.sort_variable),
                reason = "Only $(origin.n_trials) valid trials for target_trials=$(target_trials).",
            ))
            continue
        end

        order = sorted_order_for_variant(origin.events, sort_col)
        chunks = target_trial_mod_chunks(
            order,
            target_trials;
            source_row_id = Int(row.source_row_id),
            sort_variable = cellstr(row.sort_variable),
        )
        chunk_idxs = Int(row.binary_label) == 1 ? collect(eachindex(chunks)) : no_class_chunk_indices(chunks, Int(row.source_row_id))

        for chunk_idx in chunk_idxs
            chunk = chunks[chunk_idx]
            idxs = chunk.trial_indices
            events_part = origin.events[idxs, :]
            data_part = origin.data_time_trials[:, idxs]
            base_sample_id += 1

            mod_variant = chunk.reused_fill_count == 0 ?
                @sprintf("modtarget_%04d_part%03d", target_trials, chunk.chunk_index) :
                @sprintf("modtarget_%04d_remainder%03d_fill%03d", target_trials, chunk.chunk_index, chunk.reused_fill_count)

            for (augmentation_variant_index, augmentation) in enumerate(AUGMENTATION_VARIANTS)
                img = preprocess_model_image(data_part, events_part, sort_col, augmentation)
                variant = "$(mod_variant)__$(augmentation.name)"
                parent_image_id = join([cellstr(row.dataset_key), cellstr(row.channel_name), cellstr(row.sort_variable), mod_variant], "::")

                push!(rows, (
                    sample_id = length(rows) + 1,
                    base_sample_id = Int(base_sample_id),
                    parent_image_id = parent_image_id,
                    source_row_id = Int(row.source_row_id),
                    dataset_key = cellstr(row.dataset_key),
                    dataset_label = cellstr(row.dataset_label),
                    channel_name = cellstr(row.channel_name),
                    channel_idx = Int(origin.channel_idx),
                    sort_variable = cellstr(row.sort_variable),
                    erp_class = cellstr(row.erp_class),
                    erp_class_id = Int(row.erp_class_id),
                    binary_label = Int(row.binary_label),
                    target_trials = Int(target_trials),
                    mod_variant = mod_variant,
                    augmentation_variant_index = Int(augmentation_variant_index),
                    augmentation_name = String(augmentation.name),
                    augmentation_label = String(augmentation.label),
                    inverse_sort = Bool(augmentation.inverse_sort),
                    inverse_polarity = Bool(augmentation.inverse_polarity),
                    variant = variant,
                    chunk_index = Int(chunk.chunk_index),
                    chunk_count = Int(chunk.chunk_count),
                    chunk_role = String(chunk.chunk_role),
                    full_mod_split_k = Int(chunk.full_mod_split_k),
                    remainder_trials = Int(chunk.remainder_trials),
                    unique_trial_count_before_fill = Int(chunk.unique_trial_count_before_fill),
                    reused_fill_count = Int(chunk.reused_fill_count),
                    filler_indices = join(string.(chunk.filler_indices), " "),
                    n_trials = Int(length(idxs)),
                    origin_n_trials = Int(origin.n_trials),
                    origin_n_timepoints = Int(origin.n_timepoints),
                    trial_indices = join(string.(idxs), " "),
                ))
                push!(images, img)
            end
        end
    end

    sample_df = DataFrame(rows)
    sample_df.processed_img = images
    return (
        sample_df = sample_df,
        skipped_df = DataFrame(skipped),
    )
end

# %% [markdown]
# ## Fold assignment and training
#
# The fold assignment keeps the four augmentation variants of the same parent
# image in separate folds when possible. That preserves the Week-23 validation
# policy while still allowing a final model to be trained on all augmented data
# for the reference-dataset explorer.

# %%
function assign_stratified_folds!(sample_df::DataFrame; k::Int = K_FOLDS, seed::Int = GLOBAL_SEED)
    k >= length(AUGMENTATION_VARIANTS) || error("k must be >= $(length(AUGMENTATION_VARIANTS)).")
    rng = MersenneTwister(seed)
    folds = zeros(Int, nrow(sample_df))
    total_counts = zeros(Int, k)
    class_counts = [Dict{String, Int}() for _ in 1:k]
    augmentation_counts = [Dict{String, Int}() for _ in 1:k]

    function assign!(idx::Int, fold::Int)
        cls = cellstr(sample_df.erp_class[idx])
        aug = cellstr(sample_df.augmentation_name[idx])
        folds[idx] = fold
        total_counts[fold] += 1
        class_counts[fold][cls] = get(class_counts[fold], cls, 0) + 1
        augmentation_counts[fold][aug] = get(augmentation_counts[fold], aug, 0) + 1
    end

    for cls in sort(unique(cellstr.(sample_df.erp_class)))
        cls_indices = findall(==(cls), cellstr.(sample_df.erp_class))
        base_ids = shuffle(rng, unique(Int.(sample_df.base_sample_id[cls_indices])))

        for base_id in base_ids
            idxs = findall(i -> cellstr(sample_df.erp_class[i]) == cls && Int(sample_df.base_sample_id[i]) == base_id, 1:nrow(sample_df))
            available_folds = sort(
                collect(1:k);
                by = fold -> (
                    get(class_counts[fold], cls, 0),
                    total_counts[fold],
                    fold,
                ),
            )[1:min(k, length(idxs))]

            used_folds = Set{Int}()
            for idx in shuffle(rng, idxs)
                aug = cellstr(sample_df.augmentation_name[idx])
                candidates = [fold for fold in available_folds if !(fold in used_folds)]
                isempty(candidates) && (candidates = [fold for fold in 1:k if !(fold in used_folds)])
                isempty(candidates) && (candidates = collect(1:k))

                best_fold = candidates[1]
                best_key = (typemax(Int), typemax(Int), typemax(Int), typemax(Int))
                for fold in candidates
                    key = (
                        get(augmentation_counts[fold], aug, 0),
                        get(class_counts[fold], cls, 0),
                        total_counts[fold],
                        fold,
                    )
                    if key < best_key
                        best_key = key
                        best_fold = fold
                    end
                end
                assign!(idx, best_fold)
                push!(used_folds, best_fold)
            end
        end
    end

    sample_df.fold = folds
    all(folds .>= 1) || error("Some samples were not assigned to a fold.")
    return sample_df
end

function fold_distribution_tables(sample_df::DataFrame)
    fold_binary_df = combine(groupby(sample_df, [:fold, :binary_label]), nrow => :count)
    sort!(fold_binary_df, [:fold, :binary_label])

    fold_class_df = combine(groupby(sample_df, [:fold, :erp_class]), nrow => :count)
    sort!(fold_class_df, [:erp_class, :fold])

    fold_dataset_df = combine(groupby(sample_df, [:fold, :dataset_key, :binary_label]), nrow => :count)
    sort!(fold_dataset_df, [:fold, :dataset_key, :binary_label])

    fold_augmentation_df = combine(groupby(sample_df, [:fold, :augmentation_name]), nrow => :count)
    sort!(fold_augmentation_df, [:augmentation_name, :fold])

    return fold_binary_df, fold_class_df, fold_dataset_df, fold_augmentation_df
end

images_to_tensor(sample_df::DataFrame) = CNNUtils.images_to_tensor(sample_df.processed_img)

function metrics_for_indices(model, X::Array{Float32, 4}, y::Vector{Int}, idxs::Vector{Int};
        batchsize::Int,
        device::Function)

    logits, probs = Generalization.predict_logits_probs(model, X[:, :, :, idxs]; batchsize = batchsize, device = device)
    y_true = y[idxs]
    y_pred = [probs[2, i] >= probs[1, i] ? 1 : 0 for i in axes(probs, 2)]
    metrics = CNNUtils.compute_metrics(y_pred, y_true)
    return metrics, logits, probs, y_true, y_pred
end

function prediction_rows(sample_df::DataFrame, idxs::Vector{Int}, fold::Int,
        logits::AbstractMatrix, probs::AbstractMatrix, y_true::Vector{Int}, y_pred::Vector{Int})

    rows = NamedTuple[]
    for (j, idx) in enumerate(idxs)
        r = sample_df[idx, :]
        prob_no = Float32(probs[1, j])
        prob_class = Float32(probs[2, j])
        push!(rows, (
            model_name = MODEL_NAME,
            fold = fold,
            sample_id = Int(r.sample_id),
            base_sample_id = Int(r.base_sample_id),
            parent_image_id = cellstr(r.parent_image_id),
            source_row_id = Int(r.source_row_id),
            dataset_key = cellstr(r.dataset_key),
            dataset_label = cellstr(r.dataset_label),
            channel_name = cellstr(r.channel_name),
            channel_idx = Int(r.channel_idx),
            sort_variable = cellstr(r.sort_variable),
            erp_class = cellstr(r.erp_class),
            erp_class_id = Int(r.erp_class_id),
            true_binary_label = Int(y_true[j]),
            predicted_binary_label = Int(y_pred[j]),
            predicted_class_binary = y_pred[j] == 1 ? "class" : "no_class",
            correct = Int(y_true[j] == y_pred[j]),
            logit_no_class = Float32(logits[1, j]),
            logit_class = Float32(logits[2, j]),
            prob_no_class = prob_no,
            prob_class = prob_class,
            confidence = Float32(max(prob_no, prob_class)),
            class_margin = Float32(prob_class - prob_no),
            mod_variant = cellstr(r.mod_variant),
            augmentation_variant_index = Int(r.augmentation_variant_index),
            augmentation_name = cellstr(r.augmentation_name),
            inverse_sort = Bool(r.inverse_sort),
            inverse_polarity = Bool(r.inverse_polarity),
            variant = cellstr(r.variant),
            chunk_index = Int(r.chunk_index),
            chunk_count = Int(r.chunk_count),
            n_trials = Int(r.n_trials),
            origin_n_trials = Int(r.origin_n_trials),
            reused_fill_count = Int(r.reused_fill_count),
        ))
    end
    return rows
end

function run_resnet18_cv(sample_df::DataFrame;
        nepochs::Int = TRAIN_EPOCHS,
        lr::Float32 = TRAIN_LR,
        seed::Int = GLOBAL_SEED)

    X = images_to_tensor(sample_df)
    y = Int.(sample_df.binary_label)
    device, use_cuda = Generalization.setup_device()
    batchsize = use_cuda ? Generalization.TRAIN_BATCHSIZE_GPU : Generalization.TRAIN_BATCHSIZE_CPU

    metric_rows = NamedTuple[]
    history_parts = DataFrame[]
    pred_rows = NamedTuple[]

    for fold in 1:K_FOLDS
        train_idx = findall(!=(fold), Int.(sample_df.fold))
        val_idx = findall(==(fold), Int.(sample_df.fold))

        Random.seed!(seed + fold)
        model, pretrained_params_loaded = Generalization.build_resnet_single_channel_pretrained(18)
        model = device(model)

        println("$(MODEL_NAME) | CV fold $(fold)/$(K_FOLDS) | train=$(length(train_idx)) | val=$(length(val_idx))")
        model, history_df, train_time_s = Generalization.train_full_model!(
            model,
            X[:, :, :, train_idx],
            y[train_idx];
            model_name = "$(MODEL_NAME)_fold$(fold)",
            nepochs = nepochs,
            lr = lr,
            batchsize = batchsize,
            seed = seed + fold,
            device = device,
        )
        history_df.fold = fill(fold, nrow(history_df))
        push!(history_parts, history_df)

        train_metrics, _, _, _, _ = metrics_for_indices(
            model, X, y, train_idx;
            batchsize = Generalization.PREDICT_BATCHSIZE,
            device = device,
        )
        val_metrics, val_logits, val_probs, y_true, y_pred = metrics_for_indices(
            model, X, y, val_idx;
            batchsize = Generalization.PREDICT_BATCHSIZE,
            device = device,
        )

        append!(pred_rows, prediction_rows(sample_df, val_idx, fold, val_logits, val_probs, y_true, y_pred))

        push!(metric_rows, (
            model_name = MODEL_NAME,
            fold = fold,
            n_train = length(train_idx),
            n_val = length(val_idx),
            train_accuracy = Float64(train_metrics.accuracy),
            train_balanced_accuracy = Float64(train_metrics.balanced_accuracy),
            train_macro_f1 = Float64(train_metrics.macro_f1),
            train_precision = Float64(train_metrics.precision),
            train_recall = Float64(train_metrics.recall),
            val_accuracy = Float64(val_metrics.accuracy),
            val_balanced_accuracy = Float64(val_metrics.balanced_accuracy),
            val_macro_f1 = Float64(val_metrics.macro_f1),
            val_precision = Float64(val_metrics.precision),
            val_recall = Float64(val_metrics.recall),
            train_time_s = Float64(train_time_s),
            pretrained_params_loaded = pretrained_params_loaded,
            batchsize = batchsize,
            use_cuda = use_cuda,
        ))

        model = nothing
        CUDA.functional() && CUDA.reclaim()
        GC.gc(true)
    end

    return (
        metrics_df = DataFrame(metric_rows),
        history_df = isempty(history_parts) ? DataFrame() : vcat(history_parts...; cols = :union),
        predictions_df = DataFrame(pred_rows),
        device = device,
        use_cuda = use_cuda,
    )
end

function summarize_metrics(metrics_df::DataFrame)
    isempty(metrics_df) && return DataFrame()
    summary = combine(
        groupby(metrics_df, :model_name),
        :val_accuracy => mean => :val_accuracy_mean,
        :val_accuracy => std => :val_accuracy_std,
        :val_balanced_accuracy => mean => :val_balanced_accuracy_mean,
        :val_balanced_accuracy => std => :val_balanced_accuracy_std,
        :val_macro_f1 => mean => :val_macro_f1_mean,
        :val_macro_f1 => std => :val_macro_f1_std,
        :val_precision => mean => :val_precision_mean,
        :val_recall => mean => :val_recall_mean,
        :train_time_s => mean => :train_time_mean_s,
        :pretrained_params_loaded => maximum => :pretrained_params_loaded,
    )
    return summary
end

function train_resnet18_final(sample_df::DataFrame;
        nepochs::Int = TRAIN_EPOCHS,
        lr::Float32 = TRAIN_LR,
        seed::Int = GLOBAL_SEED + 50_000)

    X = images_to_tensor(sample_df)
    y = Int.(sample_df.binary_label)
    device, use_cuda = Generalization.setup_device()
    batchsize = use_cuda ? Generalization.TRAIN_BATCHSIZE_GPU : Generalization.TRAIN_BATCHSIZE_CPU

    Random.seed!(seed)
    model, pretrained_params_loaded = Generalization.build_resnet_single_channel_pretrained(18)
    model = device(model)

    println("$(MODEL_NAME) | final model | train=$(length(y))")
    model, history_df, train_time_s = Generalization.train_full_model!(
        model,
        X,
        y;
        model_name = "$(MODEL_NAME)_final",
        nepochs = nepochs,
        lr = lr,
        batchsize = batchsize,
        seed = seed,
        device = device,
    )

    final_metrics, _, _, _, _ = metrics_for_indices(
        model, X, y, collect(eachindex(y));
        batchsize = Generalization.PREDICT_BATCHSIZE,
        device = device,
    )

    metrics_df = DataFrame([(
        model_name = "$(MODEL_NAME)_final",
        n_train = length(y),
        train_accuracy = Float64(final_metrics.accuracy),
        train_balanced_accuracy = Float64(final_metrics.balanced_accuracy),
        train_macro_f1 = Float64(final_metrics.macro_f1),
        train_precision = Float64(final_metrics.precision),
        train_recall = Float64(final_metrics.recall),
        train_time_s = Float64(train_time_s),
        pretrained_params_loaded = pretrained_params_loaded,
        batchsize = batchsize,
        use_cuda = use_cuda,
    )])

    return (
        model = model,
        history_df = history_df,
        metrics_df = metrics_df,
        device = device,
        use_cuda = use_cuda,
        batchsize = batchsize,
    )
end

function run_training_pipeline(;
        output_dir::AbstractString = OUTPUT_DIR,
        nepochs::Int = TRAIN_EPOCHS,
        lr::Float32 = TRAIN_LR,
        k_folds::Int = K_FOLDS,
        seed::Int = GLOBAL_SEED,
        run_cv::Bool = RUN_CV)

    k_folds == K_FOLDS || error("K_FOLDS is a notebook constant; set WEEK24_RESNET18_FOLDS before running.")
    clean_output_dir!(output_dir)

    println("Loading real JLD2 labels from $(DATASETS_ROOT).")
    labels = load_all_real_labels()
    CSV.write(joinpath(output_dir, "labels_used.csv"), labels)

    label_summary_df = combine(groupby(labels, [:dataset_key, :erp_class, :binary_label]), nrow => :count)
    sort!(label_summary_df, [:dataset_key, :erp_class])
    CSV.write(joinpath(output_dir, "label_summary.csv"), label_summary_df)

    ctx = build_data_context()
    println("Materializing fixed-trial ERP images with inverse-sort/polarity variants.")
    materialized = materialize_augmented_samples(labels, ctx; target_trials = TARGET_TRIALS)
    sample_df = materialized.sample_df
    isempty(sample_df) && error("No training samples were materialized.")
    assign_stratified_folds!(sample_df; k = k_folds, seed = seed)

    CSV.write(joinpath(output_dir, "sample_plan.csv"), labels_without_images(sample_df))
    CSV.write(joinpath(output_dir, "skipped_label_rows.csv"), materialized.skipped_df)

    fold_binary_df, fold_class_df, fold_dataset_df, fold_augmentation_df = fold_distribution_tables(sample_df)
    CSV.write(joinpath(output_dir, "fold_distribution_binary.csv"), fold_binary_df)
    CSV.write(joinpath(output_dir, "fold_distribution_pattern_class.csv"), fold_class_df)
    CSV.write(joinpath(output_dir, "fold_distribution_dataset.csv"), fold_dataset_df)
    CSV.write(joinpath(output_dir, "fold_distribution_augmentation.csv"), fold_augmentation_df)

    cv = if run_cv
        println("Training and validating ResNet18 with $(k_folds)-fold CV.")
        cv_result = run_resnet18_cv(sample_df; nepochs = nepochs, lr = lr, seed = seed)
        CSV.write(joinpath(output_dir, "fold_metrics.csv"), cv_result.metrics_df)
        CSV.write(joinpath(output_dir, "train_history_cv.csv"), cv_result.history_df)
        CSV.write(joinpath(output_dir, "validation_predictions.csv"), cv_result.predictions_df)
        CSV.write(joinpath(output_dir, "metrics_summary.csv"), summarize_metrics(cv_result.metrics_df))
        cv_result
    else
        run_cv_env = get(ENV, "WEEK24_RUN_CV", "true")
        println("Skipping CV because WEEK24_RUN_CV=$(run_cv_env).")
        empty_result = (metrics_df = DataFrame(), history_df = DataFrame(), predictions_df = DataFrame(), device = cpu, use_cuda = false)
        CSV.write(joinpath(output_dir, "fold_metrics.csv"), empty_result.metrics_df)
        empty_result
    end

    println("Training final ResNet18 on all augmented samples for the explorer.")
    final = train_resnet18_final(sample_df; nepochs = nepochs, lr = lr, seed = seed + 50_000)
    CSV.write(joinpath(output_dir, "train_history_final.csv"), final.history_df)
    CSV.write(joinpath(output_dir, "final_train_metrics.csv"), final.metrics_df)

    write_json(joinpath(output_dir, "run_config.json"), Dict(
        "created_at" => string(now()),
        "datasets_root" => DATASETS_ROOT,
        "output_dir" => output_dir,
        "model_name" => MODEL_NAME,
        "architecture" => "ResNet18 single-channel, ImageNet-pretrained first convolution projected to 1 channel",
        "task" => "binary ERP pattern class vs no_class",
        "k_folds" => k_folds,
        "run_cv" => run_cv,
        "target_trials" => TARGET_TRIALS,
        "target_size" => TARGET_SIZE,
        "preprocessing" => "variant sort direction and polarity on raw fixed-trial chunk -> zscore_timepoints -> Gaussian smoothing -> resize to 64x64",
        "augmentation_variants" => [Dict(
            "name" => String(v.name),
            "label" => String(v.label),
            "inverse_sort" => Bool(v.inverse_sort),
            "inverse_polarity" => Bool(v.inverse_polarity),
        ) for v in AUGMENTATION_VARIANTS],
        "lowpass_sigma" => LOWPASS_SIGMA,
        "lowpass_kernel_size" => LOWPASS_KERNEL_SIZE,
        "filter_border" => FILTER_BORDER,
        "nepochs" => nepochs,
        "lr" => lr,
        "seed" => seed,
        "n_labeled_rows_used" => nrow(labels),
        "n_base_parent_chunks" => length(unique(Int.(sample_df.base_sample_id))),
        "n_augmented_images" => nrow(sample_df),
        "n_skipped_label_rows" => nrow(materialized.skipped_df),
    ))

    println("Output dir: ", output_dir)
    return (
        labels_df = labels,
        sample_df = sample_df,
        skipped_df = materialized.skipped_df,
        fold_binary_df = fold_binary_df,
        fold_class_df = fold_class_df,
        fold_dataset_df = fold_dataset_df,
        fold_augmentation_df = fold_augmentation_df,
        cv = cv,
        final = final,
        final_model = final.model,
        device = final.device,
        output_dir = output_dir,
    )
end



# %% [markdown]
# ## Model artifact helpers

# %%
MODEL_ARTIFACT_BASENAME = "resnet18_final_state.jld2"

function model_artifact_path(output_dir::AbstractString)
    return joinpath(output_dir, MODEL_ARTIFACT_BASENAME)
end

function cpu_trainables(model)
    return [Array(cpu(param)) for param in Flux.trainables(model)]
end

function cpu_model_state(model)
    return Flux.state(cpu(model))
end

function save_resnet18_model_artifact(path::AbstractString, model; metadata = Dict{String, Any}())
    mkpath(dirname(path))
    model_state = cpu_model_state(model)
    trainables = cpu_trainables(model)
    metadata_out = copy(Dict{String, Any}(metadata))
    metadata_out["artifact_schema_version"] = 2
    metadata_out["artifact_contents"] = "Flux.state CPU model state plus trainables checksum/debug copy"
    metadata_out["architecture"] = "ResNet18 single-channel, ImageNet-pretrained first convolution projected to 1 channel"
    metadata_out["model_name"] = MODEL_NAME
    metadata_out["target_size"] = TARGET_SIZE
    metadata_out["target_trials"] = TARGET_TRIALS
    metadata_out["augmentation_variants"] = [Dict(
        "name" => String(v.name),
        "label" => String(v.label),
        "inverse_sort" => Bool(v.inverse_sort),
        "inverse_polarity" => Bool(v.inverse_polarity),
    ) for v in AUGMENTATION_VARIANTS]
    metadata_out["created_at"] = string(now())
    JLD2.jldsave(path; model_state = model_state, trainables = trainables, metadata = metadata_out)
    return path
end

function load_trainables_into_model!(model, trainables)
    destination = Flux.trainables(model)
    length(destination) == length(trainables) || error(
        "Model artifact has $(length(trainables)) trainable arrays, but the fresh model has $(length(destination)).",
    )
    for (idx, (dst, src)) in enumerate(zip(destination, trainables))
        size(dst) == size(src) || error(
            "Trainable array $(idx) has size $(size(src)) in artifact but $(size(dst)) in model.",
        )
        copyto!(dst, src)
    end
    return model
end

function load_resnet18_model_artifact(path::AbstractString; device::Function = cpu)
    isfile(path) || error("Missing model artifact: $(path). Run resnet18_erpgnostics_train_export.ipynb first or set WEEK24_MODEL_ARTIFACT.")
    has_model_state = JLD2.jldopen(path, "r") do file
        haskey(file, "model_state")
    end
    has_model_state || error(
        "Model artifact $(path) was written by an older exporter that saved only Flux.trainables. " *
        "That is incomplete for ResNet18 because BatchNorm running state is not trainable. " *
        "Rerun notebooks/week_24/resnet18_erpgnostics_train_export.ipynb to write a schema-v2 artifact with Flux.state.",
    )

    model_state = JLD2.load(path, "model_state")
    metadata = JLD2.load(path, "metadata")
    model, pretrained_params_loaded = Generalization.build_resnet_single_channel_pretrained(18)
    Flux.loadmodel!(model, model_state)
    model = device(model)
    return (
        model = model,
        metadata = metadata,
        pretrained_params_loaded = pretrained_params_loaded,
        path = path,
    )
end

# %% [markdown]
# ## Score fixation-reference parent images

#
# The explorer does not show augmented ERP images as independent topoplot points.
# For each channel and sort variable it creates the four augmentation variants,
# predicts all four with the final ResNet18, and assigns the parent ERP image the
# mean `prob_class` across those variants.

# %%
function reference_labels(labels_df::DataFrame)
    ref = labels_df[labels_df.dataset_key .== REFERENCE_DATASET_KEY, :]
    isempty(ref) && error("No labels found for $(REFERENCE_DATASET_KEY).")
    return ref
end

function choose_reference_sort_variables(labels_df::DataFrame; n::Int = 3)
    ref = reference_labels(labels_df)
    summary = combine(
        groupby(ref, :sort_variable),
        :binary_label => sum => :n_pattern,
        nrow => :n_labels,
    )
    sort!(summary, [:n_pattern, :sort_variable], rev = [true, false])

    selected = String[]
    if "duration" in String.(summary.sort_variable)
        push!(selected, "duration")
    end

    positive_vars = [String(r.sort_variable) for r in eachrow(summary) if Int(r.n_pattern) > 0 && String(r.sort_variable) != "duration"]
    append!(selected, positive_vars)

    for candidate in REFERENCE_SORT_PRIORITY
        candidate in selected && continue
        if Symbol(candidate) in propertynames(events_for_dataset(build_data_context(), REFERENCE_DATASET_KEY).events)
            push!(selected, candidate)
        end
        length(selected) >= n && break
    end

    if count(row -> Int(row.n_pattern) > 0, eachrow(summary)) < n
        @warn "The fixation reference labels contain fewer than $(n) sort variables with manual pattern labels. The notebook fills the remaining slot(s) from finite fixation sort columns." summary
    end

    return selected[1:min(n, length(selected))], summary
end

function reference_channel_names()
    files = filter(path -> endswith(path, ".jld2"), readdir(signals_dir(REFERENCE_DATASET_KEY); join = false))
    names = replace.(files, ".jld2" => "")
    sort!(names)
    return names
end

function true_label_lookup(labels_df::DataFrame)
    ref = reference_labels(labels_df)
    lookup = Dict{Tuple{String, String}, String}()
    for row in eachrow(ref)
        lookup[(cellstr(row.sort_variable), cellstr(row.channel_name))] = cellstr(row.erp_class)
    end
    return lookup
end

function score_reference_parent_images(model, device::Function;
        labels_df::DataFrame,
        sort_variables::Vector{String},
        channels::Vector{String} = reference_channel_names(),
        batchsize::Int = Generalization.PREDICT_BATCHSIZE)

    ctx = build_data_context()
    rows = NamedTuple[]
    images = Matrix{Float32}[]
    label_lookup = true_label_lookup(labels_df)

    for sort_variable in sort_variables
        sort_col = Symbol(sort_variable)
        for channel_name in channels
            row_like = (
                dataset_key = REFERENCE_DATASET_KEY,
                channel_name = channel_name,
                sort_variable = sort_variable,
            )
            origin_raw = origin_for_label(row_like, ctx)
            origin = filtered_origin_for_sort(origin_raw, sort_col)

            for (augmentation_variant_index, augmentation) in enumerate(AUGMENTATION_VARIANTS)
                img = preprocess_model_image(origin.data_time_trials, origin.events, sort_col, augmentation)
                parent_image_id = join([REFERENCE_DATASET_KEY, channel_name, sort_variable, "full_parent"], "::")
                push!(rows, (
                    parent_image_id = parent_image_id,
                    dataset_key = REFERENCE_DATASET_KEY,
                    channel_name = channel_name,
                    channel_idx = Int(origin.channel_idx),
                    sort_variable = sort_variable,
                    true_erp_class = get(label_lookup, (sort_variable, channel_name), "unlabelled"),
                    true_binary_label = get(label_lookup, (sort_variable, channel_name), "no_class") == "no_class" ? 0 : 1,
                    augmentation_variant_index = Int(augmentation_variant_index),
                    augmentation_name = String(augmentation.name),
                    augmentation_label = String(augmentation.label),
                    inverse_sort = Bool(augmentation.inverse_sort),
                    inverse_polarity = Bool(augmentation.inverse_polarity),
                    n_trials = Int(origin.n_trials),
                    n_timepoints = Int(origin.n_timepoints),
                ))
                push!(images, img)
            end
        end
    end

    aug_df = DataFrame(rows)
    aug_df.processed_img = images
    X = CNNUtils.images_to_tensor(images)
    logits, probs = Generalization.predict_logits_probs(model, X; batchsize = batchsize, device = device)
    aug_df.logit_no_class = Float32.(logits[1, :])
    aug_df.logit_class = Float32.(logits[2, :])
    aug_df.prob_no_class = Float32.(probs[1, :])
    aug_df.prob_class = Float32.(probs[2, :])

    score_df = combine(
        groupby(aug_df, [:parent_image_id, :dataset_key, :channel_name, :channel_idx, :sort_variable, :true_erp_class, :true_binary_label]),
        :prob_class => mean => :score_class,
        :prob_class => std => :score_class_std,
        :prob_class => minimum => :score_class_min,
        :prob_class => maximum => :score_class_max,
        nrow => :n_augmentations,
    )
    sort!(score_df, [:sort_variable, :channel_idx])
    return (
        score_df = score_df,
        augmentation_df = aug_df,
    )
end


# %% [markdown]
# ## Static overview plots

#
# The topoplots are intentionally large and labelled so all 128 reference
# channels can be inspected. Manual positive labels are outlined in black; the
# fill color is the ResNet18 parent-image score.

# %%
function load_reference_positions()
    isfile(REFERENCE_POSITIONS_PATH) || error("Missing reference positions file: $(REFERENCE_POSITIONS_PATH)")
    positions = JLD2.load(REFERENCE_POSITIONS_PATH, "single_stored_object")
    rows = NamedTuple[]
    for (idx, point) in enumerate(positions)
        push!(rows, (
            channel_name = @sprintf("ch%03d", idx),
            channel_idx = idx,
            x = Float64(point[1]),
            y = Float64(point[2]),
        ))
    end
    return DataFrame(rows)
end

function draw_head_outline!(ax)
    theta = range(0, 2pi; length = 241)
    lines!(ax, 0.5 .+ 0.52 .* cos.(theta), 0.5 .+ 0.52 .* sin.(theta); color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [0.46, 0.50, 0.54], [1.01, 1.10, 1.01]; color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [-0.04, -0.10, -0.04], [0.55, 0.50, 0.45]; color = :gray35, linewidth = 2, inspectable = false)
    lines!(ax, [1.04, 1.10, 1.04], [0.55, 0.50, 0.45]; color = :gray35, linewidth = 2, inspectable = false)
    return ax
end

function closest_reference_channel_index(mouse_pos, positions::DataFrame)
    best_idx = 1
    best_dist_sq = Inf
    for i in 1:nrow(positions)
        dx = Float64(positions.x[i]) - Float64(mouse_pos[1])
        dy = Float64(positions.y[i]) - Float64(mouse_pos[2])
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq
            best_dist_sq = dist_sq
            best_idx = i
        end
    end
    return best_idx
end

function score_positions(score_df::DataFrame, sort_variable::AbstractString)
    positions = load_reference_positions()
    sub = score_df[score_df.sort_variable .== String(sort_variable), :]
    out = leftjoin(positions, sub; on = [:channel_name, :channel_idx])
    out.score_class = coalesce.(out.score_class, 0.0f0)
    out.true_binary_label = coalesce.(out.true_binary_label, 0)
    out.true_erp_class = coalesce.(out.true_erp_class, "unlabelled")
    sort!(out, :channel_idx)
    return out
end

function score_colorrange(values; mode::Symbol = :adaptive)
    vals = Float64.(collect(skipmissing(values)))
    isempty(vals) && return (0.0, 1.0)
    mode == :probability && return (0.0, 1.0)
    lo = quantile(vals, 0.02)
    hi = quantile(vals, 0.98)
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        lo = minimum(vals)
        hi = maximum(vals)
    end
    if hi <= lo
        pad = max(abs(lo) * 0.01, 1e-6)
        return (lo - pad, hi + pad)
    end
    return (lo, hi)
end

function score_range_label(values)
    vals = Float64.(collect(skipmissing(values)))
    isempty(vals) && return "score range: missing"
    return @sprintf("score range %.4g-%.4g", minimum(vals), maximum(vals))
end

function plot_reference_score_topoplots(score_df::DataFrame;
        sort_variables::Vector{String} = reference_sort_variables,
        colorrange_mode::Symbol = :adaptive)

    CairoMakie.activate!(type = "svg")
    fig = Figure(size = (700 * length(sort_variables), 820), figure_padding = 24)
    for (col, sort_variable) in enumerate(sort_variables)
        pos = score_positions(score_df, sort_variable)
        n_manual = sum(Int.(pos.true_binary_label) .== 1)
        crange = score_colorrange(pos.score_class; mode = colorrange_mode)
        colorbar_label = colorrange_mode == :probability ? "mean class probability" : "mean class probability (adaptive)"
        ax = Axis(
            fig[1, col];
            title = "$(sort_variable)\nmanual pattern labels: $(n_manual)\n$(score_range_label(pos.score_class))",
            titlesize = 28,
            aspect = DataAspect(),
        )
        hidedecorations!(ax)
        hidespines!(ax)
        xlims!(ax, -0.12, 1.12)
        ylims!(ax, -0.08, 1.14)
        draw_head_outline!(ax)

        scatter_plot = scatter!(
            ax,
            pos.x,
            pos.y;
            color = Float32.(pos.score_class),
            colormap = :viridis,
            colorrange = crange,
            markersize = 24,
            strokewidth = 0.6,
            strokecolor = :gray15,
        )

        manual_idx = findall(Int.(pos.true_binary_label) .== 1)
        if !isempty(manual_idx)
            scatter!(
                ax,
                pos.x[manual_idx],
                pos.y[manual_idx];
                color = RGBAf(0, 0, 0, 0),
                markersize = 34,
                strokewidth = 2.8,
                strokecolor = :black,
            )
        end

        text!(
            ax,
            pos.x,
            pos.y;
            text = pos.channel_name,
            align = (:center, :center),
            fontsize = 8,
            color = :white,
            inspectable = false,
        )
        Colorbar(fig[2, col], scatter_plot; label = colorbar_label, vertical = false, width = Relative(0.82))
    end
    resize_to_layout!(fig)
    return fig
end

DETAIL_CACHE = Dict{Tuple{String, String}, Any}()
INTERACTIVE_DETAIL_CACHE = Dict{Tuple{String, String, Int, Int}, Any}()

function reference_detail(sort_variable::AbstractString, channel_name::AbstractString)
    key = (String(sort_variable), String(channel_name))
    return get!(DETAIL_CACHE, key) do
        ctx = shared_data_context()
        row_like = (
            dataset_key = REFERENCE_DATASET_KEY,
            channel_name = String(channel_name),
            sort_variable = String(sort_variable),
        )
        origin_raw = origin_for_label(row_like, ctx)
        sort_col = Symbol(sort_variable)
        origin = filtered_origin_for_sort(origin_raw, sort_col)
        order = sorted_order_for_variant(origin.events, sort_col)
        data_sorted = origin.data_time_trials[:, order]
        sort_values = sortvalues_from(origin.events, sort_col)[order]

        z = zscore_timepoints_local(data_sorted)
        img = Float32.(permutedims(z, (2, 1)))
        smoothed = CNNUtils.apply_gaussian_pre_resize(
            img;
            target_size = size(img),
            low_pass_sigma = LOWPASS_SIGMA,
            lowpass_kernel_size = LOWPASS_KERNEL_SIZE,
            filter_border = FILTER_BORDER,
        )

        time_start_s = Float64(get(origin.metadata, "time_start_s", 0.0))
        time_end_s = Float64(get(origin.metadata, "time_end_s", 1.0))
        times = collect(range(time_start_s, time_end_s; length = size(data_sorted, 1)))
        trials = collect(1:size(data_sorted, 2))

        return (
            image = Matrix{Float32}(smoothed),
            raw_sorted = Matrix{Float32}(data_sorted),
            mean_wave = vec(mean(data_sorted; dims = 2)),
            sort_values = sort_values,
            times = times,
            trials = trials,
            channel_idx = origin.channel_idx,
            n_trials = size(data_sorted, 2),
            n_timepoints = size(data_sorted, 1),
        )
    end
end

function resize_detail_image(img::AbstractMatrix; max_trials::Int = 900, max_timepoints::Int = 520)
    target_size = (min(size(img, 1), max_trials), min(size(img, 2), max_timepoints))
    target_size == size(img) && return Float32.(img)
    return Float32.(imresize(Float32.(img), target_size))
end

function downsample_vector_for_display(values, target_length::Int)
    source_length = length(values)
    source_length == target_length && return collect(values)
    target_length <= 0 && return eltype(values)[]
    source_length == 0 && return eltype(values)[]
    idxs = unique(clamp.(round.(Int, range(1, source_length; length = target_length)), 1, source_length))
    return collect(values[idxs])
end

function reference_detail_interactive(sort_variable::AbstractString, channel_name::AbstractString;
        max_trials::Int = 700,
        max_timepoints::Int = 520)

    key = (String(sort_variable), String(channel_name), Int(max_trials), Int(max_timepoints))
    return get!(INTERACTIVE_DETAIL_CACHE, key) do
        ctx = shared_data_context()
        row_like = (
            dataset_key = REFERENCE_DATASET_KEY,
            channel_name = String(channel_name),
            sort_variable = String(sort_variable),
        )
        origin_raw = origin_for_label(row_like, ctx)
        sort_col = Symbol(sort_variable)
        origin = filtered_origin_for_sort(origin_raw, sort_col)
        order = sorted_order_for_variant(origin.events, sort_col)
        data_sorted = origin.data_time_trials[:, order]
        sort_values_full = sortvalues_from(origin.events, sort_col)[order]

        z = zscore_timepoints_local(data_sorted)
        img = resize_detail_image(Float32.(permutedims(z, (2, 1))); max_trials = max_trials, max_timepoints = max_timepoints)
        smoothed = CNNUtils.apply_gaussian_pre_resize(
            img;
            target_size = size(img),
            low_pass_sigma = LOWPASS_SIGMA,
            lowpass_kernel_size = LOWPASS_KERNEL_SIZE,
            filter_border = FILTER_BORDER,
        )

        time_start_s = Float64(get(origin.metadata, "time_start_s", 0.0))
        time_end_s = Float64(get(origin.metadata, "time_end_s", 1.0))
        times = collect(range(time_start_s, time_end_s; length = size(smoothed, 2)))
        trials = collect(range(1, size(data_sorted, 2); length = size(smoothed, 1)))
        sort_values = downsample_vector_for_display(sort_values_full, size(smoothed, 1))
        mean_wave_full = vec(mean(data_sorted; dims = 2))
        mean_wave = Float32.(downsample_vector_for_display(mean_wave_full, size(smoothed, 2)))

        return (
            image = smoothed,
            times = times,
            trials = trials,
            sort_values = sort_values,
            mean_wave = mean_wave,
            channel_idx = origin.channel_idx,
            n_trials = size(smoothed, 1),
            n_timepoints = size(smoothed, 2),
            full_n_trials = size(data_sorted, 2),
            full_n_timepoints = size(data_sorted, 1),
        )
    end
end

function erp_image_color_stats(img::AbstractMatrix; q_low::Float64 = 0.02, q_high::Float64 = 0.98)
    clipped, colorrange, tick_vals, tick_labels, cmap =
        CNNUtils.clipped_color_stats_quantile_zero_ticks(Float32.(img); q_low = q_low, q_high = q_high)
    return (
        image = Matrix{Float32}(clipped),
        colorrange = colorrange,
        ticks = (tick_vals, tick_labels),
        colormap = cmap,
    )
end

function score_row(score_df::DataFrame, sort_variable::AbstractString, channel_name::AbstractString)
    idx = findfirst(i -> score_df.sort_variable[i] == String(sort_variable) && score_df.channel_name[i] == String(channel_name), 1:nrow(score_df))
    idx === nothing && return nothing
    return score_df[idx, :]
end

function numeric_sort_values(values)
    try
        return Float64.(values), String("")
    catch
        levels = Dict{String, Int}()
        encoded = Float64[]
        for value in string.(values)
            if !haskey(levels, value)
                levels[value] = length(levels) + 1
            end
            push!(encoded, levels[value])
        end
        return encoded, "category code"
    end
end

function plot_reference_parent_image(score_df::DataFrame, sort_variable::AbstractString, channel_name::AbstractString;
        figure_size = (1550, 1450))

    detail = reference_detail(sort_variable, channel_name)
    row = score_row(score_df, sort_variable, channel_name)
    score = row === nothing ? NaN : Float64(row.score_class)
    true_class = row === nothing ? "unlabelled" : cellstr(row.true_erp_class)

    clipped, colorrange, tick_vals, tick_labels, cmap = CNNUtils.clipped_color_stats_quantile_zero_ticks(Float32.(detail.image))
    sort_values_numeric, sort_xlabel_suffix = numeric_sort_values(detail.sort_values)

    fig = Figure(size = figure_size, figure_padding = 28)
    title = @sprintf(
        "%s | %s | ResNet18 parent score = %.3f | manual label = %s",
        channel_name,
        sort_variable,
        score,
        true_class,
    )
    Label(fig[1, 1:3], title; fontsize = 34, tellwidth = false)

    ax_img = Axis(
        fig[2:5, 1];
        xlabel = "time after onset (s)",
        ylabel = "sorted trials",
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 23,
        yticklabelsize = 23,
    )
    hm = heatmap!(
        ax_img,
        detail.times,
        detail.trials,
        permutedims(clipped, (2, 1));
        colormap = cmap,
        colorrange = colorrange,
        rasterize = true,
    )
    Colorbar(fig[2:5, 2], hm; ticks = (tick_vals, tick_labels), label = "z-scored voltage", ticklabelsize = 20)

    ax_sort = Axis(
        fig[2:5, 3];
        xlabel = isempty(sort_xlabel_suffix) ? String(sort_variable) : "$(sort_variable) ($(sort_xlabel_suffix))",
        ylabel = "sorted trials",
        xlabelsize = 24,
        ylabelsize = 24,
        xticklabelsize = 20,
        yticklabelsize = 20,
    )
    lines!(ax_sort, sort_values_numeric, detail.trials; color = :gray20, linewidth = 2)

    ax_mean = Axis(
        fig[6, 1];
        xlabel = "time after onset (s)",
        ylabel = "mean ERP",
        xlabelsize = 24,
        ylabelsize = 24,
        xticklabelsize = 20,
        yticklabelsize = 20,
    )
    lines!(ax_mean, detail.times, detail.mean_wave; color = :black, linewidth = 2.5)
    linkxaxes!(ax_img, ax_mean)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    return fig
end

function plot_augmented_model_inputs(augmentation_score_df::DataFrame, sort_variable::AbstractString, channel_name::AbstractString;
        figure_size = (1100, 950))

    sub = augmentation_score_df[
        (augmentation_score_df.sort_variable .== String(sort_variable)) .&
        (augmentation_score_df.channel_name .== String(channel_name)),
        :,
    ]
    sort!(sub, :augmentation_variant_index)
    nrow(sub) == length(AUGMENTATION_VARIANTS) || error("Expected four augmentation rows for $(channel_name), $(sort_variable).")

    fig = Figure(size = figure_size, figure_padding = 24)
    Label(fig[1, 1:2], "Model-input augmentations | $(channel_name) | $(sort_variable)"; fontsize = 30, tellwidth = false)

    for (i, row) in enumerate(eachrow(sub))
        r = div(i - 1, 2) + 2
        c = mod(i - 1, 2) + 1
        img = Float32.(row.processed_img)
        clipped, colorrange, _, _, cmap = CNNUtils.clipped_color_stats_quantile_zero_ticks(img)
        ax = Axis(
            fig[r, c];
            title = @sprintf("%s\nprob_class = %.3f", cellstr(row.augmentation_name), Float64(row.prob_class)),
            titlesize = 22,
            aspect = DataAspect(),
        )
        hidedecorations!(ax)
        heatmap!(ax, 1:size(img, 2), 1:size(img, 1), permutedims(clipped, (2, 1)); colormap = cmap, colorrange = colorrange)
    end

    resize_to_layout!(fig)
    return fig
end


# %% [markdown]
# ## Interactive topoplot explorer

#
# The final figure uses WGLMakie. Use the dropdown to change the sort variable;
# click an electrode in the topoplot or use the channel menu to update the ERP
# image and mean waveform. The topoplot color is the parent score, i.e. the
# average `prob_class` across all four augmentations.

# %%
function interactive_reference_explorer(score_df::DataFrame, augmentation_score_df::DataFrame;
        sort_variables::Vector{String} = reference_sort_variables,
        initial_sort_variable::AbstractString = first(sort_variables),
        initial_channel::AbstractString = REFERENCE_INITIAL_CHANNEL,
        use_original_erp_images::Bool = true,
        interactive_max_trials::Int = 700,
        interactive_max_timepoints::Int = 520,
        initialize_page::Bool = true)

    initialize_page && WGLMakie.Page(offline = false, exportable = false)
    WGLMakie.activate!(; use_html_widgets = true)
    positions = load_reference_positions()
    channel_names = String.(positions.channel_name)
    initial_channel_idx = findfirst(==(String(initial_channel)), channel_names)
    initial_channel_idx === nothing && (initial_channel_idx = 1)

    selected_sort = Observable(String(initial_sort_variable))
    selected_index = Observable(Int(initial_channel_idx))
    selected_channel = lift(i -> channel_names[i], selected_index)

    function set_selected_index!(idx)
        idx isa Integer || return nothing
        1 <= Int(idx) <= nrow(positions) || return nothing
        selected_index[] = Int(idx)
        channel_menu.selection[] = channel_names[Int(idx)]
        return nothing
    end

    function scores_for_sort(sort_variable)
        pos = score_positions(score_df, sort_variable)
        return Float32.(pos.score_class)
    end

    function label_for_channel(sort_variable, channel_name)
        row = score_row(score_df, sort_variable, channel_name)
        row === nothing && return "unlabelled"
        return cellstr(row.true_erp_class)
    end

    function score_for_channel(sort_variable, channel_name)
        row = score_row(score_df, sort_variable, channel_name)
        row === nothing && return NaN
        return Float64(row.score_class)
    end

    score_values = lift(sv -> scores_for_sort(sv), selected_sort)
    score_range = lift(sv -> score_colorrange(scores_for_sort(sv); mode = :adaptive), selected_sort)
    selected_x = lift(i -> [positions.x[i]], selected_index)
    selected_y = lift(i -> [positions.y[i]], selected_index)
    selected_label = lift(ch -> [String(ch)], selected_channel)
    selected_title = lift(selected_sort, selected_channel) do sv, ch
        @sprintf("%s | %s | parent score %.3f | manual %s", ch, sv, score_for_channel(sv, ch), label_for_channel(sv, ch))
    end

    detail_obs = lift(selected_sort, selected_channel) do sv, ch
        if use_original_erp_images
            reference_detail(sv, ch)
        else
            reference_detail_interactive(
                sv,
                ch;
                max_trials = interactive_max_trials,
                max_timepoints = interactive_max_timepoints,
            )
        end
    end
    visual_obs = lift(d -> erp_image_color_stats(d.image), detail_obs)
    time_obs = lift(d -> d.times, detail_obs)
    trial_obs = lift(d -> d.trials, detail_obs)
    image_obs = lift(v -> v.image, visual_obs)
    image_colorrange_obs = lift(v -> v.colorrange, visual_obs)
    image_ticks_obs = lift(v -> v.ticks, visual_obs)
    image_colormap_obs = lift(v -> v.colormap, visual_obs)
    mean_obs = lift(d -> d.mean_wave, detail_obs)
    sort_curve_obs = lift(d -> first(numeric_sort_values(d.sort_values)), detail_obs)

    fig = Figure(size = (1760, 1020), figure_padding = (78, 30, 35, 25), backgroundcolor = :white)
    Label(fig[1, 1:4], selected_title; fontsize = 28, tellwidth = false)

    sort_menu = Menu(fig[2, 1], options = sort_variables, default = String(initial_sort_variable))
    channel_menu = Menu(fig[3, 1], options = channel_names, default = channel_names[initial_channel_idx])
    on(sort_menu.selection) do sv
        selected_sort[] = String(sv)
    end
    on(channel_menu.selection) do ch
        idx = findfirst(==(String(ch)), channel_names)
        idx === nothing || (selected_index[] = idx)
    end

    ax_topo = Axis(
        fig[4:8, 1];
        title = "Click a channel",
        titlesize = 24,
        aspect = DataAspect(),
        backgroundcolor = :white,
    )
    hidedecorations!(ax_topo)
    hidespines!(ax_topo)
    xlims!(ax_topo, -0.18, 1.18)
    ylims!(ax_topo, -0.10, 1.16)
    draw_head_outline!(ax_topo)

    topo_scatter = scatter!(
        ax_topo,
        positions.x,
        positions.y;
        color = score_values,
        colormap = :viridis,
        colorrange = score_range,
        markersize = 20,
        strokewidth = 0.6,
        strokecolor = :gray15,
        inspectable = false,
    )
    scatter!(
        ax_topo,
        positions.x,
        positions.y;
        color = RGBAf(0, 0, 0, 0),
        markersize = 34,
        strokewidth = 0,
        inspectable = false,
    )
    scatter!(
        ax_topo,
        selected_x,
        selected_y;
        color = RGBAf(0, 0, 0, 0),
        markersize = 44,
        strokewidth = 4,
        strokecolor = :black,
        inspectable = false,
    )
    text!(
        ax_topo,
        selected_x,
        selected_y;
        text = selected_label,
        align = (:center, :center),
        fontsize = 11,
        color = :white,
        inspectable = false,
    )
    Colorbar(fig[9, 1], topo_scatter; label = "mean class probability", vertical = false)

    on(events(ax_topo.scene).mousebutton, priority = 20) do event
        if event.button == Mouse.left && event.action == Mouse.press && is_mouseinside(ax_topo.scene)
            idx = closest_reference_channel_index(mouseposition(ax_topo.scene), positions)
            set_selected_index!(idx)
            return Consume(true)
        end
        return Consume(false)
    end

    ax_img = Axis(
        fig[2:8, 2];
        xlabel = "time after onset (s)",
        ylabel = "sorted trials",
        backgroundcolor = :white,
    )
    hm_img = heatmap!(
        ax_img,
        time_obs,
        trial_obs,
        lift(img -> permutedims(img, (2, 1)), image_obs);
        colormap = image_colormap_obs,
        colorrange = image_colorrange_obs,
        inspectable = false,
    )
    Colorbar(fig[2:8, 3], hm_img; label = "z-scored voltage", ticks = image_ticks_obs, width = 18)

    ax_sort = Axis(fig[2:8, 4]; xlabel = "sort value", ylabel = "sorted trials", backgroundcolor = :white)
    lines!(ax_sort, sort_curve_obs, trial_obs; color = :gray20, linewidth = 2, inspectable = false)

    ax_mean = Axis(fig[9, 2]; xlabel = "time after onset (s)", ylabel = "mean ERP", backgroundcolor = :white)
    lines!(ax_mean, time_obs, mean_obs; color = :black, linewidth = 2.5, inspectable = false)
    linkxaxes!(ax_img, ax_mean)

    on(detail_obs) do _
        autolimits!(ax_img)
        autolimits!(ax_sort)
        autolimits!(ax_mean)
    end

    colsize!(fig.layout, 1, 500)
    colsize!(fig.layout, 2, 560)
    colsize!(fig.layout, 3, 34)
    colsize!(fig.layout, 4, 500)
    rowsize!(fig.layout, 9, 125)
    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 10)
    return fig
end

const ERP_BROWSER_SERVER = Ref{Any}(nothing)

function stop_browser_reference_explorer!()
    server = ERP_BROWSER_SERVER[]
    if server !== nothing
        try
            close(server)
        catch err
            @warn "Could not close previous ERPgnostics browser server." exception = (err, catch_backtrace())
        end
        ERP_BROWSER_SERVER[] = nothing
    end
    return nothing
end

function open_url_in_browser(url::AbstractString)
    commands = Sys.islinux() ? [
        `xdg-open $(String(url))`,
        `gio open $(String(url))`,
    ] : Sys.isapple() ? [
        `open $(String(url))`,
    ] : Sys.iswindows() ? [
        `cmd /c start $(String(url))`,
    ] : Cmd[]

    for cmd in commands
        try
            run(cmd)
            return true
        catch
        end
    end
    return false
end

function start_browser_reference_explorer(score_df::DataFrame, augmentation_score_df::DataFrame;
        sort_variables::Vector{String} = reference_sort_variables,
        initial_sort_variable::AbstractString = first(sort_variables),
        initial_channel::AbstractString = REFERENCE_INITIAL_CHANNEL,
        use_original_erp_images::Bool = true,
        interactive_max_trials::Int = 700,
        interactive_max_timepoints::Int = 520,
        host::AbstractString = "127.0.0.1",
        port::Integer = 9384,
        open_browser::Bool = true)

    B = WGLMakie.Bonito
    stop_browser_reference_explorer!()
    WGLMakie.activate!(; use_html_widgets = true)

    app = B.App(title = "ERPgnostics ResNet18 Explorer") do _session
        fig = interactive_reference_explorer(
            score_df,
            augmentation_score_df;
            sort_variables = sort_variables,
            initial_sort_variable = initial_sort_variable,
            initial_channel = initial_channel,
            use_original_erp_images = use_original_erp_images,
            interactive_max_trials = interactive_max_trials,
            interactive_max_timepoints = interactive_max_timepoints,
            initialize_page = false,
        )
        return B.DOM.div(
            fig;
            style = B.Styles(
                "padding-left" => "42px",
                "padding-top" => "10px",
                "padding-bottom" => "24px",
                "background" => "white",
                "box-sizing" => "border-box",
                "overflow-x" => "auto",
                "width" => "max-content",
            ),
        )
    end

    server = B.Server(app, String(host), Int(port); verbose = -1)
    ERP_BROWSER_SERVER[] = server
    url = "http://$(host):$(server.port)/"

    if open_browser
        opened = open_url_in_browser(url)
        opened || @warn "Could not open browser automatically. Open this URL manually: $(url)"
    end

    return (
        url = url,
        server = server,
        message = "Open $(url) while this Julia kernel is running. Run stop_browser_reference_explorer!() to stop the server.",
    )
end
