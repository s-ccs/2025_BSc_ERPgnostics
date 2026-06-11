# augmentation.jl
#
# The single, shared augmentation used everywhere in the pipeline:
#   labeled training data, CV validation data, and unlabeled ERP images
#   are all preprocessed identically. This guarantees that a parent score is
#   always the mean over the same four sort/polarity variants of the same
#   Gaussian-reference pipeline.
#
# Pipeline per (parent chunk, augmentation):
#   variant sort direction + optional polarity flip on the raw trial chunk
#   -> per-timepoint z-score
#   -> transpose to (trials x time)
#   -> Gaussian-reference smoothing + resize to TARGET_SIZE

# --------------------------------------------------------------------------- #
# Sort handling
# --------------------------------------------------------------------------- #
"True when `v` is a usable sort value (a finite real, or a non-empty string)."
function is_valid_sort_value(v)
    (ismissing(v) || v === nothing) && return false
    v isa Real && return isfinite(Float64(v))
    return !isempty(strip(string(v)))
end

"Boolean mask over the trials of `events` whose `sort_col` value is usable."
function valid_sort_mask(events::DataFrame, sort_col::Symbol)
    sort_col in propertynames(events) || error("Sort column $(sort_col) missing.")
    return [is_valid_sort_value(v) for v in events[!, sort_col]]
end

"The `sort_col` column as numbers when all values are real, otherwise as strings."
function sortvalues_from(events::DataFrame, sort_col::Symbol)
    values = events[!, sort_col]
    finite_values = collect(skipmissing(values))
    all(v -> v isa Real, finite_values) && return Float64.(values)
    return string.(values)
end

"""
    sorted_order_for_variant(events, sort_col; inverse_sort=false) -> Vector{Int}

Trial indices ordered by `sort_col` (ascending, or descending when
`inverse_sort` is true). This is the ordering that defines the ERP image's
trial axis.
"""
function sorted_order_for_variant(events::DataFrame, sort_col::Symbol; inverse_sort::Bool = false)
    order = sortperm(sortvalues_from(events, sort_col))
    inverse_sort && reverse!(order)
    return order
end

# --------------------------------------------------------------------------- #
# Fixed-trial chunking
# --------------------------------------------------------------------------- #
"""
    fill_remainder_indices(order, remainder_idxs, target_trials) -> Vector{Int}

Random unique trial indices (seeded by `time_ns()`) that pad the remainder chunk
from `length(remainder_idxs)` up to `target_trials`. Returns an empty vector when
no padding is needed.
"""
function fill_remainder_indices(order::Vector{Int}, remainder_idxs::Vector{Int}, target_trials::Int)
    needed = target_trials - length(remainder_idxs)
    needed <= 0 && return Int[]

    used = Set(remainder_idxs)
    candidates = [idx for idx in order if !(idx in used)]
    length(candidates) < needed && (candidates = copy(order))

    return shuffle(MersenneTwister(time_ns()), candidates)[1:needed]
end

"""
    target_trial_mod_chunks(order, target_trials) -> Vector{NamedTuple}

Trial slicing: split the sorted trial indices into fixed-size chunks.

# Arguments
- `order::Vector{Int}`: trial indices already sorted by the chosen variable.
- `target_trials::Int`: number of trials per slice (`n` in the thesis).

# Returns
- `Vector{NamedTuple}`, one per chunk, each with `chunk_index`, `chunk_count`,
  `chunk_role`, `reused_fill_count`, and `trial_indices`. There are
  `floor(N / target_trials) + (remainder > 0)` chunks; the full chunks are
  filled round-robin so each spans the whole sort gradient, and a final remainder
  chunk is padded to `target_trials` (see [`fill_remainder_indices`](@ref)).

# Behavior
Errors when `target_trials <= 0` or fewer than `target_trials` trials are given.
"""
function target_trial_mod_chunks(order::Vector{Int}, target_trials::Int)
    n = length(order)
    target_trials <= 0 && error("target_trials must be positive.")
    n < target_trials && error("Cannot make fixed-size chunks of $(target_trials) from only $(n) trials.")

    full_chunk_count = div(n, target_trials)
    remainder_count = rem(n, target_trials)
    full_chunk_count >= 1 || error("Cannot build mod chunks for n=$(n), target_trials=$(target_trials).")

    full_bins = [Int[] for _ in 1:full_chunk_count]
    remainder = Int[]
    # Deal the sorted trials round-robin across the full bins (then the remainder
    # bin), so every full chunk spans the entire sort gradient instead of a block.
    rank = 1
    while rank <= n
        progressed = false
        for bin in full_bins
            if length(bin) < target_trials && rank <= n
                push!(bin, order[rank]); rank += 1; progressed = true
            end
        end
        if remainder_count > 0 && length(remainder) < remainder_count && rank <= n
            push!(remainder, order[rank]); rank += 1; progressed = true
        end
        progressed || break
    end

    chunks = NamedTuple[]
    for (chunk_index, idxs) in enumerate(full_bins)
        push!(chunks, (
            chunk_index = Int(chunk_index),
            chunk_count = 0,
            chunk_role = "full_mod_split",
            reused_fill_count = 0,
            trial_indices = copy(idxs),
        ))
    end
    if remainder_count > 0
        # Leftover trials become one extra chunk, padded back up to target_trials.
        filler = fill_remainder_indices(order, remainder, target_trials)
        push!(chunks, (
            chunk_index = Int(length(chunks) + 1),
            chunk_count = 0,
            chunk_role = "distributed_remainder_filled",
            reused_fill_count = Int(length(filler)),
            trial_indices = vcat(remainder, filler),
        ))
    end

    chunk_count = length(chunks)
    return [merge(c, (chunk_count = chunk_count,)) for c in chunks]
end

"""
    chunk_parent_image_id(dataset_key, channel_name, sort_variable, chunk, target_trials) -> String

Stable id for one trial slice, e.g. `dataset::channel::sort::modtarget_0200_part003`
(remainder slices also encode their fill count). Shared by labeled
materialisation and unlabeled scoring so both use the same parent ids.
"""
function chunk_parent_image_id(dataset_key, channel_name, sort_variable, chunk, target_trials::Int)
    mod_variant = chunk.reused_fill_count == 0 ?
        @sprintf("modtarget_%04d_part%03d", target_trials, chunk.chunk_index) :
        @sprintf("modtarget_%04d_remainder%03d_fill%03d", target_trials, chunk.chunk_index, chunk.reused_fill_count)
    return join([cellstr(dataset_key), cellstr(channel_name), cellstr(sort_variable), mod_variant], "::")
end

"""
    no_class_chunk_indices(chunks)

Keep only the first slice(s) of a `no_class` origin and discard the rest, so
negatives do not overwhelm the positive examples.
"""
function no_class_chunk_indices(chunks)
    isempty(chunks) && return Int[]
    keep_n = min(NO_CLASS_CHUNKS_PER_ORIGIN, length(chunks))
    return collect(1:keep_n)
end

# --------------------------------------------------------------------------- #
# Image preprocessing (the shared augmentation core)
# --------------------------------------------------------------------------- #
"""
    zscore_timepoints_local(data_time_trials) -> Matrix{Float32}

Z-score each timepoint (row) of a `(time, trials)` matrix across its trials,
guarding against zero variance on flat timepoints. Equivalent to the project's
`ERPImageUtils.zscore_timepoints`, kept local to avoid a cross-module call.
"""
function zscore_timepoints_local(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    mu = mean(x; dims = 2)
    sigma = std(x; dims = 2, corrected = true)
    # Guard against divide-by-zero on flat (constant) timepoints.
    sigma_safe = ifelse.(Float32.(sigma) .== 0f0, 1f0, Float32.(sigma))
    return Float32.((x .- Float32.(mu)) ./ sigma_safe)
end

"""
    pre_resize_augmented_image(data_time_trials, events, sort_col;
                               inverse_sort, inverse_polarity) -> Matrix{Float32}

Apply one augmentation variant to a raw `(time, trials)` chunk and return the
pre-resize `(trials, time)` image: order trials by `sort_col` (optionally
reversed), optionally flip polarity, then z-score per timepoint.
"""
function pre_resize_augmented_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol;
        inverse_sort::Bool, inverse_polarity::Bool)
    size(data_time_trials, 2) == nrow(events) || error("Trial count mismatch between signal and events.")
    order = sorted_order_for_variant(events, sort_col; inverse_sort = inverse_sort)
    data_ordered = Float32.(data_time_trials[:, order])
    inverse_polarity && (data_ordered .*= -1f0)   # polarity augmentation: flip sign
    data_z = zscore_timepoints_local(data_ordered)
    return Float32.(permutedims(data_z, (2, 1)))   # -> (trials x time) layout for the CNN
end

"""
    preprocess_model_image(data_time_trials, events, sort_col, augmentation) -> Matrix{Float32}

Produce the final model image for one augmentation of one ERP chunk: the
[`pre_resize_augmented_image`](@ref) followed by Gaussian-reference smoothing and
a resize to `TARGET_SIZE`.

# Arguments
- `data_time_trials::AbstractMatrix`: raw `(time, trials)` chunk.
- `events::DataFrame`: events for those trials (same column count).
- `sort_col::Symbol`: the sort variable.
- `augmentation`: a `NamedTuple` with `inverse_sort` and `inverse_polarity` flags
  (an entry of [`AUGMENTATION_VARIANTS`](@ref)).

# Returns
- `Matrix{Float32}` of size `TARGET_SIZE`, ready for [`images_to_tensor`](@ref).
"""
function preprocess_model_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol, augmentation)
    img_trials_time = pre_resize_augmented_image(
        data_time_trials, events, sort_col;
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

# --------------------------------------------------------------------------- #
# Origins
# --------------------------------------------------------------------------- #
"""
    origin_for_label(row, ctx) -> NamedTuple

Load the raw material for one ERP image.

# Arguments
- `row`: anything with `dataset_key` and `channel_name` fields.
- `ctx`: a [`build_data_context`](@ref) cache.

# Returns
A `NamedTuple` `(events, metadata, data_time_trials, channel_idx, n_trials,
n_timepoints)` with events and the single-channel signal trimmed to their common
trial count.
"""
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

"""
    filtered_origin_for_sort(origin, sort_col) -> NamedTuple

Return a copy of `origin` keeping only trials whose `sort_col` value is usable
(see [`is_valid_sort_value`](@ref)). Errors if no trial qualifies.
"""
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
    )
end

# --------------------------------------------------------------------------- #
# Materialize labeled training samples
# --------------------------------------------------------------------------- #
"""
    materialize_augmented_samples(labels, ctx; target_trials=TARGET_TRIALS) -> NamedTuple

Build the labeled training set: trial-slice every labeled origin and apply the
four augmentation variants.

# Arguments
- `labels::DataFrame`: the label pool (see [`load_all_real_labels`](@ref)).
- `ctx`: a [`build_data_context`](@ref) cache.
- `target_trials::Int=TARGET_TRIALS`: slice size.

# Returns
A `NamedTuple` `(sample_df, skipped_df)`. `sample_df` has one row per
augmentation image, with `sample_df.processed_img` holding the images and each
row carrying the chunk `parent_image_id` shared by its four variants. `class`
origins keep all chunks; `no_class` keeps only the first (see
[`no_class_chunk_indices`](@ref)). `skipped_df` records origins dropped for
having too few valid trials.

# Behavior
Errors if nothing could be materialised.
"""
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
            # No usable sort values -> cannot build an ERP image; record and skip.
            push!(skipped, (
                source_row_id = Int(row.source_row_id), dataset_key = cellstr(row.dataset_key),
                channel_name = cellstr(row.channel_name), sort_variable = cellstr(row.sort_variable),
                reason = sprint(showerror, err),
            ))
            continue
        end

        if origin.n_trials < target_trials
            # Too few valid trials to form even one fixed-size chunk.
            push!(skipped, (
                source_row_id = Int(row.source_row_id), dataset_key = cellstr(row.dataset_key),
                channel_name = cellstr(row.channel_name), sort_variable = cellstr(row.sort_variable),
                reason = "Only $(origin.n_trials) valid trials for target_trials=$(target_trials).",
            ))
            continue
        end

        order = sorted_order_for_variant(origin.events, sort_col)
        chunks = target_trial_mod_chunks(order, target_trials)
        # Pattern labels keep every chunk; no_class keeps only a deterministic
        # subset so negatives do not swamp the positives.
        chunk_idxs = Int(row.binary_label) == 1 ? collect(eachindex(chunks)) :
            no_class_chunk_indices(chunks)

        for chunk_idx in chunk_idxs
            chunk = chunks[chunk_idx]
            idxs = chunk.trial_indices
            events_part = origin.events[idxs, :]
            data_part = origin.data_time_trials[:, idxs]
            base_sample_id += 1

            parent_image_id = chunk_parent_image_id(row.dataset_key, row.channel_name,
                row.sort_variable, chunk, target_trials)

            # One model image per augmentation; all four share this parent_image_id.
            for (augmentation_variant_index, augmentation) in enumerate(AUGMENTATION_VARIANTS)
                img = preprocess_model_image(data_part, events_part, sort_col, augmentation)
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
                    augmentation_variant_index = Int(augmentation_variant_index),
                    augmentation_name = String(augmentation.name),
                    augmentation_label = String(augmentation.label),
                    inverse_sort = Bool(augmentation.inverse_sort),
                    inverse_polarity = Bool(augmentation.inverse_polarity),
                    n_trials = Int(length(idxs)),
                    origin_n_trials = Int(origin.n_trials),
                    origin_n_timepoints = Int(origin.n_timepoints),
                ))
                push!(images, img)
            end
        end
    end

    isempty(rows) && error("No training samples were materialized from the labels.")
    sample_df = DataFrame(rows)
    sample_df.processed_img = images
    return (sample_df = sample_df, skipped_df = DataFrame(skipped))
end

# --------------------------------------------------------------------------- #
# Trial slices for scoring (unlabeled data uses the same slicing as training)
# --------------------------------------------------------------------------- #
"""
    scoring_slices(origin, sort_col, dataset_key, channel_name, sort_variable; target_trials=TARGET_TRIALS)
        -> Vector of (parent_image_id, trial_indices)

Trial slices used to score a parent the same way training data is built: when the
origin has at least `target_trials` valid trials it is cut into fixed-size chunks
(so the model sees the trial dimension it was trained on); otherwise it falls
back to a single full-parent slice so the combination still gets a score.
"""
function scoring_slices(origin, sort_col::Symbol, dataset_key, channel_name, sort_variable;
        target_trials::Int = TARGET_TRIALS)
    if origin.n_trials >= target_trials
        order = sorted_order_for_variant(origin.events, sort_col)
        chunks = target_trial_mod_chunks(order, target_trials)
        return [(
            parent_image_id = chunk_parent_image_id(dataset_key, channel_name, sort_variable, c, target_trials),
            trial_indices = c.trial_indices,
        ) for c in chunks]
    end
    full_id = join([cellstr(dataset_key), cellstr(channel_name), cellstr(sort_variable), FULL_PARENT_TAG], "::")
    return [(parent_image_id = full_id, trial_indices = collect(1:origin.n_trials))]
end
