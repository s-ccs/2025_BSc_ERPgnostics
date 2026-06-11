if !isdefined(@__MODULE__, :default_data_root)
    include(joinpath(@__DIR__, "erp_data.jl"))
end

if !isdefined(@__MODULE__, :resize_image)
    include(joinpath(@__DIR__, "erp_processing.jl"))
end

using DataFrames
using Random: MersenneTwister, shuffle

"""
    augmentation_variants()

Define the four standard ERP image augmentation variants.

# Arguments
- None.

# Returns
- `Vector`: Named tuples describing reference, inverse sort, inverse polarity,
  and inverse sort plus inverse polarity variants.
"""
function augmentation_variants()
    return [
        (name = "reference", label = "normal sort, normal polarity", inverse_sort = false, inverse_polarity = false),
        (name = "inverse_sort", label = "inverse sort, normal polarity", inverse_sort = true, inverse_polarity = false),
        (name = "inverse_polarity", label = "normal sort, inverse polarity", inverse_sort = false, inverse_polarity = true),
        (name = "inverse_sort_inverse_polarity", label = "inverse sort, inverse polarity", inverse_sort = true, inverse_polarity = true),
    ]
end

"""
    trial_chunks(trial_order, target_trials; seed=time_ns())

Split a parent trial order into fixed-size chunks.

If a remainder exists, it is filled by sampling from unused parent trial indices.
The default seed is `time_ns()`.

# Arguments
- `trial_order`: Parent trial indices, usually sorted by the selected event column.
- `target_trials`: Number of trials per output chunk.
- `seed`: Seed for deterministic remainder filling within one call.

# Returns
- `Vector`: Named tuples with chunk metadata and `trial_indices`.

# Throws
- `ArgumentError`: If `target_trials` is not positive or the parent has too few trials.
"""
function trial_chunks(trial_order, target_trials; seed = time_ns())
    order = Int.(collect(trial_order))
    target_trials = Int(target_trials)
    target_trials > 0 || throw(ArgumentError("target_trials must be positive."))
    length(order) >= target_trials || throw(ArgumentError(
        "Cannot build chunks of $(target_trials) trials from only $(length(order)) trials.",
    ))

    chunks = NamedTuple[]
    full_count = div(length(order), target_trials)
    remainder_count = rem(length(order), target_trials)

    # Store complete chunks that already match the requested trial count.
    for chunk_index in 1:full_count
        start_index = (chunk_index - 1) * target_trials + 1
        stop_index = chunk_index * target_trials
        trial_indices = order[start_index:stop_index]
        push!(chunks, (
            chunk_index = chunk_index,
            chunk_count = 0,
            chunk_role = "full",
            target_trials = target_trials,
            remainder_trials = 0,
            unique_trial_count_before_fill = length(unique(trial_indices)),
            reused_fill_count = 0,
            filler_indices = Int[],
            trial_indices = copy(trial_indices),
            seed = string(seed),
        ))
    end

    # Fill the final partial chunk with unused trials to keep a fixed image height.
    if remainder_count > 0
        remainder = order[(full_count * target_trials + 1):end]
        needed = target_trials - length(remainder)
        used = Set(remainder)
        candidates = [index for index in order if !(index in used)]
        rng = MersenneTwister(seed)
        filler = shuffle(rng, candidates)[1:needed]
        trial_indices = vcat(remainder, filler)
        push!(chunks, (
            chunk_index = length(chunks) + 1,
            chunk_count = 0,
            chunk_role = "filled_remainder",
            target_trials = target_trials,
            remainder_trials = remainder_count,
            unique_trial_count_before_fill = length(unique(remainder)),
            reused_fill_count = length(filler),
            filler_indices = Int.(filler),
            trial_indices = Int.(trial_indices),
            seed = string(seed),
        ))
    end

    chunk_count = length(chunks)
    return [merge(chunk, (chunk_count = chunk_count,)) for chunk in chunks]
end

"""
    chunk_indices(chunk)

Extract trial indices from either a chunk metadata tuple or a plain index vector.

# Arguments
- `chunk`: Named tuple with `trial_indices`, or any iterable of indices.

# Returns
- `Vector{Int}`: Trial indices for `apply_chunk`.
"""
function chunk_indices(chunk)
    if :trial_indices in propertynames(chunk)
        return Int.(collect(chunk.trial_indices))
    end
    return Int.(collect(chunk))
end

"""
    apply_chunk(data_time_trials, events, chunk)

Apply one trial chunk to both signal data and event rows.

# Arguments
- `data_time_trials`: Signal matrix in `timepoints x trials` layout.
- `events`: Event table with one row per trial.
- `chunk`: Chunk metadata or trial index vector.

# Returns
- `NamedTuple`: `(data_time_trials, events)` for the selected chunk, preserving
  the chunk's partial order.

# Throws
- `ArgumentError`: If chunk indices are outside the available trial range.
"""
function apply_chunk(data_time_trials, events, chunk)
    indices = chunk_indices(chunk)
    n_trials = size(data_time_trials, 2)
    all(index -> 1 <= index <= n_trials, indices) || throw(ArgumentError(
        "Chunk contains indices outside 1:$(n_trials).",
    ))

    # Apply the same trial slice to signal columns and event rows.
    events_df = DataFrame(events)
    return (
        data_time_trials = Matrix{Float32}(data_time_trials[:, indices]),
        events = events_df[indices, :],
    )
end

"""
    invert_polarity(data_time_trials)

Return a non-mutating polarity-inverted copy of a signal matrix.

# Arguments
- `data_time_trials`: Signal matrix in `timepoints x trials` layout.

# Returns
- `Matrix{Float32}`: Matrix equal to `-data_time_trials`.
"""
function invert_polarity(data_time_trials)
    return Matrix{Float32}(-Float32.(data_time_trials))
end

"""
    variant_flag(variant, name)

Read a Boolean flag from an augmentation variant.

# Arguments
- `variant`: Named tuple or property-accessible object.
- `name`: Flag name, for example `:inverse_sort`.

# Returns
- `Bool`: Flag value, or `false` when the flag is absent.
"""
function variant_flag(variant, name)
    return name in propertynames(variant) ? Bool(getproperty(variant, name)) : false
end

"""
    prepare_augmented_image(data_time_trials, events, sort_variable, variant; smooth=true, resize=true, target_size=(64, 64))

Create one processed ERP image from an already sliced signal/event chunk.

# Arguments
- `data_time_trials`: Chunked signal matrix in `timepoints x trials` layout.
- `events`: Chunked event table in the same trial order as the matrix.
- `sort_variable`: Event column used to sort trials inside the chunk.
- `variant`: Augmentation variant with optional `inverse_sort` and
  `inverse_polarity` flags.
- `smooth`: Apply Gaussian smoothing when `true`.
- `resize`: Resize to `target_size` when `true`.
- `target_size`: Output image size, defaulting to `(64, 64)`.

# Returns
- `Matrix{Float32}`: Processed ERP image, normally `64 x 64`.
"""
function prepare_augmented_image(
        data_time_trials,
        events,
        sort_variable,
        variant;
        smooth = true,
        resize = true,
        target_size = DEFAULT_IMAGE_TARGET_SIZE)

    reverse_sort = variant_flag(variant, :inverse_sort)
    inverse_polarity = variant_flag(variant, :inverse_polarity)

    # Apply the variant sort direction before polarity and image processing.
    order = trial_sort_order(events, sort_variable; reverse = reverse_sort)
    data_sorted = sort_trials(data_time_trials, order)
    if inverse_polarity
        data_sorted = invert_polarity(data_sorted)
    end

    # Convert the sorted signal into the ERP image representation.
    image = data_sorted |>
        zscore_timepoints |>
        trials_time_image

    # Keep smoothing before resize so the kernel can use the source dimensions.
    if smooth
        image = smooth_image_for_target(image, target_size)
    end
    if resize
        image = resize_image(image; target_size = target_size)
    end
    return Matrix{Float32}(image)
end

"""
    label_metadata(label_rows)

Convert matching label rows into image metadata fields.

# Arguments
- `label_rows`: DataFrame returned by `labels_for`.

# Returns
- `NamedTuple`: `(erp_class, binary_label)` where both values are `missing`
  when the combination is unlabeled.

# Throws
- `ArgumentError`: If more than one matching label row exists.
"""
function label_metadata(label_rows)
    if nrow(label_rows) == 0
        # Missing labels are valid and become unlabeled metadata.
        return (erp_class = missing, binary_label = missing)
    elseif nrow(label_rows) == 1
        erp_class = cellstring(label_rows.erp_class[1])
        # Convert multiclass labels to the binary class/no-class convention.
        return (
            erp_class = erp_class,
            binary_label = erp_class == "no_class" ? 0 : 1,
        )
    end
    throw(ArgumentError("Duplicate labels found for one dataset/channel/sort-variable combination."))
end

"""
    prepare_augmented_images(dataset_key, channel_name, sort_variable; target_trials=200, data_root=default_data_root())

Materialize all chunked and augmented ERP images for one dataset/channel/sort variable.

# Arguments
- `dataset_key`: Dataset folder name.
- `channel_name`: Channel signal file name without `.jld2`.
- `sort_variable`: Event column used for parent and variant sorting.
- `target_trials`: Number of trials per chunk.
- `data_root`: Root folder containing dataset folders.

# Returns
- `NamedTuple`: `(images, metadata, chunks)` where `images` is a vector of
  `Matrix{Float32}`, `metadata` is a `DataFrame`, and `chunks` contains chunk
  metadata.
"""
function prepare_augmented_images(
        dataset_key,
        channel_name,
        sort_variable;
        target_trials = 200,
        data_root = default_data_root())

    events_bundle = load_events(dataset_key; data_root = data_root)
    signal_bundle = load_signal(dataset_key, channel_name; data_root = data_root)
    size(signal_bundle.data_time_trials, 2) == nrow(events_bundle.events) || throw(ArgumentError(
        "Signal trial count does not match events row count for $(dataset_key), channel $(channel_name).",
    ))

    label_rows = labels_for(dataset_key, channel_name, sort_variable; data_root = data_root)
    label_info = label_metadata(label_rows)

    # Build parent chunks before applying variant-level sort and polarity changes.
    parent_order = trial_sort_order(events_bundle.events, sort_variable)
    chunks = trial_chunks(parent_order, target_trials)
    variants = augmentation_variants()

    images = Matrix{Float32}[]
    rows = NamedTuple[]
    # Materialize every chunk/variant pair and keep trace metadata beside it.
    for chunk in chunks
        chunked = apply_chunk(signal_bundle.data_time_trials, events_bundle.events, chunk)
        for (variant_index, variant) in enumerate(variants)
            image = prepare_augmented_image(
                chunked.data_time_trials,
                chunked.events,
                sort_variable,
                variant;
                smooth = true,
                resize = true,
                target_size = DEFAULT_IMAGE_TARGET_SIZE,
            )
            push!(images, image)
            push!(rows, (
                image_index = length(images),
                dataset_key = String(dataset_key),
                channel_name = String(channel_name),
                sort_variable = String(sort_variable),
                erp_class = label_info.erp_class,
                binary_label = label_info.binary_label,
                target_trials = Int(target_trials),
                parent_trial_count = nrow(events_bundle.events),
                chunk_index = Int(chunk.chunk_index),
                chunk_count = Int(chunk.chunk_count),
                chunk_role = String(chunk.chunk_role),
                remainder_trials = Int(chunk.remainder_trials),
                unique_trial_count_before_fill = Int(chunk.unique_trial_count_before_fill),
                reused_fill_count = Int(chunk.reused_fill_count),
                filler_indices = join(string.(chunk.filler_indices), " "),
                trial_indices = join(string.(chunk.trial_indices), " "),
                augmentation_variant_index = Int(variant_index),
                augmentation_name = String(variant.name),
                augmentation_label = String(variant.label),
                inverse_sort = Bool(variant.inverse_sort),
                inverse_polarity = Bool(variant.inverse_polarity),
            ))
        end
    end

    return (
        images = images,
        metadata = DataFrame(rows),
        chunks = chunks,
    )
end
