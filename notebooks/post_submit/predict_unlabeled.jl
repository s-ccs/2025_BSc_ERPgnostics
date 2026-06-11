# predict_unlabeled.jl
#
# Score unlabeled ERP images with the final ResNet18.
#
# The scoring universe is every (dataset, sort_variable, channel) where the
# sort_variable appears in that dataset's manual labels (any label, class OR
# no_class), across ALL channels of the dataset. A combination is "unlabeled"
# exactly when it is in this universe but has NO label row of its own; those are
# the combinations scored here with the final model. Labelled combinations
# (those that DO have a label row) are scored out-of-fold by the CV instead and
# are passed in via `skip_combos`.
#
# Each unlabeled combination is scored with the same trial-slice x augmentation
# setup as training. Aggregation to a single per-combination score happens in
# aggregate_scores.jl.

"Canonical `(dataset, sort_variable, channel)` key as a tuple of `String`s."
combo_key(dataset, sort_variable, channel) = (String(dataset), String(sort_variable), String(channel))

"""
    target_combinations(labels_df; require_pattern=false) -> Vector{NTuple{3,String}}

The full scoring universe: every `(dataset, sort_variable, channel)` where the
sort variable carries any manual label for that dataset, taken across all
channels on disk.

# Arguments
- `labels_df::DataFrame`: the label pool.
- `require_pattern::Bool=false`: when false (the default), sort variables with
  only `no_class` labels are included too, since `no_class` is still a label.

# Returns
- `Vector{NTuple{3, String}}`: the combinations as `combo_key` tuples.
"""
function target_combinations(labels_df::DataFrame; require_pattern::Bool = false)
    sort_map = dataset_sort_variable_map(labels_df; require_pattern = require_pattern)
    combos = NTuple{3, String}[]
    # Universe = (labeled/configured sort variables) x (all channels on disk).
    for dataset_key in sort(collect(keys(sort_map)))
        channels = dataset_channel_names(dataset_key)
        for sort_variable in sort_map[dataset_key], channel_name in channels
            push!(combos, combo_key(dataset_key, sort_variable, channel_name))
        end
    end
    return combos
end

"""
    score_unlabeled_combinations(model, device; labels_df, skip_combos,
                                 require_pattern=false, batchsize=PREDICT_BATCHSIZE)
        -> (aug_df, skipped_df)

Score the unlabeled combinations with the final `model`.

# Arguments
- `model`, `device::Function`: the final model and its device mover.
- `labels_df::DataFrame`: the label pool (defines the universe and the ground
  truth annotation).
- `skip_combos::Set{NTuple{3, String}}`: combinations to skip — the labeled ones,
  which are scored by the CV instead.
- `require_pattern::Bool=false`: passed to [`target_combinations`](@ref).
- `batchsize::Int`: prediction batch size.

# Returns
- `aug_df::DataFrame`: one row per trial-slice × augmentation, with `prob_class`.
  Each combination is trial-sliced like the training data (whole-parent fallback
  below `TARGET_TRIALS` trials).
- `skipped_df::DataFrame`: combinations whose ERP image could not be built.
"""
function score_unlabeled_combinations(model, device::Function;
        labels_df::DataFrame,
        skip_combos::Set{NTuple{3, String}},
        require_pattern::Bool = false,
        batchsize::Int = Generalization.PREDICT_BATCHSIZE)

    ctx = build_data_context()
    label_lookup = combined_label_lookup(labels_df)
    combos = target_combinations(labels_df; require_pattern = require_pattern)

    # Drop the labeled combinations (handled by CV) -> only unlabeled remain.
    todo = [c for c in combos if !(c in skip_combos)]
    log_step("Unlabeled scoring | $(length(combos)) target combos | $(length(combos) - length(todo)) already covered | $(length(todo)) to score")

    rows = NamedTuple[]
    images = Matrix{Float32}[]
    skipped = NamedTuple[]

    for (dataset_key, sort_variable, channel_name) in todo
        sort_col = Symbol(sort_variable)
        origin = try
            filtered_origin_for_sort(origin_for_label(
                (dataset_key = dataset_key, channel_name = channel_name, sort_variable = sort_variable), ctx),
                sort_col)
        catch err
            # ERP image cannot be built (e.g. no valid sort values) -> record and skip.
            push!(skipped, (dataset_key = dataset_key, sort_variable = sort_variable,
                channel_name = channel_name, reason = sprint(showerror, err)))
            continue
        end

        dataset_label = String(get(origin.metadata, "dataset_label", dataset_key))
        key = combo_key(dataset_key, sort_variable, channel_name)
        true_erp_class = get(label_lookup.erp_class, key, "unlabeled")
        true_binary_label = get(label_lookup.binary_label, key, 0)
        n_manual_labels = get(label_lookup.n_manual_labels, key, 0)

        # Trial-slice exactly like the training data (200-trial chunks), then
        # score every slice x augmentation; the per-combination mean is taken in
        # aggregate_scores.jl. This keeps the model on the trial dimension it was
        # trained on instead of one whole-parent image.
        slices = scoring_slices(origin, sort_col, dataset_key, channel_name, sort_variable)
        for slice in slices
            events_part = origin.events[slice.trial_indices, :]
            data_part = origin.data_time_trials[:, slice.trial_indices]
            for (augmentation_variant_index, augmentation) in enumerate(AUGMENTATION_VARIANTS)
                push!(rows, (
                    parent_image_id = slice.parent_image_id,
                    dataset_key = dataset_key,
                    dataset_label = dataset_label,
                    channel_name = channel_name,
                    channel_idx = Int(origin.channel_idx),
                    sort_variable = sort_variable,
                    true_erp_class = true_erp_class,
                    true_binary_label = Int(true_binary_label),
                    has_manual_label = n_manual_labels > 0,
                    augmentation_variant_index = Int(augmentation_variant_index),
                    augmentation_name = String(augmentation.name),
                    inverse_sort = Bool(augmentation.inverse_sort),
                    inverse_polarity = Bool(augmentation.inverse_polarity),
                    n_trials = Int(length(slice.trial_indices)),
                    n_timepoints = Int(origin.n_timepoints),
                ))
                push!(images, preprocess_model_image(data_part, events_part, sort_col, augmentation))
            end
        end
    end

    skipped_df = DataFrame(skipped)
    if isempty(rows)
        log_step("Unlabeled scoring | nothing new to score")
        return DataFrame(), skipped_df
    end

    aug_df = DataFrame(rows)
    # Single batched forward pass over all augmentation images -> P(class).
    X = images_to_tensor(images)
    _, probs = predict_probs(model, X; device = device, batchsize = batchsize)
    aug_df.prob_no_class = Float32.(probs[1, :])
    aug_df.prob_class = Float32.(probs[2, :])
    n_combos = nrow(unique(aug_df[:, [:dataset_key, :sort_variable, :channel_name]]))
    log_step("Unlabeled scoring | scored $(nrow(aug_df)) augmentations across ",
        "$(length(unique(aug_df.parent_image_id))) slices / $(n_combos) combinations")
    return aug_df, skipped_df
end
