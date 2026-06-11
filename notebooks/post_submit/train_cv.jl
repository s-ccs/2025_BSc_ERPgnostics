# train_cv.jl
#
# 5-fold cross validation over the labelled augmented samples.
#
# Folds are assigned per pattern class and per parent chunk so that the four
# augmentations of one parent land in different folds where possible (matching
# the Week-23/24 validation policy). Each fold trains a fresh pretrained
# ResNet18 on its training split and predicts its validation split. The union of
# all validation predictions gives an out-of-fold (OOF) prob_class for every
# labelled augmentation. Parent OOF scores (aggregated later) are the mean of a
# parent's augmentation OOF scores, even though those augmentations come from
# different folds.

"""
    assign_stratified_folds!(sample_df; k=K_FOLDS)

Adds a `fold` column. Keeps augmentation variants of a parent chunk in distinct
folds and balances class/augmentation counts across folds.
"""
function assign_stratified_folds!(sample_df::DataFrame; k::Int = K_FOLDS)
    k >= length(AUGMENTATION_VARIANTS) || error("k must be >= $(length(AUGMENTATION_VARIANTS)).")
    rng = MersenneTwister(time_ns())
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

    # Process one class at a time, then one parent chunk (base_sample_id) at a
    # time, so that chunk's four augmentations get spread across distinct folds.
    for cls in sort(unique(cellstr.(sample_df.erp_class)))
        cls_indices = findall(==(cls), cellstr.(sample_df.erp_class))
        base_ids = shuffle(rng, unique(Int.(sample_df.base_sample_id[cls_indices])))
        for base_id in base_ids
            idxs = findall(i -> cellstr(sample_df.erp_class[i]) == cls && Int(sample_df.base_sample_id[i]) == base_id, 1:nrow(sample_df))
            # Prefer the folds currently holding the fewest samples of this class.
            available_folds = sort(collect(1:k);
                by = fold -> (get(class_counts[fold], cls, 0), total_counts[fold], fold),
            )[1:min(k, length(idxs))]

            used_folds = Set{Int}()
            for idx in shuffle(rng, idxs)
                aug = cellstr(sample_df.augmentation_name[idx])
                # One fold per augmentation of this chunk; fall back if exhausted.
                candidates = [fold for fold in available_folds if !(fold in used_folds)]
                isempty(candidates) && (candidates = [fold for fold in 1:k if !(fold in used_folds)])
                isempty(candidates) && (candidates = collect(1:k))

                # Greedily pick the fold lightest on (this augmentation, class, total).
                best_fold = candidates[1]
                best_key = (typemax(Int), typemax(Int), typemax(Int), typemax(Int))
                for fold in candidates
                    key = (get(augmentation_counts[fold], aug, 0), get(class_counts[fold], cls, 0), total_counts[fold], fold)
                    if key < best_key
                        best_key = key; best_fold = fold
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

"""
    oof_prediction_rows(sample_df, idxs, fold, probs)

Build out-of-fold augmentation-level prediction rows for one fold's validation
set.
"""
function oof_prediction_rows(sample_df::DataFrame, idxs::Vector{Int}, fold::Int, probs::AbstractMatrix)
    rows = NamedTuple[]
    for (j, idx) in enumerate(idxs)
        r = sample_df[idx, :]
        push!(rows, (
            fold = fold,
            parent_image_id = cellstr(r.parent_image_id),
            dataset_key = cellstr(r.dataset_key),
            dataset_label = cellstr(r.dataset_label),
            channel_name = cellstr(r.channel_name),
            channel_idx = Int(r.channel_idx),
            sort_variable = cellstr(r.sort_variable),
            erp_class = cellstr(r.erp_class),
            true_binary_label = Int(r.binary_label),
            augmentation_variant_index = Int(r.augmentation_variant_index),
            augmentation_name = cellstr(r.augmentation_name),
            inverse_sort = Bool(r.inverse_sort),
            inverse_polarity = Bool(r.inverse_polarity),
            n_trials = Int(r.n_trials),
            prob_no_class = Float32(probs[1, j]),
            prob_class = Float32(probs[2, j]),
        ))
    end
    return rows
end

"""
    run_cross_validation(sample_df; nepochs, lr) -> (oof_df, metrics_df)

Train one ResNet18 per fold and collect OOF augmentation predictions plus
per-fold metrics. `sample_df` must already carry a `fold` column.
"""
function run_cross_validation(sample_df::DataFrame;
        nepochs::Int = TRAIN_EPOCHS, lr::Float32 = TRAIN_LR)
    :fold in propertynames(sample_df) || error("sample_df has no fold column; call assign_stratified_folds! first.")

    X = images_to_tensor(sample_df)
    y = Int.(sample_df.binary_label)
    device, use_cuda, batchsize = setup_pipeline_device()

    oof_rows = NamedTuple[]
    metric_rows = NamedTuple[]

    for fold in 1:K_FOLDS
        # Train on the other folds, predict this fold -> out-of-fold scores.
        train_idx = findall(!=(fold), Int.(sample_df.fold))
        val_idx = findall(==(fold), Int.(sample_df.fold))
        (isempty(train_idx) || isempty(val_idx)) && continue

        # Fresh pretrained model per fold (no leakage between folds).
        model, pretrained_loaded = build_pretrained_resnet18()
        model = device(model)

        log_step("CV fold $(fold)/$(K_FOLDS) | train=$(length(train_idx)) | val=$(length(val_idx))")
        model, _, train_time_s = train_resnet18!(
            model, X[:, :, :, train_idx], y[train_idx];
            model_name = "$(MODEL_NAME)_fold$(fold)", nepochs = nepochs, lr = lr,
            batchsize = batchsize, device = device,
        )

        train_metrics, _, _, _, _ = binary_metrics(model, X, y, train_idx; device = device)
        val_metrics, _, val_probs, _, _ = binary_metrics(model, X, y, val_idx; device = device)
        append!(oof_rows, oof_prediction_rows(sample_df, val_idx, fold, val_probs))

        push!(metric_rows, (
            model_name = MODEL_NAME, fold = fold,
            n_train = length(train_idx), n_val = length(val_idx),
            train_accuracy = Float64(train_metrics.accuracy),
            train_balanced_accuracy = Float64(train_metrics.balanced_accuracy),
            val_accuracy = Float64(val_metrics.accuracy),
            val_balanced_accuracy = Float64(val_metrics.balanced_accuracy),
            val_macro_f1 = Float64(val_metrics.macro_f1),
            val_precision = Float64(val_metrics.precision),
            val_recall = Float64(val_metrics.recall),
            train_time_s = Float64(train_time_s),
            pretrained_params_loaded = pretrained_loaded,
            batchsize = batchsize, use_cuda = use_cuda,
        ))
        log_step("CV fold $(fold) | val_acc=$(round(Float64(val_metrics.accuracy); digits=4)) | val_bacc=$(round(Float64(val_metrics.balanced_accuracy); digits=4))")

        # Release the fold model + GPU memory before building the next one.
        model = nothing
        CUDA.functional() && CUDA.reclaim()
        GC.gc(true)
    end

    oof_df = DataFrame(oof_rows)
    metrics_df = DataFrame(metric_rows)
    return oof_df, metrics_df
end
