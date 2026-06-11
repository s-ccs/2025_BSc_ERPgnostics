# run_pipeline.jl
#
# Entry point for the post-submit ResNet18 ERP-scoring pipeline. Every run
# recomputes everything from scratch and overwrites the output CSVs.
#
#   julia notebooks/post_submit/run_pipeline.jl
#
# Steps:
#   1. load labels and materialise the 4-augmentation labelled samples
#   2. 5-fold cross validation -> out-of-fold scores for labelled combinations
#   3. train a final ResNet18 on all labelled samples
#   4. score the unlabelled combinations with the final model
#   5. write the lean per-combination and per-augmentation CSVs
#
# Env overrides: POST_SUBMIT_EPOCHS, POST_SUBMIT_LR, POST_SUBMIT_FOLDS,
# POST_SUBMIT_TARGET_TRIALS, POST_SUBMIT_BATCHSIZE_GPU, POST_SUBMIT_LABEL_SMOOTHING.

const _PS_DIR = @__DIR__
include(joinpath(_PS_DIR, "config.jl"))
include(joinpath(_PS_DIR, "data_loading.jl"))
include(joinpath(_PS_DIR, "augmentation.jl"))
include(joinpath(_PS_DIR, "model.jl"))
include(joinpath(_PS_DIR, "train_cv.jl"))
include(joinpath(_PS_DIR, "train_final.jl"))
include(joinpath(_PS_DIR, "predict_unlabeled.jl"))
include(joinpath(_PS_DIR, "aggregate_scores.jl"))

# Combinations with a manual label row -> scored by CV; everything else is unlabelled.
labeled_combo_set(labels::DataFrame) =
    Set(combo_key(String(r.dataset_key), String(r.sort_variable), String(r.channel_name)) for r in eachrow(labels))

function run_pipeline()
    print_config_banner()

    # 1. Labels + materialised labelled samples
    log_step("Step 1 | loading labels and materialising $(TARGET_TRIALS)-trial samples")
    labels = load_all_real_labels()
    sample_df = materialize_augmented_samples(labels, build_data_context(); target_trials = TARGET_TRIALS).sample_df
    assign_stratified_folds!(sample_df; k = K_FOLDS)
    log_step("Step 1 | $(nrow(labels)) labels | $(length(unique(sample_df.base_sample_id))) chunks | $(nrow(sample_df)) augmented images")

    # 2. 5-fold CV -> out-of-fold labelled scores
    log_step("Step 2 | $(K_FOLDS)-fold cross validation")
    cv_aug, cv_metrics = run_cross_validation(sample_df; nepochs = TRAIN_EPOCHS, lr = TRAIN_LR)
    CSV.write(joinpath(OUTPUT_DIR, "cv_fold_metrics.csv"), cv_metrics)
    isempty(cv_metrics) || log_step("Step 2 | mean val accuracy = $(round(mean(cv_metrics.val_accuracy); digits = 4))")

    # 3. Final model on all labelled samples
    log_step("Step 3 | training final model on all labelled samples")
    final = train_final_model(sample_df; nepochs = TRAIN_EPOCHS, lr = TRAIN_LR)
    CSV.write(joinpath(OUTPUT_DIR, "final_train_metrics.csv"), final.metrics_df)

    # 4. Score the unlabelled combinations with the final model
    new_aug, _ = score_unlabeled_combinations(final.model, final.device;
        labels_df = labels, skip_combos = labeled_combo_set(labels))

    # 5. Merge + write lean CSVs (overwrites existing)
    lean_aug, lean_parent, report = merge_all_scores(cv_aug = cv_aug, new_aug = new_aug, labels = labels)
    paths = write_lean_outputs(lean_aug, lean_parent)
    for (k, v) in pairs(report)
        log_step("Step 5 | ", k, " = ", v)
    end
    log_step("Wrote $(paths.parent)")
    log_step("Wrote $(paths.augmentation)")
    log_step("Done.")
    return (labels = labels, lean_aug = lean_aug, lean_parent = lean_parent, report = report)
end

run_pipeline()
