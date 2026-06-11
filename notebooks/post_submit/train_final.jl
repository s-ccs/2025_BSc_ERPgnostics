# train_final.jl
#
# Train one fresh pretrained ResNet18 on ALL labelled augmented samples. This
# final model is used to score new/unlabelled ERP images (see
# predict_unlabeled.jl). It is a separate model from the CV fold models.

"""
    train_final_model(sample_df; nepochs, lr) -> NamedTuple

Train the final ResNet18 on every labelled augmented sample. Returns the model,
its device, the training history, and a one-row training-fit metrics frame.
"""
function train_final_model(sample_df::DataFrame;
        nepochs::Int = TRAIN_EPOCHS, lr::Float32 = TRAIN_LR)
    X = images_to_tensor(sample_df)
    y = Int.(sample_df.binary_label)
    device, use_cuda, batchsize = setup_pipeline_device()

    # Fresh pretrained model, distinct from the CV fold models.
    model, pretrained_loaded = build_pretrained_resnet18()
    model = device(model)

    log_step("Final model | training on all $(length(y)) augmented labelled samples")
    model, history_df, train_time_s = train_resnet18!(
        model, X, y;
        model_name = "$(MODEL_NAME)_final", nepochs = nepochs, lr = lr,
        batchsize = batchsize, device = device,
    )

    # In-sample fit only (diagnostic); honest labelled scores come from the CV.
    fit, _, _, _, _ = binary_metrics(model, X, y, collect(eachindex(y)); device = device)
    metrics_df = DataFrame([(
        model_name = "$(MODEL_NAME)_final",
        n_train = length(y),
        train_accuracy = Float64(fit.accuracy),
        train_balanced_accuracy = Float64(fit.balanced_accuracy),
        train_macro_f1 = Float64(fit.macro_f1),
        train_precision = Float64(fit.precision),
        train_recall = Float64(fit.recall),
        train_time_s = Float64(train_time_s),
        pretrained_params_loaded = pretrained_loaded,
        batchsize = batchsize, use_cuda = use_cuda,
    )])
    log_step("Final model | train_acc=$(round(Float64(fit.accuracy); digits=4)) | train_bacc=$(round(Float64(fit.balanced_accuracy); digits=4))")

    return (model = model, device = device, history_df = history_df, metrics_df = metrics_df)
end
