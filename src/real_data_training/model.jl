# model.jl
#
# Thin wrappers around the Week-20 ResNet18 engine, plus model-artifact save/load
# for resumability.
#
# The model is an ImageNet-pretrained Metalhead ResNet18 whose first convolution
# is projected to a single input channel and whose classifier head is replaced
# by a 2-way (no_class / class) Dense layer. See
# https://fluxml.ai/Metalhead.jl/stable/api/resnet/

"Stack ERP images (a vector, or `sample_df.processed_img`) into a `(H, W, 1, N)` tensor."
images_to_tensor(imgs) = CNNUtils.images_to_tensor(imgs)
images_to_tensor(sample_df::DataFrame) = CNNUtils.images_to_tensor(sample_df.processed_img)

"""
    setup_pipeline_device() -> (device, use_cuda, batchsize)

Resolve the compute device once.

# Returns
- `device::Function`: `gpu` or `cpu` mover.
- `use_cuda::Bool`: whether a CUDA device is in use.
- `batchsize::Int`: matching training batch size ([`TRAIN_BATCHSIZE_GPU`](@ref) /
  `TRAIN_BATCHSIZE_CPU`, the thesis values).
"""
function setup_pipeline_device()
    device, use_cuda = Generalization.setup_device()
    # Thesis batch sizes (64 on GPU), not the Week-20 engine defaults.
    batchsize = use_cuda ? TRAIN_BATCHSIZE_GPU : TRAIN_BATCHSIZE_CPU
    return device, use_cuda, batchsize
end

"""
    build_pretrained_resnet18() -> (model, pretrained_params_loaded::Bool)

A fresh ImageNet-pretrained single-channel ResNet18 with a 2-class head.
"""
build_pretrained_resnet18() = Generalization.build_resnet_single_channel_pretrained(18)

"""
    train_resnet18!(model, X, y; model_name, nepochs, lr, batchsize, device,
                    label_smoothing=LABEL_SMOOTHING) -> (model, history_df, train_time_s)

Train `model` in place with Adam and logit cross-entropy.

# Arguments
- `model`: a model from [`build_pretrained_resnet18`](@ref), already on `device`.
- `X::Array{Float32, 4}`: `(H, W, 1, N)` image tensor.
- `y::Vector{Int}`: binary labels (0 = no_class, 1 = class).
- `model_name::String`: label used in the per-epoch log lines.
- `nepochs`, `lr`, `batchsize`: training schedule.
- `device::Function`: `gpu`/`cpu` mover.
- `label_smoothing::Float32=LABEL_SMOOTHING`: softens the one-hot targets
  (thesis: 0.02); the Week-20 engine's loss does not do this.

# Returns
`(model, history_df, train_time_s)` where `history_df` has one row per epoch with
`avg_loss`. The RNG is seeded from `time_ns()` each call (not reproducible).
"""
function train_resnet18!(model, X::Array{Float32, 4}, y::Vector{Int};
        model_name::String, nepochs::Int, lr::Float32, batchsize::Int, device::Function,
        label_smoothing::Float32 = LABEL_SMOOTHING)
    Random.seed!(time_ns())
    # Soften the hard 0/1 targets once up front (targets are fixed across epochs).
    y_oh = Float32.(Flux.onehotbatch(y, 0:1))
    y_target = label_smoothing > 0f0 ? Float32.(Flux.label_smoothing(y_oh, label_smoothing)) : y_oh
    loader = Flux.DataLoader((X, y_target); batchsize = batchsize, shuffle = true)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    history = NamedTuple[]
    Flux.trainmode!(model)
    total_time_s = @elapsed begin
        for epoch in 1:nepochs
            running_loss = 0f0
            n_batches = 0
            for (xb_cpu, yb_cpu) in loader
                xb = device(xb_cpu)
                yb = device(yb_cpu)
                loss_val, grads = Flux.withgradient(m -> Flux.Losses.logitcrossentropy(m(xb), yb), model)
                opt_state, model = Flux.update!(opt_state, model, grads[1])
                running_loss += Float32(loss_val)
                n_batches += 1
            end
            avg_loss = Float64(running_loss / max(1, n_batches))
            push!(history, (model_name = model_name, epoch = epoch, avg_loss = avg_loss, n_batches = n_batches))
            log_step("$(model_name) | epoch $(epoch)/$(nepochs) | loss=$(round(avg_loss; digits = 5))")
        end
    end
    return model, DataFrame(history), total_time_s
end

"""
    predict_probs(model, X; device, batchsize=PREDICT_BATCHSIZE) -> (logits, probs)

Run `model` over the image tensor `X` in batches.

# Returns
- `logits::Matrix`, `probs::Matrix`, both `2 × N`. Row 1 is `no_class`, row 2 is
  `class`, so `probs[2, :]` is P(class).
"""
function predict_probs(model, X::Array{Float32, 4}; device::Function,
        batchsize::Int = Generalization.PREDICT_BATCHSIZE)
    return Generalization.predict_logits_probs(model, X; batchsize = batchsize, device = device)
end

"""
    binary_metrics(model, X, y, idxs; device, batchsize=PREDICT_BATCHSIZE)
        -> (metrics, logits, probs, y_true, y_pred)

Score the columns `idxs` of `X` and compare to `y`. Predicts `class` when
P(class) ≥ P(no_class). Returns the classification `metrics` (accuracy, balanced
accuracy, macro-F1, precision, recall) plus the raw `logits`/`probs` and the
true/predicted label vectors for that subset.
"""
function binary_metrics(model, X::Array{Float32, 4}, y::Vector{Int}, idxs::Vector{Int};
        device::Function, batchsize::Int = Generalization.PREDICT_BATCHSIZE)
    logits, probs = predict_probs(model, X[:, :, :, idxs]; device = device, batchsize = batchsize)
    y_true = y[idxs]
    # Predict "class" (1) when P(class) >= P(no_class).
    y_pred = [probs[2, i] >= probs[1, i] ? 1 : 0 for i in axes(probs, 2)]
    metrics = CNNUtils.compute_metrics(y_pred, y_true)
    return metrics, logits, probs, y_true, y_pred
end

"""
    save_final_model(path, model) -> path

Save the model's full CPU `Flux.state` (weights + BatchNorm running stats, which
matter for ResNet18). Reload later via `build_pretrained_resnet18()` +
`Flux.loadmodel!`.
"""
function save_final_model(path::AbstractString, model)
    mkpath(dirname(path))
    JLD2.jldsave(path; model_state = Flux.state(cpu(model)))
    return path
end
