# model_engine.jl
#
# Self-contained model engine for the real-data ResNet18 ERP-scoring pipeline.
#
# This is a trimmed, vendored copy of exactly the parts of the former
# `notebooks/week_20` + `notebooks/utils` engine that this pipeline actually
# uses:
#   * the pretrained single-channel ResNet18 builder,
#   * device setup and batched prediction,
#   * the Gaussian-reference image pipeline, tensor packing and metrics,
#   * the shared image/training constants.
#
# Everything else from the original engine (dataset preparation, plotting,
# full-experiment orchestration) is intentionally omitted so that `src/` does
# not depend on `notebooks/` or any external package environment. The function
# bodies are copied verbatim to keep the scoring behaviour identical.

module RealDataModelEngine

using CUDA
using Flux
using Metalhead
using Statistics: mean

# --------------------------------------------------------------------------- #
# Image / training constants (verbatim from the original engine).
# --------------------------------------------------------------------------- #
const TARGET_SIZE = (64, 64)
const LOWPASS_SIGMA = 75.0f0
const LOWPASS_KERNEL_SIZE = (21, 21)
const FILTER_BORDER = "reflect"
const TRAIN_EPOCHS = 8
const TRAIN_LR = 3f-4
const PREDICT_BATCHSIZE = 64

# --------------------------------------------------------------------------- #
# Image pipeline, tensor packing and metrics (formerly `ERPCNNExperimentUtils`,
# with `gaussian_kernel` from the former `ERPImageUtils`). Kept as a submodule
# so callers can keep using `<engine>.ERPCNNExperimentUtils`.
# --------------------------------------------------------------------------- #
module ERPCNNExperimentUtils

using ImageFiltering: imfilter, KernelFactors
using Images: imresize
using StatisticalMeasures
using StatisticalMeasures: macro_avg

"""
    gaussian_kernel(low_pass_factor, in_size, out_size, kernel_size=(21, 21))

Create a 2D Gaussian kernel using ImageFiltering's built-in kernel generator.
Sigma is scaled per axis by `low_pass_factor * in_dim / out_dim`.
"""
function gaussian_kernel(low_pass_factor::Real,
                         in_size::Tuple{Int, Int},
                         out_size::Tuple{Int, Int},
                         kernel_size::Tuple{Int, Int} = (21, 21))
    in_h, in_w = in_size
    out_h, out_w = out_size
    k_h, k_w = kernel_size

    if !isodd(k_h) || !isodd(k_w)
        throw(ArgumentError("kernel_size must be odd in both dimensions, got $(kernel_size)"))
    end

    sigma_h = max(Float32(low_pass_factor) * Float32(in_h) / Float32(out_h), 1f-3)
    sigma_w = max(Float32(low_pass_factor) * Float32(in_w) / Float32(out_w), 1f-3)

    return KernelFactors.gaussian((sigma_h, sigma_w), kernel_size)
end

function apply_filter_n(data::AbstractMatrix, filter_fn::Function; repeats::Int = 1)
    out = Float32.(data)
    for _ in 1:repeats
        out = Float32.(filter_fn(out))
    end
    return out
end

function apply_gaussian_pre_resize(img_trials_time::AbstractMatrix;
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    kernel = gaussian_kernel(low_pass_sigma, size(img_trials_time), target_size, lowpass_kernel_size)
    return Float32.(imfilter(Float32.(img_trials_time), kernel, filter_border))
end

resize_processed_image(img::AbstractMatrix, target_size::Tuple{Int, Int}) = Float32.(imresize(Float32.(img), target_size))

function apply_pipeline_to_image(img_trials_time::AbstractMatrix;
    pipeline_name::Symbol,
    filter_fn::Union{Nothing, Function} = nothing,
    filter_repeats::Int = 1,
    target_size::Tuple{Int, Int},
    low_pass_sigma::Float32,
    lowpass_kernel_size::Tuple{Int, Int},
    filter_border::String)

    if pipeline_name == :gaussian_reference
        smoothed = apply_gaussian_pre_resize(
            img_trials_time;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        return resize_processed_image(smoothed, target_size)
    end

    filter_fn === nothing && error("A filter function is required for pipeline $(pipeline_name).")

    if pipeline_name == :gaussian_then_filter
        smoothed = apply_gaussian_pre_resize(
            img_trials_time;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        filtered = apply_filter_n(smoothed, filter_fn; repeats = filter_repeats)
        return resize_processed_image(filtered, target_size)
    elseif pipeline_name == :filter_then_gaussian
        filtered = apply_filter_n(img_trials_time, filter_fn; repeats = filter_repeats)
        smoothed = apply_gaussian_pre_resize(
            filtered;
            target_size = target_size,
            low_pass_sigma = low_pass_sigma,
            lowpass_kernel_size = lowpass_kernel_size,
            filter_border = filter_border,
        )
        return resize_processed_image(smoothed, target_size)
    elseif pipeline_name == :filter_only
        filtered = apply_filter_n(img_trials_time, filter_fn; repeats = filter_repeats)
        return resize_processed_image(filtered, target_size)
    else
        error("Unsupported pipeline name: $(pipeline_name)")
    end
end

function images_to_tensor(imgs)
    h, w = size(imgs[1])
    n = length(imgs)
    x = Array{Float32}(undef, h, w, 1, n)
    for (i, img) in enumerate(imgs)
        @assert size(img) == (h, w) "All ERP images must share the same trial/time layout."
        # Preserve the matrix orientation for the CNN:
        # height = trials (y-axis), width = time (x-axis).
        x[:, :, 1, i] = Float32.(img)
    end
    return x
end

function compute_metrics(y_pred::Vector{Int}, y_true::Vector{Int})
    acc = StatisticalMeasures.Accuracy()(y_pred, y_true)
    bacc = StatisticalMeasures.BalancedAccuracy()(y_pred, y_true)
    macro_f1 = StatisticalMeasures.MulticlassFScore(; average = macro_avg)(y_pred, y_true)
    precision = StatisticalMeasures.MulticlassPositivePredictiveValue()(y_pred, y_true)
    recall = StatisticalMeasures.MulticlassTruePositiveRate()(y_pred, y_true)

    return (
        accuracy = acc,
        balanced_accuracy = bacc,
        macro_f1 = macro_f1,
        precision = precision,
        recall = recall,
    )
end

end # module ERPCNNExperimentUtils

using .ERPCNNExperimentUtils

# --------------------------------------------------------------------------- #
# Device setup, pretrained ResNet18 and batched prediction (formerly the
# Week-20 engine). Bodies copied verbatim.
# --------------------------------------------------------------------------- #
function setup_device()
    if CUDA.functional()
        CUDA.allowscalar(false)
        CUDA.device!(0)
        println("CUDA device: ", CUDA.name(CUDA.device()))
        return gpu, true
    end
    println("CUDA is not functional; running on CPU with a smaller batch size.")
    return cpu, false
end

function collect_arrays_recursive_local(x, acc = Vector{Any}())
    if x isa AbstractArray
        push!(acc, x)
    elseif x isa NamedTuple
        for k in keys(x)
            collect_arrays_recursive_local(getfield(x, k), acc)
        end
    elseif x isa Tuple
        for xi in x
            collect_arrays_recursive_local(xi, acc)
        end
    end
    return acc
end

function project_first_conv_weights(src_weight::AbstractArray, dst_inchannels::Int)
    @assert ndims(src_weight) == 4 "Expected a 4D convolution kernel."
    src_inchannels = size(src_weight, 3)
    src_inchannels == dst_inchannels && return copy(src_weight)
    projected = mean(src_weight; dims = 3)
    return repeat(projected, 1, 1, dst_inchannels, 1)
end

function load_resnet_pretrained_project_firstconv!(model, weight_key::AbstractString)
    src_state = Metalhead.loadweights(weight_key)
    dst_arrays = Flux.trainables(model)
    src_arrays = collect_arrays_recursive_local(src_state)

    @assert !isempty(dst_arrays) "Destination model has no trainable arrays."
    @assert !isempty(src_arrays) "Source pretrained state has no arrays."

    matched = 0
    first_dst = dst_arrays[1]
    first_src = src_arrays[1]

    if ndims(first_dst) == 4 && ndims(first_src) == 4 &&
       size(first_dst, 1) == size(first_src, 1) &&
       size(first_dst, 2) == size(first_src, 2) &&
       size(first_dst, 4) == size(first_src, 4)

        projected = project_first_conv_weights(first_src, size(first_dst, 3))
        @assert size(projected) == size(first_dst) "Projected first convolution has wrong size."
        copyto!(first_dst, projected)
        matched += 1
        dst_start = 2
        src_start = 2
    else
        dst_start = 1
        src_start = 1
    end

    j = src_start
    for i in dst_start:length(dst_arrays)
        d = dst_arrays[i]
        while j <= length(src_arrays) && size(src_arrays[j]) != size(d)
            j += 1
        end
        j <= length(src_arrays) || error("Failed to map pretrained weights for destination size $(size(d)).")
        copyto!(d, src_arrays[j])
        matched += 1
        j += 1
    end

    return matched
end

resnet_backbone(model) = isdefined(Metalhead, :backbone) ? getfield(Metalhead, :backbone)(model) : model.layers.layers[1]
resnet_classifier(model) = isdefined(Metalhead, :classifier) ? getfield(Metalhead, :classifier)(model) : model.layers.layers[2]

function build_resnet_single_channel_pretrained(depth::Int; n_classes::Int = 2, in_channels::Int = 1)
    weight_key = "resnet$(depth)-IMAGENET1K_V1"
    base = Metalhead.ResNet(depth; pretrain = false, inchannels = in_channels, nclasses = 1000)
    matched = load_resnet_pretrained_project_firstconv!(base, weight_key)

    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(
        old_head.layers[1],
        old_head.layers[2],
        Dense(in_dim => n_classes),
    )

    return Chain(features, new_head), matched
end

function predict_logits_probs(model, X::Array{Float32, 4}; batchsize::Int, device::Function)
    Flux.testmode!(model, true)
    n = size(X, 4)
    logits_all = Array{Float32}(undef, 2, n)
    probs_all = Array{Float32}(undef, 2, n)
    for start_idx in 1:batchsize:n
        idx = start_idx:min(start_idx + batchsize - 1, n)
        logits = Array(cpu(model(device(X[:, :, :, idx]))))
        probs = Flux.softmax(Float32.(logits); dims = 1)
        logits_all[:, idx] .= Float32.(logits)
        probs_all[:, idx] .= Float32.(probs)
    end
    return logits_all, probs_all
end

end # module RealDataModelEngine
