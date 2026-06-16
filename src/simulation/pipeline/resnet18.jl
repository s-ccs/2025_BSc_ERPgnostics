# =============================================================================
# Single-channel ResNet18
#
# ERP images are single-channel, but the pretrained ResNet18 expects three input
# channels. These builders adapt a Metalhead ResNet18 to one input channel and a
# two-class head, optionally seeding it with ImageNet weights whose first
# convolution is averaged down to one channel.
# =============================================================================

"""
    resnet_backbone(model)

Return the feature-extraction part of a Metalhead ResNet, across Metalhead
versions that expose either a `backbone` function or a two-layer chain.
"""
function resnet_backbone(model)
    isdefined(Metalhead, :backbone) && return getfield(Metalhead, :backbone)(model)
    return model.layers.layers[1]
end

"""
    resnet_classifier(model)

Return the classifier head of a Metalhead ResNet, across Metalhead versions that
expose either a `classifier` function or a two-layer chain.
"""
function resnet_classifier(model)
    isdefined(Metalhead, :classifier) && return getfield(Metalhead, :classifier)(model)
    return model.layers.layers[2]
end

"""
    project_first_conv_weights(src_weight, dst_inchannels)

Project a pretrained first-convolution kernel onto `dst_inchannels` input
channels. When the channel counts already match, the kernel is copied; otherwise
the channels are averaged and repeated, which adapts RGB weights to grayscale.

# Arguments
- `src_weight::AbstractArray`: 4D source convolution kernel.
- `dst_inchannels::Int`: desired number of input channels.

# Returns
- `Array`: a kernel with `dst_inchannels` input channels.
"""
function project_first_conv_weights(src_weight::AbstractArray, dst_inchannels::Int)
    ndims(src_weight) == 4 || error("Expected a 4D convolution kernel.")
    src_inchannels = size(src_weight, 3)
    dst_inchannels == src_inchannels && return copy(src_weight)
    # Average across the source channels, then repeat to the target channel count.
    projected = mean(src_weight; dims = 3)
    return repeat(projected, 1, 1, dst_inchannels, 1)
end

"""
    collect_arrays_recursive(x, acc = Any[])

Collect every numeric array reachable inside nested `NamedTuple`/`Tuple`
structures into `acc`, preserving traversal order. Used to line up pretrained
weight arrays with the destination model's trainable arrays.

# Arguments
- `x`: a value, possibly nesting arrays inside tuples/named tuples.
- `acc::Vector{Any}`: accumulator, appended in place.

# Returns
- `Vector{Any}`: all arrays found, in order.
"""
function collect_arrays_recursive(x, acc = Vector{Any}())
    if x isa AbstractArray
        push!(acc, x)
    elseif x isa NamedTuple
        for k in keys(x)
            collect_arrays_recursive(getfield(x, k), acc)
        end
    elseif x isa Tuple
        for xi in x
            collect_arrays_recursive(xi, acc)
        end
    end
    return acc
end

"""
    load_resnet18_pretrained_firstconv!(model) -> Int

Copy ImageNet ResNet18 weights into `model` in place, projecting the first
convolution onto the model's input-channel count and matching the remaining
arrays by shape.

# Arguments
- `model`: the destination single-channel ResNet18.

# Returns
- `Int`: the number of weight arrays that were copied.
"""
function load_resnet18_pretrained_firstconv!(model)
    src_state = Metalhead.loadweights("resnet18-IMAGENET1K_V1")
    dst_arrays = Flux.trainables(model)
    src_arrays = collect_arrays_recursive(src_state)
    isempty(dst_arrays) && error("Destination model has no trainable arrays.")
    isempty(src_arrays) && error("Source pretrained state has no arrays.")

    matched = 0

    # The first convolution differs in channel count, so project it explicitly.
    first_dst = dst_arrays[1]
    first_src = src_arrays[1]
    if ndims(first_dst) == 4 && ndims(first_src) == 4 &&
       size(first_dst, 1) == size(first_src, 1) &&
       size(first_dst, 2) == size(first_src, 2) &&
       size(first_dst, 4) == size(first_src, 4)
        projected = project_first_conv_weights(first_src, size(first_dst, 3))
        size(projected) == size(first_dst) || error("Projected first conv weight has wrong size.")
        copyto!(first_dst, projected)
        matched += 1
        dst_start, src_start = 2, 2
    else
        dst_start, src_start = 1, 1
    end

    # Match every remaining destination array to the next source array of equal shape.
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

"""
    build_resnet18_single_channel(n_classes, in_channels) -> Chain

Build a single-channel ResNet18 with a fresh `n_classes` head, returned as a
plain `Chain`. Keeping it as a `Chain` matches the GPU path used elsewhere and
avoids leaving parts of a raw Metalhead model on the CPU.

# Arguments
- `n_classes::Int`: number of output classes.
- `in_channels::Int`: number of input channels.

# Returns
- `Tuple{Chain, Function}`: the model and a closure to (optionally) load weights.
"""
function build_resnet18_single_channel(n_classes::Int, in_channels::Int)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 1000)
    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(old_head.layers[1], old_head.layers[2], Dense(in_dim => n_classes))
    return Chain(features, new_head), base
end

"""
    build_resnet18_random(; n_classes = 2, in_channels = 1) -> Chain

Build a randomly initialised single-channel ResNet18.
"""
function build_resnet18_random(; n_classes::Int = 2, in_channels::Int = 1)
    model, _ = build_resnet18_single_channel(n_classes, in_channels)
    return model
end

"""
    build_resnet18_pretrained(; n_classes = 2, in_channels = 1)

Build a single-channel ResNet18 seeded with projected ImageNet weights.

# Returns
- `Tuple{Chain, Int}`: the model and the number of pretrained arrays loaded.
"""
function build_resnet18_pretrained(; n_classes::Int = 2, in_channels::Int = 1)
    base = Metalhead.ResNet(18; pretrain = false, inchannels = in_channels, nclasses = 1000)
    matched = load_resnet18_pretrained_firstconv!(base)
    features = resnet_backbone(base)
    old_head = resnet_classifier(base)
    in_dim = size(old_head.layers[3].weight, 2)
    new_head = Chain(old_head.layers[1], old_head.layers[2], Dense(in_dim => n_classes))
    return Chain(features, new_head), matched
end

"""
    build_resnet18_for_profile(profile)

Build the ResNet18 requested by a training profile.

# Arguments
- `profile`: named tuple whose `model_init` is `:pretrained` or `:random`.

# Returns
- `Tuple{Chain, Int}`: the model and the number of pretrained arrays loaded
  (`0` for a random initialisation).
"""
function build_resnet18_for_profile(profile)
    if profile.model_init == :pretrained
        return build_resnet18_pretrained(n_classes = 2, in_channels = 1)
    elseif profile.model_init == :random
        return build_resnet18_random(n_classes = 2, in_channels = 1), 0
    end
    error("Unsupported model_init=$(profile.model_init).")
end
