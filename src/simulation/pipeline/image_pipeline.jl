# =============================================================================
# Shared ERP image pipeline
#
# Turns a raw timepoints-by-trials matrix into the 64x64 image the classifier
# sees. The exact same steps run on real fixation channels and on simulated
# trials, which is what makes the sim-to-real comparison fair:
#     sort trials -> z-score per timepoint -> Gaussian smooth -> resize.
# =============================================================================

"""
    sort_values(events::DataFrame, sort_col::Symbol)

Return the values of `sort_col` used to order trials along the image y-axis.

Numeric columns are returned as `Float64`; any other column is returned as-is
so that tuple- or permutation-valued sort keys keep working.

# Arguments
- `events::DataFrame`: per-trial event table.
- `sort_col::Symbol`: column whose values define the trial order.

# Returns
- A vector of sort values, one per trial.
"""
function sort_values(events::DataFrame, sort_col::Symbol)
    values = events[!, sort_col]
    if eltype(values) <: Number
        return Float64.(values)
    end
    return collect(values)
end

"""
    zscore_per_timepoint(data_time_trials::AbstractMatrix)

Z-score every timepoint (row) across trials. Rows with zero variance are
stabilised with `σ = 1` to avoid `NaN` values.

# Arguments
- `data_time_trials::AbstractMatrix`: EEG values in timepoints-by-trials layout.

# Returns
- `Matrix{Float32}`: the per-timepoint z-scored matrix, same shape as the input.
"""
function zscore_per_timepoint(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    μ, σ = mean_and_std(x, 2; corrected = true)
    σ_safe = ifelse.(Float32.(σ) .== 0.0f0, 1.0f0, Float32.(σ))
    return Float32.(zscore(x, μ, σ_safe))
end

"""
    sorted_zscore_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol)

Sort trials by `sort_col`, z-score each timepoint across trials, and return the
ERP image in trials-by-time layout (rows = trials, columns = time).

# Arguments
- `data_time_trials::AbstractMatrix`: EEG values in timepoints-by-trials layout.
- `events::DataFrame`: per-trial event table; must contain `sort_col`.
- `sort_col::Symbol`: column that defines the trial order.

# Returns
- `Matrix{Float32}`: the sorted, z-scored ERP image (trials x time).
"""
function sorted_zscore_image(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol)
    size(data_time_trials, 2) == nrow(events) || error("Trial count mismatch between matrix and events.")
    sort_col in propertynames(events) || error("Sort column $(sort_col) not found in events table.")

    order = sortperm(sort_values(events, sort_col))
    data_sorted = data_time_trials[:, order]
    data_z = zscore_per_timepoint(data_sorted)
    # Convention across the whole project: rows = trials (y), columns = time (x).
    return Float32.(permutedims(data_z, (2, 1)))
end

"""
    gaussian_lowpass_kernel(low_pass_factor, in_size, out_size, kernel_size)

Build a separable Gaussian smoothing kernel whose width scales with the
downsampling ratio from `in_size` to `out_size`. This is the anti-aliasing
filter applied before the resize step.

# Arguments
- `low_pass_factor`: smoothing strength multiplier.
- `in_size::Tuple{Int,Int}`: size of the image before resizing.
- `out_size::Tuple{Int,Int}`: target size after resizing.
- `kernel_size::Tuple{Int,Int}`: kernel extent; must be odd in both dimensions.

# Returns
- A `KernelFactors` Gaussian kernel suitable for `imfilter`.
"""
function gaussian_lowpass_kernel(low_pass_factor, in_size::Tuple{Int, Int}, out_size::Tuple{Int, Int}, kernel_size::Tuple{Int, Int})
    in_h, in_w = in_size
    out_h, out_w = out_size
    k_h, k_w = kernel_size
    isodd(k_h) && isodd(k_w) || throw(ArgumentError("kernel_size must be odd in both dimensions, got $(kernel_size)"))

    # Wider Gaussian when the image shrinks more, so high frequencies are removed.
    sigma_h = max(Float32(low_pass_factor) * Float32(in_h) / Float32(out_h), 1.0f-3)
    sigma_w = max(Float32(low_pass_factor) * Float32(in_w) / Float32(out_w), 1.0f-3)
    return KernelFactors.gaussian((sigma_h, sigma_w), kernel_size)
end

"""
    gaussian_smoothed_image(img_trials_time, config::RunConfig)

Apply the configured Gaussian low-pass filter to an ERP image, sized for the
downscale to `config.target_size`.

# Arguments
- `img_trials_time::AbstractMatrix`: ERP image in trials-by-time layout.
- `config::RunConfig`: supplies the smoothing factor, kernel size, and border.

# Returns
- `Matrix{Float32}`: the smoothed image, same shape as the input.
"""
function gaussian_smoothed_image(img_trials_time::AbstractMatrix, config::RunConfig)
    kernel = gaussian_lowpass_kernel(
        config.low_pass_factor,
        size(img_trials_time),
        config.target_size,
        config.lowpass_kernel_size,
    )
    return Float32.(imfilter(Float32.(img_trials_time), kernel, config.filter_border))
end

"""
    pipeline_image(img_trials_time, config::RunConfig)

Run the configured image pipeline on a sorted, z-scored ERP image and return
the final fixed-size image. Only the `:gaussian_reference` pipeline (smooth then
resize) is supported, matching the E4 experiment definition.

# Arguments
- `img_trials_time::AbstractMatrix`: sorted, z-scored ERP image (trials x time).
- `config::RunConfig`: supplies the pipeline name and target size.

# Returns
- `Matrix{Float32}`: the `config.target_size` image.
"""
function pipeline_image(img_trials_time::AbstractMatrix, config::RunConfig)
    config.pipeline_name == :gaussian_reference ||
        error("Unsupported pipeline $(config.pipeline_name); only :gaussian_reference is defined for E4.")
    smoothed = gaussian_smoothed_image(img_trials_time, config)
    return Float32.(imresize(smoothed, config.target_size))
end

"""
    shared_preprocess(data_time_trials, events, sort_col, config::RunConfig)

Full shared preprocessing: sort and z-score the trials, then apply the image
pipeline. This single function is the only path used for both real and
simulated data, which guarantees identical treatment.

# Arguments
- `data_time_trials::AbstractMatrix`: EEG values in timepoints-by-trials layout.
- `events::DataFrame`: per-trial event table containing `sort_col`.
- `sort_col::Symbol`: column that defines the trial order.
- `config::RunConfig`: image pipeline settings.

# Returns
- `Matrix{Float32}`: the `config.target_size` ERP image.
"""
function shared_preprocess(data_time_trials::AbstractMatrix, events::DataFrame, sort_col::Symbol, config::RunConfig)
    img_trials_time = sorted_zscore_image(data_time_trials, events, sort_col)
    return pipeline_image(img_trials_time, config)
end

"""
    images_to_tensor(images)

Stack a vector of equally sized ERP images into a 4D Flux tensor with layout
`height x width x channel x sample` and a single channel.

# Arguments
- `images`: vector of `Matrix{Float32}`, all sharing the same trial/time layout.

# Returns
- `Array{Float32,4}`: the batched tensor (`h, w, 1, n`).
"""
function images_to_tensor(images)
    h, w = size(images[1])
    n = length(images)
    tensor = Array{Float32}(undef, h, w, 1, n)
    for (i, img) in enumerate(images)
        size(img) == (h, w) || error("All ERP images must share the same trial/time layout.")
        # height = trials (y-axis), width = time (x-axis); keep orientation intact.
        tensor[:, :, 1, i] = Float32.(img)
    end
    return tensor
end
