module ERPImageProcessing

using CairoMakie: cgrad
using ImageFiltering: KernelFactors, imfilter
using Printf: @sprintf
using StatsBase: mean_and_std, quantile, zscore

export zscore_timepoints
export trials_time_image
export gaussian_kernel
export gaussian_smooth_image
export erp_image_sort_zscore_smooth
export make_diverging_cmap_zero_anchored
export clipped_color_stats_quantile_zero_ticks

const DEFAULT_GAUSSIAN_KERNEL_SIZE = (21, 21)

"""
    zscore_timepoints(data_time_trials)

Compute z-scores per timepoint across trials. The input convention is
`timepoints x trials`; zero-variance rows are stabilized with `sigma = 1`.
"""
function zscore_timepoints(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    mu, sigma = mean_and_std(x, 2; corrected = true)
    sigma_safe = ifelse.(Float32.(sigma) .== 0f0, 1f0, Float32.(sigma))
    return Float32.(zscore(x, mu, sigma_safe))
end

"""
    trials_time_image(data_time_trials)

Convert a `timepoints x trials` matrix to the ERP-image layout
`trials x timepoints`.
"""
function trials_time_image(data_time_trials::AbstractMatrix)
    return Float32.(permutedims(data_time_trials, (2, 1)))
end

"""
    gaussian_kernel(sigma_factor, in_size, out_size; kernel_size=(21, 21))

Create a 2D Gaussian kernel with per-axis sigma scaled by the ratio between
input and output dimensions.
"""
function gaussian_kernel(sigma_factor::Real,
                         in_size::Tuple{Int, Int},
                         out_size::Tuple{Int, Int},
                         kernel_size::Tuple{Int, Int} = DEFAULT_GAUSSIAN_KERNEL_SIZE)
    k_h, k_w = kernel_size
    (!isodd(k_h) || !isodd(k_w)) &&
        throw(ArgumentError("kernel_size must be odd in both dimensions, got $(kernel_size)."))

    in_h, in_w = in_size
    out_h, out_w = out_size
    sigma_h = max(Float32(sigma_factor) * Float32(in_h) / Float32(out_h), 1f-3)
    sigma_w = max(Float32(sigma_factor) * Float32(in_w) / Float32(out_w), 1f-3)
    return KernelFactors.gaussian((sigma_h, sigma_w), kernel_size)
end

function gaussian_kernel(sigma_factor::Real,
                         in_size::Tuple{Int, Int},
                         out_size::Tuple{Int, Int};
                         kernel_size::Tuple{Int, Int})
    return gaussian_kernel(sigma_factor, in_size, out_size, kernel_size)
end

"""
    gaussian_smooth_image(img; sigma_factor, output_size=size(img), kernel_size=(21, 21))

Apply Gaussian smoothing to an ERP image. This step expects image layout
`trials x timepoints`.
"""
function gaussian_smooth_image(img_trials_time::AbstractMatrix;
                               sigma_factor::Real,
                               output_size::Tuple{Int, Int} = size(img_trials_time),
                               kernel_size::Tuple{Int, Int} = DEFAULT_GAUSSIAN_KERNEL_SIZE)
    img = Float32.(img_trials_time)
    min(size(img)...) <= 1 && return img

    kernel = gaussian_kernel(
        sigma_factor,
        size(img),
        output_size;
        kernel_size = kernel_size,
    )
    return Float32.(imfilter(img, kernel, "reflect"))
end

"""
    erp_image_sort_zscore_smooth(data_time_trials, trial_order; sigma_factor, smooth=true)

Convenience wrapper for the current ERP image pipeline:
sort trials -> z-score per timepoint -> convert to image -> Gaussian smooth.
The individual steps are exposed separately for custom pipelines.
"""
function erp_image_sort_zscore_smooth(data_time_trials::AbstractMatrix,
                                      trial_order;
                                      sigma_factor::Real,
                                      smooth::Bool = true,
                                      kernel_size::Tuple{Int, Int} = DEFAULT_GAUSSIAN_KERNEL_SIZE)
    order = Int.(collect(trial_order))
    length(order) == size(data_time_trials, 2) ||
        throw(ArgumentError("trial_order length $(length(order)) does not match trial count $(size(data_time_trials, 2))."))

    data_sorted = Float32.(data_time_trials[:, order])
    data_z = zscore_timepoints(data_sorted)
    img_trials_time = trials_time_image(data_z)
    smooth || return img_trials_time

    return gaussian_smooth_image(
        img_trials_time;
        sigma_factor = sigma_factor,
        kernel_size = kernel_size,
    )
end

function make_diverging_cmap_zero_anchored(vmin::Real, vmax::Real; n_steps::Int = 64)
    vmin_f = Float64(vmin)
    vmax_f = Float64(vmax)
    vmax_f <= vmin_f && (vmax_f = vmin_f + 1e-6)

    src = cgrad(:RdBu, rev = true)
    zero_pos = clamp((0.0 - vmin_f) / (vmax_f - vmin_f), 0.0, 1.0)
    n_steps = max(2, n_steps)

    if zero_pos <= 0.0
        colors = [src[0.5 + 0.5 * i / n_steps] for i in 0:n_steps]
        positions = collect(range(0.0, 1.0; length = n_steps + 1))
        return cgrad(colors, positions)
    elseif zero_pos >= 1.0
        colors = [src[0.0 + 0.5 * i / n_steps] for i in 0:n_steps]
        positions = collect(range(0.0, 1.0; length = n_steps + 1))
        return cgrad(colors, positions)
    end

    colors = Vector{Any}(undef, 2 * n_steps + 1)
    positions = Vector{Float64}(undef, 2 * n_steps + 1)

    for i in 0:n_steps
        src_t = 0.5 * i / n_steps
        dst_t = zero_pos * i / n_steps
        colors[i + 1] = src[src_t]
        positions[i + 1] = dst_t
    end

    for i in 1:n_steps
        src_t = 0.5 + 0.5 * i / n_steps
        dst_t = zero_pos + (1.0 - zero_pos) * i / n_steps
        colors[n_steps + 1 + i] = src[src_t]
        positions[n_steps + 1 + i] = dst_t
    end

    return cgrad(colors, positions)
end

function clipped_color_stats_quantile_zero_ticks(data::AbstractMatrix;
                                                 q_low::Float64 = 0.01,
                                                 q_high::Float64 = 0.99)
    vals = Float32[]
    for v in data
        fv = Float32(v)
        isfinite(fv) && push!(vals, fv)
    end
    isempty(vals) && push!(vals, 0f0)

    ql = Float32(quantile(vals, q_low))
    qh = Float32(quantile(vals, q_high))
    qh < ql && (qh = ql)

    all_nonneg = all(x -> x >= 0f0, vals)
    all_nonpos = all(x -> x <= 0f0, vals)

    vmin = ql
    vmax = qh
    if all_nonneg
        vmin = 0f0
        vmax = max(qh, 1f-6)
    elseif all_nonpos
        vmin = min(ql, -1f-6)
        vmax = 0f0
    else
        vmin = min(ql, 0f0)
        vmax = max(qh, 0f0)
    end

    if vmax <= vmin
        delta = max(abs(vmin), abs(vmax), 1f0) * 1f-6
        vmin -= delta
        vmax += delta
    end

    clipped = clamp.(Float32.(data), vmin, vmax)
    crange = (vmin, vmax)
    cmap = make_diverging_cmap_zero_anchored(vmin, vmax)

    pairs = collect(zip(
        Float32[ql, 0f0, qh],
        [@sprintf("%.3f", ql), @sprintf("%.3f", 0f0), @sprintf("%.3f", qh)],
    ))
    sort!(pairs; by = first)

    tick_vals = Float32[pairs[1][1], pairs[2][1], pairs[3][1]]
    tick_labels = [pairs[1][2], pairs[2][2], pairs[3][2]]

    for i in 2:3
        if tick_vals[i] <= tick_vals[i - 1]
            tick_vals[i] = nextfloat(tick_vals[i - 1])
        end
    end

    return clipped, crange, tick_vals, tick_labels, cmap
end

end

using .ERPImageProcessing: clipped_color_stats_quantile_zero_ticks
using .ERPImageProcessing: erp_image_sort_zscore_smooth
using .ERPImageProcessing: gaussian_kernel
using .ERPImageProcessing: gaussian_smooth_image
using .ERPImageProcessing: make_diverging_cmap_zero_anchored
using .ERPImageProcessing: trials_time_image
using .ERPImageProcessing: zscore_timepoints
