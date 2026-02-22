module ERPImageUtils

using ImageFiltering: KernelFactors
using StatsBase: mean_and_std, zscore, quantile
using Printf: @sprintf
using CairoMakie: cgrad, Reverse

export gaussian_kernel, zscore_timepoints
export make_diverging_cmap_zero_centered, clipped_color_stats, clipped_color_stats_filter_row

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

"""
    zscore_timepoints(data_time_trials)

Compute z-score row-wise (dims = 2), typically "per timepoint over trials".
Rows with zero variance are stabilized with `σ = 1` to avoid NaN values.
"""
function zscore_timepoints(data_time_trials::AbstractMatrix)
    x = Float32.(data_time_trials)
    μ, σ = mean_and_std(x, 2; corrected = true)
    σ_safe = ifelse.(Float32.(σ) .== 0f0, 1f0, Float32.(σ))
    return Float32.(zscore(x, μ, σ_safe))
end

"""
    make_diverging_cmap_zero_centered(vmin, vmax)

Build a Blue→White→Red colormap where white sits at the fractional position of 0 in [vmin, vmax].
Samples 11 colors from the reversed RdBu colormap and stretches positions so the center (white) lands at 0.
"""
function make_diverging_cmap_zero_centered(vmin::Real, vmax::Real)
    f = clamp(Float64(-vmin / (vmax - vmin)), 0.02, 0.98)
    src_cmap = cgrad(:RdBu, rev=true)
    n_half = 5
    colors = Vector{Any}(undef, 2 * n_half + 1)
    positions = Vector{Float64}(undef, 2 * n_half + 1)
    # Blue half: map source [0, 0.5] → destination [0, f]
    for i in 0:n_half
        src_t = 0.5 * i / n_half
        dst_t = f * i / n_half
        colors[i+1] = src_cmap[src_t]
        positions[i+1] = dst_t
    end
    # Red half: map source (0.5, 1.0] → destination (f, 1.0]
    for i in 1:n_half
        src_t = 0.5 + 0.5 * i / n_half
        dst_t = f + (1.0 - f) * i / n_half
        colors[n_half + 1 + i] = src_cmap[src_t]
        positions[n_half + 1 + i] = dst_t
    end
    return cgrad(colors, positions)
end

"""
    clipped_color_stats(data; q_low=0.01, q_high=0.99)

Compute symmetric color stats for the reference (top) row.
Returns `(clipped, colorrange, tick_vals, tick_labels)`.
"""
function clipped_color_stats(data::AbstractMatrix; q_low::Float64=0.01, q_high::Float64=0.99)
    x = Float32.(vec(data))
    ql = Float32(quantile(x, q_low))
    qh = Float32(quantile(x, q_high))

    m = max(abs(ql), abs(qh))
    m = m == 0f0 ? 1f-6 : m

    clipped = clamp.(Float32.(data), -m, m)
    tick_vals = Float32[-m, 0f0, m]
    tick_labels = [@sprintf("%.3f", t) for t in tick_vals]
    return clipped, (-m, m), tick_vals, tick_labels
end

"""
    clipped_color_stats_filter_row(data; q_low=0.01, q_high=0.99)

Compute asymmetric color stats for the filtered (bottom) row.
Always places white at 0 by extending the color range to include 0 and building a custom colormap.
Returns `(clipped, colorrange, tick_vals, tick_labels, colormap)`.
"""
function clipped_color_stats_filter_row(data::AbstractMatrix; q_low::Float64=0.01, q_high::Float64=0.99)
    x = Float32.(vec(data))
    ql = Float32(quantile(x, q_low))
    qh = Float32(quantile(x, q_high))

    if qh <= ql
        qh = ql + 1f-6
    end

    # Always include 0 in the range so white maps to 0
    vmin = min(ql, 0f0)
    vmax = max(qh, 0f0)

    # Prevent degenerate range
    if vmax <= vmin
        vmax = vmin + 1f-6
    end

    clipped = clamp.(Float32.(data), vmin, vmax)
    crange = (vmin, vmax)
    cmap = make_diverging_cmap_zero_centered(vmin, vmax)

    # Ticks: always show min, 0, max
    if vmin == 0f0
        # All positive: show 0, midpoint, max
        mid = vmax / 2f0
        tick_vals = Float32[0f0, mid, vmax]
    elseif vmax == 0f0
        # All negative: show min, midpoint, 0
        mid = vmin / 2f0
        tick_vals = Float32[vmin, mid, 0f0]
    else
        # Mixed: show min, 0, max
        tick_vals = Float32[vmin, 0f0, vmax]
    end

    tick_labels = [@sprintf("%.3f", t) for t in tick_vals]
    return clipped, crange, tick_vals, tick_labels, cmap
end

end
