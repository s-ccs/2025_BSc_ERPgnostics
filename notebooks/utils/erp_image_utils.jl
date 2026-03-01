module ERPImageUtils

using ImageFiltering: KernelFactors
using StatsBase: mean_and_std, zscore, quantile
using Printf: @sprintf
using CairoMakie: cgrad, Reverse

export gaussian_kernel, zscore_timepoints
export make_diverging_cmap_zero_centered, clipped_color_stats, clipped_color_stats_filter_row
export make_diverging_cmap_zero_anchored, clipped_color_stats_quantile_zero_ticks

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
    make_diverging_cmap_zero_anchored(vmin, vmax; n_steps=64)

Build a diverging colormap where value 0 is mapped to white exactly.
Works for mixed-sign, only-positive, and only-negative ranges.
"""
function make_diverging_cmap_zero_anchored(vmin::Real, vmax::Real; n_steps::Int = 64)
    vmin_f = Float64(vmin)
    vmax_f = Float64(vmax)
    vmax_f <= vmin_f && (vmax_f = vmin_f + 1e-6)

    src = cgrad(:RdBu, rev = true) # blue -> white -> red
    zero_pos = clamp((0.0 - vmin_f) / (vmax_f - vmin_f), 0.0, 1.0)
    n_steps = max(2, n_steps)

    if zero_pos <= 0.0
        # Only positive range: white at lower bound, then red branch.
        colors = [src[0.5 + 0.5 * i / n_steps] for i in 0:n_steps]
        positions = collect(range(0.0, 1.0; length = n_steps + 1))
        return cgrad(colors, positions)
    elseif zero_pos >= 1.0
        # Only negative range: blue branch, then white at upper bound.
        colors = [src[0.0 + 0.5 * i / n_steps] for i in 0:n_steps]
        positions = collect(range(0.0, 1.0; length = n_steps + 1))
        return cgrad(colors, positions)
    end

    # Mixed-sign range: stretch both halves to hit white exactly at 0.
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

"""
    clipped_color_stats_quantile_zero_ticks(data; q_low=0.01, q_high=0.99)

Asymmetric quantile clipping with explicit zero anchoring:
- mixed-sign data: clip to [q_low, q_high]
- only-positive data: clip to [0, q_high]
- only-negative data: clip to [q_low, 0]

Returns `(clipped, colorrange, tick_vals, tick_labels, colormap)` where ticks
represent `q_low`, `0`, `q_high` (sorted for plotting stability).
"""
function clipped_color_stats_quantile_zero_ticks(data::AbstractMatrix; q_low::Float64 = 0.01, q_high::Float64 = 0.99)
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

    # Keep semantic ticks (q_low, 0, q_high), but sort for colorbar rendering.
    raw_ticks = Float32[ql, 0f0, qh]
    raw_labels = [@sprintf("%.3f", ql), @sprintf("%.3f", 0f0), @sprintf("%.3f", qh)]
    pairs = collect(zip(raw_ticks, raw_labels))
    sort!(pairs; by = first)

    tick_vals = Float32[pairs[1][1], pairs[2][1], pairs[3][1]]
    tick_labels = [pairs[1][2], pairs[2][2], pairs[3][2]]

    # Avoid duplicated tick positions for plotting backends.
    for i in 2:3
        if tick_vals[i] <= tick_vals[i - 1]
            tick_vals[i] = nextfloat(tick_vals[i - 1])
        end
    end

    return clipped, crange, tick_vals, tick_labels, cmap
end

end
