module ERPImageUtils

using ImageFiltering: KernelFactors
using StatsBase: mean_and_std, zscore

export gaussian_kernel, zscore_timepoints

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

end
