using DataFrames
using ImageFiltering: KernelFactors, imfilter
using ImageTransformations: imresize
using Interpolations: Linear
using StatsBase: mean_and_std, zscore

const DEFAULT_IMAGE_TARGET_SIZE = (64, 64)

"""
    sort_column_name(events, sort_variable)

Resolve and validate the event column used for trial sorting.

# Arguments
- `events`: Event table or table-like object with column names.
- `sort_variable`: Column name as `String`, `Symbol`, or another symbol-like value.

# Returns
- `Symbol`: Validated column name.

# Throws
- `ArgumentError`: If the column is not present in `events`.
"""
function sort_column_name(events, sort_variable)
    column = Symbol(sort_variable)
    column in propertynames(events) || throw(ArgumentError("Sort variable $(sort_variable) is not present in events."))
    return column
end

"""
    trial_sort_order(events, sort_variable; reverse=false)

Compute the trial index order from an event column without filtering rows.

# Arguments
- `events`: Event table with one row per trial.
- `sort_variable`: Event column used for sorting.
- `reverse`: Sort descending when `true`, ascending when `false`.

# Returns
- `Vector{Int}`: Row indices that sort the trials. The vector has one entry per
  event row.

# Throws
- `ArgumentError`: If the sort column is missing or its values cannot be sorted.
"""
function trial_sort_order(events, sort_variable; reverse = false)
    events_df = DataFrame(events)
    column = sort_column_name(events_df, sort_variable)

    # Keep the original row index so sorting returns trial positions.
    order_df = DataFrame(
        row_index = collect(1:nrow(events_df)),
        sort_value = events_df[!, column],
    )

    # Use the row index as a stable tie-breaker for equal sort values.
    try
        sort!(order_df, [:sort_value, :row_index]; rev = [Bool(reverse), false])
    catch err
        message = sprint(showerror, err)
        throw(ArgumentError("Cannot sort events by $(sort_variable): $(message)"))
    end

    return Int.(order_df[!, :row_index])
end

"""
    sort_trials(data_time_trials, trial_order)

Reorder signal trials by column while preserving `timepoints x trials` layout.

# Arguments
- `data_time_trials`: Signal matrix with timepoints in rows and trials in columns.
- `trial_order`: Trial column indices to select in the desired order.

# Returns
- `Matrix{Float32}`: Reordered signal matrix in `timepoints x trials` layout.

# Throws
- `ArgumentError`: If the order length or index range does not match the matrix.
"""
function sort_trials(data_time_trials, trial_order)
    order = Int.(collect(trial_order))
    n_trials = size(data_time_trials, 2)
    length(order) == n_trials || throw(ArgumentError(
        "Trial order length $(length(order)) does not match trial count $(n_trials).",
    ))
    all(index -> 1 <= index <= n_trials, order) || throw(ArgumentError(
        "Trial order contains indices outside 1:$(n_trials).",
    ))
    return Matrix{Float32}(data_time_trials[:, order])
end

"""
    zscore_timepoints(data_time_trials)

Z-score each timepoint across trials.

# Arguments
- `data_time_trials`: Signal matrix in `timepoints x trials` layout.

# Returns
- `Matrix{Float32}`: Z-scored matrix in the same layout. Zero or non-finite
  standard deviations are stabilized to `1`.
"""
function zscore_timepoints(data_time_trials)
    data = Float32.(data_time_trials)
    mean_value, std_value = mean_and_std(data, 2; corrected = true)

    # Replace zero or non-finite standard deviations before calling zscore.
    std_safe = map(std_value) do value
        value32 = Float32(value)
        isfinite(value32) && value32 > 0f0 ? value32 : 1f0
    end
    return Matrix{Float32}(zscore(data, Float32.(mean_value), std_safe))
end

"""
    trials_time_image(data_time_trials)

Convert signal layout into ERP image layout.

# Arguments
- `data_time_trials`: Matrix in `timepoints x trials` layout.

# Returns
- `Matrix{Float32}`: Matrix in `trials x timepoints` layout.
"""
function trials_time_image(data_time_trials)
    return Matrix{Float32}(permutedims(data_time_trials, (2, 1)))
end

"""
    gaussian_kernel_for_target(sigma_factor, in_size, target_size, kernel_size)

Create a 2D Gaussian kernel scaled from source image size to target image size.

# Arguments
- `sigma_factor`: Smoothing factor.
- `in_size`: Input image size as `(trials, timepoints)`.
- `target_size`: Target image size used to scale sigma per axis.
- `kernel_size`: Odd kernel dimensions, for example `(21, 21)`.

# Returns
- Gaussian kernel object from `ImageFiltering.KernelFactors.gaussian`.

# Throws
- `ArgumentError`: If either kernel dimension is even.
"""
function gaussian_kernel_for_target(sigma_factor, in_size, target_size, kernel_size)
    kernel_size = Tuple(Int.(kernel_size))
    target_size = Tuple(Int.(target_size))
    all(isodd, kernel_size) || throw(ArgumentError("kernel_size must be odd in both dimensions, got $(kernel_size)."))

    # Scale sigma so smoothing strength stays comparable after resizing.
    sigma_trials = max(Float32(sigma_factor) * Float32(in_size[1]) / Float32(target_size[1]), 1f-3)
    sigma_time = max(Float32(sigma_factor) * Float32(in_size[2]) / Float32(target_size[2]), 1f-3)
    return KernelFactors.gaussian((sigma_trials, sigma_time), kernel_size)
end

"""
    smooth_image_for_target(image, target_size; sigma_factor=75f0, kernel_size=(21, 21), border="reflect")

Apply Gaussian smoothing while scaling sigma for a later resize target.

# Arguments
- `image`: ERP image in `trials x timepoints` layout.
- `target_size`: Final target size used for sigma scaling.
- `sigma_factor`: Smoothing factor.
- `kernel_size`: Odd Gaussian kernel dimensions.
- `border`: ImageFiltering border handling mode.

# Returns
- `Matrix{Float32}`: Smoothed image with the same size as `image`.
"""
function smooth_image_for_target(image, target_size; sigma_factor = 75f0, kernel_size = (21, 21), border = "reflect")
    image32 = Float32.(image)

    # Degenerate images cannot support a meaningful two-dimensional kernel.
    min(size(image32)...) <= 1 && return Matrix{Float32}(image32)

    kernel = gaussian_kernel_for_target(sigma_factor, size(image32), target_size, kernel_size)
    return Matrix{Float32}(imfilter(image32, kernel, border))
end

"""
    smooth_image(image; sigma_factor=75f0, kernel_size=(21, 21), border="reflect")

Apply the default Gaussian smoothing to an ERP image.

# Arguments
- `image`: ERP image in `trials x timepoints` layout.
- `sigma_factor`: Smoothing factor.
- `kernel_size`: Odd Gaussian kernel dimensions.
- `border`: ImageFiltering border handling mode.

# Returns
- `Matrix{Float32}`: Smoothed image with the same size as `image`.
"""
function smooth_image(image; sigma_factor = 75f0, kernel_size = (21, 21), border = "reflect")
    return smooth_image_for_target(
        image,
        DEFAULT_IMAGE_TARGET_SIZE;
        sigma_factor = sigma_factor,
        kernel_size = kernel_size,
        border = border,
    )
end

"""
    resize_image(image; target_size=(64, 64), method=Linear())

Resize an ERP image to the model-ready target size.

# Arguments
- `image`: ERP image matrix.
- `target_size`: Output size as `(rows, columns)`.
- `method`: Interpolation method, defaulting to `Interpolations.Linear()`.

# Returns
- `Matrix{Float32}`: Resized image with size `target_size`.
"""
function resize_image(image; target_size = DEFAULT_IMAGE_TARGET_SIZE, method = Linear())
    # Normalize target dimensions before handing them to ImageTransformations.
    target_size = Tuple(Int.(target_size))
    return Matrix{Float32}(imresize(Float32.(image), target_size; method = method))
end
