# Build an ERP image (trials x time) with sorting, optional dropout, and optional z-score.
function erpimage_sorted(data_all::AbstractMatrix, sortvalues;
        rng::AbstractRNG = MersenneTwister(time_ns()),
        dropout_trials_rate::Real = 0.0,
        zscore_timepoints::Bool = true)

    n_time, n_trials = size(data_all, 1), size(data_all, 2)
    if n_time == 0 || n_trials == 0
        throw(ArgumentError("erpimage_sorted received empty data; cannot proceed."))
    end

    if sortvalues === nothing
        throw(ArgumentError("sortvalues must be provided for all patterns (including :no_class)."))
    end
    if length(sortvalues) != n_trials
        throw(ArgumentError("sortvalues length does not match number of trials; ensure each trial has a sort value."))
    end
    idx = sortperm(sortvalues)
    data_sorted = data_all[:, idx]
    if size(data_sorted, 1) == 0 || size(data_sorted, 2) == 0
        throw(ArgumentError("erpimage_sorted produced empty sorted data; cannot proceed."))
    end

    data_sorted, dropout_info = dropout_trials(data_sorted, rng, dropout_trials_rate)

    if zscore_timepoints
        data_sorted = Float32.(Normalization.normalize(Float64.(data_sorted), ZScore; dims = 2))
    end

    img = Float32.(permutedims(data_sorted, (2, 1)))

    return img, dropout_info
end

# Crop the time axis by independently sampling start/end offsets in ms.
function crop_time_axis(data_time_trials::AbstractMatrix, rng::AbstractRNG, crop_start_dist, crop_end_dist,
        sampling_rate::Real)
    n_time = size(data_time_trials, 1)
    if n_time <= 1
        return data_time_trials, (crop_start_ms = 0, crop_end_ms = 0, crop_start_samples = 0, crop_end_samples = 0)
    end

    start_ms = Int(round(max(0, rand(Random.Xoshiro(time_ns()), crop_start_dist))))
    end_ms = Int(round(max(0, rand(Random.Xoshiro(time_ns()), crop_end_dist))))
    start_samples = Int(round(start_ms * sampling_rate / 1000))
    end_samples = Int(round(end_ms * sampling_rate / 1000))

    start_samples = clamp(start_samples, 0, n_time - 1)
    end_samples = clamp(end_samples, 0, n_time - 1 - start_samples)

    t_start = 1 + start_samples
    t_end = n_time - end_samples
    cropped = data_time_trials[t_start:t_end, :]
    return cropped, (crop_start_ms = start_ms, crop_end_ms = end_ms,
        crop_start_samples = start_samples, crop_end_samples = end_samples)
end

# Randomly drop trials using a rounded rate.
function dropout_trials(data_time_trials::AbstractMatrix, rng::AbstractRNG, dropout_trials_rate::Real)
    rng = Random.Xoshiro(time_ns())
    n_time, n_trials = size(data_time_trials, 1), size(data_time_trials, 2)
    if n_time == 0 || n_trials == 0
        throw(ArgumentError("dropout_trials received empty data; cannot proceed."))
    end

    drop_trials = clamp(Int(round(dropout_trials_rate)), 0, max(0, n_trials - 1))

    keep_trials = trues(n_trials)
    if drop_trials > 0
        drop_idx = randperm(Random.Xoshiro(time_ns()), n_trials)[1:drop_trials]
        keep_trials[drop_idx] .= false
    end

    kept_trials = findall(keep_trials)
    if isempty(kept_trials)
        throw(ArgumentError("dropout_trials removed all trials; cannot proceed."))
    end

    dropped = data_time_trials[:, kept_trials]
    return dropped, (dropout_trials = drop_trials,)
end
