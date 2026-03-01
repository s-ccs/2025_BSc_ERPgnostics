# Build an ERP image (trials x time) with sorting, optional dropout, and optional z-score.
function build_sorted_erpimage(data_all::AbstractMatrix, sortvalues;
        rng::AbstractRNG = fresh_rng(),
        dropout_trials_rate::Real = 0.0,
        zscore_timepoints::Bool = true)
    return maybe_diag(:build_sorted_erpimage) do
        n_time, n_trials = size(data_all, 1), size(data_all, 2)
        if n_time == 0 || n_trials == 0
            throw(ArgumentError("build_sorted_erpimage received empty data; cannot proceed."))
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
            throw(ArgumentError("build_sorted_erpimage produced empty sorted data; cannot proceed."))
        end

        data_sorted, dropout_info = apply_trial_dropout(data_sorted, fresh_rng(rng), dropout_trials_rate)

        if zscore_timepoints
            data_sorted = maybe_diag(:zscore_timepoints) do
                Float32.(Normalization.normalize(Float64.(data_sorted), ZScore; dims = 2))
            end
        end

        img = Float32.(permutedims(data_sorted, (2, 1)))

        return img, dropout_info
    end
end

# Crop the time axis by independently sampling start/end offsets in ms.
function crop_time_window(data_time_trials::AbstractMatrix, rng::AbstractRNG, crop_start_dist, crop_end_dist,
        sampling_rate::Real)
    return maybe_diag(:crop_time_window) do
        n_time = size(data_time_trials, 1)
        if n_time <= 1
            return data_time_trials, (crop_start_ms = 0, crop_end_ms = 0, crop_start_samples = 0, crop_end_samples = 0)
        end

        crop_rng = fresh_rng(rng)
        start_ms = Int(round(max(0, rand(crop_rng, crop_start_dist))))
        end_ms = Int(round(max(0, rand(crop_rng, crop_end_dist))))
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
end

function crop_time_window(data_time_trials::AbstractMatrix, rng::AbstractRNG,
        processing::ProcessingConfig, sampling_rate::Real)
    return maybe_diag(:crop_time_window_processing) do
        return crop_time_window(data_time_trials, rng,
            processing.crop_start_dist, processing.crop_end_dist, sampling_rate)
    end
end

# Randomly drop trials using a rounded rate.
function apply_trial_dropout(data_time_trials::AbstractMatrix, rng::AbstractRNG, dropout_trials_rate::Real)
    return maybe_diag(:apply_trial_dropout) do
        n_time, n_trials = size(data_time_trials, 1), size(data_time_trials, 2)
        if n_time == 0 || n_trials == 0
            throw(ArgumentError("apply_trial_dropout received empty data; cannot proceed."))
        end

        dropout_rng = fresh_rng(rng)
        drop_trials = clamp(Int(round(dropout_trials_rate)), 0, max(0, n_trials - 1))

        keep_trials = trues(n_trials)
        if drop_trials > 0
            drop_idx = randperm(dropout_rng, n_trials)[1:drop_trials]
            keep_trials[drop_idx] .= false
        end

        kept_trials = findall(keep_trials)
        if isempty(kept_trials)
            throw(ArgumentError("apply_trial_dropout removed all trials; cannot proceed."))
        end

        dropped = data_time_trials[:, kept_trials]
        return dropped, (dropout_trials = drop_trials,)
    end
end

# Build per-pattern ERP images (sorted/filtered/resized); updates images/labels/metadata in-place.
function render_pattern_images!(images::AbstractVector{Matrix{Float32}},
        labels::AbstractVector{Symbol},
        metadata::AbstractVector{NamedTuple},
        base::Int,
        data::AbstractMatrix{<:Real},
        sim_result,
        mu::Real,
        sigma::Real,
        epoch_duration_s::Real,
        sampling_rate::Int,
        noise::NoiseConfig,
        processing::ProcessingConfig,
        crop_info::NamedTuple,
        generated_size::Tuple{Int, Int},
        patterns::AbstractVector{Symbol};
        rng::AbstractRNG = fresh_rng())
    return maybe_diag(:render_pattern_images!) do
        # Sample concrete dropout counts once per simulation.
        dropout_rng = fresh_rng(rng)
        dropout_trials_rate = rand(dropout_rng, processing.dropout_trials_rate_dist)
        dropout_trials_rate = max(0, round(Int, dropout_trials_rate))

        # Render each pattern with its own sorting rule and metadata.
        for (pidx, pname) in enumerate(patterns)
            pattern_rng = fresh_rng(rng)
            sortvalues = pattern_sort_values(pname, sim_result.events, pattern_rng)
            img, dropout_info = build_sorted_erpimage(data, sortvalues;
                rng = fresh_rng(pattern_rng),
                dropout_trials_rate = dropout_trials_rate,
                zscore_timepoints = processing.zscore_timepoints,
            )

            # Low-pass prefilter before downsampling to reduce aliasing artifacts.
            filtered = img
            if processing.resize_antialias && processing.low_pass_factor > 0 && min(size(filtered)...) > 1
                needs_resize = processing.target_height > 0 && processing.target_width > 0 &&
                               size(filtered) != (processing.target_height, processing.target_width)
                if needs_resize
                    filtered = maybe_diag(:low_pass_filter) do
                        antialias_sigma = (processing.low_pass_factor * size(filtered, 1) / processing.target_height,
                            processing.low_pass_factor * size(filtered, 2) / processing.target_width)
                        kernel = KernelFactors.gaussian(antialias_sigma)
                        Float32.(imfilter(filtered, kernel, FILTER_BORDER))
                    end
                end
            end
            resized = maybe_diag(:resize_img) do
                if size(filtered, 1) == 0 || size(filtered, 2) == 0
                    throw(ArgumentError("render_pattern_images! received empty image; cannot resize."))
                end
                out = Float32.(filtered)
                if processing.target_height <= 0 || processing.target_width <= 0 ||
                        size(out) == (processing.target_height, processing.target_width)
                    return out
                end
                return Float32.(imresize(out, (processing.target_height, processing.target_width);
                    method = processing.resize_method))
            end

            raw_size = generated_size
            processed_size = size(filtered)

            # Build trial-order variants (normal/reversed) and inverted counterparts.
            # For :no_class, the randomization happens in pattern_sort_values before filtering.
            base_img = resized
            reversed_img = reverse(resized, dims = 1)
            variant_imgs = (base_img, reversed_img, -base_img, -reversed_img)
            for (vidx, spec) in enumerate(VARIANT_SPECS)
                idx = base + (pidx - 1) * VARIANT_COUNT + vidx
                images[idx] = variant_imgs[vidx]
                labels[idx] = pname
                metadata[idx] = (
                    pattern = pname,
                    variant = spec.name,
                    trial_order = spec.trial_order,
                    inverted = spec.inverted,
                    mu = mu,
                    sigma = sigma,
                    epoch_duration_s = epoch_duration_s,
                    sampling_rate = sampling_rate,
                    crop_start_ms = crop_info.crop_start_ms,
                    crop_end_ms = crop_info.crop_end_ms,
                    crop_start_samples = crop_info.crop_start_samples,
                    crop_end_samples = crop_info.crop_end_samples,
                    dropout_trials_rate = dropout_trials_rate,
                    dropout_trials = dropout_info.dropout_trials,
                    erpimage_raw_size = raw_size,
                    erpimage_processed_size = processed_size,
                    noise = map(n -> string(typeof(n)), noise.noise_pool),
                    noiselevels = sim_result.noiselevels,
                    p1_beta = sim_result.p1_beta,
                    p3_beta = sim_result.p3_beta,
                    n1_betas = sim_result.n1_betas,
                    p100_width = sim_result.hanning_params.p100_width,
                    p100_window_center = sim_result.hanning_params.p100_window_center,
                    p100_offset = sim_result.hanning_params.p100_offset,
                    p300_width = sim_result.hanning_params.p300_width,
                    p300_window_center = sim_result.hanning_params.p300_window_center,
                    p300_offset = sim_result.hanning_params.p300_offset,
                    n170_width = sim_result.hanning_params.n170_width,
                    n170_window_center = sim_result.hanning_params.n170_window_center,
                    n170_offset = sim_result.hanning_params.n170_offset,
                )
            end
        end
        return nothing
    end
end
