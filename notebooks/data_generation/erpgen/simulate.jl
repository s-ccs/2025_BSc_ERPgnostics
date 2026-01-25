# Simulate ERP data (time x trials); outputs data, events, and noise levels.
function simulate_erp(rng::AbstractRNG = MersenneTwister(time_ns()),
        mu::Real = 3.2,
        sigma::Real = 0.5;
        epoch_duration_s::Real = 1.0,
        sampling_rate::Real = 100,
        condition_levels::Int = 8,
        noise_pool::AbstractVector{<:UnfoldSim.AbstractNoise} = DEFAULT_NOISE_POOL,
        noiselevel_dists::AbstractDict{<:DataType, <:Distribution} = DEFAULT_NOISELEVEL_DISTS,
        p100_width_dist::Distribution = Normal(0.1, 0.015),
        p100_offset_dist::Distribution = Normal(0.1, 0.015),
        p300_width_dist::Distribution = Normal(0.3, 0.045),
        p300_offset_dist::Distribution = Normal(0.3, 0.045),
        n170_width_dist::Distribution = Normal(0.15, 0.0225),
        n170_offset_dist::Distribution = Normal(0.17, 0.0255),
        p1_beta_dist::Distribution = Normal(5.0, 1.0),
        p3_beta_dist::Distribution = Normal(5.0, 0.75),
        n1_beta1_dist::Distribution = Normal(5.0, 0.75),
        n1_beta2_dist::Distribution = Normal(3.0, 0.45),
        n1_beta3_dist::Distribution = Normal(2.0, 0.3),
        componentA_amp_dist::Distribution = Normal(5.0, 1.0),
        componentB_amp_dist::Distribution = Normal(-10.0, 1.0),
        componentC_amp_dist::Distribution = Normal(5.0, 1.0),
        one_sided_fan_duration_start_dist::Distribution = Normal(20.0, 3.0),
        one_sided_fan_duration_end_dist::Distribution = Normal(100.0, 15.0),
        two_sided_fan_duration_start_dist::Distribution = Normal(10.0, 1.5),
        two_sided_fan_duration_end_dist::Distribution = Normal(30.0, 4.5),
        tilted_bar_duration_start_dist::Distribution = Normal(5.0, 0.75),
        tilted_bar_duration_end_dist::Distribution = Normal(40.0, 6.0),
        hourglass_continuous_start_dist::Distribution = Normal(-2.0, 0.3),
        hourglass_continuous_end_dist::Distribution = Normal(2.0, 0.3))
    return diag_call(:simulate_6patterns) do
        with_logger(NullLogger()) do
            data, simulated_events, noiselevels, p1_beta, p3_beta, n1_betas, hanning_params = simulate_6patterns(mu, sigma;
                epoch_duration_s = epoch_duration_s,
                sampling_rate = sampling_rate,
                condition_levels = condition_levels,
                noise_pool = noise_pool,
                noiselevel_dists = noiselevel_dists,
                p100_width_dist = p100_width_dist,
                p100_offset_dist = p100_offset_dist,
                p300_width_dist = p300_width_dist,
                p300_offset_dist = p300_offset_dist,
                n170_width_dist = n170_width_dist,
                n170_offset_dist = n170_offset_dist,
                p1_beta_dist = p1_beta_dist,
                p3_beta_dist = p3_beta_dist,
                n1_beta1_dist = n1_beta1_dist,
                n1_beta2_dist = n1_beta2_dist,
                n1_beta3_dist = n1_beta3_dist,
                componentA_amp_dist = componentA_amp_dist,
                componentB_amp_dist = componentB_amp_dist,
                componentC_amp_dist = componentC_amp_dist,
                one_sided_fan_duration_start_dist = one_sided_fan_duration_start_dist,
                one_sided_fan_duration_end_dist = one_sided_fan_duration_end_dist,
                two_sided_fan_duration_start_dist = two_sided_fan_duration_start_dist,
                two_sided_fan_duration_end_dist = two_sided_fan_duration_end_dist,
                tilted_bar_duration_start_dist = tilted_bar_duration_start_dist,
                tilted_bar_duration_end_dist = tilted_bar_duration_end_dist,
                hourglass_continuous_start_dist = hourglass_continuous_start_dist,
                hourglass_continuous_end_dist = hourglass_continuous_end_dist,
                rng = rng,
            )
            return data, simulated_events, noiselevels, p1_beta, p3_beta, n1_betas, hanning_params
        end
    end
end

# Build per-pattern ERP images (sorted/filtered/resized); updates images/labels/metadata in-place.
function build_pattern_images!(images::AbstractVector{Matrix{Float32}},
        labels::AbstractVector{Symbol},
        metadata::AbstractVector{NamedTuple},
        base::Int,
        data::AbstractMatrix{<:Real},
        events::AbstractDataFrame,
        mu::Real,
        sigma::Real,
        epoch_duration_s::Real,
        sampling_rate::Real,
        condition_levels::Int,
        noise_pool::AbstractVector{<:UnfoldSim.AbstractNoise},
        noiselevels::AbstractDict{Symbol, <:Real},
        p1_beta::Real,
        p3_beta::Real,
        n1_betas::Tuple{<:Real, <:Real, <:Real},
        hanning_params::NamedTuple,
        crop_info::NamedTuple,
        generated_size::Tuple{Int, Int},
        dropout_trials_rate_dist::Distribution = Normal(2000, 250);
        rng::AbstractRNG = MersenneTwister(time_ns()),
        zscore_timepoints::Bool = true,
        gauss_smooth::Bool = false,
        filter_sigma::Real = 1.0,
        gauss_kernel_len::Int = 20,
        resize_antialias::Bool = false,
        antialias_factor::Real = 0.75,
        resize_method = Interpolations.Linear(),
        target_height::Int = 64,
        target_width::Int = 64)

    # Sample concrete dropout counts once per simulation.
    dropout_trials_rate = rand(Random.Xoshiro(time_ns()), dropout_trials_rate_dist)
    dropout_trials_rate = max(0, round(Int, dropout_trials_rate))
    # Render each pattern with its own sorting rule and metadata.
    for (pidx, pname) in enumerate(PATTERN_NAMES)
        sortvalues = sortvalues_for_pattern(pname, events, rng)
        img, dropout_info = diag_call(:erpimage_sorted) do
            erpimage_sorted(data, sortvalues;
                rng = rng,
                dropout_trials_rate = dropout_trials_rate,
                zscore_timepoints = zscore_timepoints,
            )
        end

        # Apply optional Gaussian smoothing and low-pass prefilter (prevents aliasing), then resize.
        filtered = img
        if gauss_smooth && filter_sigma > 0 && min(size(filtered)...) > 1
            kernel_len = isodd(gauss_kernel_len) ? gauss_kernel_len : gauss_kernel_len + 1
            kernel2d = KernelFactors.gaussian((filter_sigma, filter_sigma), (kernel_len, kernel_len))
            filtered = Float32.(imfilter(filtered, kernel2d, FILTER_BORDER))
        end
        # Low-pass prefilter before downsampling to reduce aliasing artifacts.
        if resize_antialias && antialias_factor > 0 && min(size(filtered)...) > 1
            needs_resize = target_height > 0 && target_width > 0 &&
                           size(filtered) != (target_height, target_width)
            if needs_resize
                antialias_sigma = (antialias_factor * size(filtered, 1) / target_height,
                    antialias_factor * size(filtered, 2) / target_width)
                kernel = KernelFactors.gaussian(antialias_sigma)
                filtered = Float32.(imfilter(filtered, kernel, FILTER_BORDER))
            end
        end
        resized = diag_call(:resize_img) do
            if size(filtered, 1) == 0 || size(filtered, 2) == 0
                throw(ArgumentError("build_pattern_images! received empty image; cannot resize."))
            end
            out = Float32.(filtered)
            if target_height <= 0 || target_width <= 0 || size(out) == (target_height, target_width)
                return out
            end
            return Float32.(imresize(out, (target_height, target_width); method = resize_method))
        end

        raw_size = generated_size
        processed_size = size(filtered)

        # Build trial-order variants (normal/reversed) and inverted counterparts.
        # For :no_class, the randomization happens in sortvalues_for_pattern before filtering.
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
                condition_levels = condition_levels,
                noise = map(n -> string(typeof(n)), noise_pool),
                noiselevels = noiselevels,
                p1_beta = p1_beta,
                p3_beta = p3_beta,
                n1_betas = n1_betas,
                p100_width = hanning_params.p100_width,
                p100_offset = hanning_params.p100_offset,
                p300_width = hanning_params.p300_width,
                p300_offset = hanning_params.p300_offset,
                n170_width = hanning_params.n170_width,
                n170_offset = hanning_params.n170_offset,
            )
        end
    end
    return nothing
end
