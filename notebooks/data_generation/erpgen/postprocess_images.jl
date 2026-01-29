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
        rng::AbstractRNG = MersenneTwister(time_ns()))

    # Sample concrete dropout counts once per simulation.
    dropout_trials_rate = rand(rng, processing.dropout_trials_rate_dist)
    dropout_trials_rate = max(0, round(Int, dropout_trials_rate))

    # Render each pattern with its own sorting rule and metadata.
    for (pidx, pname) in enumerate(patterns)
        sortvalues = pattern_sort_values(pname, sim_result.events, rng)
        img, dropout_info = diag_call(:build_sorted_erpimage) do
            build_sorted_erpimage(data, sortvalues;
                rng = rng,
                dropout_trials_rate = dropout_trials_rate,
                zscore_timepoints = processing.zscore_timepoints,
            )
        end

        # Low-pass prefilter before downsampling to reduce aliasing artifacts.
        filtered = img
        if processing.resize_antialias && processing.low_pass_factor > 0 && min(size(filtered)...) > 1
            needs_resize = processing.target_height > 0 && processing.target_width > 0 &&
                           size(filtered) != (processing.target_height, processing.target_width)
            if needs_resize
                antialias_sigma = (processing.low_pass_factor * size(filtered, 1) / processing.target_height,
                    processing.low_pass_factor * size(filtered, 2) / processing.target_width)
                kernel = KernelFactors.gaussian(antialias_sigma)
                filtered = Float32.(imfilter(filtered, kernel, FILTER_BORDER))
            end
        end
        resized = diag_call(:resize_img) do
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
                p100_offset = sim_result.hanning_params.p100_offset,
                p300_width = sim_result.hanning_params.p300_width,
                p300_offset = sim_result.hanning_params.p300_offset,
                n170_width = sim_result.hanning_params.n170_width,
                n170_offset = sim_result.hanning_params.n170_offset,
            )
        end
    end
    return nothing
end
