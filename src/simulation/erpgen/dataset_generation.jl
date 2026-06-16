# Periodically log progress for threaded execution.
function start_generation_logger!(reps_done::Threads.Atomic{Int}, n_per_pattern::Int, n_classes::Int, progress_every::Int)
    return maybe_diag(:start_generation_logger!) do
        progress_every <= 0 && return nothing
        return @async begin
            last = 0
            while true
                done = reps_done[]
                if done >= n_per_pattern
                    println("Progress: ", n_per_pattern, "/", n_per_pattern,
                        " reps (per class=", n_per_pattern, ", total images=", n_per_pattern * n_classes, ")")
                    break
                elseif done - last >= progress_every
                    println("Progress: ", done, "/", n_per_pattern,
                        " reps (per class=", done, ", total images=", done * n_classes, ")")
                    last = done
                end
                sleep(0.25)
            end
        end
    end
end

function _patterns_with_no_class(patterns::AbstractVector{Symbol})
    return :no_class in patterns ? collect(patterns) : vcat(collect(patterns), [:no_class])
end

# Stage 1: Simulate raw ERP data.
function simulate_raw_erp(config::GenerationConfig, rng::AbstractRNG)
    return maybe_diag(:simulate_raw_erp) do
        simulate_rng = fresh_rng(rng)
        sim = config.sim
        comp = config.components
        pat = config.patterns

        mu = max(1, time_seeded_rand(sim.mu_dist))
        sigma = max(0.01, time_seeded_rand(sim.sigma_dist))
        n_trials = Int(ceil(time_seeded_rand(sim.n_trials_dist)))
        n_trials = max(100, n_trials)
        n_trials += isodd(n_trials) ? 1 : 0
        sampling_rate = max(1, Int(round(time_seeded_rand(sim.sampling_rate_dist))))
        epoch_duration_s = time_seeded_rand(sim.epoch_duration_dist)
        epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))

        sim_result = simulate_erp_trials(
            simulate_rng, mu, sigma, n_trials, sampling_rate, epoch_duration_s,
            comp.p100_width_dist, comp.p100_window_offset_dist, comp.p100_n170_gap_dist, comp.n170_p300_gap_dist,
            comp.n170_width_dist, comp.p300_width_dist, comp.p1_beta_dist, comp.p3_beta_dist,
            comp.n1_beta1_dist, comp.n1_beta2_dist, comp.n1_beta3_dist,
            comp.componentA_amp_dist, comp.componentB_amp_dist, comp.componentC_amp_dist,
            comp.tilted_bar_hanning_length_dist,
            comp.one_sided_fan_duration_divisor_dist,
            comp.one_sided_fan_log_mu_offset_dist,
            comp.one_sided_fan_log_sigma_dist,
            comp.one_sided_fan_support_max_dist,
            config.noise,
            pat.loaded_patterns, pat.covariate_dists, pat.diverging_bar_levels,
        )

        raw_size = (size(sim_result.data, 2), size(sim_result.data, 1))

        params = (
            mu = mu,
            sigma = sigma,
            epoch_duration_s = epoch_duration_s,
            sampling_rate = sampling_rate,
            erpimage_raw_size = raw_size,
            noise = map(n -> string(typeof(n)), config.noise.noise_pool),
            loaded_patterns = collect(pat.loaded_patterns),
            noiselevels = sim_result.noiselevels,
            p1_beta = sim_result.p1_beta,
            p3_beta = sim_result.p3_beta,
            n1_betas = sim_result.n1_betas,
            componentA_amp = sim_result.component_amps.componentA_amp,
            componentB_amp = sim_result.component_amps.componentB_amp,
            componentC_amp = sim_result.component_amps.componentC_amp,
            p100_width = sim_result.hanning_params.p100_width,
            p100_window_center = sim_result.hanning_params.p100_window_center,
            p100_offset = sim_result.hanning_params.p100_offset,
            p100_n170_gap = sim_result.hanning_params.p100_n170_gap,
            n170_width = sim_result.hanning_params.n170_width,
            n170_window_center = sim_result.hanning_params.n170_window_center,
            n170_offset = sim_result.hanning_params.n170_offset,
            n170_p300_gap = sim_result.hanning_params.n170_p300_gap,
            p300_width = sim_result.hanning_params.p300_width,
            p300_window_center = sim_result.hanning_params.p300_window_center,
            p300_offset = sim_result.hanning_params.p300_offset,
            tilted_bar_hanning_length = sim_result.basis_shape_params.tilted_bar_hanning_length,
            one_sided_fan_duration_divisor = sim_result.basis_shape_params.one_sided_fan_duration_divisor,
            one_sided_fan_log_mu_offset = sim_result.basis_shape_params.one_sided_fan_log_mu_offset,
            one_sided_fan_log_sigma = sim_result.basis_shape_params.one_sided_fan_log_sigma,
            one_sided_fan_support_max = sim_result.basis_shape_params.one_sided_fan_support_max,
        )

        return (data = sim_result.data, events = sim_result.events, params = params)
    end
end

# Stage 2: Apply trial dropout (before cropping).
function apply_trial_dropout(data::AbstractMatrix, events, processing::ProcessingConfig, rng::AbstractRNG)
    return maybe_diag(:apply_trial_dropout) do
        dropout_trials_rate = time_seeded_rand(processing.dropout_trials_rate_dist)
        dropout_trials_rate = max(0, round(Int, dropout_trials_rate))

        n_trials = size(data, 2)
        drop_trials = clamp(Int(dropout_trials_rate), 0, max(0, n_trials - 1))

        keep_trials = trues(n_trials)
        if drop_trials > 0
            drop_idx = time_seeded_randperm(n_trials)[1:drop_trials]
            keep_trials[drop_idx] .= false
        end

        kept_trials = findall(keep_trials)
        if isempty(kept_trials)
            throw(ArgumentError("apply_trial_dropout removed all trials; cannot proceed."))
        end

        dropped_data = data[:, kept_trials]
        dropped_events = events[kept_trials, :]

        params = (
            dropout_trials_rate = dropout_trials_rate,
            dropout_trials = drop_trials,
        )

        return (data = dropped_data, events = dropped_events, params = params)
    end
end

# Stage 3: Crop time window.
function apply_cropping(data::AbstractMatrix, processing::ProcessingConfig, rng::AbstractRNG, sampling_rate::Real)
    return maybe_diag(:apply_cropping) do
        cropped, crop_info = crop_time_window(data, fresh_rng(rng), processing, sampling_rate)
        return (data = cropped, params = crop_info)
    end
end

# Post-processing: apply z-score normalization after resize.
function apply_zscore(images::Dict{Symbol, Matrix{Float32}}, processing::ProcessingConfig)
    return maybe_diag(:apply_zscore) do
        if !processing.zscore_timepoints
            return images
        end

        normalized = Dict{Symbol, Matrix{Float32}}()
        for (pname, img) in images
            normalized[pname] = maybe_diag(:zscore_timepoints) do
                # After transposition, rows=trials and cols=timepoints.
                Float32.(Normalization.normalize(Float64.(img), ZScore; dims = 1))
            end
        end
        return normalized
    end
end

# Stage 4: Sort trials by pattern-specific sort values.
function sort_by_patterns(data::AbstractMatrix, events, patterns::Vector{Symbol}, rng::AbstractRNG)
    return maybe_diag(:sort_by_patterns) do
        n_trials = size(data, 2)
        images = Dict{Symbol, Matrix{Float32}}()

        for pname in patterns
            sortvalues = pattern_sort_values(pname, events, fresh_rng(rng))
            if sortvalues === nothing
                throw(ArgumentError("sortvalues must be provided for all patterns (including :no_class)."))
            end
            if length(sortvalues) != n_trials
                throw(ArgumentError("sortvalues length does not match number of trials; ensure each trial has a sort value."))
            end

            idx = sortperm(sortvalues)
            data_sorted = data[:, idx]
            images[pname] = Float32.(permutedims(data_sorted, (2, 1)))
        end

        return images
    end
end

# Stage 5: Image processing (filter + resize).
function process_images(images::Dict{Symbol, Matrix{Float32}}, processing::ProcessingConfig)
    return maybe_diag(:process_images) do
        processed = Dict{Symbol, Matrix{Float32}}()
        processed_size = nothing

        for (pname, img) in images
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

            processed_size === nothing && (processed_size = size(filtered))

            resized = maybe_diag(:resize_img) do
                if size(filtered, 1) == 0 || size(filtered, 2) == 0
                    throw(ArgumentError("process_images received empty image; cannot resize."))
                end
                out = Float32.(filtered)
                if processing.target_height <= 0 || processing.target_width <= 0 ||
                        size(out) == (processing.target_height, processing.target_width)
                    return out
                end
                return Float32.(imresize(out, (processing.target_height, processing.target_width);
                    method = processing.resize_method))
            end

            processed[pname] = resized
        end

        processed_size === nothing && (processed_size = (0, 0))
        return (images = processed, processed_size = processed_size)
    end
end

# Stage 7: Create variants.
function create_variants(images::Dict{Symbol, Matrix{Float32}}, patterns::Vector{Symbol})
    return maybe_diag(:create_variants) do
        variants = Vector{NamedTuple}(undef, length(patterns) * VARIANT_COUNT)
        idx = 1
        for pname in patterns
            base_img = images[pname]
            reversed_img = reverse(base_img, dims = 1)
            variant_imgs = (base_img, reversed_img, -base_img, -reversed_img)
            for (vidx, spec) in enumerate(VARIANT_SPECS)
                variants[idx] = (
                    image = variant_imgs[vidx],
                    pattern = pname,
                    variant = spec.name,
                    trial_order = spec.trial_order,
                    inverted = spec.inverted,
                )
                idx += 1
            end
        end
        return variants
    end
end

# Stage 8: Attach metadata.
function attach_metadata(variants::Vector{NamedTuple}, all_params::NamedTuple)
    return maybe_diag(:attach_metadata) do
        n = length(variants)
        images = Vector{Matrix{Float32}}(undef, n)
        labels = Vector{Symbol}(undef, n)
        metadata = Vector{NamedTuple}(undef, n)

        for i in 1:n
            v = variants[i]
            images[i] = v.image
            labels[i] = v.pattern
            metadata[i] = merge(all_params, (
                pattern = v.pattern,
                variant = v.variant,
                trial_order = v.trial_order,
                inverted = v.inverted,
            ))
        end

        return images, labels, metadata
    end
end

# Pipeline runner (single repetition).
function run_pipeline(config::GenerationConfig, rng::AbstractRNG, patterns_with_no_class::Vector{Symbol})
    return maybe_diag(:run_pipeline) do
        raw = simulate_raw_erp(config, fresh_rng(rng))
        dropped = apply_trial_dropout(raw.data, raw.events, config.processing, fresh_rng(rng))
        cropped = apply_cropping(dropped.data, config.processing, fresh_rng(rng), raw.params.sampling_rate)
        sorted = sort_by_patterns(cropped.data, dropped.events, patterns_with_no_class, fresh_rng(rng))
        processed = process_images(sorted, config.processing)
        normalized = apply_zscore(processed.images, config.processing)
        variants = create_variants(normalized, patterns_with_no_class)

        all_params = merge(raw.params, dropped.params, cropped.params, (
            erpimage_processed_size = processed.processed_size,
        ))

        return attach_metadata(variants, all_params)
    end
end

function _flatten_results(results::Vector{Tuple}, total::Int)
    images = Vector{Matrix{Float32}}(undef, total)
    labels = Vector{Symbol}(undef, total)
    metadata = Vector{NamedTuple}(undef, total)

    idx = 1
    for (imgs, lbls, metas) in results
        for j in eachindex(imgs)
            images[idx] = imgs[j]
            labels[idx] = lbls[j]
            metadata[idx] = metas[j]
            idx += 1
        end
    end

    return images, labels, metadata
end

# Generate ERP images (single process, optional threading).
function generate_erp_images(n_per_pattern::Int;
        config::GenerationConfig = GenerationConfig(),
        enable_diagnostics::Bool = false)
    DIAGNOSTICS_ENABLED[] = enable_diagnostics
    enable_diagnostics!(enable_diagnostics; propagate = false)
    if enable_diagnostics
        reset_diagnostics!()
    end
    verify_unfoldsim_version!()

    result = maybe_diag(:generate_erp_images) do
        threaded = config.runtime.threaded
        n_threads = threaded ? Threads.nthreads() : 1

        if threaded && n_threads < 16
            error("threaded=true requires 16 Julia threads; restart the kernel with JULIA_NUM_THREADS=16.")
        end

        BLAS.set_num_threads(config.runtime.blas_threads)

        patterns = _patterns_with_no_class(config.patterns.patterns)
        n_classes = length(patterns) * VARIANT_COUNT
        total = n_classes * n_per_pattern

        results_per_thread = [Vector{Tuple}() for _ in 1:n_threads]

        progress_counter = config.runtime.show_progress && config.runtime.progress_every > 0 ?
            Threads.Atomic{Int}(0) : nothing
        progress_task = progress_counter === nothing ? nothing :
            start_generation_logger!(progress_counter, n_per_pattern, n_classes, config.runtime.progress_every)

        chunk_size = cld(n_per_pattern, n_threads)

        function process_chunk!(chunk_id::Int, start_idx::Int, end_idx::Int)
            local_results = results_per_thread[chunk_id]
            for _ in start_idx:end_idx
                result = run_pipeline(config, fresh_rng(), patterns)
                push!(local_results, result)

                if progress_counter !== nothing
                    Threads.atomic_add!(progress_counter, 1)
                end
            end
        end

        if threaded
            Threads.@threads :static for chunk_id in 1:n_threads
                start_idx = (chunk_id - 1) * chunk_size + 1
                end_idx = min(chunk_id * chunk_size, n_per_pattern)
                if start_idx <= end_idx
                    process_chunk!(chunk_id, start_idx, end_idx)
                end
            end
        else
            process_chunk!(1, 1, n_per_pattern)
        end

        progress_task !== nothing && wait(progress_task)

        all_results = vcat(results_per_thread...)
        images, labels, metadata = _flatten_results(all_results, total)

        return images, labels, metadata
    end

    if enable_diagnostics
        print_diagnostics_tree()
    end

    return result
end
