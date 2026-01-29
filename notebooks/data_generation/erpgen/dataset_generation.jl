# Periodically log progress for threaded execution.
function start_generation_logger!(reps_done::Threads.Atomic{Int}, n_per_pattern::Int, n_classes::Int, progress_every::Int)
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

# Generate ERP images (single process, optional threading) with legacy keyword args.
function generate_erp_images(; n_per_pattern::Int = 10,
        mu_dist::Distribution = Normal(3.2, 0.3),
        sigma_dist::Distribution = Normal(0.5, 0.1),
        epoch_duration_dist::Distribution = Normal(1.0, 0.25),
        sampling_rate_dist::Distribution = Normal(100, 5),
        dropout_trials_rate_dist::Distribution = DEFAULT_DROPOUT_RATE_DIST,
        n_trials_dist::Distribution = DEFAULT_N_TRIALS_DIST,
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
        patterns = PATTERN_NAMES,
        covariate_dists = default_pattern_covariates(),
        target_height::Int = 64,
        target_width::Int = 64,
        zscore_timepoints::Bool = true,
        resize_antialias::Bool = true,
        low_pass_factor::Real = 0.75,
        resize_method = Interpolations.Linear(),
        noise_pool = DEFAULT_NOISE_POOL,
        noiselevel_dists = DEFAULT_NOISELEVEL_DISTS,
        crop_start_dist = DEFAULT_CROP_START_DIST,
        crop_end_dist = DEFAULT_CROP_END_DIST,
        threaded::Bool = false,
        blas_threads::Int = 1,
        progress_every::Int = 10)

    ensure_latest_unfoldsim!(propagate = false)

    patterns = collect(patterns)

    noise = NoiseConfig(
        noise_pool = noise_pool,
        noiselevel_dists = noiselevel_dists,
    )

    processing = ProcessingConfig(
        dropout_trials_rate_dist = dropout_trials_rate_dist,
        crop_start_dist = crop_start_dist,
        crop_end_dist = crop_end_dist,
        zscore_timepoints = zscore_timepoints,
        resize_antialias = resize_antialias,
        low_pass_factor = low_pass_factor,
        resize_method = resize_method,
        target_height = target_height,
        target_width = target_width,
    )

    n_classes = length(patterns) * VARIANT_COUNT
    total = n_classes * n_per_pattern

    images = Vector{Matrix{Float32}}(undef, total)
    labels = Vector{Symbol}(undef, total)
    metadata = Vector{NamedTuple}(undef, total)

    if threaded
        n = Threads.nthreads()
        required = 16
        if n < required
            error("threaded=true requires $(required) Julia threads; restart the kernel with JULIA_NUM_THREADS=$(required).")
        end
    end
    BLAS.set_num_threads(blas_threads)

    nthreads = Threads.nthreads()
    active_threads = max(1, Int(threaded) * nthreads)
    rngs = [Random.Xoshiro(UInt(time_ns() + i)) for i in 1:active_threads]
    reps_done = progress_every > 0 ? Threads.Atomic{Int}(0) : nothing
    progress_task = progress_every > 0 ?
        start_generation_logger!(reps_done, n_per_pattern, n_classes, progress_every) : nothing
    progress_stride = progress_every > 0 ? max(1, progress_every) : 0

    chunk = cld(n_per_pattern, active_threads)

    # Build a chunk of images on a single thread.
    function build_chunk!(chunk_id::Int)
        rep_start = (chunk_id - 1) * chunk + 1
        rep_end = min(n_per_pattern, chunk_id * chunk)
        rep_start > rep_end && return

        local_rng = rngs[chunk_id]
        local_done = 0

        for rep in rep_start:rep_end
            mu = max(1, rand(local_rng, mu_dist))
            sigma = max(0.01, rand(local_rng, sigma_dist))
            n_trials = Int(ceil(rand(local_rng, n_trials_dist)))
            n_trials = max(100, n_trials)
            n_trials += isodd(n_trials) ? 1 : 0
            sampling_rate = max(1, Int(round(rand(local_rng, sampling_rate_dist))))
            epoch_duration_s = rand(local_rng, epoch_duration_dist)
            epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))

            sim_result = simulate_erp_trials(local_rng, mu, sigma, n_trials, sampling_rate, epoch_duration_s,
                p100_width_dist, p100_offset_dist, p300_width_dist, p300_offset_dist,
                n170_width_dist, n170_offset_dist, p1_beta_dist, p3_beta_dist,
                n1_beta1_dist, n1_beta2_dist, n1_beta3_dist,
                componentA_amp_dist, componentB_amp_dist, componentC_amp_dist,
                noise,
                patterns, covariate_dists)
            generated_size = (size(sim_result.data, 2), size(sim_result.data, 1))
            cropped, crop_info = crop_time_window(sim_result.data, local_rng, processing, sampling_rate)
            base = (rep - 1) * n_classes

            render_pattern_images!(
                images, labels, metadata, base,
                cropped, sim_result, mu, sigma, epoch_duration_s, sampling_rate,
                noise, processing, crop_info, generated_size, patterns;
                rng = local_rng,
            )

            if reps_done !== nothing
                local_done += 1
                if local_done >= progress_stride
                    Threads.atomic_add!(reps_done, local_done)
                    local_done = 0
                end
            end
        end

        if reps_done !== nothing && local_done > 0
            Threads.atomic_add!(reps_done, local_done)
        end
    end

    if threaded
        Threads.@threads :static for chunk_id in 1:active_threads
            build_chunk!(chunk_id)
        end
    else
        build_chunk!(1)
    end

    progress_task !== nothing && wait(progress_task)

    return images, labels, metadata
end
