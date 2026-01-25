# Periodically log progress for threaded execution.
function start_progress_logger!(reps_done::Threads.Atomic{Int}, n_per_pattern::Int, n_classes::Int, progress_every::Int)
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

# Sample simulation parameters from configured distributions/ranges.
function sample_sim_params(mu_dist, sigma_dist,
        condition_levels_dist, sampling_rate_dist, epoch_duration_dist)
    mu = max(1, rand(Random.Xoshiro(time_ns()), mu_dist))
    sigma = max(0.01, rand(Random.Xoshiro(time_ns()), sigma_dist))
    condition_levels = max(1, Int(round(rand(Random.Xoshiro(time_ns()), condition_levels_dist))))
    sampling_rate = max(1, Int(round(rand(Random.Xoshiro(time_ns()), sampling_rate_dist))))
    epoch_duration_s = rand(Random.Xoshiro(time_ns()), epoch_duration_dist)
    epoch_duration_s <= 0 && throw(ArgumentError("epoch_duration_s must be > 0"))
    return mu, sigma, condition_levels, sampling_rate, epoch_duration_s
end

# Generate ERP images (single process, optional threading).
function generate_erp_images(; n_per_pattern::Int = 10,
        mu_dist::Distribution = Normal(3.2, 0.3),
        sigma_dist::Distribution = Normal(0.5, 0.1),
        epoch_duration_dist::Distribution = Normal(1.0, 0.25),
        sampling_rate_dist::Distribution = Normal(100, 5),
        dropout_trials_rate_dist::Distribution = Normal(2000, 250),
        condition_levels_dist::Distribution = DiscreteUniform(3, 10),
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
        hourglass_continuous_end_dist::Distribution = Normal(2.0, 0.3),
        target_height::Int = 64,
        target_width::Int = 64,
        zscore_timepoints::Bool = true,
        filter_sigma::Real = 1.0,
        gauss_smooth::Bool = false,
        gauss_kernel_len::Int = 20,
        resize_antialias::Bool = false,
        antialias_factor::Real = 0.75,
        resize_method = Interpolations.Linear(),
        noise_pool = DEFAULT_NOISE_POOL,
        noiselevel_dists = DEFAULT_NOISELEVEL_DISTS,
        crop_start_dist = DEFAULT_CROP_DIST,
        crop_end_dist = DEFAULT_CROP_DIST,
        threaded::Bool = false,
        blas_threads::Int = 1,
        progress_every::Int = 10)

    ensure_latest_unfoldsim!(propagate = false)

    n_classes = length(PATTERN_NAMES) * VARIANT_COUNT
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
    progress_task = progress_every > 0 ? start_progress_logger!(reps_done, n_per_pattern, n_classes, progress_every) : nothing
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
            mu, sigma, condition_levels, sampling_rate, epoch_duration_s = sample_sim_params(
                mu_dist, sigma_dist, condition_levels_dist,
                sampling_rate_dist, epoch_duration_dist)
            data, simulated_events, noiselevels, p1_beta, p3_beta, n1_betas, hanning_params = simulate_erp(local_rng, mu, sigma;
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
            )
            generated_size = (size(data, 2), size(data, 1))
            cropped, crop_info = crop_time_axis(data, local_rng, crop_start_dist, crop_end_dist, sampling_rate)
            base = (rep - 1) * n_classes

            build_pattern_images!(
                images, labels, metadata, base,
                cropped, simulated_events, mu, sigma, epoch_duration_s, sampling_rate, condition_levels, noise_pool, noiselevels, p1_beta, p3_beta, n1_betas, hanning_params, crop_info, generated_size,
                dropout_trials_rate_dist;
                rng = local_rng,
                zscore_timepoints = zscore_timepoints,
                gauss_smooth = gauss_smooth,
                filter_sigma = filter_sigma,
                gauss_kernel_len = gauss_kernel_len,
                resize_antialias = resize_antialias,
                antialias_factor = antialias_factor,
                resize_method = resize_method,
                target_height = target_height,
                target_width = target_width,
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
