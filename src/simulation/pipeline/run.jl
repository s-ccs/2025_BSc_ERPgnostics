# =============================================================================
# Orchestration
#
# Ties the pieces together into one E4 parameter search: load the real data,
# build the dataset-matched baseline, run the sanity gate, sweep the three
# strategies, pick the best candidate, and write all outputs.
# =============================================================================

"""
    gpu_available() -> Bool

Return whether a functional CUDA device is available in this module.
"""
gpu_available() = isdefined(@__MODULE__, :CUDA) && CUDA.functional()

"""
    configure_device!(use_gpu::Bool)

Prepare and announce the compute device. On GPU this disables scalar indexing
and selects CUDA device 0, mirroring the setup in `src/real_data_training`. Only
the ResNet18 training and inference run on the GPU; the ERPGen simulator always
runs on the CPU.

# Arguments
- `use_gpu::Bool`: whether the run should use a CUDA device.

# Returns
- `nothing`.
"""
function configure_device!(use_gpu::Bool)
    if use_gpu
        CUDA.allowscalar(false)
        CUDA.device!(0)
        println("Device = CUDA GPU (", CUDA.name(CUDA.device()), ")")
    else
        println("Device = CPU")
    end
    return nothing
end

"""
    configure_simulation_threads!()

Set BLAS to a single thread so it does not oversubscribe the CPU while the
simulator runs across Julia threads, and report how many threads are available.
The simulation parallelises over `Threads.nthreads()`; start Julia with
`-t auto` to use all CPU threads.

# Returns
- `nothing`.
"""
function configure_simulation_threads!()
    # The ResNet18 runs on the GPU, so keep BLAS single-threaded and let the
    # CPU cores go to the parallel simulation instead.
    BLAS.set_num_threads(1)
    n = Threads.nthreads()
    println("Simulation threads = ", n, n == 1 ? " (start Julia with `-t auto` to parallelise the simulation)" : "")
    return nothing
end

"""
    validate_base_config(base_cfg, real)

Assert that the parameter space has the expected size and that the baseline
configuration's geometry matches the real recording. Fails fast before any
expensive training if an assumption is broken.

# Arguments
- `base_cfg::ERPGen.GenerationConfig`: the dataset-matched baseline.
- `real::RealValidationData`: the real dataset dimensions.

# Returns
- `Tuple`: `(param_symbols, param_ranges, base_params)` for reuse downstream.
"""
function validate_base_config(base_cfg::ERPGen.GenerationConfig, real::RealValidationData)
    specs = parameter_specs(base_cfg)
    param_ranges = parameter_ranges(base_cfg)
    param_symbols = parameter_symbols(base_cfg)
    base_params = parameter_vector_from_cfg(base_cfg, base_cfg)

    length(specs) == 24 || error("Expected 24 Normal-valued simulator distributions, got $(length(specs)).")
    length(param_ranges) == 48 || error("Expected a 48-dimensional parameter space, got $(length(param_ranges)).")
    length(param_symbols) == 48 || error("Expected 48 parameter symbols, got $(length(param_symbols)).")
    Int(round(mean(base_cfg.sim.n_trials_dist))) == real.n_trials || error("Baseline trial count does not match the real dataset.")
    Int(round(mean(base_cfg.sim.epoch_duration_dist) * mean(base_cfg.sim.sampling_rate_dist))) + 1 == real.n_timepoints ||
        error("Baseline timepoint count does not match the real dataset.")
    return param_symbols, param_ranges, base_params
end

"""
    run_strategy_sweep(base_cfg, real, config, profile, param_symbols, param_ranges, base_params)

Evaluate the baseline and every strategy candidate, returning the per-repeat and
per-candidate result tables.

# Returns
- `Tuple{DataFrame,DataFrame}`: `(repeat_results_df, strategy_results_df)`.
"""
function run_strategy_sweep(base_cfg::ERPGen.GenerationConfig, real::RealValidationData, config::RunConfig, profile,
        param_symbols::Vector{Symbol}, param_ranges, base_params::Vector{Float64})
    all_repeat_results = DataFrame[]
    aggregate_rows = NamedTuple[]

    # Baseline ("starting parameters") reference run.
    println("\n=== Baseline base_cfg reference run ===")
    base_record = baseline_record(base_cfg, base_params)
    baseline_df = evaluate_record(base_record, config.baseline_repeats, real, config, profile)
    push!(all_repeat_results, baseline_df)
    push!(aggregate_rows, aggregate_record(base_record, baseline_df, config.baseline_repeats, config, profile, param_symbols))

    # Each strategy in turn.
    for method in SEARCH_METHODS
        n_candidates = config.strategy_budgets[method]
        println("\n==============================")
        println("Strategy: ", method, " (", n_candidates, " candidates)")
        println("==============================")
        records = make_candidate_records(method, n_candidates, base_cfg, base_params, param_ranges, param_symbols)
        for record in records
            repeat_df = evaluate_record(record, config.eval_repeats, real, config, profile)
            push!(all_repeat_results, repeat_df)
            push!(aggregate_rows, aggregate_record(record, repeat_df, config.eval_repeats, config, profile, param_symbols))
        end
    end

    repeat_results_df = vcat(all_repeat_results...; cols = :union)
    strategy_results_df = sort(DataFrame(aggregate_rows), [:search_method, :candidate_index])
    return repeat_results_df, strategy_results_df
end

"""
    print_summary(strategy_results_df, best_row, param_symbols)

Print the per-strategy best candidates, the overall winner, and the winning
parameter values to standard output.

# Returns
- `nothing`.
"""
function print_summary(strategy_results_df::DataFrame, best_row, param_symbols::Vector{Symbol})
    println("\nPer-strategy top results:")
    for row in eachrow(strategy_top_rows(strategy_results_df))
        if ismissing(row.candidate_index)
            println("  ", row.search_method, ": no non-collapsed candidate")
        else
            println("  ", row.search_method, ": candidate=", row.candidate_index,
                " | BAcc=", round(row.balanced_accuracy_mean, digits = 4),
                " | macro_F1=", round(row.macro_f1_mean, digits = 4),
                " | train_time_s=", round(row.train_time_s_mean, digits = 2),
                " | classification_time_s=", round(row.classification_time_s_mean, digits = 4),
                " | valid_repeats=", row.n_valid_repeats)
        end
    end

    println("\nOverall winner:")
    println("  search_method=", best_row.search_method, " | candidate_index=", best_row.candidate_index,
        " | balanced_accuracy_mean=", round(Float64(best_row.balanced_accuracy_mean), digits = 4),
        " | macro_f1_mean=", round(Float64(best_row.macro_f1_mean), digits = 4))
    println("\nWinner parameters:")
    for sym in param_symbols
        println("  ", sym, " = ", best_row[sym])
    end
    return nothing
end

"""
    render_strategy_previews(strategy_results_df, base_cfg, real, config, param_symbols)

Render one preview figure per search strategy using that strategy's best
non-collapsed candidate. A strategy with no valid candidate is skipped. The
candidate's parameter vector is read back from `strategy_results_df` and turned
into a simulator configuration to draw the simulated sigmoid panels from.

# Returns
- `nothing`.
"""
function render_strategy_previews(strategy_results_df::DataFrame, base_cfg::ERPGen.GenerationConfig,
        real::RealValidationData, config::RunConfig, param_symbols::Vector{Symbol})
    for method in SEARCH_METHODS
        df = strategy_results_df[(strategy_results_df.search_method .== String(method)) .&
            .!ismissing.(strategy_results_df.balanced_accuracy_mean), :]
        if nrow(df) == 0
            println("  ", method, ": no non-collapsed candidate; skipping preview")
            continue
        end
        sort!(df, [:balanced_accuracy_mean, :macro_f1_mean]; rev = [true, true])
        best = df[1, :]
        params = Float64[Float64(best[sym]) for sym in param_symbols]
        cfg = build_cfg_from_params(base_cfg, params)
        heading = "$(method) — best candidate $(Int(best.candidate_index)) " *
            "(BAcc=$(round(Float64(best.balanced_accuracy_mean), digits = 3)))"
        path = render_preview(cfg, real, config; name = String(method), heading = heading)
        println("  saved preview: ", path)
    end
    return nothing
end

"""
    run_search(config::RunConfig)

Run the full E4 (64x64 ResNet18) simulator parameter search and write all
outputs to `config.output_dir`.

Steps: select the compute device and threads, load the real validation set,
build and validate the dataset-matched baseline, optionally render the preview,
run the sanity gate to pick a training profile, sweep the three strategies,
select the best candidate, and write `best_run.csv` plus the post-hoc and extra
exports.

# Arguments
- `config::RunConfig`: the experiment configuration.

# Returns
- `NamedTuple`: handles to the key results
  (`best_run_path`, `strategy_results`, `repeat_results`, `selected_profile`).
"""
function run_search(config::RunConfig)
    println("Output directory = ", config.output_dir)
    configure_device!(config.use_gpu)
    configure_simulation_threads!()
    mkpath(config.output_dir)
    # Remove the previous single-figure preview so only the new per-setting files remain.
    legacy_preview = joinpath(config.output_dir, "preview_sim_vs_real.png")
    isfile(legacy_preview) && rm(legacy_preview; force = true)

    real = load_real_validation_data(config)
    println("Real validation tensor: ", size(real.tensor), " (height, width, channel, sample)")
    println("Real source dimensions: trials=", real.n_trials, ", timepoints=", real.n_timepoints, ", sampling_rate=", real.sampling_rate)

    base_cfg = build_base_config(config, real)
    param_symbols, param_ranges, base_params = validate_base_config(base_cfg, real)
    println("Parameter dimensions: ", length(param_symbols))

    if config.write_preview
        path = render_preview(base_cfg, real, config; name = "default", heading = "default parameters (baseline)")
        println("Saved default preview: ", path)
    end

    selected_profile, sanity_df = run_sanity_gate(base_cfg, real, config)
    println("Selected training profile: ", selected_profile.name)

    repeat_results_df, strategy_results_df = run_strategy_sweep(base_cfg, real, config, selected_profile, param_symbols, param_ranges, base_params)

    best_row = select_best_row(strategy_results_df)
    best_run_path = write_best_run!(config.output_dir, best_row, param_symbols)
    write_posthoc_exports!(config.output_dir, strategy_results_df, repeat_results_df, sanity_df, config, selected_profile)
    write_extra_exports!(config.output_dir, strategy_results_df, repeat_results_df, param_symbols, param_ranges, base_params, config)

    if config.write_preview
        println("\nRendering per-strategy previews...")
        render_strategy_previews(strategy_results_df, base_cfg, real, config, param_symbols)
    end

    print_summary(strategy_results_df, best_row, param_symbols)
    println("\nWrote best run CSV: ", best_run_path)

    return (
        best_run_path = best_run_path,
        strategy_results = strategy_results_df,
        repeat_results = repeat_results_df,
        selected_profile = selected_profile,
    )
end
