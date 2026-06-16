# =============================================================================
# Evaluation loop, aggregation, and CSV reporting
#
# Each candidate is evaluated over several repeats, aggregated into a per-
# candidate summary, and the best non-collapsed candidate is written to
# `best_run.csv`. The post-hoc and extra exports reproduce the analysis tables
# cited in the thesis (per-strategy summaries, rankings, runtime budget, and
# parameter-performance correlations).
# =============================================================================

# --- Small numeric helpers that ignore missing/non-finite values --------------

"""
    finite_values(xs) -> Vector{Float64}

Collect the finite, non-missing values of `xs` as `Float64`.
"""
function finite_values(xs)
    vals = Float64[]
    for x in xs
        ismissing(x) && continue
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    return vals
end

mean_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : mean(v))
std_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : (length(v) == 1 ? 0.0 : std(v)))
median_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : median(v))
sum_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : sum(v))
max_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : maximum(v))
min_or_missing(xs) = (v = finite_values(xs); isempty(v) ? missing : minimum(v))
quantile_or_missing(xs, q) = (v = finite_values(xs); isempty(v) ? missing : quantile(v, q))

"""
    repeat_row(record, repeat_index, repeat_seed, result, profile, error_message)

Assemble one fixed-schema repeat row from a successful `result` (or `nothing`
for a failed repeat). A constant schema lets successful and failed repeats live
in the same `DataFrame`.

# Arguments
- `record::CandidateRecord`: the evaluated candidate.
- `repeat_index::Int`, `repeat_seed`: repeat identity.
- `result`: the `train_and_score_candidate` result, or `nothing` on failure.
- `profile`: the training profile used.
- `error_message::AbstractString`: failure message, empty when successful.

# Returns
- `NamedTuple`: the repeat row.
"""
function repeat_row(record::CandidateRecord, repeat_index::Int, repeat_seed, result, profile, error_message::AbstractString)
    base = (
        search_method = record.search_method,
        candidate_index = Int(record.candidate_index),
        repeat_index = repeat_index,
        repeat_seed = repeat_seed,
        parameter_subset_size = Int(record.parameter_subset_size),
    )
    if result === nothing
        return merge(base, (
            balanced_accuracy = missing, macro_f1 = missing,
            train_time_s = missing, classification_time_s = missing, generation_time_s = missing,
            n_train = missing, n_val_real = missing, train_pos = missing, train_neg = missing,
            pred_count_0 = missing, pred_count_1 = missing, true_count_0 = missing, true_count_1 = missing,
            pretrained_params_loaded = missing,
            training_profile = String(profile.name), model_init = String(profile.model_init),
            collapsed = true, error_message = error_message,
        ))
    end
    return merge(base, merge(result, (error_message = error_message,)))
end

"""
    evaluate_record(record, repeats, real, config, profile, param_symbols)

Evaluate one candidate over `repeats` repeats. A repeat that throws is recorded
as a collapsed row with its error message so a single bad candidate cannot abort
the whole sweep.

# Arguments
- `record::CandidateRecord`: candidate to evaluate.
- `repeats::Int`: number of repeats.
- `real::RealValidationData`: real validation set.
- `config::RunConfig`: experiment settings.
- `profile`: selected training profile.

# Returns
- `DataFrame`: one row per repeat.
"""
function evaluate_record(record::CandidateRecord, repeats::Int, real::RealValidationData, config::RunConfig, profile)
    rows = NamedTuple[]
    for repeat_index in 1:repeats
        repeat_seed = new_seed()
        run_tag = "$(record.search_method)/candidate$(record.candidate_index)/repeat$(repeat_index)"
        println("\n=== $(run_tag) | seed=$(repeat_seed) ===")
        # A failed candidate is caught so the sweep continues; the failure is logged.
        try
            result = train_and_score_candidate(record.cfg, real, profile, config, repeat_seed; run_tag = run_tag)
            push!(rows, repeat_row(record, repeat_index, repeat_seed, result, profile, ""))
        catch err
            @warn "Candidate repeat failed" search_method = record.search_method candidate_index = record.candidate_index repeat_index = repeat_index exception = (err, catch_backtrace())
            push!(rows, repeat_row(record, repeat_index, repeat_seed, nothing, profile, sprint(showerror, err)))
            cleanup_device!(config.use_gpu)
        end
    end
    return DataFrame(rows)
end

"""
    aggregate_record(record, repeat_df, repeats, config, profile, param_symbols)

Aggregate a candidate's repeats into one summary row: mean and std of the metrics
over the non-collapsed repeats, mean timings over all repeats, repeat/collapse
counts, and the candidate's parameter values.

# Returns
- `NamedTuple`: the per-candidate summary row.
"""
function aggregate_record(record::CandidateRecord, repeat_df::DataFrame, repeats::Int, config::RunConfig, profile, param_symbols::Vector{Symbol})
    valid_mask = [(!ismissing(row.balanced_accuracy)) && !Bool(row.collapsed) for row in eachrow(repeat_df)]
    valid_df = repeat_df[valid_mask, :]
    row_pairs = Pair{Symbol, Any}[
        :search_method => String(record.search_method),
        :candidate_index => Int(record.candidate_index),
        :balanced_accuracy_mean => mean_or_missing(valid_df.balanced_accuracy),
        :macro_f1_mean => mean_or_missing(valid_df.macro_f1),
        :balanced_accuracy_std => std_or_missing(valid_df.balanced_accuracy),
        :macro_f1_std => std_or_missing(valid_df.macro_f1),
        :train_time_s_mean => mean_or_missing(repeat_df.train_time_s),
        :classification_time_s_mean => mean_or_missing(repeat_df.classification_time_s),
        :n_valid_repeats => nrow(valid_df),
        :n_collapsed_repeats => count(Bool.(coalesce.(repeat_df.collapsed, true))),
        :eval_repeats => repeats,
        :n_per_pattern => config.n_per_pattern,
        :training_profile => String(profile.name),
        :model_init => String(profile.model_init),
        :parameter_subset_size => Int(record.parameter_subset_size),
    ]
    append!(row_pairs, [param_symbols[j] => record.params[j] for j in eachindex(record.params)])
    return (; row_pairs...)
end

# --- Best-run selection and writers -------------------------------------------

"""
    select_best_row(strategy_results_df, param_symbols)

Return the best strategy candidate (highest mean balanced accuracy, macro F1 as
tie-break) among the non-baseline strategies. Errors when no candidate passed the
collapse filter.

# Returns
- `DataFrameRow`: the winning candidate's summary row.
"""
function select_best_row(strategy_results_df::DataFrame)
    method_names = String.(collect(SEARCH_METHODS))
    df = copy(strategy_results_df[in.(strategy_results_df.search_method, Ref(method_names)), :])
    df = df[.!ismissing.(df.balanced_accuracy_mean) .& .!ismissing.(df.macro_f1_mean), :]
    nrow(df) > 0 || error("No strategy candidate passed the collapse filter. Refusing to write best_run.csv.")
    sort!(df, [:balanced_accuracy_mean, :macro_f1_mean]; rev = [true, true])
    return df[1, :]
end

"""
    write_best_run!(output_dir, best_row, param_symbols) -> String

Write `best_run.csv` for the overall winning candidate, after clearing any stale
CSVs directly in `output_dir`. Returns the written path.
"""
function write_best_run!(output_dir::AbstractString, best_row, param_symbols::Vector{Symbol})
    best_pairs = Pair{Symbol, Any}[
        :search_method => String(best_row.search_method),
        :candidate_index => Int(best_row.candidate_index),
        :balanced_accuracy_mean => Float64(best_row.balanced_accuracy_mean),
        :macro_f1_mean => Float64(best_row.macro_f1_mean),
        :balanced_accuracy_std => Float64(best_row.balanced_accuracy_std),
        :train_time_s_mean => Float64(best_row.train_time_s_mean),
        :classification_time_s_mean => Float64(best_row.classification_time_s_mean),
    ]
    append!(best_pairs, [sym => Float64(best_row[sym]) for sym in param_symbols])

    mkpath(output_dir)
    for path in readdir(output_dir; join = true)
        isfile(path) && endswith(path, ".csv") && rm(path; force = true)
    end
    best_path = joinpath(output_dir, "best_run.csv")
    CSV.write(best_path, DataFrame([(; best_pairs...)]))
    return best_path
end

"""
    strategy_top_rows(strategy_results_df)

Return the best candidate per search strategy (or a missing-filled row when a
strategy has no non-collapsed candidate).
"""
function strategy_top_rows(strategy_results_df::DataFrame)
    rows = NamedTuple[]
    for method in String.(collect(SEARCH_METHODS))
        df = copy(strategy_results_df[strategy_results_df.search_method .== method, :])
        df = df[.!ismissing.(df.balanced_accuracy_mean) .& .!ismissing.(df.macro_f1_mean), :]
        if nrow(df) == 0
            push!(rows, (search_method = method, candidate_index = missing, balanced_accuracy_mean = missing,
                macro_f1_mean = missing, train_time_s_mean = missing, classification_time_s_mean = missing, n_valid_repeats = 0))
        else
            sort!(df, [:balanced_accuracy_mean, :macro_f1_mean]; rev = [true, true])
            r = df[1, :]
            push!(rows, (search_method = String(r.search_method), candidate_index = Int(r.candidate_index),
                balanced_accuracy_mean = Float64(r.balanced_accuracy_mean), macro_f1_mean = Float64(r.macro_f1_mean),
                train_time_s_mean = Float64(r.train_time_s_mean), classification_time_s_mean = Float64(r.classification_time_s_mean),
                n_valid_repeats = Int(r.n_valid_repeats)))
        end
    end
    return DataFrame(rows)
end

"""
    write_posthoc_exports!(output_dir, strategy_results_df, repeat_results_df, sanity_df, config, profile) -> String

Write the post-hoc analysis CSVs (candidate summaries, rankings, per-method
summary, unfiltered repeat summary, baseline summary, sanity results, and run
metadata) into `output_dir/posthoc_exports`. Returns that directory.
"""
function write_posthoc_exports!(output_dir::AbstractString, strategy_results_df::DataFrame, repeat_results_df::DataFrame, sanity_df::DataFrame, config::RunConfig, profile)
    posthoc_dir = joinpath(output_dir, "posthoc_exports")
    mkpath(posthoc_dir)
    methods = String.(collect(SEARCH_METHODS))

    candidate_summary = copy(strategy_results_df)
    repeat_raw = copy(repeat_results_df)
    CSV.write(joinpath(posthoc_dir, "all_candidates_filtered_summary.csv"), candidate_summary)
    CSV.write(joinpath(posthoc_dir, "all_repeats_raw.csv"), repeat_raw)

    strategy_only = candidate_summary[in.(candidate_summary.search_method, Ref(methods)), :]
    valid_strategy = strategy_only[.!ismissing.(strategy_only.balanced_accuracy_mean) .& .!ismissing.(strategy_only.macro_f1_mean), :]

    top_per_strategy = strategy_top_rows(candidate_summary)
    top_per_strategy = top_per_strategy[.!ismissing.(top_per_strategy.candidate_index), :]
    CSV.write(joinpath(posthoc_dir, "top_per_strategy.csv"), top_per_strategy)
    CSV.write(joinpath(posthoc_dir, "candidate_ranking_all.csv"), sort(valid_strategy, [:balanced_accuracy_mean, :macro_f1_mean]; rev = [true, true]))

    # Unfiltered per-candidate summary (collapsed repeats included).
    unfiltered_rows = NamedTuple[]
    for g in groupby(repeat_raw, [:search_method, :candidate_index])
        in(String(g.search_method[1]), methods) || continue
        push!(unfiltered_rows, (
            search_method = String(g.search_method[1]), candidate_index = Int(g.candidate_index[1]),
            balanced_accuracy_mean_unfiltered = mean_or_missing(g.balanced_accuracy),
            balanced_accuracy_std_unfiltered = std_or_missing(g.balanced_accuracy),
            macro_f1_mean_unfiltered = mean_or_missing(g.macro_f1),
            macro_f1_std_unfiltered = std_or_missing(g.macro_f1),
            train_time_s_mean_unfiltered = mean_or_missing(g.train_time_s),
            classification_time_s_mean_unfiltered = mean_or_missing(g.classification_time_s),
            n_repeats_recorded = nrow(g),
            n_collapsed_repeats = count(x -> !ismissing(x) && Bool(x), g.collapsed),
            n_missing_bacc = count(ismissing, g.balanced_accuracy),
        ))
    end
    CSV.write(joinpath(posthoc_dir, "all_candidates_unfiltered_repeat_summary.csv"), DataFrame(unfiltered_rows))

    # Per-method summary combining candidate and repeat views.
    method_rows = NamedTuple[]
    for method in methods
        cand = strategy_only[strategy_only.search_method .== method, :]
        rep = repeat_raw[repeat_raw.search_method .== method, :]
        valid = cand[.!ismissing.(cand.balanced_accuracy_mean), :]
        top = top_per_strategy[top_per_strategy.search_method .== method, :]
        push!(method_rows, (
            search_method = method, n_candidates = nrow(cand), n_valid_candidates = nrow(valid), n_repeats = nrow(rep),
            n_collapsed_repeats = count(x -> !ismissing(x) && Bool(x), rep.collapsed),
            collapsed_repeat_frac = nrow(rep) == 0 ? missing : count(x -> !ismissing(x) && Bool(x), rep.collapsed) / nrow(rep),
            top_candidate_index = nrow(top) == 0 ? missing : top.candidate_index[1],
            top_bacc = nrow(top) == 0 ? missing : top.balanced_accuracy_mean[1],
            top_macro_f1 = nrow(top) == 0 ? missing : top.macro_f1_mean[1],
            top_train_time_s = nrow(top) == 0 ? missing : top.train_time_s_mean[1],
            top_classification_time_s = nrow(top) == 0 ? missing : top.classification_time_s_mean[1],
            mean_bacc_across_candidates = mean_or_missing(valid.balanced_accuracy_mean),
            median_bacc_across_candidates = median_or_missing(valid.balanced_accuracy_mean),
            std_bacc_across_candidates = std_or_missing(valid.balanced_accuracy_mean),
            mean_macro_f1_across_candidates = mean_or_missing(valid.macro_f1_mean),
            median_macro_f1_across_candidates = median_or_missing(valid.macro_f1_mean),
        ))
    end
    CSV.write(joinpath(posthoc_dir, "method_summary.csv"), DataFrame(method_rows))

    CSV.write(joinpath(posthoc_dir, "sanity_results.csv"), sanity_df)
    CSV.write(joinpath(posthoc_dir, "baseline_summary.csv"), candidate_summary[candidate_summary.search_method .== "baseline", :])

    metadata = DataFrame(
        key = ["exported_at", "simulation_threads", "target_size", "low_pass_factor", "sanity_bacc_min",
            "sanity_class_balance_frac", "eval_repeats", "n_per_pattern", "selected_profile"],
        value = [string(now()), string(Threads.nthreads()), string(config.target_size), string(config.low_pass_factor),
            string(config.sanity_bacc_min), string(config.sanity_class_balance_frac), string(config.eval_repeats),
            string(config.n_per_pattern), string(profile.name)],
    )
    CSV.write(joinpath(posthoc_dir, "run_metadata.csv"), metadata)
    return posthoc_dir
end

"""
    write_extra_exports!(output_dir, strategy_results_df, repeat_results_df, param_symbols, param_ranges, base_params, config) -> String

Write the supplementary analysis CSVs (strategy efficiency curve, distribution
summary, repeat stability, prediction balance, long-format parameters, parameter
correlations, and runtime budget) into `output_dir/extra_exports`. Returns that
directory.
"""
function write_extra_exports!(output_dir::AbstractString, strategy_results_df::DataFrame, repeat_results_df::DataFrame,
        param_symbols::Vector{Symbol}, param_ranges, base_params::Vector{Float64}, config::RunConfig)
    extra_dir = joinpath(output_dir, "extra_exports")
    mkpath(extra_dir)
    methods = String.(collect(SEARCH_METHODS))

    candidate_df = copy(strategy_results_df)
    repeat_df = copy(repeat_results_df)
    strategy_candidates = candidate_df[in.(candidate_df.search_method, Ref(methods)), :]
    valid_candidates = strategy_candidates[.!ismissing.(strategy_candidates.balanced_accuracy_mean) .& .!ismissing.(strategy_candidates.macro_f1_mean), :]

    # 1. Strategy efficiency curve: best-so-far balanced accuracy by candidate order.
    eff_rows = NamedTuple[]
    for method in methods
        sub = sort(strategy_candidates[strategy_candidates.search_method .== method, :], :candidate_index)
        best_bacc, best_macro, best_candidate = -Inf, missing, missing
        for row in eachrow(sub)
            if !ismissing(row.balanced_accuracy_mean)
                bacc = Float64(row.balanced_accuracy_mean)
                mf1 = ismissing(row.macro_f1_mean) ? -Inf : Float64(row.macro_f1_mean)
                if bacc > best_bacc || (bacc == best_bacc && !ismissing(best_macro) && mf1 > best_macro)
                    best_bacc, best_macro, best_candidate = bacc, mf1, Int(row.candidate_index)
                end
            end
            push!(eff_rows, (search_method = method, candidate_index = Int(row.candidate_index),
                balanced_accuracy_mean = row.balanced_accuracy_mean, macro_f1_mean = row.macro_f1_mean,
                n_valid_repeats = row.n_valid_repeats, n_collapsed_repeats = row.n_collapsed_repeats,
                best_candidate_so_far = best_candidate, best_bacc_so_far = isfinite(best_bacc) ? best_bacc : missing,
                best_macro_f1_at_best_so_far = best_macro))
        end
    end
    CSV.write(joinpath(extra_dir, "strategy_efficiency_curve.csv"), DataFrame(eff_rows))

    # 2. Score distribution per strategy.
    dist_rows = NamedTuple[]
    for method in methods
        cand = strategy_candidates[strategy_candidates.search_method .== method, :]
        valid = cand[.!ismissing.(cand.balanced_accuracy_mean), :]
        push!(dist_rows, (search_method = method, n_candidates = nrow(cand), n_valid_candidates = nrow(valid),
            n_candidates_with_all_repeats_collapsed = count(==(0), cand.n_valid_repeats),
            bacc_min = min_or_missing(valid.balanced_accuracy_mean), bacc_q25 = quantile_or_missing(valid.balanced_accuracy_mean, 0.25),
            bacc_median = quantile_or_missing(valid.balanced_accuracy_mean, 0.50), bacc_q75 = quantile_or_missing(valid.balanced_accuracy_mean, 0.75),
            bacc_max = max_or_missing(valid.balanced_accuracy_mean), bacc_mean = mean_or_missing(valid.balanced_accuracy_mean),
            bacc_std = std_or_missing(valid.balanced_accuracy_mean), macro_f1_min = min_or_missing(valid.macro_f1_mean),
            macro_f1_median = quantile_or_missing(valid.macro_f1_mean, 0.50), macro_f1_max = max_or_missing(valid.macro_f1_mean),
            macro_f1_mean = mean_or_missing(valid.macro_f1_mean)))
    end
    CSV.write(joinpath(extra_dir, "strategy_distribution_summary.csv"), DataFrame(dist_rows))

    # 3. Repeat stability and collapse diagnostics per candidate.
    stab_rows = NamedTuple[]
    for g in groupby(repeat_df, [:search_method, :candidate_index])
        in(String(g.search_method[1]), methods) || continue
        valid = g[.!ismissing.(g.balanced_accuracy) .& .!Bool.(coalesce.(g.collapsed, true)), :]
        bacc_valid = finite_values(valid.balanced_accuracy)
        push!(stab_rows, (search_method = String(g.search_method[1]), candidate_index = Int(g.candidate_index[1]),
            n_repeats = nrow(g), n_valid_repeats = nrow(valid), n_collapsed_repeats = count(x -> !ismissing(x) && Bool(x), g.collapsed),
            bacc_mean_valid = mean_or_missing(valid.balanced_accuracy), bacc_std_valid = std_or_missing(valid.balanced_accuracy),
            bacc_min_valid = min_or_missing(valid.balanced_accuracy), bacc_max_valid = max_or_missing(valid.balanced_accuracy),
            bacc_range_valid = isempty(bacc_valid) ? missing : maximum(bacc_valid) - minimum(bacc_valid),
            macro_f1_mean_valid = mean_or_missing(valid.macro_f1), macro_f1_std_valid = std_or_missing(valid.macro_f1),
            train_time_s_mean = mean_or_missing(g.train_time_s), classification_time_s_mean = mean_or_missing(g.classification_time_s),
            generation_time_s_mean = mean_or_missing(g.generation_time_s)))
    end
    CSV.write(joinpath(extra_dir, "repeat_stability_by_candidate.csv"), DataFrame(stab_rows))

    # 4. Per-repeat prediction balance against the collapse floor.
    pred_rows = NamedTuple[]
    for row in eachrow(repeat_df)
        n = ismissing(row.n_val_real) ? missing : Float64(row.n_val_real)
        frac(c) = ismissing(c) || ismissing(n) || n == 0 ? missing : Float64(c) / n
        pred0, pred1 = frac(row.pred_count_0), frac(row.pred_count_1)
        push!(pred_rows, (search_method = String(row.search_method), candidate_index = Int(row.candidate_index),
            repeat_index = Int(row.repeat_index), repeat_seed = row.repeat_seed, balanced_accuracy = row.balanced_accuracy,
            macro_f1 = row.macro_f1, collapsed = row.collapsed, pred_frac_no_class = pred0, pred_frac_sigmoid = pred1,
            pred_min_class_frac = ismissing(pred0) || ismissing(pred1) ? missing : min(pred0, pred1),
            pred_majority_frac = ismissing(pred0) || ismissing(pred1) ? missing : max(pred0, pred1),
            true_frac_no_class = frac(row.true_count_0), true_frac_sigmoid = frac(row.true_count_1),
            passes_prediction_balance_floor = ismissing(pred0) || ismissing(pred1) ? missing : min(pred0, pred1) >= config.sanity_class_balance_frac))
    end
    CSV.write(joinpath(extra_dir, "prediction_balance_per_repeat.csv"), DataFrame(pred_rows))

    # 5. Long-format parameter table for plotting.
    param_rows = NamedTuple[]
    for row in eachrow(strategy_candidates)
        for (j, p) in enumerate(param_symbols)
            lo, hi = param_ranges[j]
            value = Float64(row[p])
            base_value = Float64(base_params[j])
            push!(param_rows, (search_method = String(row.search_method), candidate_index = Int(row.candidate_index),
                parameter = String(p), value = value, base_value = base_value, delta_from_base = value - base_value,
                normalized_value = hi == lo ? missing : (value - lo) / (hi - lo), range_low = Float64(lo), range_high = Float64(hi),
                balanced_accuracy_mean = row.balanced_accuracy_mean, macro_f1_mean = row.macro_f1_mean,
                n_valid_repeats = row.n_valid_repeats, n_collapsed_repeats = row.n_collapsed_repeats))
        end
    end
    CSV.write(joinpath(extra_dir, "parameter_values_long.csv"), DataFrame(param_rows))

    # 6. Parameter-performance correlations (Pearson and Spearman).
    CSV.write(joinpath(extra_dir, "parameter_performance_correlations.csv"),
        parameter_performance_correlations(valid_candidates, param_symbols, methods))

    # 7. Runtime budget accounting.
    budget_rows = NamedTuple[]
    for method in vcat(["baseline"], methods)
        sub = repeat_df[repeat_df.search_method .== method, :]
        push!(budget_rows, (search_method = method, n_repeats = nrow(sub),
            total_train_time_s = sum_or_missing(sub.train_time_s), total_classification_time_s = sum_or_missing(sub.classification_time_s),
            total_generation_time_s = sum_or_missing(sub.generation_time_s), mean_train_time_s = mean_or_missing(sub.train_time_s),
            mean_classification_time_s = mean_or_missing(sub.classification_time_s), mean_generation_time_s = mean_or_missing(sub.generation_time_s)))
    end
    CSV.write(joinpath(extra_dir, "budget_accounting.csv"), DataFrame(budget_rows))
    return extra_dir
end

"""
    parameter_performance_correlations(valid_candidates, param_symbols, methods)

Correlate each parameter with balanced accuracy and macro F1, both overall and
per strategy, using Pearson and (rank-based) Spearman coefficients.

# Returns
- `DataFrame`: one row per `(scope, parameter)`.
"""
function parameter_performance_correlations(valid_candidates::DataFrame, param_symbols::Vector{Symbol}, methods)
    ordinal_ranks(x) = (order = sortperm(x); ranks = similar(x, Float64); for (rank, idx) in enumerate(order); ranks[idx] = Float64(rank); end; ranks)
    safe_cor(x, y) = (length(x) < 3 || std(x) == 0 || std(y) == 0) ? missing : cor(x, y)

    rows = NamedTuple[]
    for scope in vcat(["all"], methods)
        df = scope == "all" ? valid_candidates : valid_candidates[valid_candidates.search_method .== scope, :]
        for p in param_symbols
            xs, ys_bacc, ys_f1 = Float64[], Float64[], Float64[]
            for row in eachrow(df)
                if !ismissing(row.balanced_accuracy_mean) && !ismissing(row.macro_f1_mean)
                    push!(xs, Float64(row[p])); push!(ys_bacc, Float64(row.balanced_accuracy_mean)); push!(ys_f1, Float64(row.macro_f1_mean))
                end
            end
            push!(rows, (scope = scope, parameter = String(p), n = length(xs),
                pearson_bacc = safe_cor(xs, ys_bacc), spearman_bacc = safe_cor(ordinal_ranks(xs), ordinal_ranks(ys_bacc)),
                pearson_macro_f1 = safe_cor(xs, ys_f1), spearman_macro_f1 = safe_cor(ordinal_ranks(xs), ordinal_ranks(ys_f1))))
        end
    end
    return DataFrame(rows)
end
