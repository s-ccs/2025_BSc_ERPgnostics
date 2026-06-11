# aggregate_scores.jl
#
# Combine the two fresh score sources into the final, lean, duplicate-free CSVs.
#
#   * cv_labeled    -> out-of-fold scores for labelled combinations (per 200-trial
#                      chunk; a labelled combination can have several chunks)
#   * new_unlabeled -> final-model scores for unlabelled combinations (whole
#                      parent, four augmentations)
#
# These two sources are disjoint by construction (a combination is labelled XOR
# unlabelled), so every (dataset, sorting_variable, channel) gets exactly ONE
# model score.
#
# Output files (written next to this script):
#   lean_parent_scores.csv     -> ONE row per (dataset, sorting_variable,
#       channel): the model score and the manual label (or "unlabelled").
#   lean_augmentation_scores.csv -> the per-augmentation detail behind each score
#       (one row per dataset/sort/channel/parent/augmentation).

const LEAN_COMBO_KEY = [:dataset, :sorting_variable, :channel]
const LEAN_AUG_KEY = [:dataset, :sorting_variable, :channel, :parent, :augmentation_id]

"""
    to_lean_aug(df, source) -> DataFrame

Project a source augmentation frame onto the lean augmentation schema. Expects
`dataset_key`, `sort_variable`, `channel_name`, `parent_image_id`,
`augmentation_variant_index`, `augmentation_name`, `prob_class`.
"""
function to_lean_aug(df::DataFrame, source::AbstractString)
    isempty(df) && return DataFrame(
        dataset = String[], sorting_variable = String[], channel = String[], parent = String[],
        augmentation_id = Int[], augmentation_name = String[], score = Float64[], source = String[],
    )
    return DataFrame(
        dataset = String.(df.dataset_key),
        sorting_variable = String.(df.sort_variable),
        channel = String.(df.channel_name),
        parent = String.(df.parent_image_id),
        augmentation_id = Int.(df.augmentation_variant_index),
        augmentation_name = String.(df.augmentation_name),
        score = Float64.(df.prob_class),
        source = fill(String(source), nrow(df)),
    )
end

"""
    dedup_aug(aug) -> DataFrame

Keep one row per (dataset, sorting_variable, channel, parent, augmentation_id).
"""
function dedup_aug(aug::DataFrame)
    isempty(aug) && return aug
    return combine(first, groupby(aug, LEAN_AUG_KEY))
end

"""
    manual_label_lookup(labels) -> Dict{NTuple{3,String}, String}

Maps each labelled (dataset, sorting_variable, channel) to its manual erp_class.
There is at most one manual label per combination.
"""
function manual_label_lookup(labels::DataFrame)
    lk = combined_label_lookup(labels).erp_class
    return Dict(combo_key(k[1], k[2], k[3]) => v for (k, v) in lk)
end

"""
    aggregate_combos(aug, labels) -> DataFrame

One row per (dataset, sorting_variable, channel). The score is the mean over all
of that combination's augmentation scores (across chunks for labelled data). The
manual label is attached when present, otherwise "unlabelled".
"""
function aggregate_combos(aug::DataFrame, labels::DataFrame)
    isempty(aug) && return DataFrame(
        dataset = String[], sorting_variable = String[], channel = String[], parent = String[],
        manual_label = String[], has_manual_label = Bool[],
        score = Float64[],
    )
    # One score per combination = mean over all its augmentation scores.
    combo_df = combine(groupby(aug, LEAN_COMBO_KEY),
        :score => mean => :score,
    )

    # Attach the canonical full-parent id and the manual label (or "unlabelled").
    labmap = manual_label_lookup(labels)
    combo_df.parent = [join([r.dataset, r.channel, r.sorting_variable, FULL_PARENT_TAG], "::") for r in eachrow(combo_df)]
    combo_df.manual_label = [get(labmap, combo_key(r.dataset, r.sorting_variable, r.channel), "unlabelled") for r in eachrow(combo_df)]
    combo_df.has_manual_label = combo_df.manual_label .!= "unlabelled"

    combo_df = select(combo_df, [:dataset, :sorting_variable, :channel, :parent,
        :manual_label, :has_manual_label, :score])
    sort!(combo_df, LEAN_COMBO_KEY)
    return combo_df
end

"""
    merge_all_scores(; cv_aug, new_aug, labels) -> (lean_aug, combo_df, report)

Build the final per-augmentation and per-combination frames and a small report.
Asserts that every combination has exactly one source and appears once.
"""
function merge_all_scores(; cv_aug::DataFrame, new_aug::DataFrame, labels::DataFrame)
    cv_lean = to_lean_aug(cv_aug, "cv_labeled")
    new_lean = to_lean_aug(new_aug, "new_unlabeled")

    # Stack the two disjoint sources into one augmentation table (dedup is a guard).
    lean_aug = dedup_aug(vcat(cv_lean, new_lean; cols = :union))
    sort!(lean_aug, LEAN_AUG_KEY)
    combo_df = aggregate_combos(lean_aug, labels)

    # Consistency guards.
    n_combo = nrow(combo_df)
    n_unique = nrow(unique(combo_df[:, LEAN_COMBO_KEY]))
    n_combo == n_unique || error("Combination scores are not unique: $(n_combo) rows, $(n_unique) unique keys.")
    multi_source = combine(groupby(lean_aug, LEAN_COMBO_KEY), :source => (s -> length(unique(s))) => :n_src)
    bad = multi_source[multi_source.n_src .> 1, :]
    isempty(bad) || error("$(nrow(bad)) combinations have more than one source; labelled/unlabelled overlap.")

    report = (
        n_cv_aug = nrow(cv_lean),
        n_new_aug = nrow(new_lean),
        n_final_aug = nrow(lean_aug),
        n_combinations = n_combo,
        n_labelled = count(combo_df.has_manual_label),
        n_unlabelled = count(.!combo_df.has_manual_label),
    )
    return lean_aug, combo_df, report
end

"""
    write_lean_outputs(lean_aug, combo_df) -> NamedTuple

Write the per-combination and per-augmentation CSVs; return their paths.
"""
function write_lean_outputs(lean_aug::DataFrame, combo_df::DataFrame)
    CSV.write(LEAN_PARENT_SCORES_PATH, combo_df)
    CSV.write(LEAN_AUGMENTATION_SCORES_PATH, lean_aug)
    return (parent = LEAN_PARENT_SCORES_PATH, augmentation = LEAN_AUGMENTATION_SCORES_PATH)
end
