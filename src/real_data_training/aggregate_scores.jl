# aggregate_scores.jl
#
# Combine the score rows into the final lean, duplicate-free CSVs.
#
#   * labeled   -> combinations that occur in the dataset label files under
#                  datasets/. A label is one of the six ERP image classes or
#                  no_class.
#   * unlabeled -> dataset/event-variable/channel combinations in the scoring
#                  universe that do not occur in those label files.
#
# These two groups are disjoint by construction, so every
# (dataset, sorting_variable, channel) gets exactly ONE model score.
#
# Output files (written next to this script):
#   lean_parent_scores.csv     -> ONE row per (dataset, sorting_variable,
#       channel): the model score and the manual label (or "unlabeled").
#   lean_augmentation_scores.csv -> the per-augmentation detail behind each score
#       (one row per dataset/sort/channel/parent/augmentation).

const LEAN_COMBO_KEY = [:dataset, :sorting_variable, :channel]
const LEAN_AUG_KEY = [:dataset, :sorting_variable, :channel, :parent, :augmentation_id]

"""
    to_lean_aug(df, source) -> DataFrame

Project a source augmentation frame onto the lean augmentation schema.

# Arguments
- `df::DataFrame`: a scored frame with `dataset_key`, `sort_variable`,
  `channel_name`, `parent_image_id`, `augmentation_variant_index`,
  `augmentation_name`, `prob_class`.
- `source::AbstractString`: provenance tag written into the `source` column
  (e.g. `"labeled"` or `"unlabeled"`).

# Returns
- `DataFrame` with columns `dataset, sorting_variable, channel, parent,
  augmentation_id, augmentation_name, score, source`. An empty `df` yields an
  empty frame with the same schema.
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

Keep one row per `(dataset, sorting_variable, channel, parent, augmentation_id)`,
guarding against accidental duplicate augmentation rows.
"""
function dedup_aug(aug::DataFrame)
    isempty(aug) && return aug
    return combine(first, groupby(aug, LEAN_AUG_KEY))
end

"""
    manual_label_lookup(labels) -> Dict{NTuple{3, String}, String}

Map each labeled `(dataset, sorting_variable, channel)` to its manual
`erp_class` (at most one label per combination). Built from
[`combined_label_lookup`](@ref).
"""
function manual_label_lookup(labels::DataFrame)
    lk = combined_label_lookup(labels).erp_class
    return Dict(combo_key(k[1], k[2], k[3]) => v for (k, v) in lk)
end

"""
    aggregate_combos(aug, labels) -> DataFrame

Collapse the lean augmentation table to one score per combination.

# Arguments
- `aug::DataFrame`: lean augmentation rows (see [`to_lean_aug`](@ref)).
- `labels::DataFrame`: the label pool, used to attach the manual label.

# Returns
- `DataFrame` with one row per `(dataset, sorting_variable, channel)`:
  `parent` (canonical full-parent id), `manual_label` (or `"unlabeled"`),
  `has_manual_label`, and `score` = the mean of that combination's augmentation
  scores.
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

    # Attach the canonical full-parent id and the manual label (or "unlabeled").
    labmap = manual_label_lookup(labels)
    combo_df.parent = [join([r.dataset, r.channel, r.sorting_variable, FULL_PARENT_TAG], "::") for r in eachrow(combo_df)]
    combo_df.manual_label = [get(labmap, combo_key(r.dataset, r.sorting_variable, r.channel), "unlabeled") for r in eachrow(combo_df)]
    combo_df.has_manual_label = combo_df.manual_label .!= "unlabeled"

    combo_df = select(combo_df, [:dataset, :sorting_variable, :channel, :parent,
        :manual_label, :has_manual_label, :score])
    sort!(combo_df, LEAN_COMBO_KEY)
    return combo_df
end

"""
    merge_all_scores(; cv_aug, new_aug, labels) -> (lean_aug, combo_df, report)

Combine the two score sources into the final lean tables.

# Arguments
- `cv_aug::DataFrame`: labeled score rows.
- `new_aug::DataFrame`: unlabeled score rows.
- `labels::DataFrame`: the label pool, for the manual-label column.

# Returns
- `lean_aug::DataFrame`: the deduplicated per-augmentation table.
- `combo_df::DataFrame`: one score per combination ([`aggregate_combos`](@ref)).
- `report::NamedTuple`: row counts (labeled/unlabeled/total augmentations and
  combinations).

# Behavior
Errors if a combination is not unique or appears in both groups.
"""
function merge_all_scores(; cv_aug::DataFrame, new_aug::DataFrame, labels::DataFrame)
    cv_lean = to_lean_aug(cv_aug, "labeled")
    new_lean = to_lean_aug(new_aug, "unlabeled")

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
    isempty(bad) || error("$(nrow(bad)) combinations have more than one source; labeled/unlabeled overlap.")

    report = (
        n_labeled_aug = nrow(cv_lean),
        n_unlabeled_aug = nrow(new_lean),
        n_final_aug = nrow(lean_aug),
        n_combinations = n_combo,
        n_labeled = count(combo_df.has_manual_label),
        n_unlabeled = count(.!combo_df.has_manual_label),
    )
    return lean_aug, combo_df, report
end

"""
    write_lean_outputs(lean_aug, combo_df) -> NamedTuple

Write the per-combination and per-augmentation CSVs (overwriting any existing
files) and return their paths as `(parent, augmentation)`.
"""
function write_lean_outputs(lean_aug::DataFrame, combo_df::DataFrame)
    mkpath(dirname(LEAN_PARENT_SCORES_PATH))
    mkpath(dirname(LEAN_AUGMENTATION_SCORES_PATH))
    CSV.write(LEAN_PARENT_SCORES_PATH, combo_df)
    CSV.write(LEAN_AUGMENTATION_SCORES_PATH, lean_aug)
    return (parent = LEAN_PARENT_SCORES_PATH, augmentation = LEAN_AUGMENTATION_SCORES_PATH)
end
