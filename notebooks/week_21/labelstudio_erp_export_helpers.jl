module Week21LabelStudioERPExport

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
    ]))
    for candidate in candidates
        if isdir(joinpath(candidate, "notebooks")) && isdir(joinpath(candidate, "scripts"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir).")
end

const REPO_ROOT = find_repo_root()
const NOTEBOOK_DIR = joinpath(REPO_ROOT, "notebooks", "week_21")
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
const TEST_EXPORT_ROOT = joinpath(NOTEBOOK_DIR, "labelstudio_export_test")
const FULL_EXPORT_ROOT = joinpath(NOTEBOOK_DIR, "labelstudio_export_full")

Pkg.activate(MODEL_ENV_DIR)

using CSV
using CairoMakie
using DataFrames
using Dates
using HDF5
using JSON3
using Printf: @sprintf
using Statistics

include(joinpath(REPO_ROOT, "notebooks", "utils", "erp_image_utils.jl"))
using .ERPImageUtils: clipped_color_stats_quantile_zero_ticks

include(joinpath(REPO_ROOT, "notebooks", "week_15", "try_new_data_helpers.jl"))
using .Week15TryNewData

export REPO_ROOT
export NOTEBOOK_DIR
export TEST_EXPORT_ROOT
export FULL_EXPORT_ROOT
export SOURCE_SPECS
export EXCLUDED_DATASET_KEYS
export source_overview_df
export export_labelstudio_sources
export export_labelstudio_test
export export_labelstudio_full

const TIME_WINDOW_S = Week15TryNewData.REAL_PREVIEW_TIME_WINDOW_S
const LOWPASS_SIGMA_FACTOR = Week15TryNewData.LOWPASS_SIGMA_FACTOR
const PNG_SIZE = (860, 640)

const SORT_COLUMN_BLOCKLIST = Set([
    :dataset_key,
    :component,
    :subject_id,
    :subject_label,
    :source_subject_label,
    :epoch_index,
    :sample_index,
    :source_epoch_index,
    :source_part_index,
    :source_file,
    :source_set_relpath,
    :source_eventlist_relpath,
    :source_event_item,
    :stimulus_onset_s,
    :response_onset_s,
    :trial_onset_s,
    :fixation_onset_s,
    :fixation_offset_s,
    :saccade_onset_s,
    :saccade_offset_s,
    :image_id,
    :class_id,
    :subject,
    :session,
    :run,
    :run_label,
    :session_label,
    :source_note,
    :fixation_index,
    :trial_index,
    :word_index,
    :item_id,
    :page_num,
    :fixated_word,
    :fixated_word_key,
    :story_name,
])

const SOURCE_SPECS = [
    (
        dataset_key = "eye_eeg_freeviewing_fixations",
        label = "EYE-EEG Freeviewing Fixations",
        baseline_correct = false,
        sort_cols = [:fixation_duration_ms, :gaze_x, :gaze_y, :pupil, :condition, :trial_block_index],
    ),
    (
        dataset_key = "erp_core_n170_clean",
        label = "N170",
        baseline_correct = true,
        sort_cols = [:reaction_time_ms, :condition, :stimulus_family, :accuracy],
    ),
    (
        dataset_key = "02_new_eegeyenet_saccades",
        label = "EEGEyeNet minimally processed saccades",
        baseline_correct = false,
        sort_cols = [:saccade_amplitude, :saccade_duration, :saccade_latency_ms, :fixation_duration, :trial_event_type, :condition],
    ),
    (
        dataset_key = "02_new_eeget_rsod",
        label = "EEGET-RSOD raw visual-search fixations",
        baseline_correct = false,
        sort_cols = [:fixation_duration, :saccade_amplitude, :target_present, :saccade_duration, :gaze_x, :gaze_y, :pupil],
    ),
    (
        dataset_key = "eye_eeg_reading_fixations",
        label = "EYE-EEG Reading Fixations",
        baseline_correct = false,
        sort_cols = [:fixation_duration_ms, :gaze_x, :gaze_y, :pupil, :condition, :trial_block_index],
    ),
    (
        dataset_key = "eegeyenet_saccades",
        label = "EEGEyeNet Saccades",
        baseline_correct = false,
        sort_cols = [:saccade_amplitude_px, :saccade_latency_ms, :saccade_duration_ms, :trial_event_type, :condition],
    ),
    (
        dataset_key = "erp_core_n2pc_clean",
        label = "N2pc",
        baseline_correct = true,
        sort_cols = [:reaction_time_ms, :condition, :accuracy, :response_code],
    ),
    (
        dataset_key = "02_new_roamm_reading",
        label = "ROAMM reading fixations with attention labels",
        baseline_correct = false,
        sort_cols = [:fixation_duration, :attention_state, :pupil, :gaze_x, :gaze_y, :condition, :mw_dur],
    ),
    (
        dataset_key = "02_new_unfold_facefreeview",
        label = "Unfold face free-viewing saccades",
        baseline_correct = false,
        sort_cols = [:fixation_duration, :saccade_amplitude, :saccade_duration, :face_condition, :saccade_latency_ms, :saccade_angle],
    ),
    (
        dataset_key = "eye_eeg_sceneviewing_tobii_fixations",
        label = "EYE-EEG Sceneviewing Tobii Fixations",
        baseline_correct = false,
        sort_cols = [:fixation_duration_ms, :gaze_x, :gaze_y, :pupil, :condition, :trial_block_index],
    ),
    (
        dataset_key = "nod_eeg_public",
        label = "NOD-EEG Visual",
        baseline_correct = true,
        sort_cols = [:rt, :super_class, :stim_is_animate, :resp_is_right],
    ),
]

const LABELING_INTERFACE_XML = raw"""<View style="display: flex; flex-direction: column; align-items: center; width: 100%;">
  <Style>
    .erp-choice-row {
      font-size: 26px !important;
      line-height: 1.7;
    }
    .erp-choice-row .ant-checkbox-wrapper,
    .erp-choice-row .ant-radio-wrapper,
    .erp-choice-row .ant-checkbox + span,
    .erp-choice-row .ant-radio + span,
    .erp-choice-row label,
    .erp-choice-row span {
      font-size: 26px !important;
      line-height: 1.7 !important;
    }
    .erp-choice-row .ant-checkbox,
    .erp-choice-row .ant-radio,
    .erp-choice-row input {
      transform: scale(1.55);
      margin-right: 12px;
    }
    .erp-choice-row .ant-checkbox-wrapper,
    .erp-choice-row .ant-radio-wrapper,
    .erp-choice-row label {
      margin-right: 30px !important;
    }
  </Style>
  <View style="width: 98vw; max-width: 2400px;">
    <Header value="$metadata"/>
    <Image name="image" value="$image" zoom="true" defaultZoom="original" width="100%" maxWidth="2400px"/>
  </View>
  <View className="erp-choice-row" style="width: 98vw; max-width: 2400px; margin-top: 10px; padding: 18px 24px; background: #f7f7f7; border-radius: 4px;">
    <Choices name="erp_class" toName="image" choice="single" required="true" layout="inline">
      <Choice value="sigmoid"/>
      <Choice value="one_sided_fan"/>
      <Choice value="two_sided_fan"/>
      <Choice value="diverging_bar"/>
      <Choice value="hourglass"/>
      <Choice value="tilted_bar"/>
      <Choice value="no_class"/>
    </Choices>
  </View>
</View>
"""

const EXCLUDED_DATASET_KEYS = Set([
    # This is the already labelled training source used in
    # notebooks/week_20/resnet_fixation_generalization.ipynb.
    "fixations_dataset",
    "model_test_fixations_dataset",
])

function stable_slug(x)
    y = lowercase(String(x))
    y = replace(y, r"[^a-z0-9]+" => "_")
    y = replace(y, r"(^_+|_+$)" => "")
    return isempty(y) ? "item" : y
end

function assert_unique_sources!(specs)
    keys = String.(getfield.(specs, :dataset_key))
    dupes = sort([key for key in unique(keys) if count(==(key), keys) > 1])
    isempty(dupes) || error("Duplicate data source keys are not allowed: $(join(dupes, ", "))")
    excluded = sort([key for key in keys if key in EXCLUDED_DATASET_KEYS])
    isempty(excluded) || error(
        "Excluded already-labelled data source present in export specs: $(join(excluded, ", "))"
    )
    return nothing
end

function get_meta_string(meta, key::Symbol; default::AbstractString = "")
    key in propertynames(meta) || return default
    value = getproperty(meta, key)
    value === nothing && return default
    return String(value)
end

function get_meta_value(meta, key::Symbol)
    key in propertynames(meta) || return nothing
    return getproperty(meta, key)
end

function week19_notebook_path(dataset_key::AbstractString)
    dir = joinpath(REPO_ROOT, "notebooks", "week_19", "data_sources")
    isdir(dir) || return ""
    for path in sort(filter(p -> endswith(p, ".ipynb"), readdir(dir; join = true)))
        text = read(path, String)
        occursin("DATASET_KEY", text) && occursin(dataset_key, text) && return path
    end
    return ""
end

function source_reference_dict(spec, bundle)
    meta = bundle.metadata
    return Dict(
        "dataset_key" => String(spec.dataset_key),
        "dataset_label" => String(spec.label),
        "component" => get_meta_string(meta, :component; default = String(spec.label)),
        "excluded_already_labelled_training_source" => "notebooks/model_test/real_data_sets/fixations_dataset",
        "source_component" => get_meta_string(meta, :source_component),
        "source_processing_scripts" => get_meta_string(meta, :source_processing_scripts),
        "reader_docs" => get_meta_string(meta, :reader_docs),
        "official_source_examples" => get_meta_value(meta, :official_source_examples),
        "notes" => get_meta_value(meta, :notes),
        "recommended_sort_columns" => get_meta_value(meta, :recommended_sort_columns),
        "selected_sort_columns" => String.(selected_sort_columns(spec, bundle)),
        "metadata_path" => bundle.metadata_path,
        "events_path" => bundle.events_path,
        "h5_path" => bundle.h5_path,
        "week19_notebook_path" => week19_notebook_path(spec.dataset_key),
    )
end

function selected_sort_columns(spec, bundle)
    present = Set(propertynames(bundle.events))
    cols = Symbol[]
    for col in Symbol.(spec.sort_cols)
        col in present || continue
        col in SORT_COLUMN_BLOCKLIST && continue
        Week15TryNewData.unique_nonmissing_count(bundle.events[!, col]) > 1 || continue
        push!(cols, col)
    end
    isempty(cols) && error("No selected sort columns available for $(spec.dataset_key).")
    return cols
end

function load_subject_cache(bundle)
    caches = NamedTuple[]
    for subject_label in bundle.subject_labels
        subj = Week15TryNewData.load_subject_data(bundle.h5_path, subject_label)
        events_subset = Week15TryNewData.select_subject_events(bundle, subject_label)
        epoch_indices = Int.(events_subset.epoch_index)
        post_idx = Week15TryNewData.post_stim_indices(subj.times_s; time_window_s = TIME_WINDOW_S)
        post_times_s = subj.times_s[post_idx]
        push!(caches, (
            subject_label = String(subject_label),
            epochs = subj.epochs,
            times_s = subj.times_s,
            channel_names = String.(subj.channel_names),
            n_timepoints = Int(subj.n_timepoints),
            sfreq_hz = Float64(subj.sfreq_hz),
            events = events_subset,
            epoch_indices = epoch_indices,
            post_idx = post_idx,
            post_time_start_s = Float32(first(post_times_s)),
            post_time_end_s = Float32(last(post_times_s)),
        ))
    end
    return caches
end

function merged_channel_trials_from_cache(bundle, subject_caches, channel_name::AbstractString;
        baseline_correct::Bool)
    data_parts = Matrix{Float32}[]
    event_parts = DataFrame[]
    subject_labels = String[]
    channel_indices = Int[]
    post_len = nothing
    sfreq_hz = nothing
    time_start_s = nothing
    time_end_s = nothing

    for cache in subject_caches
        channel_idx = findfirst(==(channel_name), cache.channel_names)
        channel_idx === nothing && continue

        post_len === nothing || post_len == length(cache.post_idx) ||
            error("Cannot merge $(bundle.dataset_key): post-stimulus length differs across subjects.")
        sfreq_hz === nothing || sfreq_hz == cache.sfreq_hz ||
            error("Cannot merge $(bundle.dataset_key): sampling rate differs across subjects.")
        time_start_s === nothing || time_start_s == cache.post_time_start_s ||
            error("Cannot merge $(bundle.dataset_key): preview start time differs across subjects.")
        time_end_s === nothing || time_end_s == cache.post_time_end_s ||
            error("Cannot merge $(bundle.dataset_key): preview end time differs across subjects.")

        post_len = length(cache.post_idx)
        sfreq_hz = cache.sfreq_hz
        time_start_s = cache.post_time_start_s
        time_end_s = cache.post_time_end_s

        data_full_time_trials = reshape(
            Float32.(cache.epochs[channel_idx, :, cache.epoch_indices]),
            cache.n_timepoints,
            length(cache.epoch_indices),
        )
        if baseline_correct
            data_full_time_trials = Week15TryNewData.baseline_correct_time_trials(
                data_full_time_trials,
                cache.times_s,
            )
        end

        data_time_trials = reshape(
            Float32.(data_full_time_trials[cache.post_idx, :]),
            length(cache.post_idx),
            length(cache.epoch_indices),
        )

        push!(data_parts, data_time_trials)
        push!(event_parts, cache.events)
        push!(subject_labels, cache.subject_label)
        push!(channel_indices, Int(channel_idx))
    end

    isempty(data_parts) && error("Channel $(channel_name) not found in dataset $(bundle.dataset_key).")
    data_time_trials = hcat(data_parts...)
    events_merged = vcat(event_parts...; cols = :union)
    @assert size(data_time_trials, 2) == nrow(events_merged) "Trial count mismatch after merging subjects."

    return (
        data_time_trials = data_time_trials,
        events = events_merged,
        subject_label = length(subject_labels) == 1 ? first(subject_labels) : "merged_experiment",
        channel_idx = first(channel_indices),
        n_trials = nrow(events_merged),
        n_timepoints_post = post_len,
        time_start_s = Float32(time_start_s),
        time_end_s = Float32(time_end_s),
        sampling_rate_hz = Float64(sfreq_hz),
    )
end

function evenly_spaced_ints(first_value::Int, last_value::Int, count::Int)
    count <= 0 && return Int[]
    first_value > last_value && return Int[]
    count == 1 && return [first_value]

    raw = round.(Int, range(first_value, last_value; length = count))
    out = Int[]
    seen = Set{Int}()
    for x in raw
        if !(x in seen)
            push!(out, x)
            push!(seen, x)
        end
    end
    if length(out) < count
        for x in first_value:last_value
            if !(x in seen)
                push!(out, x)
                push!(seen, x)
                length(out) == count && break
            end
        end
    end
    return out
end

function choose_window_len(n_trials::Int, variants_per_origin::Int)
    variants_per_origin <= 1 && return n_trials
    n_trials <= 96 && return n_trials
    return clamp(round(Int, 0.60 * n_trials), min(96, n_trials), n_trials)
end

function variant_descriptors(events::DataFrame, sort_col::Symbol, variants_per_origin::Int)
    n = nrow(events)
    n == 0 && return NamedTuple[]
    if variants_per_origin <= 1
        return [(
            variant = "full",
            variant_index = 1,
            window_start_rank = 1,
            window_end_rank = n,
            window_len = n,
        )]
    end

    window_len = choose_window_len(n, variants_per_origin)
    max_start = max(1, n - window_len + 1)
    starts = evenly_spaced_ints(1, max_start, variants_per_origin - 1)
    variants = NamedTuple[(
        variant = "full",
        variant_index = 1,
        window_start_rank = 1,
        window_end_rank = n,
        window_len = n,
    )]

    for (j, start_rank) in enumerate(starts)
        end_rank = min(n, start_rank + window_len - 1)
        push!(variants, (
            variant = @sprintf("window_%03d", j),
            variant_index = j + 1,
            window_start_rank = Int(start_rank),
            window_end_rank = Int(end_rank),
            window_len = Int(end_rank - start_rank + 1),
        ))
    end
    return variants
end

function trial_indices_for_variant(events::DataFrame, sort_col::Symbol, row)
    if String(row.variant) == "full"
        return collect(1:nrow(events))
    end
    order = Week15TryNewData.trial_sort_order(events, sort_col)
    start_rank = clamp(Int(row.window_start_rank), 1, length(order))
    end_rank = clamp(Int(row.window_end_rank), start_rank, length(order))
    return Int.(order[start_rank:end_rank])
end

function build_candidate_plan(spec, bundle; target_count::Int)
    sort_cols = selected_sort_columns(spec, bundle)
    channel_names = String.(bundle.channel_names)
    origins_count = length(sort_cols) * length(channel_names)
    variants_per_origin = max(1, cld(max(target_count, 1), max(origins_count, 1)))

    rows = NamedTuple[]
    for (sort_rank, sort_col) in enumerate(sort_cols)
        descriptors = variant_descriptors(bundle.events, sort_col, variants_per_origin)
        for (channel_idx, channel_name) in enumerate(channel_names)
            origin_id = join([spec.dataset_key, channel_name, String(sort_col)], "::")
            for desc in descriptors
                image_id = join([origin_id, desc.variant], "::")
                push!(rows, (
                    dataset_key = String(spec.dataset_key),
                    dataset_label = String(spec.label),
                    channel_name = String(channel_name),
                    channel_idx = Int(channel_idx),
                    sort_col = String(sort_col),
                    sort_col_rank = Int(sort_rank),
                    variant = String(desc.variant),
                    variant_index = Int(desc.variant_index),
                    window_start_rank = Int(desc.window_start_rank),
                    window_end_rank = Int(desc.window_end_rank),
                    window_len = Int(desc.window_len),
                    origin_id = origin_id,
                    image_id = image_id,
                ))
            end
        end
    end

    df = DataFrame(rows)
    sort!(df, [:sort_col_rank, :variant_index, :channel_idx])
    return df
end

function select_balanced_candidates(candidates::DataFrame, target_count::Int)
    target = min(target_count, nrow(candidates))
    target <= 0 && return candidates[1:0, :]

    selected = Int[]
    sort_cols = sort(unique(String.(candidates.sort_col)))
    grouped_indices = Dict{String, Vector{Int}}()
    pointers = Dict{String, Int}()

    for sort_col in sort_cols
        idxs = findall(==(sort_col), String.(candidates.sort_col))
        sort!(idxs; by = idx -> (
            Int(candidates.variant_index[idx]),
            Int(candidates.channel_idx[idx]),
            String(candidates.image_id[idx]),
        ))
        grouped_indices[sort_col] = idxs
        pointers[sort_col] = 1
    end

    while length(selected) < target
        progressed = false
        for sort_col in sort_cols
            idxs = grouped_indices[sort_col]
            ptr = pointers[sort_col]
            ptr <= length(idxs) || continue
            push!(selected, idxs[ptr])
            pointers[sort_col] = ptr + 1
            progressed = true
            length(selected) == target && break
        end
        progressed || break
    end

    out = candidates[selected, :]
    out.export_id = collect(1:nrow(out))
    return out
end

function smooth_erp_image(data_time_trials::AbstractMatrix, events_trials::DataFrame, sort_col::Symbol)
    img = Week15TryNewData.build_base_image(data_time_trials, events_trials, sort_col)
    return Week15TryNewData.process_erp_image(
        img,
        nothing;
        lowpass = true,
        sigma_factor = LOWPASS_SIGMA_FACTOR,
    )
end

function axis_ticks(n::Int)
    n <= 1 && return ([1], ["1"])
    mid = Int(round((n + 1) / 2))
    vals = unique([1, mid, n])
    return (vals, string.(vals))
end

function save_erp_png(img::AbstractMatrix, output_path::AbstractString; title::AbstractString)
    clipped, colorrange, tick_vals, tick_labels, cmap = clipped_color_stats_quantile_zero_ticks(img)
    n_trials, n_time = size(clipped)
    xticks = axis_ticks(n_time)
    yticks = axis_ticks(n_trials)

    fig = Figure(size = PNG_SIZE, figure_padding = 18)
    ax = Axis(fig[1, 1];
        title = title,
        titlesize = 14,
        xlabel = "post-stimulus time samples",
        ylabel = "sorted trials",
        xlabelsize = 13,
        ylabelsize = 13,
        xticklabelsize = 11,
        yticklabelsize = 11,
    )
    ax.xticks = xticks
    ax.yticks = yticks

    hm = heatmap!(
        ax,
        1:n_time,
        1:n_trials,
        permutedims(Float32.(clipped), (2, 1));
        colormap = cmap,
        colorrange = colorrange,
    )
    Colorbar(fig[1, 2], hm;
        width = 18,
        ticklabelsize = 10,
        ticks = (tick_vals, tick_labels),
    )
    colgap!(fig.layout, 12)
    resize_to_layout!(fig)
    CairoMakie.save(output_path, fig)
    return nothing
end

function posix_relpath(path::AbstractString, root::AbstractString)
    rel = relpath(path, root)
    startswith(rel, "..") && error("Path $(path) is outside local file document root $(root).")
    return replace(rel, '\\' => '/')
end

function write_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

function write_dataset_readme(path::AbstractString, spec, manifest_path::AbstractString, tasks_path::AbstractString;
        target_count::Int, local_file_document_root::AbstractString)
    text = """
    # $(spec.label) ($(spec.dataset_key))

    This folder was generated by `notebooks/week_21/export_erp_images_labelstudio_week21.ipynb`.

    Export settings:
    - requested tasks: $(target_count)
    - ERP matrix preprocessing: sort -> timepoint z-score -> Gaussian smoothing
    - no 64x64 matrix resizing
    - Label Studio local file root: $(local_file_document_root)

    Files:
    - images/: rendered ERP image PNG files
    - $(basename(manifest_path)): task metadata table
    - $(basename(tasks_path)): Label Studio JSON tasks
    - source_reference.json: source URLs and local reference paths for thesis citation checks
    - labeling_interface.xml: ERP image pattern labels

    For local storage imports in Label Studio, set:
    - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(local_file_document_root)

    Then import $(basename(tasks_path)) or add this folder as a Local Files source storage
    with import method `Tasks`.
    """
    open(path, "w") do io
        write(io, text)
    end
    return path
end

function source_overview_df(specs = SOURCE_SPECS; target_count::Int = 1000)
    assert_unique_sources!(specs)
    rows = NamedTuple[]
    for spec in specs
        bundle = Week15TryNewData.load_clean_dataset_bundle(spec.dataset_key)
        sort_cols = selected_sort_columns(spec, bundle)
        n_origins = length(sort_cols) * length(bundle.channel_names)
        variants_per_origin = max(1, cld(max(target_count, 1), max(n_origins, 1)))
        push!(rows, (
            dataset_key = String(spec.dataset_key),
            dataset_label = String(spec.label),
            n_subjects = length(bundle.subject_labels),
            n_channels = length(bundle.channel_names),
            n_trials = nrow(bundle.events),
            selected_sort_columns = join(string.(sort_cols), ", "),
            n_selected_sort_columns = length(sort_cols),
            n_origin_channel_sort_pairs = n_origins,
            variants_per_origin_for_target = variants_per_origin,
            candidate_capacity_for_target = n_origins * variants_per_origin,
            baseline_correct = Bool(spec.baseline_correct),
        ))
    end
    return DataFrame(rows)
end

function export_one_dataset(spec;
        target_count::Int,
        export_root::AbstractString,
        local_file_document_root::AbstractString = NOTEBOOK_DIR,
        clean::Bool = true)

    bundle = Week15TryNewData.load_clean_dataset_bundle(spec.dataset_key)
    candidate_plan = build_candidate_plan(spec, bundle; target_count = target_count)
    selections = select_balanced_candidates(candidate_plan, target_count)

    dataset_dir = joinpath(export_root, spec.dataset_key)
    images_dir = joinpath(dataset_dir, "images")
    if clean
        rm(dataset_dir; recursive = true, force = true)
    end
    mkpath(images_dir)

    subject_caches = load_subject_cache(bundle)
    channel_cache = Dict{String, Any}()
    manifest_rows = NamedTuple[]
    tasks = Dict{String, Any}[]

    for row in eachrow(selections)
        channel_name = String(row.channel_name)
        origin = get!(channel_cache, channel_name) do
            merged_channel_trials_from_cache(
                bundle,
                subject_caches,
                channel_name;
                baseline_correct = Bool(spec.baseline_correct),
            )
        end

        sort_col = Symbol(row.sort_col)
        trial_idxs = trial_indices_for_variant(origin.events, sort_col, row)
        events_part = origin.events[trial_idxs, :]
        data_part = origin.data_time_trials[:, trial_idxs]
        img = smooth_erp_image(data_part, events_part, sort_col)

        filename = @sprintf(
            "%s_%04d_ch%03d_%s_%s.png",
            stable_slug(spec.dataset_key),
            Int(row.export_id),
            Int(row.channel_idx),
            stable_slug(row.sort_col),
            stable_slug(row.variant),
        )
        output_path = joinpath(images_dir, filename)
        title = "$(spec.label) | ch=$(channel_name) | sort=$(row.sort_col) | $(row.variant)"
        save_erp_png(img, output_path; title = title)

        local_rel = posix_relpath(output_path, local_file_document_root)
        image_url = "/data/local-files/?d=$(local_rel)"
        metadata = "$(spec.label) ($(spec.dataset_key)) | ch=$(channel_name) | sort=$(row.sort_col) | $(row.variant)"

        manifest_row = (
            id = Int(row.export_id),
            image = image_url,
            image_file = filename,
            dataset_key = String(spec.dataset_key),
            dataset_label = String(spec.label),
            subject_label = String(origin.subject_label),
            channel_name = channel_name,
            channel_idx = Int(row.channel_idx),
            sort_variable = String(row.sort_col),
            variant = String(row.variant),
            window_start_rank = Int(row.window_start_rank),
            window_end_rank = Int(row.window_end_rank),
            n_trials = Int(size(img, 1)),
            n_origin_trials = Int(origin.n_trials),
            n_timepoints = Int(size(img, 2)),
            time_start_s = Float32(origin.time_start_s),
            time_end_s = Float32(origin.time_end_s),
            sampling_rate_hz = Float64(origin.sampling_rate_hz),
            baseline_correct = Bool(spec.baseline_correct),
            source_h5 = String(bundle.h5_path),
            source_events = String(bundle.events_path),
            origin_id = String(row.origin_id),
            image_id = String(row.image_id),
            metadata = metadata,
        )
        push!(manifest_rows, manifest_row)

        push!(tasks, Dict(
            "id" => Int(row.export_id),
            "data" => Dict(
                "image" => image_url,
                "metadata" => metadata,
                "dataset_key" => String(spec.dataset_key),
                "dataset_label" => String(spec.label),
                "channel_name" => channel_name,
                "channel_idx" => Int(row.channel_idx),
                "sort_variable" => String(row.sort_col),
                "variant" => String(row.variant),
                "n_trials" => Int(size(img, 1)),
                "n_timepoints" => Int(size(img, 2)),
                "image_id" => String(row.image_id),
            ),
        ))
    end

    manifest_df = DataFrame(manifest_rows)
    manifest_path = joinpath(dataset_dir, "manifest.csv")
    tasks_path = joinpath(dataset_dir, @sprintf("tasks_%s_%04d.json", stable_slug(spec.dataset_key), nrow(manifest_df)))
    interface_path = joinpath(dataset_dir, "labeling_interface.xml")
    config_path = joinpath(dataset_dir, "source_config.json")
    reference_path = joinpath(dataset_dir, "source_reference.json")
    readme_path = joinpath(dataset_dir, "README.md")

    CSV.write(manifest_path, manifest_df)
    write_json(tasks_path, tasks)
    open(interface_path, "w") do io
        write(io, LABELING_INTERFACE_XML)
    end
    reference = source_reference_dict(spec, bundle)
    write_json(reference_path, reference)
    write_json(config_path, Dict(
        "created_at" => string(now()),
        "dataset_key" => String(spec.dataset_key),
        "dataset_label" => String(spec.label),
        "target_count" => target_count,
        "exported_count" => nrow(manifest_df),
        "selected_sort_columns" => unique(String.(manifest_df.sort_variable)),
        "baseline_correct" => Bool(spec.baseline_correct),
        "local_file_document_root" => local_file_document_root,
        "preprocessing" => "sort -> zscore_timepoints -> Gaussian smoothing; no 64x64 matrix resize",
        "label_studio_image_field" => "image",
        "source_reference_json" => reference_path,
        "source_reference" => reference,
    ))
    write_dataset_readme(readme_path, spec, manifest_path, tasks_path;
        target_count = target_count,
        local_file_document_root = local_file_document_root)

    return (
        dataset_key = String(spec.dataset_key),
        dataset_dir = dataset_dir,
        images_dir = images_dir,
        manifest_path = manifest_path,
        tasks_path = tasks_path,
        interface_path = interface_path,
        config_path = config_path,
        reference_path = reference_path,
        readme_path = readme_path,
        exported_count = nrow(manifest_df),
        manifest_df = manifest_df,
    )
end

function export_labelstudio_sources(;
        specs = SOURCE_SPECS,
        target_count::Int,
        export_root::AbstractString,
        local_file_document_root::AbstractString = NOTEBOOK_DIR,
        clean::Bool = true)

    assert_unique_sources!(specs)
    mkpath(export_root)
    if clean
        for spec in specs
            rm(joinpath(export_root, spec.dataset_key); recursive = true, force = true)
        end
    end

    results = NamedTuple[]
    for spec in specs
        println("Exporting ", spec.dataset_key, " -> ", target_count, " tasks")
        result = export_one_dataset(
            spec;
            target_count = target_count,
            export_root = export_root,
            local_file_document_root = local_file_document_root,
            clean = clean,
        )
        push!(results, result)
        GC.gc(false)
    end

    summary_df = DataFrame([(
        dataset_key = r.dataset_key,
        dataset_dir = r.dataset_dir,
        images_dir = r.images_dir,
        manifest_path = r.manifest_path,
        tasks_path = r.tasks_path,
        reference_path = r.reference_path,
        exported_count = r.exported_count,
    ) for r in results])
    summary_path = joinpath(export_root, "summary.csv")
    CSV.write(summary_path, summary_df)

    all_tasks = Dict{String, Any}[]
    global_task_id = 0
    for r in results
        tasks = JSON3.read(read(r.tasks_path, String), Vector{Dict{String, Any}})
        for task in tasks
            global_task_id += 1
            task["id"] = global_task_id
            task["data"]["global_id"] = global_task_id
            push!(all_tasks, task)
        end
    end
    all_tasks_path = joinpath(export_root, @sprintf("tasks_all_sources_%04d_per_source.json", target_count))
    write_json(all_tasks_path, all_tasks)

    return (
        export_root = export_root,
        summary_path = summary_path,
        all_tasks_path = all_tasks_path,
        summary_df = summary_df,
        results = results,
    )
end

function export_labelstudio_test(; kwargs...)
    return export_labelstudio_sources(;
        target_count = 10,
        export_root = TEST_EXPORT_ROOT,
        kwargs...,
    )
end

function export_labelstudio_full(; kwargs...)
    return export_labelstudio_sources(;
        target_count = 1000,
        export_root = FULL_EXPORT_ROOT,
        kwargs...,
    )
end

end
