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
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")
Pkg.activate(MODEL_ENV_DIR)

using CSV
using DataFrames
using Dates
using HDF5
using JSON3
using Printf: @sprintf
using Statistics

include(joinpath(REPO_ROOT, "notebooks", "week_20", "resnet18_data_source_screening.jl"))
using .Week20ResNet18DataSourceScreening

include(joinpath(REPO_ROOT, "notebooks", "week_21", "labelstudio_erp_export_helpers.jl"))
using .Week21LabelStudioERPExport

const Screening = Week20ResNet18DataSourceScreening
const Export = Week21LabelStudioERPExport
const Week15 = Export.Week15TryNewData

const EXPORT_BATCH_ID = "week21_pattern_positive_followup_1000"
const EXPORT_ROOT = joinpath(Export.NOTEBOOK_DIR, "labelstudio_export_pattern_positive_followup_1000")
const LOCAL_FILE_DOCUMENT_ROOT = Export.NOTEBOOK_DIR
const TARGET_COUNT_PER_SOURCE = 1000
const ANNOTATIONS_CSV = joinpath(Export.NOTEBOOK_DIR, "labelstudio_annotations_all.csv")
const POSITIVE_SORTS_CSV = joinpath(Export.NOTEBOOK_DIR, "labelstudio_positive_sort_variables.csv")
const MODEL_PREDICTIONS_CSV = joinpath(Export.NOTEBOOK_DIR, "labelstudio_export_model_prioritized_200", "model_predictions_all_candidates.csv")
const REFERENCE_DATASET_KEY = "fixations_dataset"
const REFERENCE_LABEL = "Reference Fixation Dataset"
const REFERENCE_DATA_DIR = joinpath(REPO_ROOT, "notebooks", "model_test", "real_data_sets", REFERENCE_DATASET_KEY)
const REFERENCE_H5_PATH = joinpath(REFERENCE_DATA_DIR, "data_fixations.hdf5")
const REFERENCE_EVENTS_PATH = joinpath(REFERENCE_DATA_DIR, "events.csv")
const REFERENCE_SAMPLING_RATE = 512.0
const REFERENCE_PRE_STIM_S = 0.5
const REFERENCE_TIME_ZERO_IDX = Int(round(REFERENCE_PRE_STIM_S * REFERENCE_SAMPLING_RATE)) + 1

const PATTERN_CLASSES = Set([
    "sigmoid",
    "one_sided_fan",
    "two_sided_fan",
    "diverging_bar",
    "hourglass",
    "tilted_bar",
])

function stable_slug(x)
    return Export.stable_slug(String(x))
end

function write_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

function cellstr(x)
    return ismissing(x) ? "" : string(x)
end

function truthy(x)
    lowercase(strip(cellstr(x))) in ("true", "1", "yes")
end

function tracking_key(dataset_key, channel_name, sort_variable)
    return join(String.([dataset_key, channel_name, sort_variable]), "||")
end

function source_status_row(source_status_df::DataFrame, dataset_key::AbstractString)
    idx = findfirst(k -> !ismissing(k) && String(k) == String(dataset_key), source_status_df.dataset_key)
    idx === nothing && error("Dataset $(dataset_key) is not present in source_status_df.")
    return source_status_df[idx, :]
end

function source_label_from_status(row)
    ismissing(row.component) ? String(row.dataset_key) : String(row.component)
end

function dataset_spec_from_status(row, sort_cols)
    return (
        dataset_key = String(row.dataset_key),
        label = source_label_from_status(row),
        baseline_correct = Bool(row.baseline_correct),
        sort_cols = Symbol.(sort_cols),
    )
end

function positive_sort_map(annotations_df::DataFrame)
    out = Dict{String, Set{String}}()
    for row in eachrow(annotations_df)
        hasproperty(row, :is_pattern_class) || continue
        truthy(row.is_pattern_class) || continue
        dataset_key = cellstr(row.dataset_key)
        sort_variable = cellstr(row.sort_variable)
        isempty(sort_variable) && continue
        push!(get!(out, dataset_key, Set{String}()), sort_variable)
    end
    return out
end

function existing_tracking_keys(annotations_df::DataFrame)
    keys = Set{String}()
    for row in eachrow(annotations_df)
        if hasproperty(row, :tracking_key) && !ismissing(row.tracking_key) && !isempty(cellstr(row.tracking_key))
            push!(keys, cellstr(row.tracking_key))
        elseif hasproperty(row, :dataset_key) && hasproperty(row, :channel_name) && hasproperty(row, :sort_variable)
            push!(keys, tracking_key(cellstr(row.dataset_key), cellstr(row.channel_name), cellstr(row.sort_variable)))
        end
    end
    return keys
end

function sort_order_for_dataset(prediction_df::DataFrame, dataset_key::AbstractString, sort_col::AbstractString)
    sdf = prediction_df[
        (String.(prediction_df.dataset_key) .== String(dataset_key)) .&
        (String.(prediction_df.sort_col) .== String(sort_col)),
        :,
    ]
    isempty(sdf) && return typemax(Int)
    return minimum(Int.(sdf.sort_col_rank))
end

function select_balanced_scored_candidates(prediction_df::DataFrame, dataset_key::AbstractString,
        positive_sorts::Set{String}, existing_keys::Set{String};
        target_count::Int = TARGET_COUNT_PER_SOURCE)

    mask = [
        String(row.dataset_key) == String(dataset_key) &&
        String(row.sort_col) in positive_sorts &&
        !(String(row.tracking_key) in existing_keys)
        for row in eachrow(prediction_df)
    ]
    sdf = prediction_df[mask, :]
    isempty(sdf) && return sdf

    sort_cols = sort(collect(positive_sorts); by = col -> sort_order_for_dataset(prediction_df, dataset_key, col))
    sort_cols = [col for col in sort_cols if any(==(col), String.(sdf.sort_col))]
    grouped_indices = Dict{String, Vector{Int}}()
    pointers = Dict{String, Int}()

    for sort_col in sort_cols
        idxs = findall(==(sort_col), String.(sdf.sort_col))
        sort!(idxs; by = idx -> (
            -Float64(sdf.prob_class[idx]),
            -Float64(sdf.confidence[idx]),
            Int(sdf.channel_idx[idx]),
            String(sdf.image_id[idx]),
        ))
        grouped_indices[sort_col] = idxs
        pointers[sort_col] = 1
    end

    selected = Int[]
    target = min(target_count, nrow(sdf))
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

    out = sdf[selected, :]
    out.export_id = collect(1:nrow(out))
    out.export_batch = fill(EXPORT_BATCH_ID, nrow(out))
    out.selection_policy = fill("positive_labeled_sort_vars_round_robin_then_prob_class_desc", nrow(out))
    return out
end

function annotation_history_rows(annotations_df::DataFrame, dataset_key::AbstractString)
    rows = NamedTuple[]
    for row in eachrow(annotations_df[cellstr.(annotations_df.dataset_key) .== String(dataset_key), :])
        push!(rows, (
            tracking_key = cellstr(row.tracking_key),
            dataset_key = cellstr(row.dataset_key),
            dataset_label = cellstr(row.dataset_label),
            channel_name = cellstr(row.channel_name),
            channel_idx = cellstr(row.channel_idx),
            sort_variable = cellstr(row.sort_variable),
            export_batch = cellstr(row.export_batch),
            export_root = cellstr(row.export_root),
            manifest_path = hasproperty(row, :manifest_path) ? cellstr(row.manifest_path) : "",
            image_file = cellstr(row.image_file),
            label_status = cellstr(row.label_status),
            label_studio_project_id = cellstr(row.label_studio_project_id),
            label_studio_project_title = cellstr(row.label_studio_project_title),
            label_studio_task_id = cellstr(row.label_studio_task_id),
            annotation_id = cellstr(row.annotation_id),
            annotator_id = cellstr(row.annotator_id),
            erp_class = cellstr(row.erp_class),
            erp_class_raw = cellstr(row.erp_class_raw),
            erp_class_id = cellstr(row.erp_class_id),
            is_pattern_class = cellstr(row.is_pattern_class),
            annotation_created_at = cellstr(row.annotation_created_at),
            annotation_updated_at = cellstr(row.annotation_updated_at),
            annotation_lead_time = cellstr(row.annotation_lead_time),
        ))
    end
    return rows
end

function write_source_tracking(dataset_dir::AbstractString, annotations_df::DataFrame,
        dataset_key::AbstractString, new_tracking_rows::Vector{NamedTuple})

    rows = vcat(annotation_history_rows(annotations_df, dataset_key), new_tracking_rows)
    df = DataFrame(rows)
    CSV.write(joinpath(dataset_dir, "classified_combinations.csv"), df)
    return df
end

function export_model_dataset(selected_df::DataFrame, source_status_df::DataFrame,
        annotations_df::DataFrame; export_root::AbstractString = EXPORT_ROOT)

    dataset_key = String(selected_df.dataset_key[1])
    status_row = source_status_row(source_status_df, dataset_key)
    bundle = Week15.load_clean_dataset_bundle(dataset_key)
    sort_cols = unique(String.(selected_df.sort_col))
    spec = dataset_spec_from_status(status_row, sort_cols)

    dataset_dir = joinpath(export_root, dataset_key)
    images_dir = joinpath(dataset_dir, "images")
    rm(dataset_dir; recursive = true, force = true)
    mkpath(images_dir)

    subject_caches = Export.load_subject_cache(bundle)
    channel_cache = Dict{String, Any}()
    manifest_rows = NamedTuple[]
    tasks = Dict{String, Any}[]
    new_tracking_rows = NamedTuple[]

    for row in eachrow(selected_df)
        channel_name = String(row.channel_name)
        origin = get!(channel_cache, channel_name) do
            Export.merged_channel_trials_from_cache(
                bundle,
                subject_caches,
                channel_name;
                baseline_correct = Bool(row.baseline_correct),
            )
        end

        sort_col = Symbol(row.sort_col)
        img = Export.smooth_erp_image(origin.data_time_trials, origin.events, sort_col)
        filename = @sprintf(
            "%s_%04d_ch%03d_%s_positive_followup.png",
            stable_slug(dataset_key),
            Int(row.export_id),
            Int(row.channel_idx),
            stable_slug(row.sort_col),
        )
        output_path = joinpath(images_dir, filename)
        title = "$(source_label_from_status(status_row)) | ch=$(channel_name) | sort=$(row.sort_col) | p=$(round(Float64(row.prob_class); digits = 3))"
        Export.save_erp_png(img, output_path; title = title)

        local_rel = Export.posix_relpath(output_path, LOCAL_FILE_DOCUMENT_ROOT)
        image_url = "/data/local-files/?d=$(local_rel)"
        metadata = join([
            "batch=$(EXPORT_BATCH_ID)",
            "dataset=$(dataset_key)",
            "label=$(source_label_from_status(status_row))",
            "ch=$(channel_name)",
            "sort=$(row.sort_col)",
            "positive_sort_from_prior_labels=true",
            "model_prob_class=$(round(Float64(row.prob_class); digits = 4))",
        ], " | ")

        tr_key = String(row.tracking_key)
        manifest_row = (
            id = Int(row.export_id),
            image = image_url,
            image_file = filename,
            export_batch = EXPORT_BATCH_ID,
            dataset_key = dataset_key,
            dataset_label = source_label_from_status(status_row),
            source_notebook = String(row.source_notebook),
            subject_label = String(row.subject_label),
            channel_name = channel_name,
            channel_idx = Int(row.channel_idx),
            sort_variable = String(row.sort_col),
            variant = "full",
            n_trials = Int(origin.n_trials),
            n_timepoints = Int(size(img, 2)),
            time_start_s = Float32(origin.time_start_s),
            time_end_s = Float32(origin.time_end_s),
            sampling_rate_hz = Float64(origin.sampling_rate_hz),
            baseline_correct = Bool(row.baseline_correct),
            model_name = String(row.model_name),
            model_prob_class = Float32(row.prob_class),
            model_prob_no_class = Float32(row.prob_no_class),
            model_confidence = Float32(row.confidence),
            model_class_margin = Float32(row.class_margin),
            model_predicted_class = String(row.predicted_class),
            selection_policy = String(row.selection_policy),
            positive_sort_from_prior_labels = true,
            tracking_key = tr_key,
            source_h5 = String(bundle.h5_path),
            source_events = String(bundle.events_path),
            origin_id = tracking_key(dataset_key, channel_name, row.sort_col),
            image_id = String(row.image_id),
            metadata = metadata,
        )
        push!(manifest_rows, manifest_row)

        push!(tasks, Dict(
            "id" => Int(row.export_id),
            "data" => Dict(
                "image" => image_url,
                "metadata" => metadata,
                "export_batch" => EXPORT_BATCH_ID,
                "dataset_key" => dataset_key,
                "dataset_label" => source_label_from_status(status_row),
                "source_notebook" => String(row.source_notebook),
                "channel_name" => channel_name,
                "channel_idx" => Int(row.channel_idx),
                "sort_variable" => String(row.sort_col),
                "variant" => "full",
                "n_trials" => Int(origin.n_trials),
                "n_timepoints" => Int(size(img, 2)),
                "tracking_key" => tr_key,
                "image_id" => String(row.image_id),
                "model_name" => String(row.model_name),
                "model_prob_class" => Float64(row.prob_class),
                "model_confidence" => Float64(row.confidence),
                "model_predicted_class" => String(row.predicted_class),
                "positive_sort_from_prior_labels" => true,
            ),
        ))

        push!(new_tracking_rows, (
            tracking_key = tr_key,
            dataset_key = dataset_key,
            dataset_label = source_label_from_status(status_row),
            channel_name = channel_name,
            channel_idx = string(Int(row.channel_idx)),
            sort_variable = String(row.sort_col),
            export_batch = EXPORT_BATCH_ID,
            export_root = export_root,
            manifest_path = joinpath(dataset_dir, "manifest.csv"),
            image_file = filename,
            label_status = "exported_for_labeling",
            label_studio_project_id = "",
            label_studio_project_title = "",
            label_studio_task_id = "",
            annotation_id = "",
            annotator_id = "",
            erp_class = "",
            erp_class_raw = "",
            erp_class_id = "",
            is_pattern_class = "",
            annotation_created_at = "",
            annotation_updated_at = "",
            annotation_lead_time = "",
        ))
    end

    manifest_df = DataFrame(manifest_rows)
    manifest_path = joinpath(dataset_dir, "manifest.csv")
    tasks_path = joinpath(dataset_dir, @sprintf("tasks_%s_%04d.json", stable_slug(dataset_key), nrow(manifest_df)))
    interface_path = joinpath(dataset_dir, "labeling_interface.xml")
    reference_path = joinpath(dataset_dir, "source_reference.json")
    config_path = joinpath(dataset_dir, "source_config.json")

    CSV.write(manifest_path, manifest_df)
    write_json(tasks_path, tasks)
    open(interface_path, "w") do io
        write(io, Export.LABELING_INTERFACE_XML)
    end
    write_json(reference_path, Export.source_reference_dict(spec, bundle))
    write_source_tracking(dataset_dir, annotations_df, dataset_key, new_tracking_rows)

    write_json(config_path, Dict(
        "created_at" => string(now()),
        "export_batch" => EXPORT_BATCH_ID,
        "dataset_key" => dataset_key,
        "dataset_label" => source_label_from_status(status_row),
        "target_upper_bound_per_source" => TARGET_COUNT_PER_SOURCE,
        "exported_count" => nrow(manifest_df),
        "positive_sort_variables_from_prior_labels" => sort(collect(Set(String.(manifest_df.sort_variable)))),
        "selection_policy" => "Only sort variables with at least one prior non-no_class label; balanced by sort variable; model-ranked within each sort.",
        "overlap_rule" => "No repeated dataset_key + channel_name + sort_variable from previous Label Studio annotations or this export.",
        "preprocessing" => "sort -> zscore_timepoints -> Gaussian smoothing; no 64x64 matrix resize for exported PNGs",
        "model_predictions_csv" => MODEL_PREDICTIONS_CSV,
        "source_reference_json" => reference_path,
        "manifest_path" => manifest_path,
        "tasks_path" => tasks_path,
    ))

    open(joinpath(dataset_dir, "README.md"), "w") do io
        println(io, "# $(source_label_from_status(status_row)) ($(dataset_key))")
        println(io)
        println(io, "Export batch: `$(EXPORT_BATCH_ID)`")
        println(io)
        println(io, "- Exported images: $(nrow(manifest_df))")
        println(io, "- Sort variables: $(join(sort(collect(Set(String.(manifest_df.sort_variable)))), ", "))")
        println(io, "- Selection: prior non-`no_class` sort variables only; model-ranked within balanced sort-variable rounds.")
        println(io, "- Overlap rule: no repeated `dataset_key + channel_name + sort_variable` from previous annotations.")
    end

    return (
        dataset_key = dataset_key,
        dataset_dir = dataset_dir,
        images_dir = images_dir,
        manifest_path = manifest_path,
        tasks_path = tasks_path,
        reference_path = reference_path,
        exported_count = nrow(manifest_df),
        selected_sort_variables = join(sort(collect(Set(String.(manifest_df.sort_variable)))), ", "),
    )
end

function load_reference_erps()
    h5open(REFERENCE_H5_PATH, "r") do fid
        return read(fid["data"]["data_fixations.hdf5"])
    end
end

function reference_channel_name(channel_idx::Integer)
    return @sprintf("ch%03d", Int(channel_idx))
end

function select_reference_candidates(erps, events::DataFrame, positive_sorts::Set{String},
        existing_keys::Set{String}; target_count::Int = TARGET_COUNT_PER_SOURCE)

    valid_sorts = [s for s in sort(collect(positive_sorts)) if Symbol(s) in propertynames(events)]
    rows = NamedTuple[]
    for sort_col in valid_sorts
        for channel_idx in 1:size(erps, 1)
            channel_name = reference_channel_name(channel_idx)
            tr_key = tracking_key(REFERENCE_DATASET_KEY, channel_name, sort_col)
            tr_key in existing_keys && continue
            push!(rows, (
                dataset_key = REFERENCE_DATASET_KEY,
                dataset_label = REFERENCE_LABEL,
                channel_name = channel_name,
                channel_idx = Int(channel_idx),
                sort_col = sort_col,
                tracking_key = tr_key,
            ))
        end
    end
    df = DataFrame(rows)
    isempty(df) && return df

    sort_cols = valid_sorts
    grouped_indices = Dict{String, Vector{Int}}()
    pointers = Dict{String, Int}()
    for sort_col in sort_cols
        idxs = findall(==(sort_col), String.(df.sort_col))
        sort!(idxs; by = idx -> Int(df.channel_idx[idx]))
        grouped_indices[sort_col] = idxs
        pointers[sort_col] = 1
    end

    selected = Int[]
    target = min(target_count, nrow(df))
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
    out = df[selected, :]
    out.export_id = collect(1:nrow(out))
    out.export_batch = fill(EXPORT_BATCH_ID, nrow(out))
    out.selection_policy = fill("reference_positive_labeled_sort_vars_round_robin_by_channel", nrow(out))
    return out
end

function export_reference_dataset(selected_df::DataFrame, annotations_df::DataFrame;
        export_root::AbstractString = EXPORT_ROOT)

    dataset_key = REFERENCE_DATASET_KEY
    dataset_dir = joinpath(export_root, dataset_key)
    images_dir = joinpath(dataset_dir, "images")
    rm(dataset_dir; recursive = true, force = true)
    mkpath(images_dir)

    erps = load_reference_erps()
    events = CSV.read(REFERENCE_EVENTS_PATH, DataFrame)
    n = min(size(erps, 3), nrow(events))
    events_n = events[1:n, :]

    manifest_rows = NamedTuple[]
    tasks = Dict{String, Any}[]
    new_tracking_rows = NamedTuple[]

    for row in eachrow(selected_df)
        channel_idx = Int(row.channel_idx)
        channel_name = String(row.channel_name)
        sort_col = Symbol(row.sort_col)
        data_time_trials = Float32.(erps[channel_idx, REFERENCE_TIME_ZERO_IDX:end, 1:n])
        img = Export.smooth_erp_image(data_time_trials, events_n, sort_col)

        filename = @sprintf(
            "%s_%04d_ch%03d_%s_positive_followup.png",
            stable_slug(dataset_key),
            Int(row.export_id),
            channel_idx,
            stable_slug(row.sort_col),
        )
        output_path = joinpath(images_dir, filename)
        title = "$(REFERENCE_LABEL) | ch=$(channel_name) | sort=$(row.sort_col)"
        Export.save_erp_png(img, output_path; title = title)

        local_rel = Export.posix_relpath(output_path, LOCAL_FILE_DOCUMENT_ROOT)
        image_url = "/data/local-files/?d=$(local_rel)"
        metadata = join([
            "batch=$(EXPORT_BATCH_ID)",
            "dataset=$(dataset_key)",
            "label=$(REFERENCE_LABEL)",
            "ch=$(channel_name)",
            "sort=$(row.sort_col)",
            "positive_sort_from_prior_labels=true",
        ], " | ")

        tr_key = String(row.tracking_key)
        manifest_row = (
            id = Int(row.export_id),
            image = image_url,
            image_file = filename,
            export_batch = EXPORT_BATCH_ID,
            dataset_key = dataset_key,
            dataset_label = REFERENCE_LABEL,
            source_notebook = "notebooks/model_test/manual_labeling_prepare.ipynb",
            subject_label = "reference_fixations",
            channel_name = channel_name,
            channel_idx = channel_idx,
            sort_variable = String(row.sort_col),
            variant = "full",
            n_trials = Int(size(img, 1)),
            n_timepoints = Int(size(img, 2)),
            time_start_s = 0.0f0,
            time_end_s = Float32((size(img, 2) - 1) / REFERENCE_SAMPLING_RATE),
            sampling_rate_hz = REFERENCE_SAMPLING_RATE,
            baseline_correct = false,
            selection_policy = String(row.selection_policy),
            positive_sort_from_prior_labels = true,
            tracking_key = tr_key,
            source_h5 = REFERENCE_H5_PATH,
            source_events = REFERENCE_EVENTS_PATH,
            origin_id = tracking_key(dataset_key, channel_name, row.sort_col),
            image_id = "$(dataset_key)::$(channel_name)::$(row.sort_col)::full",
            metadata = metadata,
        )
        push!(manifest_rows, manifest_row)

        push!(tasks, Dict(
            "id" => Int(row.export_id),
            "data" => Dict(
                "image" => image_url,
                "metadata" => metadata,
                "export_batch" => EXPORT_BATCH_ID,
                "dataset_key" => dataset_key,
                "dataset_label" => REFERENCE_LABEL,
                "source_notebook" => "notebooks/model_test/manual_labeling_prepare.ipynb",
                "channel_name" => channel_name,
                "channel_idx" => channel_idx,
                "sort_variable" => String(row.sort_col),
                "variant" => "full",
                "n_trials" => Int(size(img, 1)),
                "n_timepoints" => Int(size(img, 2)),
                "tracking_key" => tr_key,
                "image_id" => "$(dataset_key)::$(channel_name)::$(row.sort_col)::full",
                "positive_sort_from_prior_labels" => true,
            ),
        ))

        push!(new_tracking_rows, (
            tracking_key = tr_key,
            dataset_key = dataset_key,
            dataset_label = REFERENCE_LABEL,
            channel_name = channel_name,
            channel_idx = string(channel_idx),
            sort_variable = String(row.sort_col),
            export_batch = EXPORT_BATCH_ID,
            export_root = export_root,
            manifest_path = joinpath(dataset_dir, "manifest.csv"),
            image_file = filename,
            label_status = "exported_for_labeling",
            label_studio_project_id = "",
            label_studio_project_title = "",
            label_studio_task_id = "",
            annotation_id = "",
            annotator_id = "",
            erp_class = "",
            erp_class_raw = "",
            erp_class_id = "",
            is_pattern_class = "",
            annotation_created_at = "",
            annotation_updated_at = "",
            annotation_lead_time = "",
        ))
    end

    manifest_df = DataFrame(manifest_rows)
    manifest_path = joinpath(dataset_dir, "manifest.csv")
    tasks_path = joinpath(dataset_dir, @sprintf("tasks_%s_%04d.json", stable_slug(dataset_key), nrow(manifest_df)))
    reference_path = joinpath(dataset_dir, "source_reference.json")

    CSV.write(manifest_path, manifest_df)
    write_json(tasks_path, tasks)
    open(joinpath(dataset_dir, "labeling_interface.xml"), "w") do io
        write(io, Export.LABELING_INTERFACE_XML)
    end
    write_json(reference_path, Dict(
        "dataset_key" => dataset_key,
        "label" => REFERENCE_LABEL,
        "local_h5" => REFERENCE_H5_PATH,
        "local_events_csv" => REFERENCE_EVENTS_PATH,
        "source_notebook" => joinpath(REPO_ROOT, "notebooks", "model_test", "manual_labeling_prepare.ipynb"),
        "previous_label_projects" => ["week 11", "ERP image 400 more"],
    ))
    write_source_tracking(dataset_dir, annotations_df, dataset_key, new_tracking_rows)

    write_json(joinpath(dataset_dir, "source_config.json"), Dict(
        "created_at" => string(now()),
        "export_batch" => EXPORT_BATCH_ID,
        "dataset_key" => dataset_key,
        "dataset_label" => REFERENCE_LABEL,
        "target_upper_bound_per_source" => TARGET_COUNT_PER_SOURCE,
        "exported_count" => nrow(manifest_df),
        "positive_sort_variables_from_prior_labels" => sort(collect(Set(String.(manifest_df.sort_variable)))),
        "selection_policy" => "Only reference sort variables with at least one prior non-no_class label; balanced by sort variable.",
        "overlap_rule" => "No repeated fixations_dataset + channel_name + sort_variable from previous Label Studio annotations or this export.",
        "preprocessing" => "sort -> zscore_timepoints -> Gaussian smoothing; no 64x64 matrix resize for exported PNGs",
        "source_reference_json" => reference_path,
        "manifest_path" => manifest_path,
        "tasks_path" => tasks_path,
    ))

    open(joinpath(dataset_dir, "README.md"), "w") do io
        println(io, "# $(REFERENCE_LABEL) ($(dataset_key))")
        println(io)
        println(io, "Export batch: `$(EXPORT_BATCH_ID)`")
        println(io)
        println(io, "- Exported images: $(nrow(manifest_df))")
        println(io, "- Sort variables: $(join(sort(collect(Set(String.(manifest_df.sort_variable)))), ", "))")
        println(io, "- Overlap rule: no repeated `dataset_key + channel_name + sort_variable` from previous annotations.")
    end

    return (
        dataset_key = dataset_key,
        dataset_dir = dataset_dir,
        images_dir = images_dir,
        manifest_path = manifest_path,
        tasks_path = tasks_path,
        reference_path = reference_path,
        exported_count = nrow(manifest_df),
        selected_sort_variables = join(sort(collect(Set(String.(manifest_df.sort_variable)))), ", "),
    )
end

function write_all_tasks(export_root::AbstractString, results)
    all_tasks = Dict{String, Any}[]
    global_id = 0
    for r in results
        tasks = JSON3.read(read(r.tasks_path, String), Vector{Dict{String, Any}})
        for task in tasks
            global_id += 1
            task["id"] = global_id
            task["data"]["global_id"] = global_id
            push!(all_tasks, task)
        end
    end
    path = joinpath(export_root, "tasks_all_sources_pattern_positive_followup.json")
    write_json(path, all_tasks)
    return path
end

function assert_no_overlap!(selected_parts::Vector{DataFrame}, existing_keys::Set{String})
    selected_keys = String[]
    for df in selected_parts
        isempty(df) && continue
        append!(selected_keys, String.(df.tracking_key))
    end
    length(selected_keys) == length(unique(selected_keys)) ||
        error("Selection contains duplicate dataset/channel/sort combinations.")
    overlap = [key for key in selected_keys if key in existing_keys]
    isempty(overlap) || error("Selection overlaps existing annotations: $(join(overlap[1:min(end, 10)], ", "))")
    return nothing
end

function write_tracking_table(export_root::AbstractString, annotations_df::DataFrame, results)
    dfs = DataFrame[]
    for dataset_key in unique(cellstr.(annotations_df.dataset_key))
        rows = annotation_history_rows(annotations_df, dataset_key)
        isempty(rows) || push!(dfs, DataFrame(rows))
    end
    for r in results
        path = joinpath(r.dataset_dir, "classified_combinations.csv")
        isfile(path) || continue
        df = CSV.read(path, DataFrame)
        new_df = df[String.(df.export_batch) .== EXPORT_BATCH_ID, :]
        isempty(new_df) || push!(dfs, new_df)
    end
    out = isempty(dfs) ? DataFrame() : vcat(dfs...; cols = :union)
    path = joinpath(export_root, "already_classified_tracking.csv")
    CSV.write(path, out)
    return path
end

function run_export()
    isfile(ANNOTATIONS_CSV) || error("Missing annotations CSV. Run notebooks/week_21/update_labelstudio_annotation_tracking.py first.")
    isfile(MODEL_PREDICTIONS_CSV) || error("Missing model predictions CSV: $(MODEL_PREDICTIONS_CSV)")

    mkpath(EXPORT_ROOT)
    annotations_df = CSV.read(ANNOTATIONS_CSV, DataFrame)
    positive_map = positive_sort_map(annotations_df)
    existing_keys = existing_tracking_keys(annotations_df)
    prediction_df = CSV.read(MODEL_PREDICTIONS_CSV, DataFrame)
    source_status_df = Screening.discover_week19_data_sources()

    model_dataset_keys = sort(unique(String.(prediction_df.dataset_key)))
    selected_parts = DataFrame[]
    summary_rows = NamedTuple[]

    for dataset_key in model_dataset_keys
        pos_sorts = get(positive_map, dataset_key, Set{String}())
        if isempty(pos_sorts)
            push!(summary_rows, (
                dataset_key = dataset_key,
                dataset_label = dataset_key,
                requested_upper_bound = TARGET_COUNT_PER_SOURCE,
                positive_sort_variables = "",
                total_candidate_channel_sort_pairs = count(==(dataset_key), String.(prediction_df.dataset_key)),
                existing_combinations_excluded = count(k -> startswith(k, dataset_key * "||"), existing_keys),
                available_after_exclusion = 0,
                exported_count = 0,
                note = "Skipped: no prior non-no_class labels for this data source.",
            ))
            continue
        end

        selected = select_balanced_scored_candidates(
            prediction_df,
            dataset_key,
            pos_sorts,
            existing_keys;
            target_count = TARGET_COUNT_PER_SOURCE,
        )
        push!(selected_parts, selected)
        status_row = source_status_row(source_status_df, dataset_key)
        push!(summary_rows, (
            dataset_key = dataset_key,
            dataset_label = source_label_from_status(status_row),
            requested_upper_bound = TARGET_COUNT_PER_SOURCE,
            positive_sort_variables = join(sort(collect(pos_sorts)), ", "),
            total_candidate_channel_sort_pairs = count(==(dataset_key), String.(prediction_df.dataset_key)),
            existing_combinations_excluded = count(k -> startswith(k, dataset_key * "||"), existing_keys),
            available_after_exclusion = nrow(selected),
            exported_count = nrow(selected),
            note = nrow(selected) == 0 ?
                "No remaining unique dataset/channel/sort combinations after excluding previous annotations." :
                "Exported remaining unique positive-sort candidates up to the per-source upper bound.",
        ))
    end

    if haskey(positive_map, REFERENCE_DATASET_KEY)
        erps = load_reference_erps()
        events = CSV.read(REFERENCE_EVENTS_PATH, DataFrame)
        selected_ref = select_reference_candidates(
            erps,
            events,
            positive_map[REFERENCE_DATASET_KEY],
            existing_keys;
            target_count = TARGET_COUNT_PER_SOURCE,
        )
        push!(selected_parts, selected_ref)
        push!(summary_rows, (
            dataset_key = REFERENCE_DATASET_KEY,
            dataset_label = REFERENCE_LABEL,
            requested_upper_bound = TARGET_COUNT_PER_SOURCE,
            positive_sort_variables = join(sort(collect(positive_map[REFERENCE_DATASET_KEY])), ", "),
            total_candidate_channel_sort_pairs = size(erps, 1) * length(positive_map[REFERENCE_DATASET_KEY]),
            existing_combinations_excluded = count(k -> startswith(k, REFERENCE_DATASET_KEY * "||"), existing_keys),
            available_after_exclusion = nrow(selected_ref),
            exported_count = nrow(selected_ref),
            note = nrow(selected_ref) == 0 ?
                "No remaining unique reference channel/sort combinations after excluding previous annotations." :
                "Exported remaining unique reference positive-sort candidates up to the per-source upper bound.",
        ))
    end

    assert_no_overlap!(selected_parts, existing_keys)

    results = NamedTuple[]
    for selected in selected_parts
        isempty(selected) && continue
        dataset_key = String(selected.dataset_key[1])
        println("Exporting ", dataset_key, " -> ", nrow(selected), " tasks")
        if dataset_key == REFERENCE_DATASET_KEY
            push!(results, export_reference_dataset(selected, annotations_df; export_root = EXPORT_ROOT))
        else
            push!(results, export_model_dataset(selected, source_status_df, annotations_df; export_root = EXPORT_ROOT))
        end
        GC.gc(false)
    end

    summary_df = DataFrame(summary_rows)
    CSV.write(joinpath(EXPORT_ROOT, "summary.csv"), summary_df)
    nonempty_selected = [df for df in selected_parts if !isempty(df)]
    selected_df = isempty(nonempty_selected) ? DataFrame() : vcat(nonempty_selected...; cols = :union)
    CSV.write(joinpath(EXPORT_ROOT, "selected_candidates.csv"), selected_df)
    all_tasks_path = write_all_tasks(EXPORT_ROOT, results)
    tracking_path = write_tracking_table(EXPORT_ROOT, annotations_df, results)

    write_json(joinpath(EXPORT_ROOT, "export_config.json"), Dict(
        "created_at" => string(now()),
        "export_batch" => EXPORT_BATCH_ID,
        "target_upper_bound_per_source" => TARGET_COUNT_PER_SOURCE,
        "annotations_csv" => ANNOTATIONS_CSV,
        "positive_sort_variables_csv" => POSITIVE_SORTS_CSV,
        "model_predictions_csv" => MODEL_PREDICTIONS_CSV,
        "summary_csv" => joinpath(EXPORT_ROOT, "summary.csv"),
        "selected_candidates_csv" => joinpath(EXPORT_ROOT, "selected_candidates.csv"),
        "already_classified_tracking_csv" => tracking_path,
        "all_tasks_path" => all_tasks_path,
        "local_file_document_root" => LOCAL_FILE_DOCUMENT_ROOT,
        "overlap_rule" => "dataset_key + channel_name + sort_variable is unique against previous annotations and within this export",
        "labeling_interface" => Export.LABELING_INTERFACE_XML,
    ))

    println("Export root: ", EXPORT_ROOT)
    println("All tasks: ", all_tasks_path)
    println(summary_df)
    return (
        export_root = EXPORT_ROOT,
        summary_df = summary_df,
        selected_candidates_df = selected_df,
        results = results,
        all_tasks_path = all_tasks_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_export()
end
