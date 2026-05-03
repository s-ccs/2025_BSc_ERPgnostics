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

const REQUESTED_DATASET_KEYS = [
    "eye_eeg_freeviewing_fixations",
    "erp_core_n170_clean",
    "02_new_eegeyenet_saccades",
    "eegeyenet_saccades",
    "roamm_reading_fixations",
    "02_new_roamm_reading",
    "eye_eeg_sceneviewing_tobii_fixations",
    "eye_eeg_reading_fixations",
    "02_new_unfold_facefreeview",
    "kilo_word_erp",
    "erp_core_n2pc_clean",
]

const DUPLICATE_CANONICAL = Dict(
    "eegeyenet_saccades" => "02_new_eegeyenet_saccades",
    "roamm_reading_fixations" => "02_new_roamm_reading",
)

const EXPORT_BATCH_ID = "week21_model_prioritized_200"
const EXPORT_ROOT = joinpath(Export.NOTEBOOK_DIR, "labelstudio_export_model_prioritized_200")
const PREVIOUS_EXPORT_ROOTS = [
    Export.TEST_EXPORT_ROOT,
]
const TARGET_COUNT_PER_SOURCE = 200
const LOCAL_FILE_DOCUMENT_ROOT = Export.NOTEBOOK_DIR

function stable_slug(x)
    return Export.stable_slug(String(x))
end

function write_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
    end
    return path
end

function source_status_row(source_status_df::DataFrame, dataset_key::AbstractString)
    idx = findfirst(k -> !ismissing(k) && String(k) == String(dataset_key), source_status_df.dataset_key)
    idx === nothing && error("Dataset $(dataset_key) is not present in source_status_df.")
    return source_status_df[idx, :]
end

function source_component(dataset_key::AbstractString)
    bundle = Week15.load_clean_dataset_bundle(dataset_key)
    return String(bundle.metadata.component)
end

function canonical_dataset_keys()
    out = String[]
    for key in REQUESTED_DATASET_KEYS
        haskey(DUPLICATE_CANONICAL, key) && continue
        key in out || push!(out, key)
    end
    return out
end

function duplicate_resolution_df()
    rows = NamedTuple[]
    for key in REQUESTED_DATASET_KEYS
        canonical = get(DUPLICATE_CANONICAL, key, key)
        included = key == canonical
        reason = if key == "roamm_reading_fixations"
            "Dropped: same ROAMM source URL, subject, and trial count as 02_new_roamm_reading; 02_new_roamm_reading keeps the newer +2.0 s epoch window and attention_state/fixation_duration variables."
        elseif key == "eegeyenet_saccades"
            "Dropped: same EEGEyeNet ds005872/sub-EP10 saccade source family as 02_new_eegeyenet_saccades; 02_new_eegeyenet_saccades keeps the newer minimally processed variables and longer epoch window."
        else
            "Included as unique source."
        end
        push!(rows, (
            requested_dataset_key = key,
            canonical_dataset_key = canonical,
            included = included,
            requested_component = source_component(key),
            canonical_component = source_component(canonical),
            reason = reason,
        ))
    end
    return DataFrame(rows)
end

function write_duplicate_resolution(export_root::AbstractString)
    df = duplicate_resolution_df()
    CSV.write(joinpath(export_root, "duplicate_resolution.csv"), df)
    open(joinpath(export_root, "duplicate_resolution.md"), "w") do io
        println(io, "# Duplicate source resolution")
        println(io)
        println(io, "Canonical export sources:")
        for row in eachrow(df[df.included .== true, :])
            println(io, "- `$(row.canonical_dataset_key)`: $(row.canonical_component)")
        end
        println(io)
        println(io, "Dropped aliases:")
        for row in eachrow(df[df.included .== false, :])
            println(io, "- `$(row.requested_dataset_key)` -> `$(row.canonical_dataset_key)`: $(row.reason)")
        end
    end
    return df
end

function tracking_key(dataset_key, channel_name, sort_variable)
    return join(String.([dataset_key, channel_name, sort_variable]), "||")
end

function previous_tracking_df(previous_roots::Vector{String} = PREVIOUS_EXPORT_ROOTS)
    rows = NamedTuple[]
    for root in previous_roots
        isdir(root) || continue
        manifest_paths = String[]
        for (dir, _, files) in walkdir(root)
            for file in files
                file == "manifest.csv" && push!(manifest_paths, joinpath(dir, file))
            end
        end
        for manifest_path in sort(manifest_paths)
            df = CSV.read(manifest_path, DataFrame)
            all(in.(["dataset_key", "channel_name", "sort_variable"], Ref(names(df)))) || continue
            for row in eachrow(df)
                push!(rows, (
                    tracking_key = tracking_key(row.dataset_key, row.channel_name, row.sort_variable),
                    dataset_key = String(row.dataset_key),
                    channel_name = String(row.channel_name),
                    channel_idx = Int(row.channel_idx),
                    sort_variable = String(row.sort_variable),
                    export_batch = basename(root),
                    export_root = root,
                    manifest_path = manifest_path,
                    image_file = String(row.image_file),
                    label_status = "previously_exported_for_labeling",
                    label_studio_project_id = missing,
                    created_at = string(now()),
                ))
            end
        end
    end
    return DataFrame(rows)
end

function add_tracking_key!(df::DataFrame)
    df.tracking_key = [
        tracking_key(row.dataset_key, row.channel_name, row.sort_col)
        for row in eachrow(df)
    ]
    return df
end

function source_label_from_status(row)
    ismissing(row.component) ? String(row.dataset_key) : String(row.component)
end

function dataset_spec_from_status(row, bundle, sort_cols)
    return (
        dataset_key = String(row.dataset_key),
        label = source_label_from_status(row),
        baseline_correct = Bool(row.baseline_correct),
        sort_cols = Symbol.(sort_cols),
    )
end

function build_model_predictions(source_status_df::DataFrame, dataset_keys::Vector{String})
    println("Training ResNet18 screening model.")
    training = Screening.train_resnet18_screening_model(nepochs = Screening.Generalization.TRAIN_EPOCHS)

    println("Materializing candidate images for selected canonical sources.")
    target_df = Screening.materialize_all_source_images(
        source_status_df;
        dataset_keys = dataset_keys,
        max_channels = nothing,
    )

    println("Classifying ", nrow(target_df), " candidate images.")
    prediction_df = Screening.predict_source_images(
        training.model,
        target_df;
        device = training.device,
    )
    add_tracking_key!(prediction_df)
    return prediction_df, training.train_metrics_df, training.history_df
end

function select_balanced_scored_candidates(prediction_df::DataFrame, dataset_key::AbstractString,
        previous_keys::Set{String}; target_count::Int = TARGET_COUNT_PER_SOURCE)

    sdf = prediction_df[String.(prediction_df.dataset_key) .== String(dataset_key), :]
    sdf = sdf[[!(String(k) in previous_keys) for k in sdf.tracking_key], :]
    isempty(sdf) && return sdf

    sort!(sdf, [:sort_col_rank, :sort_col, :prob_class, :confidence], rev = [false, false, true, true])
    sort_cols = sort(unique(String.(sdf.sort_col)); by = col -> minimum(Int.(sdf.sort_col_rank[String.(sdf.sort_col) .== col])))
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
    out.selection_policy = fill("round_robin_by_sort_col_then_prob_class_desc", nrow(out))
    return out
end

function save_tracking_tables!(export_root::AbstractString, previous_df::DataFrame, selected_df::DataFrame)
    new_rows = NamedTuple[]
    for row in eachrow(selected_df)
        push!(new_rows, (
            tracking_key = String(row.tracking_key),
            dataset_key = String(row.dataset_key),
            channel_name = String(row.channel_name),
            channel_idx = Int(row.channel_idx),
            sort_variable = String(row.sort_col),
            export_batch = EXPORT_BATCH_ID,
            export_root = export_root,
            manifest_path = joinpath(export_root, String(row.dataset_key), "manifest.csv"),
            image_file = "",
            label_status = "exported_for_labeling",
            label_studio_project_id = missing,
            created_at = string(now()),
        ))
    end
    new_df = DataFrame(new_rows)
    combined = vcat(previous_df, new_df; cols = :union)
    CSV.write(joinpath(export_root, "already_classified_tracking.csv"), combined)
    return combined
end

function export_selected_dataset(selected_df::DataFrame, source_status_df::DataFrame,
        previous_df::DataFrame; export_root::AbstractString = EXPORT_ROOT)

    dataset_key = String(selected_df.dataset_key[1])
    status_row = source_status_row(source_status_df, dataset_key)
    bundle = Week15.load_clean_dataset_bundle(dataset_key)
    sort_cols = unique(String.(selected_df.sort_col))
    spec = dataset_spec_from_status(status_row, bundle, sort_cols)

    dataset_dir = joinpath(export_root, dataset_key)
    images_dir = joinpath(dataset_dir, "images")
    rm(dataset_dir; recursive = true, force = true)
    mkpath(images_dir)

    subject_caches = Export.load_subject_cache(bundle)
    channel_cache = Dict{String, Any}()
    manifest_rows = NamedTuple[]
    tasks = Dict{String, Any}[]

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
            "%s_%04d_ch%03d_%s_modelrank.png",
            stable_slug(dataset_key),
            Int(row.export_id),
            Int(row.channel_idx),
            stable_slug(row.sort_col),
        )
        output_path = joinpath(images_dir, filename)
        title = "$(source_label_from_status(status_row)) | ch=$(channel_name) | sort=$(row.sort_col) | model p=$(round(Float64(row.prob_class); digits = 3))"
        Export.save_erp_png(img, output_path; title = title)

        local_rel = Export.posix_relpath(output_path, LOCAL_FILE_DOCUMENT_ROOT)
        image_url = "/data/local-files/?d=$(local_rel)"
        metadata = join([
            "batch=$(EXPORT_BATCH_ID)",
            "dataset=$(dataset_key)",
            "label=$(source_label_from_status(status_row))",
            "ch=$(channel_name)",
            "sort=$(row.sort_col)",
            "model_prob_class=$(round(Float64(row.prob_class); digits = 4))",
            "model_confidence=$(round(Float64(row.confidence); digits = 4))",
        ], " | ")

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
            tracking_key = String(row.tracking_key),
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
                "tracking_key" => String(row.tracking_key),
                "image_id" => String(row.image_id),
                "model_name" => String(row.model_name),
                "model_prob_class" => Float64(row.prob_class),
                "model_confidence" => Float64(row.confidence),
                "model_predicted_class" => String(row.predicted_class),
            ),
        ))
    end

    manifest_df = DataFrame(manifest_rows)
    manifest_path = joinpath(dataset_dir, "manifest.csv")
    tasks_path = joinpath(dataset_dir, @sprintf("tasks_%s_%04d.json", stable_slug(dataset_key), nrow(manifest_df)))
    interface_path = joinpath(dataset_dir, "labeling_interface.xml")
    reference_path = joinpath(dataset_dir, "source_reference.json")
    config_path = joinpath(dataset_dir, "source_config.json")
    tracking_path = joinpath(dataset_dir, "classified_combinations.csv")
    readme_path = joinpath(dataset_dir, "README.md")

    CSV.write(manifest_path, manifest_df)
    write_json(tasks_path, tasks)
    open(interface_path, "w") do io
        write(io, Export.LABELING_INTERFACE_XML)
    end
    write_json(reference_path, Export.source_reference_dict(spec, bundle))

    previous_source = previous_df[String.(previous_df.dataset_key) .== dataset_key, :]
    source_tracking = vcat(
        previous_source[:, intersect(names(previous_source), [
            "tracking_key", "dataset_key", "channel_name", "channel_idx", "sort_variable",
            "export_batch", "export_root", "manifest_path", "image_file", "label_status",
            "label_studio_project_id", "created_at",
        ])],
        DataFrame([(
            tracking_key = String(row.tracking_key),
            dataset_key = String(row.dataset_key),
            channel_name = String(row.channel_name),
            channel_idx = Int(row.channel_idx),
            sort_variable = String(row.sort_col),
            export_batch = EXPORT_BATCH_ID,
            export_root = export_root,
            manifest_path = manifest_path,
            image_file = String(manifest_df.image_file[Int(row.export_id)]),
            label_status = "exported_for_labeling",
            label_studio_project_id = missing,
            created_at = string(now()),
        ) for row in eachrow(selected_df)]);
        cols = :union,
    )
    CSV.write(tracking_path, source_tracking)

    write_json(config_path, Dict(
        "created_at" => string(now()),
        "export_batch" => EXPORT_BATCH_ID,
        "dataset_key" => dataset_key,
        "dataset_label" => source_label_from_status(status_row),
        "requested_count" => TARGET_COUNT_PER_SOURCE,
        "exported_count" => nrow(manifest_df),
        "previous_combinations_excluded" => nrow(previous_source),
        "selected_sort_columns" => unique(String.(manifest_df.sort_variable)),
        "selection_policy" => "balanced round-robin by sort variable, candidates within each sort ranked by ResNet18 prob_class then confidence",
        "overlap_rule" => "dataset_key + channel_name + sort_variable is unique against labelstudio_export_test and within this export",
        "preprocessing" => "sort -> zscore_timepoints -> Gaussian smoothing; no 64x64 matrix resize for exported PNGs",
        "model_preprocessing_for_selection" => "Week-20 ResNet18 screening pipeline uses 64x64 Gaussian-reference images only for ranking",
        "label_studio_image_field" => "image",
        "local_file_document_root" => LOCAL_FILE_DOCUMENT_ROOT,
        "manifest_path" => manifest_path,
        "tasks_path" => tasks_path,
        "tracking_path" => tracking_path,
        "source_reference_json" => reference_path,
    ))

    open(readme_path, "w") do io
        println(io, "# $(source_label_from_status(status_row)) ($(dataset_key))")
        println(io)
        println(io, "Export batch: `$(EXPORT_BATCH_ID)`")
        println(io)
        println(io, "- Requested images: $(TARGET_COUNT_PER_SOURCE)")
        println(io, "- Exported images: $(nrow(manifest_df))")
        println(io, "- Selection: balanced by sort variable, ranked by Week-20 ResNet18 `prob_class` within each sort.")
        println(io, "- Overlap rule: no repeated `dataset_key + channel_name + sort_variable` from `labelstudio_export_test`.")
        println(io, "- Export image preprocessing: sorted ERP matrix, timepoint z-score, Gaussian smoothing, no 64x64 resize.")
        println(io)
        println(io, "Key files:")
        println(io, "- `manifest.csv`")
        println(io, "- `classified_combinations.csv`")
        println(io, "- `$(basename(tasks_path))`")
        println(io, "- `source_reference.json`")
    end

    return (
        dataset_key = dataset_key,
        dataset_dir = dataset_dir,
        images_dir = images_dir,
        manifest_path = manifest_path,
        tasks_path = tasks_path,
        interface_path = interface_path,
        config_path = config_path,
        tracking_path = tracking_path,
        reference_path = reference_path,
        exported_count = nrow(manifest_df),
        requested_count = TARGET_COUNT_PER_SOURCE,
        available_after_previous_exclusion = nrow(selected_df),
    )
end

function assert_no_overlap!(selected_df::DataFrame, previous_df::DataFrame)
    selected_keys = String.(selected_df.tracking_key)
    length(selected_keys) == length(unique(selected_keys)) ||
        error("Selection contains duplicate dataset/channel/sort combinations.")
    previous_keys = Set(String.(previous_df.tracking_key))
    overlap = [key for key in selected_keys if key in previous_keys]
    isempty(overlap) || error("Selection overlaps previous export combinations: $(join(overlap[1:min(end, 10)], ", "))")
    return nothing
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
    all_tasks_path = joinpath(export_root, "tasks_all_sources_model_prioritized.json")
    write_json(all_tasks_path, all_tasks)
    return all_tasks_path
end

function run_export()
    rm(EXPORT_ROOT; recursive = true, force = true)
    mkpath(EXPORT_ROOT)
    duplicate_df = write_duplicate_resolution(EXPORT_ROOT)
    dataset_keys = canonical_dataset_keys()

    raw_components = [
        key for key in dataset_keys
        if occursin(r"(^|[^a-z0-9])raw([^a-z0-9]|$)", lowercase(source_component(key)))
    ]
    isempty(raw_components) || error("Raw sources are not allowed in this export: $(join(raw_components, ", "))")

    source_status_df = Screening.discover_week19_data_sources()
    previous_df = previous_tracking_df()
    previous_keys = Set(String.(previous_df.tracking_key))

    prediction_df, train_metrics_df, history_df = build_model_predictions(source_status_df, dataset_keys)
    CSV.write(joinpath(EXPORT_ROOT, "model_predictions_all_candidates.csv"), prediction_df)
    CSV.write(joinpath(EXPORT_ROOT, "model_train_metrics.csv"), train_metrics_df)
    CSV.write(joinpath(EXPORT_ROOT, "model_train_history.csv"), history_df)

    selected_parts = DataFrame[]
    selection_summary_rows = NamedTuple[]
    for dataset_key in dataset_keys
        selected = select_balanced_scored_candidates(
            prediction_df,
            dataset_key,
            previous_keys;
            target_count = TARGET_COUNT_PER_SOURCE,
        )
        push!(selected_parts, selected)
        total_candidates = nrow(prediction_df[String.(prediction_df.dataset_key) .== dataset_key, :])
        previous_count = count(==(dataset_key), String.(previous_df.dataset_key))
        push!(selection_summary_rows, (
            dataset_key = dataset_key,
            requested_count = TARGET_COUNT_PER_SOURCE,
            total_candidate_channel_sort_pairs = total_candidates,
            previous_test_combinations_excluded = previous_count,
            available_after_previous_exclusion = nrow(selected),
            exported_count = nrow(selected),
            shortage = max(0, TARGET_COUNT_PER_SOURCE - nrow(selected)),
            n_sort_variables_exported = length(unique(String.(selected.sort_col))),
            note = nrow(selected) < TARGET_COUNT_PER_SOURCE ?
                "Fewer than requested because unique dataset/channel/sort combinations are exhausted after excluding previous test export." :
                "Requested count reached.",
        ))
    end

    selected_all = vcat(selected_parts...; cols = :union)
    assert_no_overlap!(selected_all, previous_df)
    CSV.write(joinpath(EXPORT_ROOT, "selected_candidates.csv"), selected_all)

    results = NamedTuple[]
    for selected in selected_parts
        isempty(selected) && continue
        println("Exporting ", selected.dataset_key[1], " -> ", nrow(selected), " selected tasks")
        push!(results, export_selected_dataset(selected, source_status_df, previous_df; export_root = EXPORT_ROOT))
        GC.gc(false)
    end

    summary_df = DataFrame(selection_summary_rows)
    CSV.write(joinpath(EXPORT_ROOT, "summary.csv"), summary_df)
    all_tasks_path = write_all_tasks(EXPORT_ROOT, results)
    tracking_df = save_tracking_tables!(EXPORT_ROOT, previous_df, selected_all)

    write_json(joinpath(EXPORT_ROOT, "export_config.json"), Dict(
        "created_at" => string(now()),
        "export_batch" => EXPORT_BATCH_ID,
        "requested_dataset_keys" => REQUESTED_DATASET_KEYS,
        "canonical_dataset_keys" => dataset_keys,
        "duplicate_resolution_csv" => joinpath(EXPORT_ROOT, "duplicate_resolution.csv"),
        "target_count_per_source" => TARGET_COUNT_PER_SOURCE,
        "local_file_document_root" => LOCAL_FILE_DOCUMENT_ROOT,
        "previous_export_roots" => PREVIOUS_EXPORT_ROOTS,
        "summary_csv" => joinpath(EXPORT_ROOT, "summary.csv"),
        "selected_candidates_csv" => joinpath(EXPORT_ROOT, "selected_candidates.csv"),
        "model_predictions_csv" => joinpath(EXPORT_ROOT, "model_predictions_all_candidates.csv"),
        "already_classified_tracking_csv" => joinpath(EXPORT_ROOT, "already_classified_tracking.csv"),
        "all_tasks_path" => all_tasks_path,
        "labeling_interface" => Export.LABELING_INTERFACE_XML,
    ))

    println("Export root: ", EXPORT_ROOT)
    println("All tasks: ", all_tasks_path)
    println("Tracking rows: ", nrow(tracking_df))
    println(summary_df)
    return (
        export_root = EXPORT_ROOT,
        summary_df = summary_df,
        duplicate_df = duplicate_df,
        selected_candidates_df = selected_all,
        previous_tracking_df = previous_df,
        results = results,
        all_tasks_path = all_tasks_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_export()
end
