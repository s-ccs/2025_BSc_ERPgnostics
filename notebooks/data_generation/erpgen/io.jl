# Save generated ERP dataset and settings to JLD2.
function save_erp_dataset(images, labels, metadata;
        dataset_dir::AbstractString = joinpath(pwd(), "datasets"),
        prefix::AbstractString = "erp_dataset",
        settings = NamedTuple(),
        environment = NamedTuple())

    mkpath(dataset_dir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "$(prefix)_$(timestamp).jld2"
    path = joinpath(dataset_dir, filename)
    env = merge((generated_at = timestamp,), environment)

    jldsave(path;
        images = images,
        labels = labels,
        metadata = metadata,
        settings = settings,
        environment = env,
    )

    return path
end
