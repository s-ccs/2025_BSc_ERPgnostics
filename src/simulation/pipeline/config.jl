"""
    RunConfig

Holds every tunable setting for one E4 (64x64 ResNet18) parameter search.

A single immutable configuration is built once in the entry point and then
threaded through the data loading, search, training, and reporting functions.

# Fields
- `target_size::Tuple{Int,Int}`: ERP image resolution fed to the classifier.
- `pipeline_name::Symbol`: image pipeline variant (`:gaussian_reference`).
- `low_pass_factor::Float32`: Gaussian smoothing strength of the shared pipeline.
- `lowpass_kernel_size::Tuple{Int,Int}`: Gaussian kernel size (odd in both dims).
- `filter_border::String`: border handling for the smoothing filter.
- `eval_repeats::Int`: number of repeats averaged per strategy candidate.
- `baseline_repeats::Int`: repeats for the hand-crafted baseline configuration.
- `n_per_pattern::Int`: simulated images generated per class per candidate.
- `sanity_bacc_min::Float64`: minimum balanced accuracy for a non-collapsed run.
- `sanity_class_balance_frac::Float64`: minimum predicted fraction per class.
- `batchsize::Int`: training/inference batch size.
- `train_epochs::Int`: epochs for the default training profile.
- `train_lr::Float32`: learning rate for the default training profile.
- `label_smoothing::Float32`: one-hot label smoothing strength.
- `class_weights::Vector{Float32}`: per-class loss weights (`[no_class, sigmoid]`).
- `strategy_budgets`: number of candidates per search strategy.
- `datasets_root::String`: directory that holds the dataset bundles.
- `fixations_dataset_key::String`: dataset folder name of the real fixations data.
- `output_dir::String`: directory where all CSVs and figures are written.
- `use_gpu::Bool`: whether to run training and inference on CUDA.
- `write_preview::Bool`: whether to render the sim-vs-real preview figure.
"""
Base.@kwdef struct RunConfig
    target_size::Tuple{Int, Int} = (64, 64)
    pipeline_name::Symbol = :gaussian_reference
    low_pass_factor::Float32 = 75.0f0
    lowpass_kernel_size::Tuple{Int, Int} = (21, 21)
    filter_border::String = "reflect"

    eval_repeats::Int = 3
    baseline_repeats::Int = 1
    n_per_pattern::Int = 1000

    sanity_bacc_min::Float64 = 0.55
    sanity_class_balance_frac::Float64 = 0.05

    batchsize::Int = 64
    train_epochs::Int = 8
    train_lr::Float32 = 3.0f-4
    label_smoothing::Float32 = 0.02f0
    class_weights::Vector{Float32} = Float32[1.0, 1.0]

    strategy_budgets::Dict{Symbol, Int} = Dict(
        :broad_random => 12,
        :latin_hypercube => 48,
        :monte_carlo => 48,
    )

    datasets_root::String
    fixations_dataset_key::String = "fixations_dataset"
    output_dir::String

    use_gpu::Bool = false
    write_preview::Bool = true
end

# Class names and binary label encoding used throughout the pipeline.
const SIGMOID_CLASS = "sigmoid"
const NO_CLASS = "no_class"
const SEARCH_METHODS = (:broad_random, :latin_hypercube, :monte_carlo)

"""
    new_seed() -> UInt64

Return a fresh random seed from the system nanosecond clock. Each call gives a
new value, so runs are independently random by default; there is no global root
seed to manage.

# Returns
- `UInt64`: a fresh seed suitable for `Random.seed!` or `Random.Xoshiro`.
"""
new_seed() = time_ns()

"""
    default_training_profiles(config::RunConfig)

Build the ordered list of training profiles tried by the sanity gate.

The first profile that produces a non-collapsed run on the real validation set
is reused for the whole strategy sweep. Later profiles act as fallbacks with
more epochs, a lower learning rate, or a randomly initialised network.

# Arguments
- `config::RunConfig`: supplies the default epochs, learning rate, batch size,
  class weights, and label smoothing.

# Returns
- `Vector{<:NamedTuple}`: one profile per entry, in the order they are tried.
"""
function default_training_profiles(config::RunConfig)
    return [
        (name = "pretrained_default", model_init = :pretrained, nepochs = config.train_epochs, lr = config.train_lr,
            batchsize = config.batchsize, class_weights = config.class_weights, label_smoothing = config.label_smoothing),
        (name = "pretrained_longer_same_lr", model_init = :pretrained, nepochs = 16, lr = config.train_lr,
            batchsize = config.batchsize, class_weights = config.class_weights, label_smoothing = config.label_smoothing),
        (name = "pretrained_lower_lr", model_init = :pretrained, nepochs = 12, lr = 1.0f-4,
            batchsize = config.batchsize, class_weights = config.class_weights, label_smoothing = config.label_smoothing),
        (name = "random_init_resnet18", model_init = :random, nepochs = 12, lr = config.train_lr,
            batchsize = config.batchsize, class_weights = config.class_weights, label_smoothing = config.label_smoothing),
    ]
end

"""
    to_device(x, use_gpu::Bool)

Move an array or model to the GPU when `use_gpu` is true, otherwise return it
unchanged. Centralising the device choice keeps the training code readable.

# Arguments
- `x`: any Flux-movable value (array or model).
- `use_gpu::Bool`: whether a functional CUDA device should be used.

# Returns
- The value placed on the selected device.
"""
to_device(x, use_gpu::Bool) = use_gpu ? Flux.gpu(x) : x

"""
    set_all_seeds!(seed, use_gpu::Bool) -> Xoshiro

Seed Julia's global RNG (and CUDA's RNG when on GPU) and return a fresh
`Xoshiro` generator seeded the same way for local, explicit random draws.

# Arguments
- `seed`: integer seed to install.
- `use_gpu::Bool`: whether to also seed the CUDA RNG.

# Returns
- `Random.Xoshiro`: a generator seeded with `seed`.
"""
function set_all_seeds!(seed, use_gpu::Bool)
    Random.seed!(seed)
    if use_gpu && isdefined(CUDA, :seed!)
        # CUDA seeding is best-effort: a failure must not abort the run.
        try
            CUDA.seed!(seed)
        catch err
            @warn "CUDA.seed! failed; continuing after Random.seed!." exception = (err, catch_backtrace())
        end
    end
    return Random.Xoshiro(seed)
end

"""
    cleanup_device!(use_gpu::Bool)

Run garbage collection and, on GPU, reclaim CUDA memory between candidates so
long sweeps do not accumulate device allocations.

# Arguments
- `use_gpu::Bool`: whether to reclaim CUDA memory after collecting garbage.

# Returns
- `nothing`.
"""
function cleanup_device!(use_gpu::Bool)
    GC.gc()
    use_gpu && CUDA.reclaim()
    return nothing
end
