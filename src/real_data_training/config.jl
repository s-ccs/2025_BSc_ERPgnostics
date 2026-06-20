# config.jl
#
# Central configuration for the lean real-data ResNet18 ERP-scoring pipeline.
#
# This file:
#   * locates the repository root,
#   * activates this folder's own package environment (`Project.toml` /
#     `Manifest.toml`), so the pipeline is self-contained,
#   * loads the vendored model engine (`model_engine.jl`): the pretrained
#     ResNet18 + prediction + Gaussian-reference image pipeline,
#   * defines all constants (class ids, augmentation variants, seeds, paths).
#
# It is meant to be `include`d once before the other modules. All pipeline
# files share the same Main-scope definitions; `run_pipeline.jl` includes them
# in order.

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

"""
    find_repo_root(start_dir=@__DIR__) -> String

Locate the repository root.

# Arguments
- `start_dir::AbstractString`: directory to start the upward search from.

# Returns
- `String`: the first ancestor of `start_dir`/`pwd()` that contains both a
  `datasets` and a `src` directory.

# Behavior
Throws an error if no such directory is found.
"""
function find_repo_root(start_dir::AbstractString = @__DIR__)
    candidates = unique(normpath.([
        start_dir,
        pwd(),
        joinpath(start_dir, ".."),
        joinpath(start_dir, "..", ".."),
        joinpath(start_dir, "..", "..", ".."),
        joinpath(pwd(), ".."),
        joinpath(pwd(), "..", ".."),
    ]))
    # The repo root is the first candidate holding both datasets/ and src/.
    for candidate in candidates
        if isdir(joinpath(candidate, "datasets")) && isdir(joinpath(candidate, "src"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir), pwd=$(pwd()).")
end

const REPO_ROOT = find_repo_root()
const REAL_DATA_TRAINING_DIR = @__DIR__

Pkg.activate(REAL_DATA_TRAINING_DIR; io = devnull)

using CSV
using CUDA
using DataFrames
using Dates
using Flux
using ImageFiltering: imfilter
using Images: imresize
using JLD2
using JSON3
using Printf: @sprintf
using Random
using Statistics

# Load the self-contained model engine (`model_engine.jl`): a trimmed, vendored
# copy of the parts of the former Week-20 engine that this pipeline uses --
# the pretrained Metalhead ResNet18 builder, the device/prediction helpers and
# the Gaussian-reference image pipeline. It depends only on this folder's own
# environment, not on `notebooks/`.
if !isdefined(Main, :RealDataModelEngine)
    include(joinpath(REAL_DATA_TRAINING_DIR, "model_engine.jl"))
end

using .RealDataModelEngine

const Generalization = RealDataModelEngine
const CNNUtils = Generalization.ERPCNNExperimentUtils

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
const DATASETS_ROOT = joinpath(REPO_ROOT, "datasets")

# The only outputs of a run (all overwritten each time): the two lean score CSVs
# and the final model trained on all labeled data. They live next to these
# scripts so moving this folder keeps code and outputs together.
const LEAN_AUGMENTATION_SCORES_PATH = joinpath(REAL_DATA_TRAINING_DIR, "lean_augmentation_scores.csv")
const LEAN_PARENT_SCORES_PATH = joinpath(REAL_DATA_TRAINING_DIR, "lean_parent_scores.csv")
const FINAL_MODEL_PATH = joinpath(REAL_DATA_TRAINING_DIR, "final_model.jld2")

const MODEL_NAME = "resnet18_real_data_training_inverse_sort_polarity_binary"

# --------------------------------------------------------------------------- #
# Hyper-parameters (overridable via environment variables)
# --------------------------------------------------------------------------- #
env_config(name::AbstractString, default::AbstractString) = get(ENV, name, default)

const K_FOLDS = parse(Int, env_config("REAL_DATA_TRAINING_FOLDS", "5"))
const TRAIN_EPOCHS = parse(Int, env_config("REAL_DATA_TRAINING_EPOCHS", string(Generalization.TRAIN_EPOCHS)))
const TRAIN_LR = parse(Float32, env_config("REAL_DATA_TRAINING_LR", string(Generalization.TRAIN_LR)))
const TARGET_TRIALS = parse(Int, env_config("REAL_DATA_TRAINING_TARGET_TRIALS", "200"))
const NO_CLASS_CHUNKS_PER_ORIGIN = parse(Int, env_config("REAL_DATA_TRAINING_NO_CLASS_CHUNKS_PER_ORIGIN", "1"))

# Training batch size and label smoothing follow the thesis (batch size 64,
# label smoothing 0.02), overriding the Week-20 engine defaults (32 / none).
"Training batch size on a CUDA device (thesis value 64)."
const TRAIN_BATCHSIZE_GPU = parse(Int, env_config("REAL_DATA_TRAINING_BATCHSIZE_GPU", "64"))
"Training batch size on CPU."
const TRAIN_BATCHSIZE_CPU = parse(Int, env_config("REAL_DATA_TRAINING_BATCHSIZE_CPU", "8"))
"Label-smoothing strength applied to the one-hot targets (thesis value 0.02)."
const LABEL_SMOOTHING = parse(Float32, env_config("REAL_DATA_TRAINING_LABEL_SMOOTHING", "0.02"))

# Image-pipeline constants, taken straight from the Week-20 engine so training,
# CV, final scoring and unlabeled scoring all use identical preprocessing.
const TARGET_SIZE = Generalization.TARGET_SIZE
const LOWPASS_SIGMA = Generalization.LOWPASS_SIGMA
const LOWPASS_KERNEL_SIZE = Generalization.LOWPASS_KERNEL_SIZE
const FILTER_BORDER = Generalization.FILTER_BORDER

# Pattern-class vocabulary. Anything not in here is ignored when reading labels.
const CLASS_ID = Dict(
    "no_class" => 0,
    "sigmoid" => 1,
    "one_sided_fan" => 2,
    "two_sided_fan" => 3,
    "diverging_bar" => 4,
    "hourglass" => 5,
    "tilted_bar" => 6,
)

"""
The four sort/polarity augmentation variants applied identically to every ERP
image in training, CV and scoring. Each is a `NamedTuple` `(name, label,
inverse_sort, inverse_polarity)`.
"""
const AUGMENTATION_VARIANTS = [
    (name = "reference", label = "normal sort, normal polarity", inverse_sort = false, inverse_polarity = false),
    (name = "inverse_sort", label = "inverse sort, normal polarity", inverse_sort = true, inverse_polarity = false),
    (name = "inverse_polarity", label = "normal sort, inverse polarity", inverse_sort = false, inverse_polarity = true),
    (name = "inverse_sort_inverse_polarity", label = "inverse sort, inverse polarity", inverse_sort = true, inverse_polarity = true),
]

"""
Dataset-specific sort variables that should be scored even when they were not
manually labeled. Maps `dataset_key => Vector{sort_variable}`.
"""
const EXTRA_SORT_VARIABLES_BY_DATASET = Dict(
    "fixations_dataset" => [
        "duration",
        "sac_amplitude",
        "sac_endpos_x",
        "sac_endpos_y",
        "sac_startpos_x",
        "sac_startpos_y",
        "sac_vmax",
        "fix_avgpos_x",
        "fix_avgpos_y",
        "fix_avgpupilsize",
        "overlapping",
        "fix_samebox",
        "fix_type",
        "latency",
    ],
)

# Suffix used for the parent id of a whole-parent (all-trials) ERP image.
const FULL_PARENT_TAG = "full_parent"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
"""
    log_step(msg...)

Print a timestamped log line and flush, so progress is visible during long runs.
"""
function log_step(msg...)
    println("[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", msg...)
    flush(stdout)
    return nothing
end

"Convert any cell value to a `String`, mapping `missing`/`nothing` to `\"\"`."
cellstr(x) = (ismissing(x) || x === nothing) ? "" : string(x)

"Log the key configuration constants at the start of a run."
function print_config_banner()
    log_step("REPO_ROOT          = ", REPO_ROOT)
    log_step("OUTPUT_DIR         = ", REAL_DATA_TRAINING_DIR)
    log_step("DATASETS_ROOT      = ", DATASETS_ROOT)
    log_step("MODEL_NAME         = ", MODEL_NAME)
    log_step("K_FOLDS            = ", K_FOLDS)
    log_step("TARGET_TRIALS      = ", TARGET_TRIALS)
    log_step("TRAIN_EPOCHS       = ", TRAIN_EPOCHS)
    log_step("TRAIN_LR           = ", TRAIN_LR)
    log_step("BATCHSIZE_GPU      = ", TRAIN_BATCHSIZE_GPU)
    log_step("LABEL_SMOOTHING    = ", LABEL_SMOOTHING)
    log_step("TARGET_SIZE        = ", TARGET_SIZE)
    return nothing
end
