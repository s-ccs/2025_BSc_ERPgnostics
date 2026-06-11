# config.jl
#
# Central configuration for the lean post-submit ResNet18 ERP-scoring pipeline.
#
# This file:
#   * locates the repository root,
#   * activates the shared `notebooks/model_test` package environment,
#   * loads the Week-20 ResNet/Metalhead engine module (the actual pretrained
#     ResNet18 + training/prediction + Gaussian-reference image pipeline),
#   * defines all constants (class ids, augmentation variants, seeds, paths).
#
# It is meant to be `include`d once before the other modules. All pipeline
# files share the same Main-scope definitions; `run_pipeline.jl` includes them
# in order.

import Pkg

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_NUM_PRECOMPILE_TASKS"] = "1"

"""
    find_repo_root(start_dir) -> String

Walk up from `start_dir`/`pwd()` until a directory containing both `notebooks`
and `datasets` is found.
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
    # The repo root is the first candidate holding both notebooks/ and datasets/.
    for candidate in candidates
        if isdir(joinpath(candidate, "notebooks")) && isdir(joinpath(candidate, "datasets"))
            return candidate
        end
    end
    error("Could not locate repository root from start_dir=$(start_dir), pwd=$(pwd()).")
end

const REPO_ROOT = find_repo_root()
const POST_SUBMIT_DIR = joinpath(REPO_ROOT, "notebooks", "post_submit")
const MODEL_ENV_DIR = joinpath(REPO_ROOT, "notebooks", "model_test")

Pkg.activate(MODEL_ENV_DIR; io = devnull)

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

"""
    quiet_include(path)

`include` a file while suppressing its stdout/stderr (the Week-20 engine prints
a lot during load).
"""
function quiet_include(path::AbstractString)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            include(path)
        end
    end
end

# Load the Week-20 engine that provides the pretrained Metalhead ResNet18,
# the training/prediction helpers and the Gaussian-reference image pipeline.
if !isdefined(Main, :Week20ResNetFixationGeneralization)
    quiet_include(joinpath(REPO_ROOT, "notebooks", "week_20", "resnet_fixation_generalization_experiment.jl"))
end

using .Week20ResNetFixationGeneralization

const Generalization = Week20ResNetFixationGeneralization
const CNNUtils = Generalization.ERPCNNExperimentUtils

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
const DATASETS_ROOT = joinpath(REPO_ROOT, "datasets")
const OUTPUT_DIR = joinpath(POST_SUBMIT_DIR, "outputs")

# Final lean output CSVs (overwritten on every run).
const LEAN_AUGMENTATION_SCORES_PATH = joinpath(POST_SUBMIT_DIR, "lean_augmentation_scores.csv")
const LEAN_PARENT_SCORES_PATH = joinpath(POST_SUBMIT_DIR, "lean_parent_scores.csv")

const MODEL_NAME = "resnet18_post_submit_inverse_sort_polarity_binary"

# --------------------------------------------------------------------------- #
# Hyper-parameters (overridable via environment variables)
# --------------------------------------------------------------------------- #
const K_FOLDS = parse(Int, get(ENV, "POST_SUBMIT_FOLDS", "5"))
const TRAIN_EPOCHS = parse(Int, get(ENV, "POST_SUBMIT_EPOCHS", string(Generalization.TRAIN_EPOCHS)))
const TRAIN_LR = parse(Float32, get(ENV, "POST_SUBMIT_LR", string(Generalization.TRAIN_LR)))
const TARGET_TRIALS = parse(Int, get(ENV, "POST_SUBMIT_TARGET_TRIALS", "200"))
const NO_CLASS_CHUNKS_PER_ORIGIN = parse(Int, get(ENV, "POST_SUBMIT_NO_CLASS_CHUNKS_PER_ORIGIN", "1"))

# Training batch size and label smoothing follow the thesis (batch size 64,
# label smoothing 0.02), overriding the Week-20 engine defaults (32 / none).
const TRAIN_BATCHSIZE_GPU = parse(Int, get(ENV, "POST_SUBMIT_BATCHSIZE_GPU", "64"))
const TRAIN_BATCHSIZE_CPU = parse(Int, get(ENV, "POST_SUBMIT_BATCHSIZE_CPU", "8"))
const LABEL_SMOOTHING = parse(Float32, get(ENV, "POST_SUBMIT_LABEL_SMOOTHING", "0.02"))

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

# The four sort/polarity augmentations applied identically to every ERP image,
# in training, CV, final scoring and unlabeled scoring.
const AUGMENTATION_VARIANTS = [
    (name = "reference", label = "normal sort, normal polarity", inverse_sort = false, inverse_polarity = false),
    (name = "inverse_sort", label = "inverse sort, normal polarity", inverse_sort = true, inverse_polarity = false),
    (name = "inverse_polarity", label = "normal sort, inverse polarity", inverse_sort = false, inverse_polarity = true),
    (name = "inverse_sort_inverse_polarity", label = "inverse sort, inverse polarity", inverse_sort = true, inverse_polarity = true),
]

# Dataset-specific sort variables that should be scored even when they were not
# manually labelled in Label Studio.
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

mkpath(OUTPUT_DIR)

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

cellstr(x) = (ismissing(x) || x === nothing) ? "" : string(x)

function print_config_banner()
    log_step("REPO_ROOT          = ", REPO_ROOT)
    log_step("DATASETS_ROOT      = ", DATASETS_ROOT)
    log_step("OUTPUT_DIR         = ", OUTPUT_DIR)
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
