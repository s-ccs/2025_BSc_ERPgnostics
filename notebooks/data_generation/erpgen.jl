module ERPGen

using Distributed
using Distributions
using Interpolations
using Random
using ImageFiltering: KernelFactors, imfilter
using ImageTransformations: Lanczos4OpenCV
using Images: imresize
using Normalization
using Dates
using JLD2
using LinearAlgebra
using Logging
using StatsModels: @formula
using UnfoldSim

import Pkg
import DataFrames: AbstractDataFrame
import UnfoldSim.simulate_component
import Base: length

const MODULE_PATH = abspath(@__FILE__)
const PROJECT_DIR = abspath(@__DIR__)
const DELTA_LATENCY = Symbol("\u0394latency")

include(joinpath(@__DIR__, "erpgen", "unfoldsim_utils.jl"))
include(joinpath(@__DIR__, "erpgen", "diagnostics.jl"))
include(joinpath(@__DIR__, "erpgen", "generation_components.jl"))
include(joinpath(@__DIR__, "erpgen", "generation_patterns.jl"))
include(joinpath(@__DIR__, "erpgen", "config_defaults.jl"))
include(joinpath(@__DIR__, "erpgen", "designs.jl"))
include(joinpath(@__DIR__, "erpgen", "generation_simulation.jl"))
include(joinpath(@__DIR__, "erpgen", "transform_trials.jl"))
include(joinpath(@__DIR__, "erpgen", "postprocess_images.jl"))
include(joinpath(@__DIR__, "erpgen", "dataset_generation.jl"))
include(joinpath(@__DIR__, "erpgen", "dataset_io.jl"))

export PATTERN_NAMES, VARIANT_NAMES, DEFAULT_NOISE_POOL, DEFAULT_NOISELEVEL_DISTS
export RESIZE_METHOD_SPECS, DEFAULT_RESIZE_METHODS
export SimulationConfig, ComponentConfig, PatternConfig, NoiseConfig, ProcessingConfig, RuntimeConfig, GenerationConfig
export generate_erp_images, save_erp_dataset
export enable_diagnostics!, reset_diagnostics!, diagnostics_snapshot
export print_diagnostics, monitor_workers, start_monitor, stop_monitor!
export ensure_latest_unfoldsim!

end
