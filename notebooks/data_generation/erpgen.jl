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

include(joinpath(@__DIR__, "erpgen", "unfoldsim_update.jl"))
include(joinpath(@__DIR__, "erpgen", "diagnostics.jl"))
include(joinpath(@__DIR__, "erpgen", "components.jl"))
include(joinpath(@__DIR__, "erpgen", "defaults.jl"))
include(joinpath(@__DIR__, "erpgen", "patterns.jl"))
include(joinpath(@__DIR__, "erpgen", "processing.jl"))
include(joinpath(@__DIR__, "erpgen", "simulate.jl"))
include(joinpath(@__DIR__, "erpgen", "parallel.jl"))
include(joinpath(@__DIR__, "erpgen", "io.jl"))

export PATTERN_NAMES, VARIANT_NAMES, DEFAULT_NOISE_POOL, DEFAULT_NOISELEVEL_DISTS
export RESIZE_METHOD_SPECS, DEFAULT_RESIZE_METHODS
export generate_erp_images, save_erp_dataset
export enable_diagnostics!, reset_diagnostics!, diagnostics_snapshot
export print_diagnostics, monitor_workers, start_monitor, stop_monitor!
export ensure_latest_unfoldsim!

end
