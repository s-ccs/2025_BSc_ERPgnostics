# =============================================================================
# SimulationPipeline
#
# Self-contained reimplementation of the E4 (64x64 ResNet18) simulator parameter
# search. It searches simulator parameters with three strategies (broad random,
# Latin hypercube, Monte Carlo) so that a ResNet18 trained purely on simulated
# `sigmoid` / `no_class` ERP images transfers as well as possible to the real
# `fixations_dataset`.
#
# The simulator itself (`ERPGen`) is vendored verbatim from
# notebooks/data_generation; everything else in this module is a clean, fully
# documented reimplementation. See README.md for the file layout.
# =============================================================================
module SimulationPipeline

using CSV
using CairoMakie
using DataFrames
using Dates
using Distributions
using Flux
using Flux: onecold, onehotbatch, Chain, Dense
using ImageFiltering: KernelFactors, imfilter
using Images: imresize
using JLD2
using LinearAlgebra: BLAS
using MLUtils: DataLoader
using Metalhead
using Printf: @sprintf
using Random
using Statistics
using Statistics: mean, std, median, cor
using StatsBase: mean_and_std, zscore, quantile
using StatisticalMeasures
using StatisticalMeasures: macro_avg

# CUDA is optional: the pipeline falls back to CPU execution if it is missing.
try
    using CUDA
    using cuDNN
catch err
    @warn "CUDA/cuDNN could not be loaded; CPU execution will be used." exception = (err, catch_backtrace())
end

# Vendored simulator, included unchanged (see README for provenance).
include(joinpath(@__DIR__, "erpgen.jl"))
using .ERPGen

# Pipeline modules, in dependency order.
include(joinpath(@__DIR__, "pipeline", "config.jl"))
include(joinpath(@__DIR__, "pipeline", "image_pipeline.jl"))
include(joinpath(@__DIR__, "pipeline", "parameter_space.jl"))
include(joinpath(@__DIR__, "pipeline", "fixations_data.jl"))
include(joinpath(@__DIR__, "pipeline", "search_strategies.jl"))
include(joinpath(@__DIR__, "pipeline", "resnet18.jl"))
include(joinpath(@__DIR__, "pipeline", "training.jl"))
include(joinpath(@__DIR__, "pipeline", "preview.jl"))
include(joinpath(@__DIR__, "pipeline", "reporting.jl"))
include(joinpath(@__DIR__, "pipeline", "run.jl"))

export RunConfig, run_search, load_real_validation_data, build_base_config, gpu_available

end # module SimulationPipeline
