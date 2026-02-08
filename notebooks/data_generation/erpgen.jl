#=
ERPGen Module - Synthetic ERP Image Generation

FILE STRUCTURE & DEPENDENCIES:
==============================

erpgen.jl (this file)
    │
    ├── config.jl ─────────────────── Configuration structs, defaults, CovariateDesign
    │       │
    │       └─► SimulationConfig, ComponentConfig, PatternConfig,
    │           NoiseConfig, ProcessingConfig, RuntimeConfig
    │
    ├── diagnostics.jl ────────────── Performance profiling system (optional)
    │       │
    │       └─► maybe_diag(), diag_call(), print_diagnostics_tree()
    │
    ├── components.jl ─────────────── ERP components & pattern definitions
    │       │                         [depends on: config, diagnostics]
    │       │
    │       └─► TimeVaryingComponent, basis functions, PATTERN_NAMES,
    │           covariates_for_patterns, pattern_sort_values
    │
    ├── simulation.jl ─────────────── Core ERP simulation
    │       │                         [depends on: config, diagnostics, components]
    │       │
    │       └─► simulate_erp_trials, simulate_pattern_trials
    │
    ├── processing.jl ─────────────── Image processing pipeline
    │       │                         [depends on: config, diagnostics, components]
    │       │
    │       └─► crop_time_window, build_sorted_erpimage, apply_trial_dropout,
    │           render_pattern_images, low-pass filter, imresize
    │
    ├── io.jl ─────────────────────── Dataset persistence
    │       │                         [depends on: diagnostics]
    │       │
    │       └─► save_erp_dataset, load_erp_dataset
    │
    └── dataset_generation.jl ─────── Main API with multi-threading
            │                         [depends on: all above]
            │
            └─► generate_erp_images
=#
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
using Statistics
using StatsModels: @formula
using UnfoldSim

import Pkg
import DataFrames: AbstractDataFrame
import UnfoldSim.simulate_component
import Base: length

const MODULE_PATH = abspath(@__FILE__)
const PROJECT_DIR = abspath(@__DIR__)
const DELTA_LATENCY = Symbol("\u0394latency")

const UNFOLDSIM_VERIFIED = Ref(false)
const UNFOLDSIM_VERIFY_LOCK = ReentrantLock()

# Return the installed UnfoldSim version (or nothing if not installed).
function _installed_unfoldsim_version()
    return maybe_diag(:_installed_unfoldsim_version) do
        for dep in values(Pkg.dependencies())
            dep.name == "UnfoldSim" || continue
            return dep.version
        end
        return nothing
    end
end

# Return the latest UnfoldSim version from reachable registries.
function _latest_unfoldsim_version()
    return maybe_diag(:_latest_unfoldsim_version) do
        latest = nothing
        for reg in Pkg.Registry.reachable_registries()
            uuids = Pkg.Registry.uuids_from_name(reg, "UnfoldSim")
            isempty(uuids) && continue
            info = Pkg.Registry.registry_info(reg.pkgs[first(uuids)])
            versions = keys(info.version_info)
            isempty(versions) && continue
            reg_latest = maximum(versions)
            latest = latest === nothing ? reg_latest : max(latest, reg_latest)
        end
        return latest
    end
end

"""
Verify that UnfoldSim is installed and at the latest version.
Call this once from the main thread before starting parallel work.
Throws an error if the version is outdated - no automatic update.
"""
function verify_unfoldsim_version!()
    return maybe_diag(:verify_unfoldsim_version!) do
        if Threads.threadid() != 1
            error("verify_unfoldsim_version! must be called from the main thread")
        end

        lock(UNFOLDSIM_VERIFY_LOCK)
        try
            if UNFOLDSIM_VERIFIED[]
                return nothing
            end

            current = _installed_unfoldsim_version()
            if current === nothing
                error("UnfoldSim is not installed. Please run: Pkg.add(\"UnfoldSim\")")
            end

            try
                Pkg.Registry.update()
            catch err
                @warn "Could not update registries while checking UnfoldSim version. Continuing with installed version $(current)." exception = err
                UNFOLDSIM_VERIFIED[] = true
                return nothing
            end

            latest = _latest_unfoldsim_version()
            if latest === nothing
                @warn "Could not determine latest UnfoldSim version from registry. Continuing with installed version $(current)."
                UNFOLDSIM_VERIFIED[] = true
                return nothing
            end

            if current != latest
                error("""
                UnfoldSim version mismatch!
                Installed: $(current)
                Latest:    $(latest)

                Please update manually with: Pkg.update(\"UnfoldSim\")
                Then restart Julia.
                """)
            end

            UNFOLDSIM_VERIFIED[] = true
            println("UnfoldSim version $(current) verified.")
            return nothing
        finally
            unlock(UNFOLDSIM_VERIFY_LOCK)
        end
    end
end

function reset_unfoldsim_verification!()
    lock(UNFOLDSIM_VERIFY_LOCK)
    try
        UNFOLDSIM_VERIFIED[] = false
    finally
        unlock(UNFOLDSIM_VERIFY_LOCK)
    end
    return nothing
end

# Backward-compatible wrapper (no automatic update anymore).
function ensure_latest_unfoldsim!(; propagate::Bool = true)
    Base.depwarn("ensure_latest_unfoldsim! is deprecated; use verify_unfoldsim_version!()", :ensure_latest_unfoldsim!)
    verify_unfoldsim_version!()

    if propagate && nworkers() > 0
        current = _installed_unfoldsim_version()
        for p in workers()
            worker_version = try
                Distributed.remotecall_fetch(ERPGen._installed_unfoldsim_version, p)
            catch err
                throw(ErrorException("Failed to read UnfoldSim version from worker $(p): $(err)"))
            end
            if worker_version === nothing
                throw(ErrorException("Worker $(p) has no UnfoldSim installed. Restart the cell/call to reload workers."))
            end
            if current !== nothing && worker_version != current
                throw(ErrorException("Worker $(p) uses UnfoldSim $(worker_version), main uses $(current). Restart the cell/call to reload workers."))
            end
        end
    end

    return nothing
end

include(joinpath(@__DIR__, "erpgen", "config.jl"))
include(joinpath(@__DIR__, "erpgen", "diagnostics.jl"))
include(joinpath(@__DIR__, "erpgen", "components.jl"))
include(joinpath(@__DIR__, "erpgen", "simulation.jl"))
include(joinpath(@__DIR__, "erpgen", "processing.jl"))
include(joinpath(@__DIR__, "erpgen", "io.jl"))
include(joinpath(@__DIR__, "erpgen", "dataset_generation.jl"))

export PATTERN_NAMES, VARIANT_NAMES, DEFAULT_NOISE_POOL, DEFAULT_NOISELEVEL_DISTS
export RESIZE_METHOD_SPECS, DEFAULT_RESIZE_METHODS
export SimulationConfig, ComponentConfig, PatternConfig, NoiseConfig, ProcessingConfig, RuntimeConfig, GenerationConfig
export generate_erp_images, save_erp_dataset, load_erp_dataset
export enable_diagnostics!, reset_diagnostics!, diagnostics_snapshot, diagnostics_totals
export print_diagnostics, print_diagnostics_sorted, print_diagnostics_tree, monitor_workers, start_monitor, stop_monitor!
export ensure_latest_unfoldsim!
export verify_unfoldsim_version!, reset_unfoldsim_verification!

end
