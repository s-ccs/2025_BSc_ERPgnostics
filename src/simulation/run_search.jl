# =============================================================================
# Entry point: run the E4 (64x64 ResNet18) simulator parameter search.
#
# Usage (from the repository root). Use `-t auto` so the simulation parallelises
# across all available CPU threads (the ResNet18 still runs on the GPU):
#     julia -t auto --project=src/simulation src/simulation/run_search.jl
#
# This script only activates the project, loads `SimulationPipeline`, builds a
# `RunConfig`, and calls `run_search`. All experiment logic lives in the module
# under src/simulation/pipeline.
#
# The defaults below are tuned for a quick TEST run: three candidates per search
# strategy. Restore the thesis budget by setting `strategy_budgets` to
# `Dict(:broad_random => 12, :latin_hypercube => 48, :monte_carlo => 48)`.
# =============================================================================

import Pkg

const SIMULATION_DIR = @__DIR__
const REPO_ROOT = abspath(joinpath(SIMULATION_DIR, "..", ".."))

# Activate this project's own environment so all dependencies resolve here.
Pkg.activate(SIMULATION_DIR)

include(joinpath(SIMULATION_DIR, "SimulationPipeline.jl"))
using .SimulationPipeline

"""
    build_test_config() -> RunConfig

Build the configuration for a quick test run: three candidates per strategy,
writing into `src/simulation/outputs/strategy_64x64_resnet18`.

# Returns
- `RunConfig`: the test configuration.
"""
function build_test_config()
    return RunConfig(
        datasets_root = joinpath(REPO_ROOT, "datasets"),
        output_dir = joinpath(SIMULATION_DIR, "outputs", "strategy_64x64_resnet18"),
        use_gpu = gpu_available(),
        # TEST budget: three candidates per strategy (thesis uses 12 / 48 / 48).
        strategy_budgets = Dict(:broad_random => 3, :latin_hypercube => 3, :monte_carlo => 3),
    )
end

# Run only when executed as a script, not when included for interactive use.
if abspath(PROGRAM_FILE) == @__FILE__
    config = build_test_config()
    run_search(config)
end
