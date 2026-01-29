const UNFOLDSIM_UPDATED = Ref(false)
const UNFOLDSIM_LOCK = ReentrantLock()

# Return the installed UnfoldSim version (or nothing if not installed).
function _installed_unfoldsim_version()
    for dep in values(Pkg.dependencies())
        dep.name == "UnfoldSim" || continue
        return dep.version
    end
    return nothing
end

# Return the latest UnfoldSim version from reachable registries.
function _latest_unfoldsim_version()
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

# Update UnfoldSim only if the installed version is behind the registry.
function ensure_latest_unfoldsim!(; propagate::Bool = true)
    updated = false
    if !UNFOLDSIM_UPDATED[]
        lock(UNFOLDSIM_LOCK)
        try
            if !UNFOLDSIM_UPDATED[]
                Pkg.Registry.update()
                current = _installed_unfoldsim_version()
                latest = _latest_unfoldsim_version()
                if latest === nothing
                    throw(ErrorException("Unable to resolve latest UnfoldSim version from registry."))
                end
                if current === nothing || current != latest
                    Pkg.update("UnfoldSim")
                    updated = true
                end
                UNFOLDSIM_UPDATED[] = true
            end
        catch err
            UNFOLDSIM_UPDATED[] = false
            throw(ErrorException("Failed to update UnfoldSim to the latest version: $(err)"))
        finally
            unlock(UNFOLDSIM_LOCK)
        end
    end

    current = _installed_unfoldsim_version()
    if updated && current !== nothing
        println("UnfoldSim was updated to version $(current). Restart the cell/call to reload workers.")
    end

    if propagate && nworkers() > 0
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
