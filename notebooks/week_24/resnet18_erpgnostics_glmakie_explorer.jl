# Native GLMakie ERPgnostics-style explorer for saved ResNet18 scores.
#
# Run from the repository root:
#
#     julia --project=notebooks/model_test notebooks/week_24/resnet18_erpgnostics_glmakie_explorer.jl
#
# Optional environment variables:
#     WEEK24_SCORE_OUTPUT_DIR  directory containing all_parent_scores.csv
#     WEEK24_START_DATASET     initial dataset key
#     WEEK24_START_SORT        initial sort variable
#     WEEK24_START_CHANNEL     initial channel name

include(joinpath(@__DIR__, "resnet18_erpgnostics_common.jl"))

import GLMakie

GLMakie.activate!()

const NATIVE_SCORE_DIR = get(
    ENV,
    "WEEK24_SCORE_OUTPUT_DIR",
    joinpath(NOTEBOOK_DIR, "outputs", "resnet18_erpgnostics_train_export"),
)

function option_at(options, idx)
    idx isa Integer || return nothing
    isempty(options) && return nothing
    return options[clamp(Int(idx), 1, length(options))]
end

function scored_channel_names(score_df::DataFrame, dataset_key::AbstractString, sort_variable::AbstractString)
    sub = score_df[
        (score_df.dataset_key .== String(dataset_key)) .&
        (score_df.sort_variable .== String(sort_variable)),
        :,
    ]
    sort!(sub, [:channel_idx, :channel_name])
    return String.(sub.channel_name)
end

function first_valid_or(default_value::AbstractString, options::Vector{String})
    isempty(options) && error("No options available.")
    return String(default_value) in options ? String(default_value) : first(options)
end

function native_erpgnostics_explorer(;
        score_dir::AbstractString = NATIVE_SCORE_DIR,
        start_dataset_key::AbstractString = get(ENV, "WEEK24_START_DATASET", ""),
        start_sort_variable::AbstractString = get(ENV, "WEEK24_START_SORT", ""),
        start_channel::AbstractString = get(ENV, "WEEK24_START_CHANNEL", ""),
        max_trials::Int = 700,
        max_timepoints::Int = 520)

    score_run = load_parent_score_outputs(score_dir)
    score_df = score_run.score_df
    dataset_keys = score_dataset_keys(score_df)
    isempty(dataset_keys) && error("No datasets in saved score file: $(parent_scores_path(score_dir))")

    initial_dataset = isempty(start_dataset_key) ?
        initial_dataset_key(score_df) :
        first_valid_or(start_dataset_key, dataset_keys)

    initial_sorts = score_sort_variables(score_df, initial_dataset)
    initial_sort = isempty(start_sort_variable) ?
        first(initial_sorts) :
        first_valid_or(start_sort_variable, initial_sorts)

    initial_channels = scored_channel_names(score_df, initial_dataset, initial_sort)
    initial_channel_name_value = isempty(start_channel) ?
        initial_channel_name(score_df, initial_dataset, initial_sort) :
        first_valid_or(start_channel, initial_channels)

    selected_dataset = Observable(initial_dataset)
    selected_sort = Observable(initial_sort)
    selected_channel = Observable(initial_channel_name_value)

    function valid_sort(dataset_key, sort_variable)
        options = score_sort_variables(score_df, dataset_key)
        return first_valid_or(String(sort_variable), options)
    end

    function valid_channel(dataset_key, sort_variable, channel_name)
        options = scored_channel_names(score_df, dataset_key, sort_variable)
        return first_valid_or(String(channel_name), options)
    end

    valid_sort_obs = lift(selected_dataset, selected_sort) do ds, sv
        valid_sort(ds, sv)
    end

    valid_channel_obs = lift(selected_dataset, valid_sort_obs, selected_channel) do ds, sv, ch
        valid_channel(ds, sv, ch)
    end

    positions_obs = lift(ds -> load_dataset_positions(ds), selected_dataset)
    topo_df_obs = lift(selected_dataset, valid_sort_obs) do ds, sv
        score_positions(score_df, ds, sv)
    end
    topo_x = lift(df -> Float64.(df.x), topo_df_obs)
    topo_y = lift(df -> Float64.(df.y), topo_df_obs)
    topo_score = lift(df -> Float32.(df.score_class), topo_df_obs)
    topo_labels = lift(df -> String.(df.channel_name), topo_df_obs)

    detail_obs = lift(selected_dataset, valid_sort_obs, valid_channel_obs) do ds, sv, ch
        dataset_detail_interactive(
            ds,
            sv,
            ch;
            max_trials = max_trials,
            max_timepoints = max_timepoints,
        )
    end

    visual_obs = lift(d -> erp_image_color_stats(d.image), detail_obs)
    time_obs = lift(d -> d.times, detail_obs)
    trial_obs = lift(d -> d.trials, detail_obs)
    image_obs = lift(v -> v.image, visual_obs)
    image_colorrange_obs = lift(v -> v.colorrange, visual_obs)
    image_ticks_obs = lift(v -> v.ticks, visual_obs)
    image_colormap_obs = lift(v -> v.colormap, visual_obs)
    sort_curve_obs = lift(d -> first(numeric_sort_values(d.sort_values)), detail_obs)
    mean_obs = lift(d -> d.mean_wave, detail_obs)

    title_obs = lift(selected_dataset, valid_sort_obs, valid_channel_obs) do ds, sv, ch
        row = score_row(score_df, ds, sv, ch)
        score = row === nothing ? NaN : Float64(row.score_class)
        manual = row === nothing ? "unlabelled" : cellstr(row.true_erp_class)
        @sprintf("%s | %s | %s | pattern score %.3f | manual %s", ds, ch, sv, score, manual)
    end

    sort_xlabel_obs = lift(valid_sort_obs, detail_obs) do sv, d
        _, suffix = numeric_sort_values(d.sort_values)
        isempty(suffix) ? String(sv) : "$(sv) ($(suffix))"
    end

    fig = Figure(size = (1680, 950), figure_padding = (24, 28, 24, 18), backgroundcolor = :white)
    Label(fig[1, 1:4], title_obs; fontsize = 24, tellwidth = false)

    dataset_menu = Menu(fig[2, 1], options = dataset_keys, default = initial_dataset, width = 360)
    sort_menu = Menu(fig[3, 1], options = initial_sorts, default = initial_sort, width = 360)
    channel_menu = Menu(fig[4, 1], options = initial_channels, default = initial_channel_name_value, width = 360)
    updating_menus = Ref(false)

    function with_menu_update!(f)
        updating_menus[] = true
        try
            return f()
        finally
            updating_menus[] = false
        end
    end

    function set_dataset!(dataset_key::AbstractString)
        dataset_key = String(dataset_key)
        dataset_key in dataset_keys || return nothing

        sort_options = score_sort_variables(score_df, dataset_key)
        sort_variable = first(sort_options)
        channel_options = scored_channel_names(score_df, dataset_key, sort_variable)
        channel_idx = 1

        with_menu_update!() do
            selected_dataset[] = dataset_key
            selected_sort[] = sort_variable
            selected_channel[] = channel_options[Int(channel_idx)]

            sort_menu.options[] = sort_options
            sort_menu.i_selected[] = 1
            sort_menu.selection[] = sort_variable
            channel_menu.options[] = channel_options
            channel_menu.i_selected[] = Int(channel_idx)
            channel_menu.selection[] = channel_options[Int(channel_idx)]
        end
        return nothing
    end

    function set_sort!(sort_variable::AbstractString)
        sort_variable = String(sort_variable)
        sort_options = score_sort_variables(score_df, selected_dataset[])
        sort_variable in sort_options || return nothing

        channel_options = scored_channel_names(score_df, selected_dataset[], sort_variable)
        channel_name = valid_channel(selected_dataset[], sort_variable, selected_channel[])
        channel_idx = findfirst(==(channel_name), channel_options)
        channel_idx === nothing && (channel_idx = 1)

        with_menu_update!() do
            selected_sort[] = sort_variable
            selected_channel[] = channel_options[Int(channel_idx)]
            channel_menu.options[] = channel_options
            channel_menu.i_selected[] = Int(channel_idx)
            channel_menu.selection[] = channel_options[Int(channel_idx)]
        end
        return nothing
    end

    function set_channel!(channel_name::AbstractString)
        channel_name = String(channel_name)
        channel_name in scored_channel_names(score_df, selected_dataset[], valid_sort_obs[]) || return nothing
        selected_channel[] = channel_name
        return nothing
    end

    on(dataset_menu.selection) do ds
        updating_menus[] && return nothing
        ds === nothing && return nothing
        set_dataset!(String(ds))
    end
    on(dataset_menu.i_selected) do idx
        updating_menus[] && return nothing
        ds = option_at(dataset_keys, idx)
        ds === nothing || set_dataset!(ds)
    end

    on(sort_menu.selection) do sv
        updating_menus[] && return nothing
        sv === nothing && return nothing
        set_sort!(String(sv))
    end
    on(sort_menu.i_selected) do idx
        updating_menus[] && return nothing
        sv = option_at(score_sort_variables(score_df, selected_dataset[]), idx)
        sv === nothing || set_sort!(sv)
    end

    on(channel_menu.selection) do ch
        updating_menus[] && return nothing
        ch === nothing && return nothing
        set_channel!(String(ch))
    end
    on(channel_menu.i_selected) do idx
        updating_menus[] && return nothing
        ch = option_at(scored_channel_names(score_df, selected_dataset[], valid_sort_obs[]), idx)
        ch === nothing || set_channel!(ch)
    end

    ax_topo = Axis(fig[5:8, 1]; title = "Click a channel", aspect = DataAspect(), backgroundcolor = :white)
    hidedecorations!(ax_topo)
    hidespines!(ax_topo)
    xlims!(ax_topo, -0.18, 1.18)
    ylims!(ax_topo, -0.10, 1.16)
    draw_head_outline!(ax_topo)

    topo_scatter = scatter!(
        ax_topo,
        topo_x,
        topo_y;
        color = topo_score,
        colormap = :viridis,
        colorrange = (0.0f0, 1.0f0),
        markersize = 18,
        strokewidth = 0.4,
        strokecolor = :gray15,
        inspectable = false,
    )
    text!(
        ax_topo,
        topo_x,
        topo_y;
        text = topo_labels,
        align = (:center, :center),
        fontsize = 7,
        color = :white,
        inspectable = false,
    )
    Colorbar(fig[9, 1], topo_scatter; label = "mean class probability", vertical = false, width = Relative(0.90))

    on(events(ax_topo.scene).mousebutton, priority = 20) do event
        if event.button == Mouse.left && event.action == Mouse.press && is_mouseinside(ax_topo.scene)
            positions = positions_obs[]
            idx = closest_channel_index(mouseposition(ax_topo.scene), positions)
            channel_name = cellstr(positions.channel_name[idx])
            set_channel!(channel_name)
            channel_options = scored_channel_names(score_df, selected_dataset[], valid_sort_obs[])
            menu_idx = findfirst(==(channel_name), channel_options)
            if menu_idx !== nothing
                channel_menu.i_selected[] = Int(menu_idx)
                channel_menu.selection[] = channel_name
            end
            return Consume(true)
        end
        return Consume(false)
    end

    ax_img = Axis(fig[2:8, 2]; xlabel = "time after onset (s)", ylabel = "sorted trials", backgroundcolor = :white)
    hm = heatmap!(
        ax_img,
        time_obs,
        trial_obs,
        lift(img -> permutedims(img, (2, 1)), image_obs);
        colormap = image_colormap_obs,
        colorrange = image_colorrange_obs,
        inspectable = false,
    )
    Colorbar(fig[2:8, 3], hm; label = "z-scored voltage", ticks = image_ticks_obs, width = 18)

    ax_sort = Axis(fig[2:8, 4]; xlabel = sort_xlabel_obs, ylabel = "sorted trials", backgroundcolor = :white)
    lines!(ax_sort, sort_curve_obs, trial_obs; color = :gray20, linewidth = 2, inspectable = false)

    ax_mean = Axis(fig[9, 2]; xlabel = "time after onset (s)", ylabel = "mean ERP", backgroundcolor = :white)
    lines!(ax_mean, time_obs, mean_obs; color = :black, linewidth = 2.2, inspectable = false)
    linkxaxes!(ax_img, ax_mean)

    on(detail_obs) do _
        autolimits!(ax_img)
        autolimits!(ax_sort)
        autolimits!(ax_mean)
    end

    colsize!(fig.layout, 1, 420)
    colsize!(fig.layout, 2, 560)
    colsize!(fig.layout, 3, 36)
    colsize!(fig.layout, 4, 470)
    rowsize!(fig.layout, 9, 120)
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 8)

    return (
        figure = fig,
        score_df = score_df,
        selected_dataset = selected_dataset,
        selected_sort = selected_sort,
        selected_channel = selected_channel,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    app = native_erpgnostics_explorer()
    screen = display(app.figure)
    println("Native GLMakie ERPgnostics explorer is running.")
    println("Score dir: $(NATIVE_SCORE_DIR)")
    println("Close the GLMakie window to exit.")
    try
        wait(screen)
    catch
        readline()
    end
end
