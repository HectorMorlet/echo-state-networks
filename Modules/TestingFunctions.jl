module TestingFunctions

using CairoMakie
using StatsBase
using Distributed
using IJulia
using JSON
using Dates

include("ONReservoir.jl")
using .ONReservoir
include("EchoStateNetworks.jl")
using .EchoStateNetworks
include("TurningError.jl")
using .TurningError
include("EchoStateNetworksStochastic.jl")
using .EchoStateNetworksStochastic
include("EchoStateNetworksReadoutSwitching.jl")
using .EchoStateNetworksReadoutSwitching

export TestingParameters, create_testing_params, compare_preds, create_pred_for_params_single_step, create_pred_for_params_free_run, create_pred_for_params_multi_step, test_params, test_multi_step_multi_trial, graph_multi_step_RMSE_vs_n_steps, test_single_step, test_freerun, RMSE, test_multi_step_multi_trial_singular, find_test, check_if_test_exists, test_multi_step_n_steps, test_multi_step_k, save_file, chart_tests, quick_graph_series

struct TestingParameters
    mask_states_b4_readout::Bool
    stochastic::Bool
    stochastic_rescale_V_rec::Bool
    readout_switching::Bool
end

function create_testing_params(;mask_states_b4_readout=false, stochastic=false, stochastic_rescale_V_rec=false, readout_switching=false)
    return(TestingParameters(mask_states_b4_readout, stochastic, stochastic_rescale_V_rec, readout_switching))
end

function RMSE(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred) .^ 2))
end

function compare_preds(lo_test, vanilla_preds, ON_preds, x_start, x_end; calculate_error=true, ignore_first=0, offset=0, mark_every=0)
    ON_preds_cropped = ON_preds[ignore_first+1:end]
    vanilla_preds_cropped = vanilla_preds[ignore_first+1:min(length(ON_preds), end)]
    lo_test_cropped = lo_test[offset+ignore_first+1:min(length(ON_preds_cropped)+offset+ignore_first, end)]

    if calculate_error
        println("Overall RMSE:")
        println("    Vanilla: ", RMSE(vanilla_preds_cropped, lo_test_cropped))
        println("    Ordinal network reservoir: ", RMSE(ON_preds_cropped, lo_test_cropped))
        println("Turning partition RMSE:")
        println("    Vanilla: ", turning_partition_RMSE(vanilla_preds_cropped, lo_test_cropped))
        println("    Ordinal network reservoir: ", turning_partition_RMSE(ON_preds_cropped, lo_test_cropped))
    end

    fig = Figure( size = (1200,600))

    ax1 = Axis(fig[1,1], title="Vanilla")
    lines!(ax1, vanilla_preds_cropped; linewidth = 1.0, color = :royalblue)
    lines!(ax1, lo_test_cropped; linewidth = 1.0, color = Cycled(2))
    if mark_every != 0
        vlines!(ax1, x_start:mark_every:x_end; color=:gray, alpha=0.5)
    end

    ax2 = Axis(fig[1,2], title="New Approach")
    lines!(ax2, ON_preds_cropped; linewidth = 1.0, color = :darkred)
    lines!(ax2, lo_test_cropped; linewidth = 1.0, color = Cycled(2))
    if mark_every != 0
        vlines!(ax2, x_start:mark_every:x_end; color=:gray, alpha=0.5)
    end

    xlims!(ax1, x_start,x_end)
    ylims!(ax1, -25,25)
    xlims!(ax2, x_start,x_end)
    ylims!(ax2, -25,25)

    fig
end

function get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ; testing_params=create_testing_params(), R_delay=1)
    #if m > 1
    part_symbols_train, unique_partitions_train = create_ordinal_partition(lo_train, m, w, τ)
    trans_adjacency_matrix = create_transition_matrix(part_symbols_train)
    num_partitions = length(unique_partitions_train)
    #else
    #    part_symbols_train = [1]
    #    unique_partitions_train = nothing
    #    trans_adjacency_matrix = nothing
    #    num_partitions = 1
    #end
    # trans_adjacency_map = trans_adjacency_matrix_to_map(trans_adjacency_matrix, num_partitions)

    if testing_params.readout_switching
        ESN_params = create_ESN_params_readout_switching(k, d, ρ, α, η, β, num_partitions)
    else
        ESN_params = create_ESN_params(k, d, ρ, α, η, β, num_partitions=num_partitions, ON_part_adjacency=trans_adjacency_matrix)
    end

    if testing_params.stochastic
        R, train_states = train_one_step_pred_stochastic(lo_train, ESN_params, partition_symbols=part_symbols_train, ON_part_adjacency=trans_adjacency_matrix, rescale_V_rec=true, R_delay=R_delay)
    elseif testing_params.readout_switching
        R, train_states = train_one_step_pred_readout_switching(lo_train, ESN_params, testing_params, part_symbols_train, R_delay=R_delay)
    else
        R, train_states = train_one_step_pred(lo_train, ESN_params, testing_params, partition_symbols=part_symbols_train, R_delay=R_delay)
    end

    return(R, train_states[end,:], unique_partitions_train, ESN_params, part_symbols_train[end], trans_adjacency_matrix)
end

# function create_pred_for_params_single_step(lo_train, lo_test, m; k = 100, part_connection=0.5, d = k*0.05, ρ = 1.2, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 2, return_num_partitions = false, testing_params=create_testing_params(), R_delay=R_delay)
#     R, starting_state, unique_partitions_train, ESN_params, _, _ = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ, R_delay=R_delay)

#     println("Created reservoir of size: ", size(starting_state))

#     part_symbols_test, unique_partitions_test = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)
#     println("WARNING not prepared for testing params")
#     println("This function is outdated and should be created again")
#     @asssert(false)
#     preds, test_states = one_step_pred(lo_test, ESN_params, R, testing_params, S=starting_state, partition_symbols=part_symbols_test)

#     println("Should use R-delay to determine the length of preds here")
#     if !return_num_partitions
#         return(preds[1:end-length(unique_partitions_test)])
#     else
#         return(preds[1:end-length(unique_partitions_test)], length(unique_partitions_train))
#     end
# end

# function test_single_step(lo_train, lo_test, m, k; from=0, to=100, ignore_first=100, equal_total_k=true, part_connection=nothing)
#     ON_preds, num_partitions = create_pred_for_params_single_step(lo_train, lo_test, m; k = k, part_connection=part_connection, return_num_partitions=true)
#     vanilla_k = equal_total_k ? k*num_partitions : k
#     vanilla_preds = create_pred_for_params_single_step(lo_train, lo_test, 1; k = vanilla_k)
#     compare_preds(lo_test, vanilla_preds, ON_preds, from, to, ignore_first=ignore_first)
# end

function create_pred_for_params_single_step(lo_train, lo_test, m; k=100, part_connection=0.5, d=k*0.05, ρ=1.2, α=1.0, η=1/maximum(lo_train), β=0.001, w=1, τ=1, return_num_partitions=false, testing_params=create_testing_params(), R_delay=1)
    R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol, ON_part_adjacency =
        get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ, testing_params=testing_params, R_delay=R_delay)

    println("Created reservoir of size: ", size(starting_state))

    part_symbols_test, _ = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)

    if testing_params.stochastic
        preds, test_states = one_step_pred_stochastic(lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test, ON_part_adjacency=ON_part_adjacency, rescale_V_rec=testing_params.rescale_V_rec)
    elseif testing_params.readout_switching
        preds, test_states = one_step_pred_readout_switching(lo_test, ESN_params, R, part_symbols_test, S=starting_state)
    else
        preds, test_states = one_step_pred(lo_test, ESN_params, R, testing_params, S=starting_state, partition_symbols=part_symbols_test)
    end


    if return_num_partitions
        return(preds, length(unique_partitions_train))
    else
        return(preds)
    end
end

function find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions)
    @assert(length(partition_window) >= (m-1)*τ+1)

    symbols, _ = create_ordinal_partition(partition_window, m, 1, τ; unique_partitions = unique_partitions)

    if symbols[end] === nothing
        # TODO pick the nearest partition rather than a random partition
        return(rand(1:length(unique_partitions)))
    else
        return(symbols[end])
    end
end

function create_pred_for_params_free_run(lo_train, num_test_steps, m; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1, return_num_partitions=false, testing_params=create_testing_params())
    R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol, ON_part_adjacency = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ, testing_params=testing_params)
    
    println("Created reservoir of size: ", size(starting_state))

    partition_window = lo_train[end-((m-1)*τ+1):end]
    preds = [lo_train[end]]
    state = starting_state
    
    part_symbol = starting_part_symbol
    for i in 2:(num_test_steps+1)
        if testing_params.readout_switching
            pred, state = one_step_pred_readout_switching(preds[i-1], ESN_params, R, part_symbol, S = state)
        elseif testing_params.stochastic
            pred, state = one_step_pred_stochastic(
                preds[i-1], ESN_params, R, S=state, partition_symbols=part_symbol, ON_part_adjacency=ON_part_adjacency, rescale_V_rec = testing_params.stochastic_rescale_V_rec
            )
        else
            pred, state = one_step_pred(preds[i-1], ESN_params, R, testing_params, S = state, partition_symbols=part_symbol)
        end
        partition_window = [partition_window[2:end]; pred]
        part_symbol = find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions_train)
        state = state[end,:]

        push!(preds, pred[1])
    end
    
    if !return_num_partitions
        return(preds)
    else
        return(preds, length(unique_partitions_train))
    end
end

# function test_freerun(lo_train, lo_test, m, k; from=0, to=100, equal_total_k=true, part_connection=nothing)
#     ON_preds, num_partitions = create_pred_for_params_free_run(lo_train, length(lo_test), m; k = k, return_num_partitions=true)
#     vanilla_k = equal_total_k ? k*num_partitions : k
#     vanilla_preds = create_pred_for_params_free_run(lo_train, length(lo_test), 1; k = vanilla_k)
#     compare_preds(lo_test, vanilla_preds, ON_preds, from, to, calculate_error=false)
# end

# function create_pred_for_params_multi_step(lo_train, lo_test, m, chunk_length; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1, return_num_partitions=false, testing_params=create_testing_params())
#     R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol, ON_part_adjacency = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ, testing_params=testing_params)
    
#     println("Created reservoir of size: ", size(starting_state))

#     part_symbols_test, unique_partitions_test = create_ordinal_partition(
#         [lo_train[end-(m-1)*τ+1:end]; lo_test], m, w, τ, unique_partitions=unique_partitions_train)
#     part_symbols_test = part_symbols_test[(m-1)*τ+1:end]

#     # println(part_symbols_test)
#     # @assert(false)

#     @assert(all(x -> x in unique_partitions_train, unique_partitions_test))
#     @assert(count(x -> x === nothing, part_symbols_test) == 0)
#     @assert(length(part_symbols_test) == length(lo_test))

#     # why do part_symbols_test start with nothing here?

#     if testing_params.stochastic
#         _, test_states = one_step_pred_stochastic(
#             lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test, ON_part_adjacency=ON_part_adjacency, rescale_V_rec = testing_params.stochastic_rescale_V_rec
#         )
#     elseif testing_params.readout_switching
#         _, test_states = one_step_pred_readout_switching(
#             lo_test, ESN_params, R, part_symbols_test, S=starting_state
#         )
#     else
#         _, test_states = one_step_pred(lo_test, ESN_params, R, testing_params, S=starting_state, partition_symbols=part_symbols_test)
#     end

#     preds = []

#     pred = lo_train[end]
#     state = starting_state
#     @assert(starting_part_symbol !== nothing)
#     part_symbol = starting_part_symbol
#     partition_window = lo_train[end-((m-1)*τ):end]
    
#     chunk_i = 0
#     while chunk_i+chunk_length <= length(lo_test)
#         for _ in 1:chunk_length
#             if testing_params.stochastic
#                 pred, state = one_step_pred_stochastic(
#                     pred, ESN_params, R, S = state, partition_symbols=part_symbol, ON_part_adjacency=ON_part_adjacency, rescale_V_rec = testing_params.stochastic_rescale_V_rec
#                 )
#             elseif testing_params.readout_switching
#                 pred, state = one_step_pred_readout_switching(
#                     pred, ESN_params, R, part_symbol, S = state
#                 )
#             else
#                 pred, state = one_step_pred(
#                     pred, ESN_params, R, testing_params, S = state, partition_symbols=part_symbol
#                 )
#             end
            
#             pred = pred[1]
#             state = state[end,:]
#             partition_window = [partition_window[2:end]; pred]
            
#             part_symbol = find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions_train)

#             push!(preds, pred)
#         end

#         chunk_i += chunk_length

#         pred = lo_test[chunk_i]
#         state = test_states[chunk_i, :]
#         part_symbol = part_symbols_test[chunk_i]
#         @assert(part_symbol !== nothing)
#         if chunk_i-((m-1)*τ+1) > 0
#             partition_window = lo_test[chunk_i-((m-1)*τ+1):chunk_i]
#         else
#             partition_window = [lo_train[end-((m-1)*τ+1)+chunk_i:end]; lo_test[1:chunk_i]]
#         end
#     end
    
#     if !return_num_partitions
#         return(preds)
#     else
#         return(preds, length(unique_partitions_train))
#     end
# end

# function test_multi_step(lo_train, lo_test, m, layer_k; n_steps=5, from=0, to=100, equal_total_k=true, ignore_first=100, testing_params=create_testing_params())
#     ON_preds_multistep, num_partitions = create_pred_for_params_multi_step(lo_train, lo_test, 3, n_steps; k = layer_k, return_num_partitions=true, testing_params=testing_params)
#     vanilla_k = equal_total_k ? layer_k*num_partitions : layer_k
#     vanilla_preds_multistep = create_pred_for_params_multi_step(lo_train, lo_test, 1, n_steps; k = vanilla_k)
#     compare_preds(lo_test, vanilla_preds_multistep, ON_preds_multistep, from, to, offset=0, mark_every=n_steps, ignore_first=ignore_first)
# end

# function test_multi_step_multi_trial(lo_train, lo_test, m, layer_k; n_steps=5, equal_total_k=true, ignore_first=100, trials=10, verbose=true, testing_params=create_testing_params())
#     vanilla_RMSEs, ON_network_RMSEs, vanilla_turning_RMSEs, ON_network_turning_RMSEs = [], [], [], []

#     for i in 1:trials
#         println("Trial ", i, " of ", trials)
#         ON_preds, num_partitions = create_pred_for_params_multi_step(lo_train, lo_test, m, n_steps; k = layer_k, return_num_partitions=true, testing_params=testing_params)
#         if equal_total_k
#             vanilla_k = layer_k*num_partitions
#         else
#             vanilla_k = layer_k        
#         end
#         vanilla_preds = create_pred_for_params_multi_step(lo_train, lo_test, 1, n_steps; k = vanilla_k)

#         ON_preds_cropped = ON_preds[ignore_first+1:end]
#         vanilla_preds_cropped = vanilla_preds[ignore_first+1:min(length(ON_preds), end)]
#         lo_test_cropped = lo_test[ignore_first+1:min(length(ON_preds_cropped)+ignore_first, end)]

#         push!(vanilla_RMSEs, RMSE(vanilla_preds_cropped, lo_test_cropped))
#         push!(ON_network_RMSEs, RMSE(ON_preds_cropped, lo_test_cropped))
#         push!(vanilla_turning_RMSEs, turning_partition_RMSE(vanilla_preds_cropped, lo_test_cropped))
#         push!(ON_network_turning_RMSEs, turning_partition_RMSE(ON_preds_cropped, lo_test_cropped))
#     end

#     if verbose
#         println("Mean Vanilla RMSE: ", mean(vanilla_RMSEs))
#         println("Mean ON Network RMSE: ", mean(ON_network_RMSEs))
#         println("Mean Vanilla Turning RMSE: ", mean(vanilla_turning_RMSEs))
#         println("Mean ON Network Turning RMSE: ", mean(ON_network_turning_RMSEs))
#     end

#     if !verbose
#         return(
#             mean(vanilla_RMSEs),
#             mean(ON_network_RMSEs),
#             mean(vanilla_turning_RMSEs),
#             mean(ON_network_turning_RMSEs)
#         )
#     end
# end

# function graph_multi_step_RMSE_vs_n_steps(lo_train, lo_test, n_step_trials, m, layer_k; equal_total_k=true, ignore_first=100, trials=10, testing_params=create_testing_params())
#     vanilla_RMSEs, ON_network_RMSEs, vanilla_turning_RMSEs, ON_network_turning_RMSEs = [], [], [], []
#     i = 1
#     for n_steps in n_step_trials
#         vanilla_RMSE, ON_network_RMSE, vanilla_turning_RMSE, ON_network_turning_RMSE = test_multi_step_multi_trial(lo_train, lo_test, m, layer_k; n_steps=n_steps, equal_total_k=equal_total_k, ignore_first=ignore_first, trials=trials, verbose=false, testing_params=testing_params)
#         push!(vanilla_RMSEs, vanilla_RMSE)
#         push!(ON_network_RMSEs, ON_network_RMSE)
#         push!(vanilla_turning_RMSEs, vanilla_turning_RMSE)
#         push!(ON_network_turning_RMSEs, ON_network_turning_RMSE)
        
#         fig = Figure(size=(800, 600))
#         ax = Axis(fig[1,1], 
#             xlabel="Number of Steps", 
#             ylabel="RMSE",
#             title="RMSE vs Prediction Steps")

#         # Plot vanilla RMSEs in blues
#         lines!(ax, n_step_trials[1:i], vanilla_RMSEs, color=:royalblue, label="Vanilla RMSE")
#         scatter!(ax, n_step_trials[1:i], vanilla_RMSEs, color=:royalblue)

#         lines!(ax, n_step_trials[1:i], vanilla_turning_RMSEs, color=:lightblue, label="Vanilla Turning RMSE")
#         scatter!(ax, n_step_trials[1:i], vanilla_turning_RMSEs, color=:lightblue)

#         # Plot ON network RMSEs in reds  
#         lines!(ax, n_step_trials[1:i], ON_network_RMSEs, color=:darkred, label="ON Network RMSE")
#         scatter!(ax, n_step_trials[1:i], ON_network_RMSEs, color=:darkred)

#         lines!(ax, n_step_trials[1:i], ON_network_turning_RMSEs, color=:lightcoral, label="ON Network Turning RMSE")
#         scatter!(ax, n_step_trials[1:i], ON_network_turning_RMSEs, color=:lightcoral)

#         axislegend(position=(:right, :bottom))
    
#         # IJulia.clear_output(true)
#         display(fig)

#         i += 1
#     end
# end

function test_single_step_multi_trial(
        data_train, data_test;
        m::Union{Int,Nothing}=nothing, k::Union{Int,Nothing}=nothing, n_steps::Union{Int,Nothing}=nothing,
        error_metrics=[RMSE, turning_partition_RMSE, std, minimum, maximum],
        error_aggregations=[mean, std, min, max],
        ignore_first=100, trials=30, testing_params=create_testing_params()
    )
    errors = Dict(error_metric => Float64[] for error_metric in error_metrics)

    for i in 1:trials
        println("Trial ", i, " of ", trials)

        preds = create_pred_for_params_single_step(
            data_train,
            data_test,
            m,
            k=k,
            R_delay=n_steps,
            testing_params=testing_params
        )[ignore_first+1:end-n_steps]

        for metric in error_metrics
            push!(errors[metric], metric(data_test[1+n_steps+ignore_first:end], preds))
        end
    end

    return Dict(
        metric => Dict(aggregation => aggregation(values) for aggregation in error_aggregations)
        for (metric, values) in errors
    )
end

function test_multi_step_multi_trial_singular(
        data_train, data_test;
        m::Union{Int,Nothing}=nothing, k::Union{Int,Nothing}=nothing, n_steps::Union{Int,Nothing}=nothing,
        error_metrics=[RMSE, turning_partition_RMSE],
        error_aggregations=[mean, std, min, max],
        ignore_first=100, trials=30, testing_params=create_testing_params()
    )
    if isnothing(m) || isnothing(k) || isnothing(n_steps)
        error("Both m and k must be provided")
    end

    errors = Dict(error_metric => Float64[] for error_metric in error_metrics)

    for i in 1:trials
        println("Trial ", i, " of ", trials)
        
        preds = create_pred_for_params_multi_step(
            data_train, data_test, m, n_steps;
            k = k, testing_params=testing_params
        )

        preds_cropped = preds[ignore_first+1:min(length(data_test), end)]
        data_test_cropped = data_test[ignore_first+1:min(length(preds_cropped)+ignore_first, end)]

        for error_metric in error_metrics
            push!(errors[error_metric], error_metric(preds_cropped, data_test_cropped))
        end
    end

    return Dict(
        metric => Dict(aggregation => aggregation(values) for aggregation in error_aggregations)
        for (metric, values) in errors
    )
end

function find_existing_test(tests, test)
    test_copy = deepcopy(test)
    pop!(test_copy, "error_func", nothing)
    test_copy["testing_params"] = Dict{String, Any}(
        "stochastic" => test_copy["testing_params"].stochastic,
        "mask_states_b4_readout" => test_copy["testing_params"].mask_states_b4_readout,
        "stochastic_rescale_V_rec" => test_copy["testing_params"].stochastic_rescale_V_rec,
        "readout_switching" => test_copy["testing_params"].readout_switching
    )
    test_copy["error_funcs"] = ["$(err_func)" for err_func in test["error_funcs"]]

    # Convert any vectors to Any[]
    for (key, value) in test_copy
        if value isa Vector
            test_copy[key] = Any[x for x in value]
            test_copy[key] = sort(value)
        end
    end

    for existing_test in tests
        existing_test_copy = deepcopy(existing_test)
        pop!(existing_test_copy, "errors", nothing)
        pop!(existing_test_copy, "date", nothing)

        # Sort any vectors in existing test
        for (key, value) in existing_test_copy
            if value isa Vector
                existing_test_copy[key] = sort(value)
            end
        end
        
        if existing_test_copy == test_copy
            return existing_test
        end
    end
    
    return nothing
end

function name_of_func(func)
    # Try to get name from methods first
    ms = methods(func)
    if !isempty(ms)
        return first(ms).name
    end
    
    # Fallback to string representation
    str_repr = string(func)
    # Remove module prefix if present (e.g. "Main.RMSE" -> "RMSE")
    return string(split(str_repr, '.')[end])
end

function display_test_chart(testing_parameter, fixed_params, test_output)
    param_name = titlecase(testing_parameter)
    display(
        chart_tests(
            "Error vs. $param_name\n$(join(["$k = $v" for (k,v) in fixed_params], ", "))",
            param_name, "Error",
            Dict("Test in progress" => test_output)
        )
    )
end

function test_params(file_name, tests, data_train, data_test, prediction_type, data_label,
    testing_parameter::String, parameter_trials, fixed_params::Dict;
    error_funcs = [RMSE, turning_partition_RMSE],
    aggregation_funcs = [mean, std, minimum, maximum, median],
    ignore_first=100, trials=30,
    testing_params=create_testing_params(),
    do_chart=true,
    do_chart_existing=false)

    # Check for duplicates in parameter trials
    if length(unique(parameter_trials)) != length(parameter_trials)
        error("parameter_trials contains duplicate values")
    end
    
    # Calculate num_partitions only if m is not the testing parameter
    num_partitions = nothing
    if testing_parameter != "m"
        m = get(fixed_params, :m, fixed_params["m"])
        _, unique_partitions = create_ordinal_partition(data_train, m, 1, 1)
        num_partitions = length(unique_partitions)
    end

    # Create test output dictionary
    test_output = Dict(
        "prediction_type" => prediction_type,
        "testing_parameter" => testing_parameter,
        "data" => data_label,
        testing_parameter => parameter_trials,
        "aggregation_funcs" => [name_of_func(aggregation_func) for aggregation_func in aggregation_funcs],
        "error_funcs" => [name_of_func(error_func) for error_func in error_funcs],
        "ignore_first" => ignore_first,
        "trials" => trials,
        "testing_params" => testing_params,
        "num_partitions" => num_partitions
    )

    # Add fixed parameters to test output
    for (key, value) in fixed_params
        test_output[String(key)] = value
    end

    if testing_parameter != "k" && testing_parameter != "m"
        if testing_params.readout_switching
            test_output["total_k"] = test_output["k"]
        else
            test_output["total_k"] = test_output["k"] * num_partitions
        end
    else
        test_output["total_k"] = nothing
    end

    println("\n\n\n############\nRunning Test\n############\n")
    println(test_output)
    println("\n")

    existing_test = find_existing_test(tests, test_output)
    if existing_test !== nothing
        println("A test with these parameters already exists. Skipping.")

        if do_chart_existing
            println("Charting...")
            display_test_chart(testing_parameter, fixed_params, existing_test)
        end
        return(num_partitions)
    end

    if prediction_type == "multi_step"
        test_multi_trial_func = test_multi_step_multi_trial_singular
    elseif prediction_type == "single_step"
        test_multi_trial_func = test_single_step_multi_trial
    else
        error("INVALID prediction_type")
    end
    
    push!(tests, test_output)
    
    test_output["measurements"] = Dict(
        name_of_func(error_func) => Dict(
            name_of_func(aggregation_func) => Float64[]
            for aggregation_func in aggregation_funcs
        )
        for error_func in error_funcs
    )
    test_output["date"] = today()
    
    for param_value in parameter_trials
        println(findall(x -> x == param_value, parameter_trials)[1], " / ", length(parameter_trials))

        # Create kwargs dict by merging fixed params with current parameter value
        kwargs = Dict{Symbol,Any}((Symbol(k), v) for (k, v) in pairs(fixed_params))
        kwargs[Symbol(testing_parameter)] = param_value

        errors = test_multi_trial_func(
            data_train, data_test;
            error_metrics=error_funcs,
            error_aggregations=aggregation_funcs,
            ignore_first=ignore_first,
            trials=trials, testing_params=testing_params,
            kwargs...
        )

        for aggregation_func in aggregation_funcs
            for error_func in error_funcs
                push!(test_output["measurements"][name_of_func(error_func)][name_of_func(aggregation_func)], errors[error_func][aggregation_func])
            end
        end

        if do_chart
            display_test_chart(testing_parameter, fixed_params, test_output)
        end
        
        save_file(file_name, tests)
    end
    
    return(num_partitions)
end

function save_file(file_name, tests)
    println("Saving...")

    open(file_name, "w") do f
        JSON.print(f, tests)
    end
end

function find_test(test_dict)
    tests = if isfile("../../Scripts/tests.json")
        JSON.parsefile("../../Scripts/tests.json")
    else
        error("No tests.json file found")
    end

    matching_tests = []
    for existing_test in tests
        if all(get(existing_test, k, nothing) == v for (k, v) in pairs(test_dict))
            push!(matching_tests, existing_test)
        end
    end

    if isempty(matching_tests)
        error("No matching test found")
    end
    return matching_tests
end

function chart_tests(title, xlabel, ylabel, results;
    error_funcs=first(values(results))["error_funcs"],
    aggregation_funcs=first(values(results))["aggregation_funcs"],
    bottom_margin=0,
    ylim_low=nothing,
    ylim_high=nothing)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1,1],
        xlabel=xlabel,
        ylabel=ylabel,
        title=title)

    i = 1
    for (key, test) in pairs(results)
        for error_func in error_funcs
            for aggregation_func in aggregation_funcs
                if aggregation_func == "range"
                    dependent_var_min = test["measurements"][error_func]["minimum"]
                    dependent_var_max = test["measurements"][error_func]["maximum"]
                    independent_var = test[test["testing_parameter"]][1:length(dependent_var_min)]
                    
                    independent_var = Float64.(independent_var)
                    dependent_var_min = Float64.(dependent_var_min)
                    dependent_var_max = Float64.(dependent_var_max)
                    band!(independent_var, dependent_var_min, dependent_var_max, color = Cycled(Int(test["m"])), alpha=0.2, label=key * " - " * name_of_func(error_func) * " min and max")
                elseif aggregation_func == "std"
                    stds = test["measurements"][error_func]["std"]
                    mean = test["measurements"][error_func]["mean"]
                    independent_var = test[test["testing_parameter"]][1:length(stds)]
                    errorbars!(independent_var, mean, stds, color = Cycled(Int(test["m"])))
                else
                    dependent_var = test["measurements"][error_func][aggregation_func]
                    independent_var = test[test["testing_parameter"]][1:length(dependent_var)]
                    lines!(ax, independent_var, dependent_var, color=Cycled(Int(test["m"])), label=key * " - " * name_of_func(error_func) * " " * name_of_func(aggregation_func))
                    scatter!(ax, independent_var, dependent_var, color=Cycled(Int(test["m"])), markersize=0.5)
                end
                i += 1
            end
        end
    end

    ylims!(ylim_low, ylim_high)

    axislegend(position=(:right, :bottom))

    fig
end

function quick_graph_series(series, x_min=nothing, x_max=nothing, y_min=nothing, y_max=nothing, dots=false)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1,1])

    lines!(ax, series, color=Cycled(1))
    if dots
        scatter!(ax, series, color=Cycled(1))
    end

    xlims!(x_min, x_max)
    ylims!(y_min, y_max)

    fig
end

end