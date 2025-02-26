module TestingFunctions

using CairoMakie
using StatsBase
using Distributed
using IJulia

include("ONReservoir.jl")
using .ONReservoir
include("EchoStateNetworks.jl")
using .EchoStateNetworks
include("TurningError.jl")
using .TurningError
include("EchoStateNetworksStochastic.jl")
using .EchoStateNetworksStochastic

export TestingParameters, create_testing_params, compare_preds, create_pred_for_params_single_step, create_pred_for_params_free_run, create_pred_for_params_multi_step, test_multi_step, test_multi_step_multi_trial, graph_multi_step_RMSE_vs_n_steps, test_single_step, test_freerun

struct TestingParameters
    mask_states_b4_readout::Bool
    stochastic::Bool
end

function create_testing_params(;mask_states_b4_readout=false, stochastic=false)
    return(TestingParameters(mask_states_b4_readout, stochastic))
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

    ax2 = Axis(fig[1,2], title="Ordinal Network")
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

function get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ; testing_params=create_testing_params())
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

    ESN_params = create_ESN_params(k, d, ρ, α, η, β, num_partitions=num_partitions, ON_part_adjacency=trans_adjacency_matrix)

    if testing_params.stochastic
        R, train_states = train_one_step_pred_stochastic(lo_train, ESN_params, partition_symbols=part_symbols_train, ON_part_adjacency=trans_adjacency_matrix)
    else
        R, train_states = train_one_step_pred(lo_train, ESN_params, partition_symbols=part_symbols_train, mask_states_b4_readout=testing_params.mask_states_b4_readout)
    end

    return(R, train_states[end,:], unique_partitions_train, ESN_params, part_symbols_train[end], trans_adjacency_matrix)
end

function create_pred_for_params_single_step(lo_train, lo_test, m; k = 100, part_connection=0.5, d = k*0.05, ρ = 1.2, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 2, return_num_partitions = false, mask_states_b4_readout=false)
    R, starting_state, unique_partitions_train, ESN_params, _, _ = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ)

    println("Created reservoir of size: ", size(starting_state))

    part_symbols_test, unique_partitions_test = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)
    preds, test_states = one_step_pred(lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test, mask_states_b4_readout=mask_states_b4_readout)

    if !return_num_partitions
        return(preds[1:end-length(unique_partitions_test)])
    else
        return(preds[1:end-length(unique_partitions_test)], length(unique_partitions_train))
    end
end

function test_single_step(lo_train, lo_test, m, k; from=0, to=100, ignore_first=100, equal_total_k=true, part_connection=nothing)
    ON_preds, num_partitions = create_pred_for_params_single_step(lo_train, lo_test, m; k = k, part_connection=part_connection, return_num_partitions=true)
    vanilla_k = equal_total_k ? k*num_partitions : k
    vanilla_preds = create_pred_for_params_single_step(lo_train, lo_test, 1; k = vanilla_k)
    compare_preds(lo_test, vanilla_preds, ON_preds, from, to, ignore_first=ignore_first)
end

function find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions)
    @assert(length(partition_window) >= (m-1)*τ+1)

    symbols, _ = create_ordinal_partition(partition_window, m, 1, τ; unique_partitions = unique_partitions)

    return(symbols[end])
end

function create_pred_for_params_free_run(lo_train, num_test_steps, m; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1, return_num_partitions=false, mask_states_b4_readout=false)
    R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol, _ = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ)
    
    println("Created reservoir of size: ", size(starting_state))

    partition_window = lo_train[end-((m-1)*τ+1):end]
    preds = [lo_train[end]]
    state = starting_state
    
    part_symbol = starting_part_symbol
    for i in 2:(num_test_steps+1)
        pred, state = one_step_pred(preds[i-1], ESN_params, R, S = state, partition_symbols=part_symbol, mask_states_b4_readout=false)
        state = state[end,:]
        partition_window = [partition_window[2:end]; pred]
        part_symbol = find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions_train)

        push!(preds, pred[1])
    end
    
    if !return_num_partitions
        return(preds)
    else
        return(preds, length(unique_partitions_train))
    end
end

function test_freerun(lo_train, lo_test, m, k; from=0, to=100, equal_total_k=true, part_connection=nothing)
    ON_preds, num_partitions = create_pred_for_params_free_run(lo_train, length(lo_test), m; k = k, return_num_partitions=true)
    vanilla_k = equal_total_k ? k*num_partitions : k
    vanilla_preds = create_pred_for_params_free_run(lo_train, length(lo_test), 1; k = vanilla_k)
    compare_preds(lo_test, vanilla_preds, ON_preds, from, to, calculate_error=false)
end

function create_pred_for_params_multi_step(lo_train, lo_test, m, chunk_length; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1, return_num_partitions=false, testing_params=create_testing_params())
    R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol, ON_part_adjacency = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ, testing_params=testing_params)
    
    println("Created reservoir of size: ", size(starting_state))
    
    part_symbols_test, unique_partitions_test = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)
    if testing_params.stochastic
        _, test_states = one_step_pred_stochastic(lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test, ON_part_adjacency=ON_part_adjacency)
    else
        _, test_states = one_step_pred(lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test, mask_states_b4_readout=testing_params.mask_states_b4_readout)
    end

    preds = []

    pred = lo_train[end]
    state = starting_state
    part_symbol = starting_part_symbol
    partition_window = lo_train[end-((m-1)*τ+1):end]
    
    chunk_i = 0
    while chunk_i+chunk_length <= length(lo_test)
        for _ in 1:chunk_length
            if testing_params.stochastic
                pred, state = one_step_pred_stochastic(pred, ESN_params, R, S = state, partition_symbols=part_symbol, ON_part_adjacency=ON_part_adjacency)
            else
                pred, state = one_step_pred(pred, ESN_params, R, S = state, partition_symbols=part_symbol, mask_states_b4_readout=testing_params.mask_states_b4_readout)
            end
            
            pred = pred[1]
            state = state[end,:]
            partition_window = [partition_window[2:end]; pred]
            part_symbol = find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions_train)

            push!(preds, pred)
        end

        chunk_i += chunk_length

        pred = lo_test[chunk_i]
        state = test_states[chunk_i, :]
        part_symbol = part_symbols_test[chunk_i]
        if chunk_i-((m-1)*τ+1) > 0
            partition_window = lo_test[chunk_i-((m-1)*τ+1):chunk_i]
        else
            partition_window = [lo_train[end-((m-1)*τ+1)+chunk_i:end]; lo_test[1:chunk_i]]
        end
    end
    
    if !return_num_partitions
        return(preds)
    else
        return(preds, length(unique_partitions_train))
    end
end

function test_multi_step(lo_train, lo_test, m, layer_k; n_steps=5, from=0, to=100, equal_total_k=true, ignore_first=100)
    ON_preds_multistep, num_partitions = create_pred_for_params_multi_step(lo_train, lo_test, 3, n_steps; k = layer_k, return_num_partitions=true)
    vanilla_k = equal_total_k ? layer_k*num_partitions : layer_k
    vanilla_preds_multistep = create_pred_for_params_multi_step(lo_train, lo_test, 1, n_steps; k = vanilla_k)
    compare_preds(lo_test, ON_preds_multistep, vanilla_preds_multistep, from, to, offset=0, mark_every=n_steps, ignore_first=ignore_first)
end

function test_multi_step_multi_trial(lo_train, lo_test, m, layer_k; n_steps=5, equal_total_k=true, ignore_first=100, trials=10, verbose=true, testing_params=create_testing_params())
    vanilla_RMSEs, ON_network_RMSEs, vanilla_turning_RMSEs, ON_network_turning_RMSEs = [], [], [], []

    for i in 1:trials
        println("Trial ", i, " of ", trials)
        ON_preds, num_partitions = create_pred_for_params_multi_step(lo_train, lo_test, 3, n_steps; k = layer_k, return_num_partitions=true, testing_params=testing_params)
        if equal_total_k
            vanilla_k = layer_k*num_partitions
        else
            vanilla_k = layer_k        
        end
        vanilla_preds = create_pred_for_params_multi_step(lo_train, lo_test, 1, n_steps; k = vanilla_k)

        ON_preds_cropped = ON_preds[ignore_first+1:end]
        vanilla_preds_cropped = vanilla_preds[ignore_first+1:min(length(ON_preds), end)]
        lo_test_cropped = lo_test[ignore_first+1:min(length(ON_preds_cropped)+ignore_first, end)]

        push!(vanilla_RMSEs, RMSE(vanilla_preds_cropped, lo_test_cropped))
        push!(ON_network_RMSEs, RMSE(ON_preds_cropped, lo_test_cropped))
        push!(vanilla_turning_RMSEs, turning_partition_RMSE(vanilla_preds_cropped, lo_test_cropped))
        push!(ON_network_turning_RMSEs, turning_partition_RMSE(ON_preds_cropped, lo_test_cropped))
    end

    if verbose
        println("Mean Vanilla RMSE: ", mean(vanilla_RMSEs))
        println("Mean ON Network RMSE: ", mean(ON_network_RMSEs))
        println("Mean Vanilla Turning RMSE: ", mean(vanilla_turning_RMSEs))
        println("Mean ON Network Turning RMSE: ", mean(ON_network_turning_RMSEs))
    end

    if !verbose
        return(
            mean(vanilla_RMSEs),
            mean(ON_network_RMSEs),
            mean(vanilla_turning_RMSEs),
            mean(ON_network_turning_RMSEs)
        )
    end
end

function graph_multi_step_RMSE_vs_n_steps(lo_train, lo_test, n_step_trials, m, layer_k; equal_total_k=true, ignore_first=100, trials=10, testing_params=create_testing_params())
    vanilla_RMSEs, ON_network_RMSEs, vanilla_turning_RMSEs, ON_network_turning_RMSEs = [], [], [], []
    i = 1
    for n_steps in n_step_trials
        vanilla_RMSE, ON_network_RMSE, vanilla_turning_RMSE, ON_network_turning_RMSE = test_multi_step_multi_trial(lo_train, lo_test, m, layer_k; n_steps=n_steps, equal_total_k=equal_total_k, ignore_first=ignore_first, trials=trials, verbose=false, testing_params=testing_params)
        push!(vanilla_RMSEs, vanilla_RMSE)
        push!(ON_network_RMSEs, ON_network_RMSE)
        push!(vanilla_turning_RMSEs, vanilla_turning_RMSE)
        push!(ON_network_turning_RMSEs, ON_network_turning_RMSE)
        
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1,1], 
            xlabel="Number of Steps", 
            ylabel="RMSE",
            title="RMSE vs Prediction Steps")

        # Plot vanilla RMSEs in blues
        lines!(ax, n_step_trials[1:i], vanilla_RMSEs, color=:royalblue, label="Vanilla RMSE")
        scatter!(ax, n_step_trials[1:i], vanilla_RMSEs, color=:royalblue)

        lines!(ax, n_step_trials[1:i], vanilla_turning_RMSEs, color=:lightblue, label="Vanilla Turning RMSE")
        scatter!(ax, n_step_trials[1:i], vanilla_turning_RMSEs, color=:lightblue)

        # Plot ON network RMSEs in reds  
        lines!(ax, n_step_trials[1:i], ON_network_RMSEs, color=:darkred, label="ON Network RMSE")
        scatter!(ax, n_step_trials[1:i], ON_network_RMSEs, color=:darkred)

        lines!(ax, n_step_trials[1:i], ON_network_turning_RMSEs, color=:lightcoral, label="ON Network Turning RMSE")
        scatter!(ax, n_step_trials[1:i], ON_network_turning_RMSEs, color=:lightcoral)

        axislegend(position=(:right, :bottom))
    
        IJulia.clear_output(true)
        display(fig)

        i += 1
    end
end

end