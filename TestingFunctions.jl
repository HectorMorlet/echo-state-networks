module TestingFunctions

using CairoMakie

include("ONReservoir.jl")
using .ONReservoir
include("EchoStateNetworks.jl")
using .EchoStateNetworks

export compare_preds, create_pred_for_params_single_step, create_pred_for_params_free_run, create_pred_for_params_multi_step, find_ordinal_partition_symbol, get_starting_state_and_R
# TODO remove the last two functions

function compare_preds(lo_test, ON_preds, vanilla_preds, x_start, x_end; calculate_error=true, ignore_first=0, offset=1)
    ON_preds_cropped = ON_preds[ignore_first+1:end]
    vanilla_preds_cropped = vanilla_preds[ignore_first+1:min(length(ON_preds), end)]
    lo_test_cropped = lo_test[offset+ignore_first+1:min(length(ON_preds_cropped)+offset+ignore_first, end)]

    if calculate_error
        println("Ordinal network reservoir prediction RMSE: ", RMSE(ON_preds_cropped, lo_test_cropped))
        println("Vanilla prediction RMSE: ", RMSE(vanilla_preds_cropped, lo_test_cropped))
        println("Ordinal network reservoir prediction turning partition RMSE: ", turning_partition_RMSE(ON_preds_cropped, lo_test_cropped))
        println("Vanilla prediction turning partition RMSE: ", turning_partition_RMSE(vanilla_preds_cropped, lo_test_cropped))
    end

    fig = Figure( size = (1200,600))

    ax1 = Axis(fig[1,1])
    lines!(ax1, ON_preds_cropped; linewidth = 1.0, color = Cycled(1))
    lines!(ax1, lo_test_cropped; linewidth = 1.0, color = Cycled(2))

    xlims!(x_start,x_end)
    ylims!(-25,25)

    ax2 = Axis(fig[1,2])
    lines!(ax2, vanilla_preds_cropped; linewidth = 1.0, color = Cycled(1))
    lines!(ax2, lo_test_cropped; linewidth = 1.0, color = Cycled(2))

    xlims!(x_start,x_end)
    ylims!(-25,25)

    fig
end

function get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ)
    part_symbols_train, unique_partitions_train = create_ordinal_partition(lo_train, m, w, τ)
    trans_adjacency_matrix = create_transition_matrix(part_symbols_train)
    num_partitions = length(unique_partitions_train)
    # trans_adjacency_map = trans_adjacency_matrix_to_map(trans_adjacency_matrix, num_partitions)

    ESN_params = create_ESN_params(k, d, ρ, α, η, β, num_partitions=num_partitions, ON_part_adjacency=trans_adjacency_matrix)

    R, train_states = train_one_step_pred(lo_train, ESN_params, partition_symbols=part_symbols_train)

    return(R, train_states[end,:], unique_partitions_train, ESN_params, part_symbols_train[end])
end

function create_pred_for_params_single_step(lo_train, lo_test, m; k = 100, part_connection=0.5, d = k*0.05, ρ = 1.2, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 2)
    R, starting_state, unique_partitions_train, ESN_params, _ = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ)

    println("Created reservoir of size: ", size(starting_state))

    part_symbols_test, unique_partitions_test = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)
    preds, test_states = one_step_pred(lo_test, ESN_params, R, S=starting_state, partition_symbols=part_symbols_test)

    return(preds[1:end-length(unique_partitions_test)])
end

function find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions)
    @assert(length(partition_window) >= (m-1)*τ+1)

    symbols, _ = create_ordinal_partition(partition_window, m, 1, τ; unique_partitions = unique_partitions)

    return(symbols[end])
end

function create_pred_for_params_free_run(lo_train, num_test_steps, m; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1)
    R, starting_state, unique_partitions_train, ESN_params, starting_part_symbol = get_starting_state_and_R(lo_train, m, k, d, ρ, α, η, β, w, τ)

    partition_window = lo_train[end-((m-1)*τ+1):end]
    preds = [lo_train[end]]
    state = starting_state
    
    part_symbol = starting_part_symbol
    for i in 2:(num_test_steps+1)
        pred, state = one_step_pred(preds[i-1], ESN_params, R, S = state, partition_symbols=part_symbol)
        state = state[end,:]
        
        partition_window = [partition_window[2:end]; pred]
        part_symbol = find_ordinal_partition_symbol(partition_window, m, τ, unique_partitions_train)

        push!(preds, pred[1])
    end
    
    return(preds)
end

function create_pred_for_params_multi_step(lo_train, lo_test, m, chunk_length; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1)
    
end

# function multi_step_pred(initial_value, sub_part_symbols_test, state, R, ESN_params)
#     pred, state = initial_value, state
#     preds = [pred]
    
#     for symbol in sub_part_symbols_test
#         one_pred, one_state = one_step_pred(pred, ESN_params, R, S = state, partition_symbols=symbol)
#         pred = one_pred[1]
#         state = one_state[1,:]
#         push!(preds, pred)
#     end
    
#     return(preds, state)
# end

# function create_pred_for_params_multi_step(m, chunk_length; k = 100, d = k*0.05, ρ = 1.1, α = 1.0, η = 1/maximum(lo_train), β = 0.001, w = 1, τ = 1)
#     part_symbols_train, unique_partitions_train = create_ordinal_partition(lo_train, m, w, τ)
#     trans_adjacency_matrix = create_transition_matrix(part_symbols_train)
#     num_partitions = length(unique_partitions_train)
#     trans_adjacency_map = trans_adjacency_matrix_to_map(trans_adjacency_matrix, num_partitions)

#     ESN_params = create_ESN_params(k, d, ρ, α, η, β, num_partitions=num_partitions, part_connection=0.1, ON_part_adjacency=trans_adjacency_map)

#     R, train_states = train_one_step_pred(lo_train, ESN_params, partition_symbols=part_symbols_train)

#     part_symbols_test, unique_partitions_test = create_ordinal_partition(lo_test, m, w, τ, unique_partitions=unique_partitions_train)
    
#     preds = []
#     current_state = train_states[end,:]
#     current_true_value = lo_train[end]
#     i = 1
#     while i+chunk_length-1 < length(part_symbols_test)
#         new_preds, _ = multi_step_pred(current_true_value, part_symbols_test[i:i+chunk_length-2], current_state, R, ESN_params)
#         preds = vcat(preds, new_preds)

#         states = run_ESN(lo_test[i:i+chunk_length-1], ESN_params; S = current_state, partition_symbols=part_symbols_test[i:i+chunk_length-1])
#         current_state = states[1,:]
        
#         current_true_value = lo_test[i+chunk_length-1]
        
#         i = i + chunk_length
#     end
    
#     return(preds[2:end])
# end


end