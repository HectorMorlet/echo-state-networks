module EchoStateNetworksStochastic

using LinearAlgebra

include("../Modules/EchoStateNetworks.jl")
using .EchoStateNetworks
include("StandardFunctions.jl")
using .StandardFunctions

export train_one_step_pred_stochastic, one_step_pred_stochastic, run_ESN_stochastic

function randomly_mask_V_rec_for_partition(V_rec, partition_symbol, k, num_partitions, ON_part_adjacency)
    cumulative_probs = cumsum(ON_part_adjacency[partition_symbol, :])
    r = rand()
    chosen_partition = findfirst(p -> r <= p, cumulative_probs)

    masked_V_rec = copy(V_rec)
    
    # Zero out all connections except those in the chosen partition
    for i in 1:num_partitions
        for j in 1:num_partitions
            if i == j
                continue
            end

            if i != partition_symbol || j != chosen_partition
                start_idx = (i-1)*k + 1
                end_idx = i*k
                masked_V_rec[start_idx:end_idx, :] .= 0
            end
        end
    end

    return(masked_V_rec)
end

function randomly_choose_transition_partitions(num_partitions, ON_part_adjacency)
    chosen_partitions = []
    
    for i in 1:num_partitions
        # for each partition, choose a partition to transmit based on ON_part_adjacency
        cumulative_probs = cumsum(ON_part_adjacency[i, :])
        r = rand()
        chosen_partition = findfirst(p -> r <= p, cumulative_probs)
        
        push!(chosen_partitions, chosen_partition)
    end

    return(chosen_partitions)
end

function generate_possible_chosen_parts(num_partitions, ON_part_adjacency; sensitivity=0.25)
    result = [findall(p -> p > sensitivity, row) for row in eachrow(ON_part_adjacency)]
    possible_combinations = [collect(t) for t in Iterators.product(result...)]
    # println("Cacheable combinations: ", length(possible_combinations))
    return possible_combinations
end

function mask_V_rec(V_rec, k, num_partitions, chosen_partitions)
    masked_V_rec = copy(V_rec)
    
    for i in 1:num_partitions
        for j in 1:num_partitions
            if i == j || j == chosen_partitions[i]
                continue
            end

            masked_V_rec[(i-1)*k + 1:i*k, (j-1)*k + 1:j*k] .= 0
        end
    end

    return(masked_V_rec)
end

function run_ESN_stochastic(x, ESN_params; S = nothing, partition_symbols = nothing, ON_part_adjacency = nothing, rescale_V_rec = false)
    if S == nothing
        S = randn(ESN_params.k*ESN_params.num_partitions)
    end

    # i.e. if we are doing the layered ESN (which requires partition_symbols) then we need the adjacency network for the stochastic network to function
    @assert (partition_symbols === nothing || ON_part_adjacency !== nothing)
    
    if isempty(ESN_params.masked_V_recs)
        possible_chosen_parts = generate_possible_chosen_parts(ESN_params.num_partitions, ON_part_adjacency)
        for chosen_parts in possible_chosen_parts
            masked_V_rec = mask_V_rec(ESN_params.V_rec, ESN_params.k, ESN_params.num_partitions, chosen_parts)

            if rescale_V_rec
                max_abs_ev = maximum(abs.(eigen(masked_V_rec).values))
                masked_V_rec = masked_V_rec * (ESN_params.ρ / max_abs_ev)
            end

            ESN_params.masked_V_recs[chosen_parts] = masked_V_rec
        end
    end

    states = zeros(Float64, ESN_params.k*ESN_params.num_partitions, length(x))
    
    for t in 1:length(x)
        # if t > length(partition_symbols)
        #     break
        # end
        
        if partition_symbols != nothing
            if partition_symbols[t] == nothing
                continue
            end
            masked_V_in = mask_V_in_for_partition(ESN_params.V_in, partition_symbols[t], ESN_params.k, ESN_params.num_partitions)

            chosen_parts = randomly_choose_transition_partitions(ESN_params.num_partitions, ON_part_adjacency)
            if !haskey(ESN_params.masked_V_recs, chosen_parts)
                masked_V_rec = mask_V_rec(
                    ESN_params.V_rec,
                    ESN_params.k,
                    ESN_params.num_partitions,
                    chosen_parts
                )
                if rescale_V_rec
                    max_abs_ev = maximum(abs.(eigen(masked_V_rec).values))
                    masked_V_rec *= ESN_params.ρ / max_abs_ev
                end
            else
                masked_V_rec = ESN_params.masked_V_recs[chosen_parts]
            end

            # mask V_rec based on transition probabilities
            # masked_V_rec = randomly_mask_V_rec_for_partition(ESN_params.V_rec, partition_symbols[t], ESN_params.k, ESN_params.num_partitions, ON_part_adjacency)
        else
            masked_V_in = ESN_params.V_in
            masked_V_rec = ESN_params.V_rec
        end
        
        S = (1 − ESN_params.α)*S + ESN_params.α*tanh.(
            ESN_params.η*masked_V_in*x[t] + masked_V_rec*S + ESN_params.V_bias)
        states[:, t] = S
    end
    
    return(states')
end

function train_one_step_pred_stochastic(x, ESN_params; partition_symbols=nothing, ON_part_adjacency=nothing, rescale_V_rec=false, R_delay=1)
    states = run_ESN_stochastic(x, ESN_params; partition_symbols=partition_symbols, ON_part_adjacency=ON_part_adjacency, rescale_V_rec=rescale_V_rec)
    
    target_z = x[1+R_delay:length(x)]
    @assert(size(states)[1]-R_delay == length(target_z))
    predicted_states = states[1:length(target_z),:]

    # don't mask the states before readout
    # if partition_symbols != nothing
    #     mask_states!(predicted_states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    # end
    
    R = ridge_regression(target_z, predicted_states, ESN_params.β)
    
    return(R, states)
end

function one_step_pred_stochastic(x, ESN_params, R; S = nothing, partition_symbols=nothing, ON_part_adjacency=nothing, rescale_V_rec=false)
    states = run_ESN_stochastic(x, ESN_params; S = S, partition_symbols=partition_symbols, ON_part_adjacency=ON_part_adjacency, rescale_V_rec=rescale_V_rec)

    # don't mask the states before readout
    # if partition_symbols != nothing
    #     mask_states!(states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    # end

    preds = states*R
    
    return(preds, states)
end

end # module
