include("EchoStateNetworks.jl")

module EchoStateNetworksStochastic

using ..EchoStateNetworks
using LinearAlgebra

export ESNParameters, create_ESN_params, train_one_step_pred, one_step_pred, mask_states!, mask_V_in_for_partition, run_ESN

# Extend the base ESN parameters to include stochastic components
# struct StochasticESNParameters <: ESNParameters
#     V_in::Vector{Float64}
#     V_rec::Matrix{Float64} 
#     V_bias::Vector{Float64}
#     k::Int64
#     α::Float64
#     η::Float64
#     β::Float64
#     num_partitions::Int64
#     noise_amplitude::Float64  # Additional parameter for stochastic behavior
# end

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

function randomly_mask_V_rec(V_rec, k, num_partitions, ON_part_adjacency)
    masked_V_rec = copy(V_rec)
    
    for i in 1:num_partitions
        # for each partition, choose a partition to transmit based on ON_part_adjacency
        cumulative_probs = cumsum(ON_part_adjacency[i, :])
        r = rand()
        chosen_partition = findfirst(p -> r <= p, cumulative_probs)

        for j in 1:num_partitions
            if i == j || j == chosen_partition
                continue
            end

            masked_V_rec[(i-1)*k + 1:i*k, (j-1)*k + 1:j*k] .= 0
        end
    end

    return(masked_V_rec)
end

function run_ESN(x, ESN_params; S = nothing, partition_symbols = nothing, ON_part_adjacency = nothing)
    if S == nothing
        S = randn(ESN_params.k*ESN_params.num_partitions)
    end

    @assert (partition_symbols === nothing || ON_part_adjacency !== nothing)
    
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

            # mask V_rec based on transition probabilities
            # masked_V_rec = randomly_mask_V_rec_for_partition(ESN_params.V_rec, partition_symbols[t], ESN_params.k, ESN_params.num_partitions, ON_part_adjacency)
            masked_V_rec = randomly_mask_V_rec(ESN_params.V_rec, ESN_params.k, ESN_params.num_partitions, ON_part_adjacency)
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


# do I really need to have this function? The only changed function is run_ESN
function ridge_regression(x::Vector, states::Matrix, beta::Float64)
    # Ensure states is a matrix and x is a vector
    @assert size(states, 1) == length(x)
    
    # Compute the number of features
    n_features = size(states, 2)
    
    # Compute the identity matrix of size n_features
    I_test = Matrix{Float64}(I, n_features, n_features)
    
    # Compute the Ridge regression solution
    R = (states' * states + beta * I_test) \ (states' * x)
    
    return R
end

function train_one_step_pred(x, ESN_params; partition_symbols=nothing, ON_part_adjacency=nothing)
    states = run_ESN(x, ESN_params; partition_symbols=partition_symbols, ON_part_adjacency=ON_part_adjacency)
    
    target_z = x[2:length(x)]
    predicted_states = states[1:size(states)[1]-1,:]

    # don't mask the states before readout
    # if partition_symbols != nothing
    #     mask_states!(predicted_states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    # end
    
    R = ridge_regression(target_z, predicted_states, ESN_params.β)
    
    return(R, states)
end

function one_step_pred(x, ESN_params, R; S = nothing, partition_symbols=nothing, ON_part_adjacency=nothing)
    states = run_ESN(x, ESN_params; S = S, partition_symbols=partition_symbols, ON_part_adjacency=ON_part_adjacency)

    # don't mask the states before readout
    # if partition_symbols != nothing
    #     mask_states!(states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    # end

    preds = states*R
    
    return(preds, states)
end

end # module
