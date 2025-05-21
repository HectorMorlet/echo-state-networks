module EchoStateNetworks

greet() = print("EchoStateNetworks module... TODO greeting here.")

using SparseArrays
using LinearAlgebra
using Statistics


include("StandardFunctions.jl")
using .StandardFunctions

export ESNParameters, create_ESN_params, train_one_step_pred, one_step_pred, mask_states!, mask_V_in_for_partition, run_ESN, erdos_renyi_adjacency#remove last two

struct ESNParameters
    V_in::Vector{Float64}
    V_rec::Matrix{Float64}
    V_bias::Vector{Float64}
    k::Int64
    α::Float64
    η::Float64
    β::Float64
    ρ::Float64
    num_partitions::Int64 # this is the number of partitions
    masked_V_recs::Dict{Vector{Int}, Matrix{Float64}}
end

function create_ESN_params(k, d, ρ, α, η, β; num_partitions=1, ON_part_adjacency=nothing, ON_part_set_connection=nothing, testing_params=create_testing_params())
    V_in, V_rec, V_bias = create_ESN(k, d, ρ, num_partitions=num_partitions, ON_part_adjacency=ON_part_adjacency, set_connection=ON_part_set_connection, testing_params=testing_params)
    ESN_params = ESNParameters(V_in, V_rec, V_bias, k, α, η, β, ρ, num_partitions,
        Dict{Vector{Int}, Matrix{Float64}}())
    
    return(ESN_params)
end

function erdos_renyi_adjacency(k, d)
    p = d/(k-1)
    adj_mat = triu(collect(sprand(k, k, p)))
    adj_mat = adj_mat + adj_mat'
    adj_mat[diagind(adj_mat)] .= 0
    return(adj_mat)
end

# test_matrix = [
#     0.1  0.2  0.4  0.7  0.11 16
#     0.2  0.3  0.5  0.8  0.12 17
#     0.4  0.5  0.6  0.9  0.13 18
#     0.7  0.8  0.9  0.10 0.14 0.19
#     0.11 0.12 0.13 0.14 0.15 0.20
#     0.16 0.17 0.18 0.19 0.20 0.21
# ]

# test_k = 2
# test_m = 3
# test_part_connection = 1.0

# mask_adjacency!(test_matrix, test_k, test_m, test_part_connection, test_ON_part_adjacency)

# test_matrix
function mask_adjacency!(V_rec, k, num_partitions, ON_part_adjacency; set_connection = nothing, testing_params=create_testing_params())
    for part_i in 1:num_partitions
        for part_j in 1:num_partitions
            # self loops in the OTN
            if part_i == part_j && !testing_params.add_self_loops
                continue
            end

            if testing_params.layer_connections_one_to_one_constant_value || testing_params.layer_connections_fully_connected_constant_value
                avg_matrix_value = mean(V_rec)
            end
            
            if part_i != part_j
                if !(testing_params.layer_connections_sparsely_connected && ON_part_adjacency[part_i,part_j] > 0)
                    V_rec[(part_i-1)*k+1:part_i*k,(part_j-1)*k+1:part_j*k] .= 0.0
                end
            end
            
            if ON_part_adjacency[part_i,part_j] > 0
                for i in 1:k
                    if testing_params.layer_connections_one_to_one_constant_value
                        V_rec[(part_i-1)*k+i,(part_j-1)*k+i] = avg_matrix_value
                    elseif testing_params.layer_connections_one_to_one_randomised
                        V_rec[(part_i-1)*k+i,(part_j-1)*k+i] = randn()
                    elseif testing_params.layer_connections_fully_connected_trans_probs
                        for j in 1:k
                            V_rec[(part_i-1)*k+i,(part_j-1)*k+j] = ON_part_adjacency[part_i, part_j]
                        end
                    elseif testing_params.layer_connections_fully_connected_constant_value
                        for j in 1:k
                            V_rec[(part_i-1)*k+i,(part_j-1)*k+j] = avg_matrix_value
                        end
                    elseif testing_params.layer_connections_sparsely_connected
                        continue # this is done earlier in the function
                    elseif testing_params.layer_connections_disconnected
                        continue
                    else
                        V_rec[(part_i-1)*k+i,(part_j-1)*k+i] = ON_part_adjacency[part_i, part_j]
                    end
                end
            end
        end
    end
end


function create_ESN(k, d, ρ; num_partitions=1, ON_part_adjacency=nothing, set_connection=nothing, testing_params=create_testing_params())
    V_in = randn(k*num_partitions)
    
    V_rec = erdos_renyi_adjacency(k*num_partitions, d*num_partitions)
    
    if ON_part_adjacency != nothing
        # Rescale adjacency to be only as large as largest weight
        ON_part_adjacency = ON_part_adjacency/maximum(ON_part_adjacency)*maximum(V_rec)

        mask_adjacency!(V_rec, k, num_partitions, ON_part_adjacency, set_connection=set_connection, testing_params=testing_params)
    end
    
    max_abs_ev = maximum(abs.(eigen(V_rec).values))
    V_rec = V_rec * ρ / max_abs_ev
    
    V_bias = randn(k*num_partitions)
    
    return(V_in, V_rec, V_bias)
end

function one_hot_encode(partition::Int, m::Int)
    one_hot = zeros(Int, m)
    one_hot[partition] = 1
    return one_hot
end

function mask_V_in_for_partition(V_in, partition, k, num_partitions)
    masked_V_in = zeros(k*num_partitions, num_partitions)
    
    for part_i in 1:num_partitions
        masked_V_in[(part_i-1)*k+1:part_i*k, part_i] .= 1
    end
    
    masked_V_in = masked_V_in*one_hot_encode(partition, num_partitions).*V_in
    
    return(masked_V_in)
end

function run_ESN(x, ESN_params; S = nothing, partition_symbols = nothing, testing_params=create_testing_params())
    if S === nothing
        S = randn(ESN_params.k*ESN_params.num_partitions)
    end
    
    states = zeros(Float64, ESN_params.k*ESN_params.num_partitions, length(x))

    # println("Testing for nothing in partition symbols in run ESN")
    # if partition_symbols !== nothing
    #     idx_first_non = findfirst(sym -> sym !== nothing, partition_symbols)
    #     @assert idx_first_non !== nothing "partition_symbols must contain at least one non-nothing symbol"
    #     if idx_first_non > 1
    #         @assert all(sym === nothing for sym in partition_symbols[1:idx_first_non-1]) "partition_symbols must have only nothing before first non-nothing"
    #     end

    #     # Find it
    #     bad_idx = findnext(sym -> sym === nothing, partition_symbols, idx_first_non)
    #     if bad_idx !== nothing
    #         println("partition_symbols has nothing at index: ", bad_idx)
    #     else
    #         println("All clear")
    #     end

    #     @assert bad_idx === nothing "partition_symbols must not contain nothing after first non-nothing; found at index: $bad_idx"
        
    # end
    valid_symbols = partition_symbols === nothing ? [] : (partition_symbols isa AbstractVector ? filter(sym -> sym !== nothing, partition_symbols) : [partition_symbols])
    unexpected_part_symbol_count = 0
    
    for t in 1:length(x)
        # if t > length(partition_symbols)
        #     break
        # end
        
        if partition_symbols !== nothing && !testing_params.dont_mask_input_vector
            if partition_symbols[t] === nothing
                continue
                # partition_symbols[t] = rand(valid_symbols)
                # unexpected_part_symbol_count += 1
            end
            masked_V_in = mask_V_in_for_partition(ESN_params.V_in, partition_symbols[t], ESN_params.k, ESN_params.num_partitions)
        else
            masked_V_in = ESN_params.V_in
        end
        
        S = (1 − ESN_params.α)*S + ESN_params.α*tanh.(
            ESN_params.η*masked_V_in*x[t] + ESN_params.V_rec*S + ESN_params.V_bias)
        states[:, t] = S
    end

    if unexpected_part_symbol_count != 0
        println("Number of unexpected part symbols: ", unexpected_part_symbol_count)
    end
    
    return(states')
end

function mask_states!(states, partition_symbols, k, num_partitions)
    # for part_i in 1:num_partitions
    #     states[(part_i-1)*k+1:part_i*k, partition_symbols[part_i]] .= 0
    # end
    @assert size(states)[2] == k*num_partitions
    for state_i in 1:size(states)[1]
        if partition_symbols[state_i] != nothing
            states[state_i,:] = mask_V_in_for_partition(states[state_i,:], partition_symbols[state_i], k, num_partitions)
        end
    end
end

function train_one_step_pred(x, ESN_params, testing_params; partition_symbols=nothing, R_delay=1)
    states = run_ESN(x, ESN_params; partition_symbols=partition_symbols, testing_params=testing_params)
    
    target_z = x[1+R_delay:length(x)]
    @assert(size(states)[1]-R_delay == length(target_z))
    predicted_states = states[1:length(target_z),:]

    # don't mask the states before readout
    if testing_params.mask_states_b4_readout
        if partition_symbols != nothing
            mask_states!(predicted_states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
        end
    end
    
    R = ridge_regression(target_z, predicted_states, ESN_params.β)
    
    return(R, states)
end

function one_step_pred(x, ESN_params, R, testing_params; S = nothing, partition_symbols=nothing)
    states = run_ESN(x, ESN_params; S = S, partition_symbols=partition_symbols, testing_params=testing_params)

    if testing_params.mask_states_b4_readout
        if partition_symbols != nothing
            mask_states!(states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
        end
    end

    preds = states*R
    
    return(preds, states)
end

end # module EchoStateNetworks
