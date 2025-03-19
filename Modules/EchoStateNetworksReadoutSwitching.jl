module EchoStateNetworksReadoutSwitching

using LinearAlgebra

include("../Modules/EchoStateNetworks.jl")
using .EchoStateNetworks
include("StandardFunctions.jl")
using .StandardFunctions

export train_one_step_pred_readout_switching, one_step_pred_readout_switching, create_ESN_params_readout_switching

function create_ESN_params_readout_switching(k, d, ρ, α, η, β, num_partitions)
    V_in, V_rec, V_bias = create_ESN_readout_switching(k, d, ρ)
    ESN_params = ESNParameters(V_in, V_rec, V_bias, k, α, η, β, ρ, num_partitions,
        Dict{Vector{Int}, Matrix{Float64}}())
    
    return(ESN_params)
end

function create_ESN_readout_switching(k, d, ρ; num_partitions=1, ON_part_adjacency=nothing, set_connection=nothing)
    V_in = randn(k)
    V_rec = erdos_renyi_adjacency(k, d)
    max_abs_ev = maximum(abs.(eigen(V_rec).values))
    V_rec = V_rec * ρ / max_abs_ev
    V_bias = randn(k)
    
    return(V_in, V_rec, V_bias)
end

function run_ESN_readout_switching(x, ESN_params; S = nothing)
    if S === nothing
        S = randn(ESN_params.k)
    end
    
    states = zeros(Float64, ESN_params.k, length(x))
    
    for t in 1:length(x)
        S = (1 − ESN_params.α)*S + ESN_params.α*tanh.(
            ESN_params.η*ESN_params.V_in*x[t] + ESN_params.V_rec*S + ESN_params.V_bias)
        states[:, t] = S
    end
    
    return(states')
end

function train_one_step_pred_readout_switching(x, ESN_params, testing_params, partition_symbols, R_delay=1)
    states = run_ESN_readout_switching(x, ESN_params)
    
    target_z = x[1+R_delay:length(x)]
    @assert(size(states)[1]-R_delay == length(target_z))
    predicted_states = states[1:length(target_z),:]

    # if testing_params.mask_states_b4_readout
    #     if partition_symbols != nothing
    #         mask_states!(predicted_states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    #     end
    # end

    # @assert(all(partition_symbols[1:end-1] .== 1))
    # @assert(target_z[partition_symbols[1:end-1] .== 1] == target_z)
    # @assert(predicted_states[partition_symbols[1:end-1] .== 1, :] == predicted_states)
    
    R = Matrix(undef, ESN_params.num_partitions, ESN_params.k)  # Initialize an n x 3 matrix for integer vectors

    for i in 1:ESN_params.num_partitions
        R[i, :] = ridge_regression(
            target_z[partition_symbols[1:end-1] .== i],
            predicted_states[partition_symbols[1:end-1] .== i, :],
            ESN_params.β
        )
    end
    
    return(R', states)
end

function one_step_pred_readout_switching(x, ESN_params, R, partition_symbols; S = nothing)
    states = run_ESN_readout_switching(x, ESN_params; S = S)

    # if testing_params.mask_states_b4_readout
    #     if partition_symbols != nothing
    #         mask_states!(states, partition_symbols, ESN_params.k, ESN_params.num_partitions)
    #     end
    # end

    preds = states*R

    @assert(partition_symbols !== nothing)
    @assert(count(x -> x === nothing, partition_symbols) == 0)

    partition_preds = [preds[i, partition_symbols[i]] for i in 1:length(partition_symbols)]
    
    return(partition_preds, states)
end


end