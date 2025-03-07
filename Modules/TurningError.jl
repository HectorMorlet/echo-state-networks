module TurningError

using Combinatorics
using Statistics

greet() = print("ONReservoir module... TODO greeting here.")

export turning_partition_RMSE, create_turning_partition_mask

function create_partition_mask(tr, m, w, τ, partitions)
    mask = zeros(Int, trunc(Int, length(tr)/w - τ*(m-1)))
    for j in 1:1:trunc(Int, length(tr)/w - τ*(m-1))
        x = w*(j-1) + 1
        mask[j] = sortperm([tr[i] for i in x:τ:(x+τ*(m-1))]) in partitions
    end

    return([zeros(Int, τ*(m-1)); mask])
end

function find_turning_partitions(m)
    perms = collect(permutations(1:m))
    return(filter(p -> (p != 1:m && p != reverse(1:m)), perms))
end

function create_turning_partition_mask(tr, m, w, τ)
    return(create_partition_mask(tr, m, w, τ, find_turning_partitions(m)))
end

function turning_partition_RMSE(tr_pred, tr_true; m = 4, w = 1, τ = 1)
    partitions = find_turning_partitions(m)

    errors = []
    for j in 1:1:trunc(Int, length(tr_true)/w - τ*(m-1))
        x = w*(j-1) + 1
        if sortperm([tr_true[i] for i in x:τ:(x+τ*(m-1))]) in partitions
            push!(errors, tr_pred[x+τ*(m-1)] - tr_true[x])
        end
    end

    return sqrt(mean(errors .^ 2))
end

end