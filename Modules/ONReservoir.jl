module ONReservoir

greet() = print("ONReservoir module... TODO greeting here.")

export trans_adjacency_matrix_to_map, create_ordinal_partition, create_transition_matrix

function trans_adjacency_matrix_to_map(trans_matrix, num_partitions)
    return(Dict(
            i => [j for j in 1:num_partitions if (trans_matrix .> 0)[i,j]] for i in 1:num_partitions
    ))
end

function create_ordinal_partition_future(tr, m, w, τ; unique_partitions = nothing)
    rankings = zeros(Int, trunc(Int, length(tr)/w - τ*(m-1)), m)
    for j in 1:1:trunc(Int, length(tr)/w - τ*(m-1))
        x = w*(j-1) + 1
        rankings[j, :] = sortperm([tr[i] for i in x:τ:(x+τ*(m-1))])
    end
    if unique_partitions == nothing
        unique_partitions = unique(eachrow(rankings))
    end
    part_symbols = [findfirst(==(row), unique_partitions) for row in eachrow(rankings)]
    return(part_symbols, unique_partitions)
end

function create_ordinal_partition(tr, m, w, τ; unique_partitions = nothing)
    # println("\n\nCreating ordinal partition...")
    rankings = zeros(Int, trunc(Int, length(tr)/w - τ*(m-1)), m)
    for j in 1:1:trunc(Int, length(tr)/w - τ*(m-1))

        x = w*(j-1) + 1
        # if length(tr) == 4
        #     println("Testing ", [tr[i] for i in x:τ:(x+τ*(m-1))])
        #     println("Result ", sortperm([tr[i] for i in x:τ:(x+τ*(m-1))]))
        # end
        rankings[j, :] = sortperm([tr[i] for i in x:τ:(x+τ*(m-1))])
    end
    if unique_partitions === nothing
        unique_partitions = unique(eachrow(rankings))
    end
    part_symbols = [findfirst(==(row), unique_partitions) for row in eachrow(rankings)]
    part_symbols = [[nothing for _ in 1:τ*(m-1)]; part_symbols]

    return(part_symbols, unique_partitions)
end

function find_probs(partition)
    df = DataFrame(partition, :auto)
    counts = combine(groupby(df, names(df)), nrow => :count)
    counts.probability = counts.count ./ sum(counts.count)
    return(counts)
end

function create_transition_matrix(part_symbols)
    # Remove any nothing values
    part_symbols = filter(x -> x !== nothing, part_symbols)
    
    # Get unique symbols
    symbols = unique(part_symbols)

    # 2×n matrix of transitions
    transitions = [part_symbols[1:end-1] part_symbols[2:end]]

    # Initialize transition matrix
    n = length(symbols)
    transition_matrix = zeros(Float64, n, n)

    # Count transitions
    for row in 1:size(transitions, 1)
        i = findfirst(isequal(transitions[row, 1]), symbols)
        j = findfirst(isequal(transitions[row, 2]), symbols)
        transition_matrix[i, j] += 1
    end

    # Normalize rows to get transition probabilities
    for i in 1:n
        row_sum = sum(transition_matrix[i, :])
        if row_sum > 0
            transition_matrix[i, :] /= row_sum
        end
    end

    return(transition_matrix)
end

end # module ONReservoir