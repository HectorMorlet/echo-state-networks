using DynamicalSystemsBase
using DelimitedFiles

# Initial conditions for Rössler system
init_conds = [0.1, 0.1, 0.1]

# Standard parameters for Rössler system (a, b, c)
ro_variables = [0.2, 0.2, 5.7]

# Time parameters
init_time_to_ignore = 10
total_time = 500*5
time_delta = 0.01

# Define the Rössler system
function Rossler(u, p, t)
    a = p[1]
    b = p[2]
    c = p[3]
    
    du1 = -u[2] - u[3]
    du2 = u[1] + a*u[2]
    du3 = b + u[3]*(u[1] - c)
    
    return SVector(du1, du2, du3)
end

# Create the dynamical system
ro = CoupledODEs(Rossler, init_conds, ro_variables)

# Generate trajectory
ro_tr, tvec = trajectory(ro, total_time; Δt = time_delta, Ttr = init_time_to_ignore)

# Function to split dataset into training and testing sets
function SplitSet(set, ratio)
    split_point = trunc(Int, ratio*length(set))
    train_set = set[1:split_point]
    test_set = set[split_point+1:length(set)]
    return(train_set, test_set)
end

# Split x-coordinate data into training and testing sets
ro_train, ro_test = SplitSet(ro_tr[:,1], 0.8)

# Save data to files
writedlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_01.txt"), ro_train)
writedlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_01.txt"), ro_test)
