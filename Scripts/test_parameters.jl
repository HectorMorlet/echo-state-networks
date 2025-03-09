using DelimitedFiles
using JSON

include("../Modules/TurningError.jl")
using .TurningError
include("../Modules/TestingFunctions.jl")
using .TestingFunctions



# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))



# testing_params = create_testing_params(mask_states_b4_readout=true)

# test_multi_step_multi_trial_singular(
#         lo_train_0_01, lo_test_0_01, 4, 100;
#         error_metric=RMSE, trials=1,
#         testing_params=testing_params
#     )


# preds = create_pred_for_params_multi_step(
#     lo_train_0_01, lo_test_0_01, 4, 10;
#     k = 100, testing_params=testing_params
# )

# quick_graph_series(preds)

# graph_multi_step_RMSE_vs_n_steps(lo_train_0_01, lo_test_0_01, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 2, 50; ignore_first=100, trials=10)




file_name = joinpath(@__DIR__, "tests.json")

# Read existing data (or create empty array if file doesn’t exist)
tests = if isfile(file_name)
    JSON.parsefile(file_name)
else
    []
end





#####################
### deterministic ###
#####################



# k = 50, m = 3

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    300, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)






# k = 100, m = 3

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)






# k = 650, m = 1

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    650, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)





# k = 50, m = 4

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)





# k = 1300, m = 1

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    1300, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)





# k = 100, m = 4

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
)



println("\n\n\n##########\n\nTESTING Ks\n\n##########\n")
# Testing k


part_lcm = lcm(6, 13) # 78
total_ks = [part_lcm*2, part_lcm*4, part_lcm*8, part_lcm*16]


test_m = 1

println("\nTESTING m = 1\n##########\n")

num_parts = test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks, test_m, 1
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks, test_m, 5
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks, test_m, 50
)

println("Num partitions: ", num_parts)



test_m = 2

println("\nTESTING m = 2\n##########\n")

num_parts = test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 2, test_m, 1
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 2, test_m, 5
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 2, test_m, 50
)

println("Num partitions: ", num_parts)


test_m = 3


num_parts = test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 6, test_m, 1
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 6, test_m, 5
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 6, test_m, 50
)

println("Num partitions: ", num_parts)


test_m = 4


num_parts = test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 13, test_m, 1
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 13, test_m, 5
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    total_ks ./ 13, test_m, 50
)

println("Num partitions: ", num_parts)































##################
### stochastic ###
##################



testing_params = create_testing_params(stochastic=true, stochastic_rescale_V_rec=true)



# k = 50, m = 3

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)






# k = 100, m = 3

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)






# k = 50, m = 4

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)






# k = 100, m = 4

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE,
    testing_params = testing_params
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 4, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE,
    testing_params = testing_params
)








test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    [30, 50, 100, 200], 3, 1, RMSE,
    testing_params = testing_params
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    [30, 50, 100, 200], 3, 5, RMSE,
    testing_params = testing_params
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    [30, 50, 100, 200], 3, 50, RMSE,
    testing_params = testing_params
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    6 .* [30, 50, 100, 200], 3, 1, RMSE,
    testing_params = testing_params
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    6 .* [30, 50, 100, 200], 3, 5, RMSE,
    testing_params = testing_params
)

test_multi_step_k(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    6 .* [30, 50, 100, 200], 3, 50, RMSE,
    testing_params = testing_params
)














