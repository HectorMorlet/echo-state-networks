using DelimitedFiles
using JSON

include("../Modules/TurningError.jl")
using .TurningError
include("../Modules/TestingFunctions.jl")
using .TestingFunctions



# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))



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

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    300, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    300, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)






# k = 100, m = 3

num_partitions = test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    100*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)







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





