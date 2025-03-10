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

# for test in tests
#     if test["testing_parameter"] == "n_steps"
#         if test["m"] == 3
#             test["num_partitions"] = 6
#         end
#         test["total_k"] = test["k"]*test["num_partitions"]
#     end
# end

# save_file(file_name, tests)


part_lcm = lcm(6, 13) # 78


println("#####################")
println("### deterministic ###")
println("#####################")


println("\n\n\n##########\n\nTESTING n_steps\n\n##########\n")



println("TESTING total_k = 468, m = 1, 2, 3, 4\n")

n_steps_test = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 1, "k" => part_lcm*6)
)

@assert(num_partitions == 1)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 2, "k" => part_lcm*3)
)

@assert(num_partitions == 2)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 3, "k" => part_lcm)
)

@assert(num_partitions == 6)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 4, "k" => div(part_lcm*6, 13))
)

@assert(num_partitions == 13)





println("TESTING k = 50, m = 3\n")

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
    Dict("m" => 3, "k" => 50)
)

test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
    Dict("m" => 1, "k" => 300)
)






println("TESTING k = 100, m = 3\n")

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
    Dict("m" => 3, "k" => 100)
)

test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
    Dict("m" => 1, "k" => 100 * num_partitions)
)






println("TESTING k = 650, m = 1\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
    Dict("m" => 1, "k" => 650)
)





println("TESTING k = 50, m = 4\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
    Dict("m" => 4, "k" => 50)
)





println("TESTING k = 1300, m = 1\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
    Dict("m" => 1, "k" => 1300)
)

println("TESTING k = 100, m = 4\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
    Dict("m" => 4, "k" => 100)
)






println("\n\n\n##########\n\nTESTING Ks\n\n##########\n")
# Testing k


total_ks = [part_lcm*2, part_lcm*4, part_lcm*8, part_lcm*16]


test_m = 1

println("\nTESTING m = 1\n##########\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k",
    total_ks,
    Dict("m"=>test_m, "n_steps"=>1))

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks,
    Dict("m" => test_m, "n_steps" => 5)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks,
    Dict("m" => test_m, "n_steps" => 50)
)

println("Num partitions: ", num_partitions)



test_m = 2

println("\nTESTING m = 2\n##########\n")

num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 1)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 5)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 50)
)

println("Num partitions: ", num_parts)


test_m = 3

println("\nTESTING m = 3\n##########\n")


num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 1)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 5)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 50)
)

println("Num partitions: ", num_parts)


test_m = 4

println("\nTESTING m = 4\n##########\n")


num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 1)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 5)
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 50)
)

println("Num partitions: ", num_parts)































println("##################")
println("### stochastic ###")
println("##################")


testing_params = create_testing_params(stochastic=true, stochastic_rescale_V_rec=true)


println("\n\n\n##########\n\nTESTING n_steps\n\n##########\n")



println("TESTING total_k = 468, m = 1, 2, 3, 4\n")

n_steps_test = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 1, "k" => part_lcm*6),
    testing_params=testing_params
)

@assert(num_partitions == 1)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 2, "k" => part_lcm*3),
    testing_params=testing_params
)

@assert(num_partitions == 2)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 3, "k" => part_lcm),
    testing_params=testing_params
)

@assert(num_partitions == 6)

num_partitions = test_multi_step(
    file_name, tests, 
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01", 
    "n_steps", n_steps_test, 
    Dict("m" => 4, "k" => div(part_lcm*6, 13)),
    testing_params=testing_params
)

@assert(num_partitions == 13)









println("\n\n\n##########\n\nTESTING Ks\n\n##########\n")
# Testing k


total_ks = [part_lcm*2, part_lcm*4, part_lcm*8, part_lcm*16]


test_m = 1

println("\nTESTING m = 1\n##########\n")

num_partitions = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k",
    total_ks,
    Dict("m"=>test_m, "n_steps"=>1),
    testing_params=testing_params)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks,
    Dict("m" => test_m, "n_steps" => 5),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks,
    Dict("m" => test_m, "n_steps" => 50),
    testing_params=testing_params
)

println("Num partitions: ", num_partitions)



test_m = 2

println("\nTESTING m = 2\n##########\n")

num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 1),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 5),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 2,
    Dict("m" => test_m, "n_steps" => 50),
    testing_params=testing_params
)

println("Num partitions: ", num_parts)


test_m = 3

println("\nTESTING m = 3\n##########\n")


num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 1),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 5),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 6,
    Dict("m" => test_m, "n_steps" => 50),
    testing_params=testing_params
)

println("Num partitions: ", num_parts)


test_m = 4

println("\nTESTING m = 4\n##########\n")


num_parts = test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 1),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 5),
    testing_params=testing_params
)

test_multi_step(
    file_name, tests,
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    "k", total_ks .÷ 13,
    Dict("m" => test_m, "n_steps" => 50),
    testing_params=testing_params
)

println("Num partitions: ", num_parts)















