using DelimitedFiles
using JSON

include("../Modules/TurningError.jl")
using .TurningError
include("../Modules/TestingFunctions.jl")
using .TestingFunctions



# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))


ro_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_01.txt")))
ro_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_01.txt")))





file_name = joinpath(@__DIR__, "tests.json")

# Read existing data (or create empty array if file doesn’t exist)
tests = if isfile(file_name)
    JSON.parsefile(file_name)
else
    []
end

part_lcm = lcm(6, 13) # 78




n_steps_test = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
ms = [1, 2, 3, 4]



println("\n\n\n##########\n\nTESTING n_steps\n\n##########\n")

println("TESTING total_k = 468, m = 1, 2, 3, 4\n")

testing_paramss = [
    create_testing_params(),
    create_testing_params(stochastic=true, stochastic_rescale_V_rec=true)
]

ks = [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 13)]

data_label = ["Lorenz 0_01", "Rossler 0_01"]
train_data = [lo_train_0_01, ro_train_0_01]
test_data = [lo_test_0_01, ro_test_0_01]

for data_i in 1:2
    println("\nTESTING data ", data_label[data_i])
    for prediction_type in ["multi_step", "single_step"]
        println("\nTESTING ", prediction_type)
        for testing_params in testing_paramss
            println("\nTESTING ", testing_params)
            for i in 1:4
                num_partitions = test_params(
                    file_name, tests, 
                    train_data[data_i], test_data[data_i], prediction_type, data_label[data_i], 
                    "n_steps", n_steps_test, 
                    Dict("m" => ms[i], "k" => ks[i])
                )
            end
        end
    end
end



println("TESTING single step total_k = 288, m = 1, 2, 3, 4\n")

total_k = 8*6*13
ks = [total_k, total_k÷2, total_k÷6, total_k÷13]

for i in 1:4
    num_partitions = test_params(
        file_name, tests, 
        lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_01", 
        "n_steps", n_steps_test, 
        Dict("m" => ms[i], "k" => ks[i])
    )
end



println("TESTING single step total_k = 288, m = 1, 2, 3, 4\n")

min_subreservoir_k = 48

ks = [48*6, 48*3, 48]

for i in 1:3
    num_partitions = test_params(
        file_name, tests, 
        lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_01", 
        "n_steps", n_steps_test, 
        Dict("m" => ms[i], "k" => ks[i])
    )
end


# up to here


for k in [50, 100]
    println("TESTING subreservoir k = $(k), m = 1, 2, 3, 4\n")
    for m in 1:4
        num_partitions = test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
            "n_steps", n_steps_test, 
            Dict("m" => m, "k" => k)
        )
    end
end



println("\n\nTESTING readout switching\n")

testing_params = create_testing_params(readout_switching=true)
for m in 1:4
    num_partitions = test_params(
        file_name, tests, 
        lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
        "n_steps", n_steps_test, 
        Dict("m" => m, "k" => part_lcm*6),
        testing_params = testing_params
    )
end



println("\n\n\n##########\n\nTESTING Ks\n\n##########\n")

total_ks = [part_lcm, part_lcm*2, part_lcm*4, part_lcm*8, part_lcm*16]
for m in 1:4
    println("\nTESTING m = $(m)\n##########\n")

    for n_steps in [1, 5, 50]
        num_partitions = test_params(
            file_name, tests,
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01",
            "k", total_ks,
            Dict("m" => m, "n_steps" => n_steps))
        println(num_partitions)
    end
end

