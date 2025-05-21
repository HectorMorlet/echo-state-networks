using DelimitedFiles
using JSON

include("../Modules/TurningError.jl")
using .TurningError
include("../Modules/TestingFunctions.jl")
using .TestingFunctions



# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))

lo_train_0_05 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_05.txt")))
lo_test_0_05 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_05.txt")))



ro_train_0_1 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_1.txt")))
ro_test_0_1 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_1.txt")))

ro_train_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_5.txt")))
ro_test_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_5.txt")))



mg_train_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_0_5.txt")))
mg_test_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_0_5.txt")))

mg_train_2_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_2_5.txt")))
mg_test_2_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_2_5.txt")))



lo_train_0_01_extra_long = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01_extra_long.txt")))
lo_test_0_01_extra_long = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01_extra_long.txt")))




file_name = joinpath(@__DIR__, "tests.json")

# Read existing data (or create empty array if file doesn’t exist)
tests = if isfile(file_name)
    JSON.parsefile(file_name)
else
    []
end

for existing_test in tests
    if !haskey(existing_test, "noise_std")
        existing_test["noise_std"] = 0
    end

    if !haskey(existing_test["testing_params"], "layer_connections_one_to_one_constant_value")
        existing_test["testing_params"]["layer_connections_one_to_one_constant_value"] = false
        existing_test["testing_params"]["layer_connections_one_to_one_randomised"] = false
        existing_test["testing_params"]["layer_connections_fully_connected_trans_probs"] = false
        existing_test["testing_params"]["layer_connections_fully_connected_constant_value"] = false
        existing_test["testing_params"]["layer_connections_sparsely_connected"] = false
        existing_test["testing_params"]["layer_connections_disconnected"] = false
        existing_test["testing_params"]["add_self_loops"] = false
    end

    if !haskey(existing_test["testing_params"], "partition_choose_at_random")
        existing_test["testing_params"]["partition_choose_at_random"] = false
        existing_test["testing_params"]["partition_take_turns"] = false
    end

    if !haskey(existing_test["testing_params"], "dont_mask_input_vector")
        existing_test["testing_params"]["dont_mask_input_vector"] = false
    end

    if existing_test["m"] == 1
        existing_test["τ"] = 1
    end
end

part_lcm = lcm(6, 13) # 78









# Delete this
# num_partitions = test_params(
#     file_name, tests, 
#     mg_train_0_5, mg_test_0_5, "multi_step", "MG 0_5", 
#     "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
#     Dict("m" => 4, "k" => 78)
# )

train_datas = Dict()
train_datas["Lorenz 0_01"] = lo_train_0_01
train_datas["Lorenz 0_05"] = lo_train_0_05
train_datas["Rossler 0_1"] = ro_train_0_1
train_datas["Rossler 0_5"] = ro_train_0_5
train_datas["MG 0_5"] = mg_train_0_5
train_datas["MG 2_5"] = mg_train_2_5

test_datas = Dict()
test_datas["Lorenz 0_01"] = lo_test_0_01
test_datas["Lorenz 0_05"] = lo_test_0_05
test_datas["Rossler 0_1"] = ro_test_0_1
test_datas["Rossler 0_5"] = ro_test_0_5
test_datas["MG 0_5"] = mg_test_0_5
test_datas["MG 2_5"] = mg_test_2_5

data_labels_1 = ["Lorenz 0_01", "Rossler 0_1", "MG 0_5"]
data_labels_5 = ["Lorenz 0_05", "Rossler 0_5", "MG 2_5"]

lorenz_labels = ["Lorenz 0_01", "Lorenz 0_05"]
rossler_labels = ["Rossler 0_1", "Rossler 0_5"]
mg_labels = ["MG 0_5"]#, "MG 2_5"]




default_τs = Dict()

default_τs["ORSESN"] = Dict()

default_τs["ORSESN"]["multi_step"] = Dict()
default_τs["ORSESN"]["multi_step"]["Lorenz 0_01"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["ORSESN"]["multi_step"]["Lorenz 0_05"] = Dict(2 => 10, 3 => 10, 4 => 4) # 4 (but 10 also good) -> just use 4
default_τs["ORSESN"]["multi_step"]["Rossler 0_1"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["ORSESN"]["multi_step"]["Rossler 0_5"] = Dict(2 => 4, 3 => 4, 4 => 4)
default_τs["ORSESN"]["multi_step"]["MG 0_5"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["ORSESN"]["multi_step"]["MG 2_5"] = Dict(2 => 50, 3 => 50, 4 => 50) # 150 for m=2,3 and 50 for m=4 -> just use 50

default_τs["ORSESN"]["single_step"] = Dict()
default_τs["ORSESN"]["single_step"]["Lorenz 0_01"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["ORSESN"]["single_step"]["Lorenz 0_05"] = Dict(2 => 4, 3 => 4, 4 => 4)
default_τs["ORSESN"]["single_step"]["Rossler 0_1"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["ORSESN"]["single_step"]["Rossler 0_5"] = Dict(2 => 4, 3 => 4, 4 => 4)
default_τs["ORSESN"]["single_step"]["MG 0_5"] = Dict(2 => 20, 3 => 20, 4 => 20) # -> to test again with tau > 100 -> just use 20
default_τs["ORSESN"]["single_step"]["MG 2_5"] = Dict(2 => 50, 3 => 50, 4 => 20) # 50 for m=2,3 and 20 for m=4 -> just use 50

default_τs["OPESN"] = Dict()

default_τs["OPESN"]["multi_step"] = Dict()
default_τs["OPESN"]["multi_step"]["Lorenz 0_01"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["OPESN"]["multi_step"]["Lorenz 0_05"] = Dict(2 => 10, 3 => 10, 4 => 4) # 4 (but 10 good for m=2,3) -> preference just 4 but check direct
default_τs["OPESN"]["multi_step"]["Rossler 0_1"] = Dict(2 => 20, 3 => 20, 4 => 10) # for m=2,3 and = 10 for m=4 -> very  big error for tau=20,m=4
default_τs["OPESN"]["multi_step"]["Rossler 0_5"] = Dict(2 => 20, 3 => 20, 4 => 10) # for m=2,3 and = 10 for m=4 -> very  big error for tau=20,m=4
default_τs["OPESN"]["multi_step"]["MG 0_5"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["OPESN"]["multi_step"]["MG 2_5"] = Dict(2 => 50, 3 => 50, 4 => 50)

default_τs["OPESN"]["single_step"] = Dict()
default_τs["OPESN"]["single_step"]["Lorenz 0_01"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["OPESN"]["single_step"]["Lorenz 0_05"] = Dict(2 => 4, 3 => 4, 4 => 4)
default_τs["OPESN"]["single_step"]["Rossler 0_1"] = Dict(2 => 10, 3 => 20, 4 => 10) # 10 for 2,4 and 20 for 3 -> preference 20
default_τs["OPESN"]["single_step"]["Rossler 0_5"] = Dict(2 => 10, 3 => 10, 4 => 4) # for m=2,3 and = 10 for m=4 -> big error otherwise
default_τs["OPESN"]["single_step"]["MG 0_5"] = Dict(2 => 20, 3 => 20, 4 => 20)
default_τs["OPESN"]["single_step"]["MG 2_5"] = Dict(2 => 50, 3 => 50, 4 => 20) # for for m=2,3 and = 20 for m=4 -> HUGE eror otherwise




default_ρs = Dict()

default_ρs["ORSESN"] = Dict()
default_ρs["ORSESN"]["Lorenz 0_01"] = Dict()
default_ρs["ORSESN"]["Lorenz 0_01"]["multi_step"] = Dict(i => 1.7 for i in 1:4)
default_ρs["ORSESN"]["Lorenz 0_01"]["single_step"] = Dict(i => 1.7 for i in 1:4)
default_ρs["ORSESN"]["Lorenz 0_05"] = Dict()
default_ρs["ORSESN"]["Lorenz 0_05"]["multi_step"] = Dict(1=>2, 2=>2, 3=>2, 4=>1.4)
default_ρs["ORSESN"]["Lorenz 0_05"]["single_step"] = Dict(i => 2.0 for i in 1:4)
default_ρs["ORSESN"]["Rossler 0_1"] = Dict()
default_ρs["ORSESN"]["Rossler 0_1"]["multi_step"] = Dict(i => 1.7 for i in 1:4)
default_ρs["ORSESN"]["Rossler 0_1"]["single_step"] = Dict(i => 1.7 for i in 1:4)
default_ρs["ORSESN"]["Rossler 0_5"] = Dict()
default_ρs["ORSESN"]["Rossler 0_5"]["multi_step"] = Dict(1=>1.4, 2=>1.4, 3=>1.4, 4=>1.4)
default_ρs["ORSESN"]["Rossler 0_5"]["single_step"] = Dict(1=>2.0, 2=>2.0, 3=>2.0, 4=>2.0)




n_steps_test_1 = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500]
n_steps_test_5 = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]

n_steps_test_1_param_testing = [1, 20, 50, 100]
n_steps_test_5_param_testing = [1, 10, 20, 40]

ks = Dict()
ks["Lorenz 0_01"] = [part_lcm*6, part_lcm*3, part_lcm, 20]
ks["Lorenz 0_05"] = [part_lcm*6, part_lcm*3, part_lcm, 20]
ks["Rossler 0_1"] = [part_lcm*6, part_lcm*3, part_lcm, 20]
ks["Rossler 0_5"] = [part_lcm*6, part_lcm*3, part_lcm, 20]
ks["MG 0_5"] = [part_lcm*6, part_lcm*3, part_lcm, 20]
ks["MG 2_5"] = [part_lcm*6, part_lcm*3, part_lcm, 20]


# test_params(
#     file_name, tests, 
#     train_datas["Rossler 0_1"], test_datas["Rossler 0_1"], "multi_step", "Rossler 0_1", 
#     "n_steps", n_steps_test_1,
#     Dict("m" => 4, "k" => 20),
#     testing_params=create_testing_params(),
#     noise_std=0.1,
#     do_chart=false,
#     τ = default_τs["OPESN"]["multi_step"]["Rossler 0_1"][4]
# )

# for m in 1:4
#     test_params(
#         file_name, tests, 
#         train_datas["Rossler 0_1"], test_datas["Rossler 0_1"], "single_step", "Rossler 0_1", 
#         "n_steps", n_steps_test_1,
#         Dict("m" => m, "k" => 20),
#         testing_params=create_testing_params(),
#         noise_std=0.1,
#         do_chart=false,
#         τ = m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1
#     )
# end

# exit()

# One-off test
# data_label = "Lorenz 0_01"
# prediction_type = "multi_step"
# architecture = "OPESN"
# for m in 2:4
#     println("Tau: ", m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1)
#     test_params(
#         file_name, tests, 
#         train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
#         "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],#data_label in data_labels_1 ? n_steps_test_1 : n_steps_test_5,
#         Dict("m" => m, "k" => 468),
#         testing_params=create_testing_params(),
#         noise_std=0.1,
#         do_chart=false,
#         τ = m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1,
#         ρ = 1.1,# default_ρs[architecture][data_label][prediction_type][m],
#         trials = 30,
#         check_for_existing_test=false
#     )
# end

# exit()



# test_params(
#     file_name, tests, 
#     train_datas["MG 0_5"], test_datas["MG 0_5"], "multi_step", "MG 0_5", 
#     "n_steps", [200],
#     Dict("m" => 4, "k" => 500),
#     testing_params=create_testing_params(layer_connections_disconnected=true),
#     noise_std=0.1,
#     do_chart=false,
#     τ = default_τs["OPESN"]["multi_step"]["MG 0_5"][4],
#     ρ = 1.1,# default_ρs[architecture][data_label][prediction_type][m],
#     trials = 30,
#     check_for_existing_test=false
# )



# Testing for ρ
for testing_params in [create_testing_params(readout_switching=true), create_testing_params(layer_connections_disconnected=true)]
    for data_label in ["Lorenz 0_01", "Lorenz 0_05", "Rossler 0_1", "MG 0_5"]
        for prediction_type in ["multi_step", "single_step"]
            for m in 1:4
                n_steps = data_label in data_labels_1 ? n_steps_test_1 : n_steps_test_5
                if testing_params == create_testing_params(layer_connections_disconnected=true)
                    if data_label == "Lorenz 0_01" && prediction_type == "multi_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250]
                    elseif data_label == "Lorenz 0_01" && prediction_type == "single_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150]
                    elseif data_label == "Lorenz 0_05" && prediction_type == "multi_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
                    elseif data_label == "Lorenz 0_05" && prediction_type == "single_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 40]
                    elseif data_label == "Rossler 0_1" && prediction_type == "multi_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500]
                    elseif data_label == "Rossler 0_1" && prediction_type == "single_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500]
                    elseif data_label == "MG 0_5" && prediction_type == "multi_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200]
                    elseif data_label == "MG 0_5" && prediction_type == "single_step"
                        n_steps = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200]
                    end
                end

                architecture = "ORSESN"
                if testing_params == create_testing_params(layer_connections_disconnected=true)
                    # k = ks[data_label][m]
                    architecture = "OPESN"
                end
                test_params(
                    file_name, tests, 
                    train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
                    "n_steps", n_steps,
                    Dict("m" => m, "k" => 500),
                    testing_params=testing_params,
                    noise_std=0.1,
                    do_chart=false,
                    τ = m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1,
                    ρ = 1.1,# default_ρs[architecture][data_label][prediction_type][m],
                    trials = 30,
                    check_for_existing_test=true
                )
                # for ρ in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]#[0.6, 0.7, 0.75, 0.85, 0.95]
                #     # try
                #         test_params(
                #             file_name, tests, 
                #             train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
                #             "n_steps", data_label in data_labels_1 ? n_steps_test_1_param_testing : n_steps_test_5_param_testing,
                #             Dict("m" => m, "k" => 500),
                #             testing_params=testing_params,
                #             noise_std=0.1,
                #             do_chart=false,
                #             τ = m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1,
                #             ρ = ρ,
                #             trials = 3
                #         )
                #     # catch e
                #     #     println("-----------------------------------------------------------------------------------------------")
                #     #     println("Error------------------------------------------------------------------------------------------")
                #     #     println("-----------------------------------------------------------------------------------------------")
                #     #     println(e)
                #     # end
                # end
            end
        end
    end
end

exit()

# Testing OPESN connections

testing_paramss = [
    create_testing_params(),
    create_testing_params(stochastic = true),
    create_testing_params(layer_connections_one_to_one_constant_value = true),
    create_testing_params(layer_connections_one_to_one_randomised = true),
    create_testing_params(layer_connections_fully_connected_trans_probs = true),
    create_testing_params(layer_connections_fully_connected_constant_value = true),
    create_testing_params(layer_connections_sparsely_connected = true),
    create_testing_params(layer_connections_disconnected = true)
]

for prediction_type in ["multi_step", "single_step"]
    for testing_params in testing_paramss
        for m in [1,4]
            test_params(
                file_name, tests, 
                lo_train_0_01, lo_test_0_01, prediction_type, "Lorenz 0_01", 
                "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100],
                Dict("m" => m, "k" => 468),
                testing_params=testing_params,
                noise_std=0.1,
                do_chart=false,
                τ=m != 1 ? default_τs["OPESN"][prediction_type]["Lorenz 0_01"][m] : 1
            )
        end
    end
end




exit()





# architecture = "OPESN"
# # prediction_type = "single_step"
# data_label = "Rossler 0_5"
# # for ρ in [1.4, 1.5, 1.7, 2.0, 2.4]
# for prediction_type in ["multi_step", "single_step"]
#     for m in 1:4
#         for τ in [1, 2, 4, 10, 20, 50]
#             test_params(
#                 file_name, tests, 
#                 train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
#                 "n_steps", data_label in data_labels_1 ? n_steps_test_1_param_testing : n_steps_test_5_param_testing,
#                 Dict("m" => m, "k" => 468),
#                 testing_params=create_testing_params(readout_switching=(architecture=="ORSESN")),
#                 noise_std=0.1,
#                 do_chart=false,
#                 τ = τ,##m != 1 ? default_τs[architecture][prediction_type][data_label][m] : 1,
#                 ρ = 1.1,#ρ,
#                 trials = 3#,
#                 # check_for_existing_test = false
#             )
#         end
#     end
# end

exit()
# exit()


# for ρ in [1.0, 1.05, 1.1, 1.15, 1.2]
#     test_params(
#         file_name, tests, 
#         train_datas["Lorenz 0_05"], test_datas["Lorenz 0_05"], "multi_step", "Lorenz 0_05", 
#         "n_steps", [20],
#         Dict("m" => 1, "k" => 468),
#         testing_params=create_testing_params(readout_switching=true),
#         noise_std=0.1,
#         do_chart=false,
#         τ=1,
#         ρ=ρ
#     )
# end

# exit()




n_steps_test_1 = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500]
n_steps_test_5 = [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100]
# n_steps_test = [1, 2, 5, 10, 20, 30, 50, 70, 100, 200, 500]
ms = [1, 2, 3, 4]
ks = [part_lcm*6, part_lcm*3, part_lcm, 20]#div(part_lcm*6, 13)]


for testing_params in [create_testing_params(readout_switching=true), create_testing_params()]
    for prediction_type in ["multi_step", "single_step"]
        for m in 1:4
            for data_label in ["Rossler 0_5"]#lorenz_labels#[data_labels_1; data_labels_5]
                if testing_params == create_testing_params()
                    k = ks[m]
                else
                    k = 468
                end

                if data_label in data_labels_1
                    n_steps_test = n_steps_test_1
                else
                    n_steps_test = n_steps_test_5
                end

                # try
                test_params(
                    file_name, tests, 
                    train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
                    "n_steps", n_steps_test,
                    Dict("m" => m, "k" => k),
                    testing_params=testing_params,
                    trials=10,
                    noise_std=0.1,
                    do_chart=false,
                    τ=m==1 ? 1 : default_τs[(testing_params == create_testing_params() ? "OPESN" : "ORSESN")][prediction_type][data_label][m],
                    ρ=1.1
                )
                # catch e
                #     println("-----------------------------------------------------------------------------------------------")
                #     println("Error------------------------------------------------------------------------------------------")
                #     println("-----------------------------------------------------------------------------------------------")
                #     println(e)
                # end
            end
        end
    end
end

# test_params(
#     file_name, tests, 
#     train_datas["Lorenz 0_05"], test_datas["Lorenz 0_05"], "multi_step", "Lorenz 0_05", 
#     "n_steps", [1, 3, 10, 20, 40],
#     Dict("m" => 4, "k" => 468),
#     testing_params=create_testing_params(readout_switching=true),
#     noise_std=0.1,
#     do_chart=false,
#     τ=40,
#     check_for_existing_test=true
# )

# Rossler 0.1
# MG 2.5: tau = 50
# 

# Testing for tau
for testing_params in [create_testing_params(readout_switching=true), create_testing_params()]
    for prediction_type in ["multi_step", "single_step"]
        for data_label in ["MG 2_5"]#["MG 0_5", "MG 2_5", "Rossler 0_1"]#["Lorenz 0_05", "Rossler 0_1", "MG 0_5", "Rossler 0_5", "MG 2_5"]
            if data_label in data_labels_1
                τs = [120, 150, 200]#[1, 2, 4, 10, 20, 50, 100]
                n_steps_test = [1, 20, 50, 100]
            else
                τs = [200, 250, 300]#[80, 100, 150]#[1, 2, 4, 10, 20, 50]
                n_steps_test = [1, 10, 20, 40]#3, 
            end

            for τ in τs
                for m in 2:4
                    if testing_params == create_testing_params()
                        k = ks[m]
                    else
                        k = 468
                    end

                    test_params(
                        file_name, tests, 
                        train_datas[data_label], test_datas[data_label], prediction_type, data_label, 
                        "n_steps", n_steps_test,
                        Dict("m" => m, "k" => k),
                        testing_params=testing_params,
                        noise_std=0.1,
                        do_chart=false,
                        τ=τ
                    )
                end
            end
        end
    end
end

exit()

ks = [part_lcm*6, part_lcm*3, part_lcm, 20]#div(part_lcm*6, 13)]



for prediction_type in ["multi_step", "single_step"]
    for noise_std in [0, 0.1, 0.2, 0.3, 0.5, 1.0]
        for m in 1:4
            test_params(
                file_name, tests, 
                lo_train_0_01, lo_test_0_01, prediction_type, "Lorenz 0_01", 
                "n_steps", n_steps_test,
                Dict("m" => m, "k" => ks[m]),
                testing_params=create_testing_params(),
                noise_std=noise_std,
                do_chart=false,
                τ=20
            )
        end
    end
end

for prediction_type in ["multi_step", "single_step"]
    for m in 2:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, prediction_type, "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => 468),
            testing_params=create_testing_params(partition_choose_at_random = true, readout_switching = true),
            noise_std=0.1,
            do_chart=false,
            τ=20
        )
    end
end



chosen_τ = 20

testing_paramss = [
    # create_testing_params(),
    # create_testing_params(dont_mask_input_vector = true),
    # create_testing_params(mask_states_b4_readout = true),
    # create_testing_params(readout_switching = true),
    # create_testing_params(readout_switching = true, partition_choose_at_random = true),
    create_testing_params(stochastic = true),
    # create_testing_params(stochastic_rescale_V_rec = true),

    # create_testing_params(partition_choose_at_random = true),
    # create_testing_params(partition_take_turns = true),

    create_testing_params(layer_connections_one_to_one_constant_value = true),
    create_testing_params(layer_connections_one_to_one_randomised = true),
    create_testing_params(layer_connections_fully_connected_trans_probs = true),
    create_testing_params(layer_connections_fully_connected_constant_value = true),
    create_testing_params(layer_connections_sparsely_connected = true),
    create_testing_params(layer_connections_disconnected = true)
    # create_testing_params(add_self_loops = true),
    # create_testing_params(layer_connections_disconnected = true, mask_states_b4_readout = true),
    # create_testing_params(layer_connections_disconnected = true, dont_mask_input_vector = true)
]

for testing_params in testing_paramss
    for m in 2:3
        if testing_params.readout_switching
            k = 468
        else
            k = ks[m]
        end
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
            # ro_train_0_1, ro_test_0_1, "multi_step", "Rossler 0_1", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => k),
            testing_params=testing_params,
            noise_std=0.1,
            do_chart=false,
            τ=chosen_τ
        )
    end
end

for testing_params in testing_paramss
    for m in 2:3
        if testing_params.readout_switching
            k = 468
        else
            k = ks[m]
        end
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => k),
            testing_params=testing_params,
            noise_std=0.1,
            do_chart=false,
            τ=chosen_τ
        )
    end
end

# one off
# test_params(
#     file_name, tests, 
#     lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
#     "n_steps", [100],
#     Dict("m" => 4, "k" => 468),
#     testing_params=create_testing_params(readout_switching = true),
#     noise_std=0.0,
#     do_chart=false,
#     τ=20
# )

println("Noise ---------------------------------------------------------------------------------------------------------------------")

for noise_std in [0, 0.1, 0.3, 0.5, 1.0]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => 468),
            testing_params=create_testing_params(readout_switching = true),
            noise_std=noise_std,
            do_chart=false,
            τ=20
        )
    end
end

for noise_std in [0, 0.1, 0.3, 0.5, 1.0]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => 468),
            testing_params=create_testing_params(readout_switching = true),
            noise_std=noise_std,
            do_chart=false,
            τ=20
        )
    end
end


println("Tau ---------------------------------------------------------------------------------------------------------------------")

for τ in [1, 2, 5, 10, 20, 50, 100]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_05", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => 468),
            testing_params=create_testing_params(readout_switching = true),
            noise_std=0.1,
            do_chart=false,
            τ=τ
        )
    end
end

for τ in [1, 2, 5, 10, 20, 50, 100]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_05", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => 468),
            testing_params=create_testing_params(readout_switching = true),
            noise_std=0.1,
            do_chart=false,
            τ=τ
        )
    end
end





for τ in [1, 2, 5, 10, 20, 50, 100]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => ks[m]),
            testing_params=create_testing_params(),
            noise_std=0.1,
            do_chart=false,
            τ=τ
        )
    end
end

for τ in [1, 2, 5, 10, 20, 50, 100]
    for m in 1:4
        test_params(
            file_name, tests, 
            lo_train_0_01, lo_test_0_01, "single_step", "Lorenz 0_01", 
            "n_steps", n_steps_test,
            Dict("m" => m, "k" => ks[m]),
            testing_params=create_testing_params(),
            noise_std=0.1,
            do_chart=false,
            τ=τ
        )
    end
end



println("Attractors ---------------------------------------------------------------------------------------------------------------------")

# New Mackie Glass

# One off repeat

test_params(
    file_name, tests, 
    ro_train_0_1, ro_test_0_1, "multi_step", "Rossler 0_1", 
    "n_steps", n_steps_test.*2,
    Dict("m" => 4, "k" => 40),
    testing_params=create_testing_params(),
    noise_std=0.1,
    do_chart=false,
    τ=20,
    check_for_existing_test=false
)

# test_params(
#     file_name, tests, 
#     mg_train_0_5, mg_test_0_5, "multi_step", "MG 0_5", 
#     "n_steps", n_steps_test.*2,
#     Dict("m" => 1, "k" => 468),
#     testing_params=create_testing_params(),
#     noise_std=0.1,
#     do_chart=false,
#     τ=20,
#     check_for_existing_test=false
# )

# Subreservoirs
for prediction_type in ["multi_step", "single_step"]
    for m in 1:4
        try
            test_params(
                file_name, tests, 
                mg_train_0_5, mg_test_0_5, prediction_type, "MG 0_5", 
                "n_steps", n_steps_test.*2,
                Dict("m" => m, "k" => ks[m]),
                testing_params=create_testing_params(),#layer_connections_disconnected=true),
                noise_std=0.1,
                do_chart=false,
                τ=20
            )
        catch
            println("Error --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            continue
        end
    end
end

# Readout switching
for prediction_type in ["multi_step", "single_step"]
    for m in 1:4
        try
            test_params(
                file_name, tests, 
                mg_train_0_5, mg_test_0_5, prediction_type, "MG 0_5", 
                "n_steps", n_steps_test.*2,
                Dict("m" => m, "k" => 468),
                testing_params=create_testing_params(readout_switching = true),
                noise_std=0.1,
                do_chart=false,
                τ=20
            )
        catch
            println("Error --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            continue
        end
    end
end



## New Rossler testing

# Subreservoirs
for prediction_type in ["multi_step", "single_step"]
    for m in 1:4
        try
            test_params(
                file_name, tests, 
                ro_train_0_1, ro_test_0_1, prediction_type, "Rossler 0_1", 
                "n_steps", n_steps_test.*2,
                Dict("m" => m, "k" => ks[m]),
                testing_params=create_testing_params(),#layer_connections_disconnected=true),
                noise_std=0.1,
                do_chart=false,
                τ=20
            )
        catch
            println("Error --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            continue
        end
    end
end

# Readout switching
for prediction_type in ["multi_step", "single_step"]
    for m in 1:4
        try
            test_params(
                file_name, tests, 
                ro_train_0_1, ro_test_0_1, prediction_type, "Rossler 0_1", 
                "n_steps", n_steps_test.*2,
                Dict("m" => m, "k" => 468),
                testing_params=create_testing_params(readout_switching = true),
                noise_std=0.1,
                do_chart=false,
                τ=20
            )
        catch
            println("Error --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            continue
        end
    end
end

# one-off experiment for comparing the m=4 readout switching to a traditional ESN with the same number of total parameters

# println("One-off experiment 1 ---------------------------------------------------------------")
# test_params(
#     file_name, tests, 
#     lo_train_0_01, lo_test_0_01, "multi_step", "Lorenz 0_01", 
#     # ro_train_0_1, ro_test_0_1, "multi_step", "Rossler 0_1", 
#     "n_steps", [50, 70, 100],
#     Dict("m" => 3, "k" => 78),
#     testing_params=create_testing_params(),
#     noise_std=0.1,
#     do_chart=false,
#     τ=chosen_τ,
#     check_for_existing_test=false
# )



exit()


ks = [part_lcm*6, part_lcm*3, part_lcm, 20]


for m in 1:4
    test_params(
        file_name, tests, 
        lo_train_0_01_extra_long, lo_test_0_01_extra_long, "multi_step", "Lorenz 0_01 Extra Long", 
        "n_steps", n_steps_test,
        Dict("m" => m, "k" => ks[m]),
        testing_params=create_testing_params(),
        noise_std=0.1,
        do_chart=false,
        τ=chosen_τ
    )
end





exit()






println("\n\n\n##########\n\nTESTING n_steps\n\n##########\n")

println("TESTING total_k = 468, m = 1, 2, 3, 4\n")

testing_paramss = [
    create_testing_params(),
    # create_testing_params(stochastic=true, stochastic_rescale_V_rec=true),
    # create_testing_params(read)
]

ks = Dict(
    "Lorenz 0_01" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 13)],
    "Lorenz 0_05" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 13)],
    "Rossler 0_1" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 12)],
    "Rossler 0_5" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 10)],
    "MG 0_5" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 17)],
    "MG 2_5" => [part_lcm*6, part_lcm*3, part_lcm, div(part_lcm*6, 17)],
)

data_labels = [
    "Lorenz 0_01",
    "Lorenz 0_05",
    "Rossler 0_1",
    "Rossler 0_5",
    "MG 0_5",
    "MG 2_5"
]
train_datas = [
    lo_train_0_01,
    lo_train_0_05,
    ro_train_0_1,
    ro_train_0_5,
    mg_train_0_5,
    mg_train_2_5
]
test_datas = [
    lo_test_0_01,
    lo_test_0_05,
    ro_test_0_1,
    ro_test_0_5,
    mg_test_0_5,
    mg_test_2_5
]

for (data_label, train_data, test_data) in zip(data_labels, train_datas, test_datas)
    for noise_std in [0, 0.1, 0.2, 0.3, 0.5, 1.0]
        for prediction_type in ["multi_step", "single_step"]
            for testing_params in testing_paramss
                for m_i in 1:3
                    for τ in [1, 2, 5, 10, 20, 30, 50]
                        if τ != 1 && ms[m_i] == 1
                            continue
                        end

                        println("\n\n\nTESTING:")
                        println("    data ", data_label)
                        println("    ", prediction_type)
                        println("    m = ", ms[m_i], " and k = ", ks[data_label][m_i])
                        println("    noise = ", noise_std)
                        println("Params: ", testing_params)
                        test_params(
                            file_name, tests, 
                            train_data, test_data, prediction_type, data_label, 
                            "n_steps", n_steps_test,
                            Dict("m" => ms[m_i], "k" => ks[data_label][m_i]),
                            testing_params=testing_params,
                            noise_std=noise_std,
                            do_chart=false,
                            τ=τ
                        )
                    end
                end
            end
        end
    end
end










for τ in [20]#, 30, 50, 1, 2, 5, 10]
    for noise_std in [0, 0.1, 0.3, 0.5, 1.0]
        num_partitions = test_params(
            file_name, tests, 
            lo_train_0_01_extra_long, lo_test_0_01_extra_long, "multi_step", "Lorenz 0_01 Extra Long", 
            "n_steps", [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], 
            Dict("m" => 4, "k" => 36),
            noise_std=noise_std,
            do_chart=false,
            τ=τ
        )
    end
end


exit()

# Dict{String, Any}("prediction_type" => "single_step", "testing_params" => TestingParameters(false, false, false, false), "data" => "Rossler 0_1", "num_partitions" => 12, "error_funcs" => [:RMSE, :turning_partition_RMSE], "noise_std" => 0, "testing_parameter" => "n_steps", "k" => 39, "ignore_first" => 100, "trials" => 30, "m" => 4, "total_k" => 468, "aggregation_funcs" => [:mean, :std, :minimum, :maximum, :median], "n_steps" => [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100])



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

# multi_step
# Rossler
# vanilla subreservoirs
# m = 3
# k = 78
# "total_k" => 468






