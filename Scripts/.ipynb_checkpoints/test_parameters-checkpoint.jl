using DelimitedFiles
using JSON
using Dates

include("../Modules/TurningError.jl")
using .TurningError
include("../Modules/TestingFunctions.jl")
using .TestingFunctions
include("../Modules/ONReservoir.jl")
using .ONReservoir



# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm("../Data/lorenz_train_0_01.txt"))
lo_test_0_01 = vec(readdlm("../Data/lorenz_test_0_01.txt"));



file_name = "tests.json"

# Read existing data (or create empty array if file doesn’t exist)
tests = if isfile(file_name)
    JSON.parsefile(file_name)
else
    []
end




function check_if_test_exists(test_output)
    for existing_test in tests
        check_dict = deepcopy(existing_test)
        pop!(check_dict, "errors", nothing)
        pop!(check_dict, "date", nothing)
        if check_dict == test_output
            return true
        end
    end
    
    return false
end




function test_multi_step_n_steps(data_train, data_test, data_label,
    k, m, n_step_trials, error_func; ignore_first=100, trials=30,
    testing_params=create_testing_params())
    
    _, unique_partitions = create_ordinal_partition(data_train, m, 1, 1)
    num_partitions = length(unique_partitions)

    test_output = Dict(
        "prediction_type" => "multi_step",
        "testing_parameter" => "n_steps",
        "data" => data_label,
        "n_steps" => n_step_trials,
        "error_func" => first(methods(error_func)).name,
        "k" => k,
        "m" => m,
        "ignore_first" => ignore_first,
        "trials" => trials,
        "testing_params" => testing_params,
        "num_partitions" => num_partitions
    )

    if check_if_test_exists(test_output)
        println("A test with these parameters already exists. Skipping.")
        return
    end
    
    println("\nTesting")
    println(test_output)
    
    push!(tests, test_output)
    
    test_output["errors"] = []
    test_output["date"] = today()
    
    for n_steps in n_step_trials
        test_output["errors"] = [
            test_output["errors"];
            test_multi_step_multi_trial_singular(
                data_train, data_test, m, k;
                error_metric=RMSE, n_steps=n_steps, ignore_first=ignore_first,
                trials=trials, testing_params=testing_params
            )
        ]
        
        save_file()
    end
    
    return(num_partitions)
end

function save_file()
    open(file_name, "w") do f
        JSON.print(f, tests)
    end
end

num_partitions = test_multi_step_n_steps(
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)

test_multi_step_n_steps(
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
)

test_multi_step_n_steps(
    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
    50*num_partitions, 1, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
)



#test_multi_step_n_steps(
#    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
#    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], RMSE
#)
#
#test_multi_step_n_steps(
#    lo_train_0_01, lo_test_0_01, "Lorenz 0_01",
#    50, 3, [1, 2, 3, 5, 10, 20, 30, 40, 50, 70, 100], turning_partition_RMSE
#)



save_file()