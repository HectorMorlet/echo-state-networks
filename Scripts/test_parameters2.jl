using DelimitedFiles
using JSON
using StatsBase

include("/Users/hectormorlet/Desktop/Uni/Honours Research/echo-state-networks/Modules/TurningError.jl")
using .TurningError
include("/Users/hectormorlet/Desktop/Uni/Honours Research/echo-state-networks/Modules/TestingFunctions.jl")
using .TestingFunctions

# read in lo_train and lo_test
lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "Data", "lorenz_test_0_01.txt")))

n_trials = 30

R_delay = 100
metrics = [RMSE, turning_partition_RMSE]

function trial_single_step(R_delays, m, k)
    metric_result_means = Dict()
    for metric in metrics
        metric_result_means[metric] = []
    end

    for R_delay in R_delays
        println("\nR_delay: ", R_delay)

        metric_results = Dict()
        for metric in metrics
            metric_results[metric] = []
        end

        for i in 1:n_trials
            println("Trial ", i, " of ", n_trials)

            preds = create_pred_for_params_single_step(
                lo_train_0_01,
                lo_test_0_01,
                m,
                k=k,
                R_delay=R_delay
            )[1:end-R_delay]

            for metric in metrics
                push!(metric_results[metric], metric(lo_test_0_01[1+R_delay:end], preds))
            end
        end

        for (metric, metric_result_series) in metric_results
            push!(metric_result_means[metric], mean(metric_result_series))
        end
    end

    return(metric_result_means)
end

R_delays = [1, 2, 3, 5, 10, 20, 30, 50, 70, 100]

part_lcm = lcm(6, 13) # 78
total_k = 78*8 # 624

println("\n\n\n\nTesting m = 1")
R_delays_1 = trial_single_step(R_delays, 1, 624)
println("\n\n\n\nTesting m = 2")
R_delays_2 = trial_single_step(R_delays, 2, 624 ÷ 2)
println("\n\n\n\nTesting m = 3")
R_delays_3 = trial_single_step(R_delays, 3, 624 ÷ 6)
println("\n\n\n\nTesting m = 4")
R_delays_3 = trial_single_step(R_delays, 4, 624 ÷ 13)


# TODO:
# - Fix beginning jitteryness - is it starting with the test partitions correctly?
# - Saving to json
# - 