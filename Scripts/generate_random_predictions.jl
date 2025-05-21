using DelimitedFiles
using StatsBase

include("../Modules/TestingFunctions.jl")
using .TestingFunctions
include("../Modules/TurningError.jl")
using .TurningError





# Lorenz

lo_0_01 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))
]
lo_0_05 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_05.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_05.txt")))
]
ro_0_1 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_1.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_1.txt")))
]
ro_0_5 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_5.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_5.txt")))
]
mg_0_5 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_0_5.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_0_5.txt")))
]
mg_2_5 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_2_5.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_2_5.txt")))
]

seriess = [lo_0_01, lo_0_05, ro_0_1, ro_0_5, mg_0_5, mg_2_5]
series_names = ["lo_0_01", "lo_0_05", "ro_0_1", "ro_0_5", "mg_0_5", "mg_2_5"]

# seriess = [lo_0_01, ro_0_1, mg_0_5]
# series_names = ["lo_0_01", "ro_0_1", "mg_0_5"]

n = 25000000

for i in eachindex(seriess)
    println("\n", series_names[i])
    println("RMSE:")
    println(RMSE(rand(seriess[i], n), rand(seriess[i], n)))
    println("Turning RMSE:")
    println(turning_partition_RMSE(rand(seriess[i], n), rand(seriess[i], n)))
end


# Rossler RMSE: 7.28
# Rossler turning RMSE: 7.20
# Lorenz RMSE: 11.20
# Lorenz turning RMSE: 11.07
# MG RMSE: 
# 