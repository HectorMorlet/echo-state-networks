using Plots
using DelimitedFiles

include("../Modules/TestingFunctions.jl")
using .TestingFunctions
include("../Modules/TurningError.jl")
using .TurningError

lo_0_01 = [
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")));
    vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))
]

plot(lo_0_01)

function add_gaussian_noise(series, std_dev)
    noisy_series = series .+ randn(length(series)) .* std_dev
    return noisy_series
end

