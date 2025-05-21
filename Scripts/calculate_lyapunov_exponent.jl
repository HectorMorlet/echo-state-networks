using DelimitedFiles
using DynamicalSystems
using CairoMakie
using DelayEmbeddings: embed
using StatsBase

lo_train_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_01.txt")))
lo_test_0_01 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_01.txt")))
lo_0_01 = [lo_train_0_01; lo_test_0_01]

lo_train_0_05 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_train_0_05.txt")))
lo_test_0_05 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "lorenz_test_0_05.txt")))
lo_0_05 = [lo_train_0_05; lo_test_0_05]

mg_train_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_0_5.txt")))
mg_test_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_0_5.txt")))
mg_0_5 = [mg_train_0_5; mg_test_0_5]

mg_train_2_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_2_5.txt")))
mg_test_2_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_2_5.txt")))
mg_2_5 = [mg_train_2_5; mg_test_2_5]

ro_train_0_1 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_1.txt")))
ro_test_0_1 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_1.txt")))
ro_0_1 = [ro_train_0_1; ro_test_0_1]

ro_train_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_train_0_5.txt")))
ro_test_0_5 = vec(readdlm(joinpath(@__DIR__, "..", "Data", "rossler_test_0_5.txt")))
ro_0_5 = [ro_train_0_5; ro_test_0_5]

datasets = Dict(
    "lo_0_01" => lo_0_01,
    "lo_0_05" => lo_0_05,
    "mg_0_5" => mg_0_5,
    "mg_2_5" => mg_2_5,
    "ro_0_1" => ro_0_1,
    "ro_0_5" => ro_0_5
)





function plot_lyapunov(x, ks, Δt; ds=[4, 8], τs=[7, 15], ntype = NeighborNumber(5))
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="k ($(Δt)×t)", ylabel="E - E(0)")
    ntype = NeighborNumber(5) #5 nearest neighbors of each state
    λs = []

    for d in ds, τ in τs
        r = embed(x, d, τ)

        # E1 = lyapunov_from_data(r, ks1; ntype)
        # λ1 = ChaosTools.linreg(ks1 .* Δt, E1)[2]
        # plot(ks1,E1.-E1[1], label = "dense, d=$(d), τ=$(τ), λ=$(round(λ1, 3))")

        E2 = lyapunov_from_data(r, ks; ntype)
        λ2 = ChaosTools.linreg(ks .* Δt, E2)[2]
        lines!(ks, E2.-E2[1]; label = "d=$(d), τ=$(τ), λ=$(round(λ2, digits = 3))")

        push!(λs, λ2)
    end
    println("mean λ: ", mean(λs))
    println("mean 1/λ: ", 1/mean(λs))
    if length(ds)*length(τs) <= 9
        axislegend(ax; position = :lt)
    end
    ax.title = "Continuous Reconstruction Lyapunov"
    fig
end


plot_lyapunov(lo_0_01, 0:4:750, 0.01)
plot_lyapunov(lo_0_01, 150:1:380, 0.01, τs=7:1:15, ds=4:1:8)
# λ ≈ 0.896
# 1/λ ≈ 1.116

plot_lyapunov(lo_0_05, 0:1:150, 0.05, τs=1:1:3, ds=3:1:6)
plot_lyapunov(lo_0_05, 30:1:70, 0.05, τs=1:1:3, ds=3:1:6)
# λ ≈ 0.881
# 1/λ ≈ 1.134

plot_lyapunov(ro_0_1, 0:4:750, 0.1)
plot_lyapunov(ro_0_1, 200:1:350, 0.1, τs=7:1:15, ds=4:1:8)
# λ ≈ 0.062
# 1/λ ≈ 16.098

plot_lyapunov(ro_0_5, 0:10:1000, 0.5)
plot_lyapunov(ro_0_5, 200:10:500, 0.5, τs=1:1:3, ds=4:2:8)
# λ ≈ 0.0123
# 1/λ ≈ 80.981

plot_lyapunov(mg_0_5, 0:4:2000, 0.5)
plot_lyapunov(mg_0_5, 300:1:1200, 0.5, τs=7:1:15, ds=4:1:8)
# λ ≈ 0.00392
# 1/λ ≈ 255.369

plot_lyapunov(mg_2_5, 0:4:2000, 2.5, τs=[35, 75])
plot_lyapunov(mg_2_5, 0:4:2000, 2.5, τs=7:1:15, ds=4:2:8)
# λ ≈ 
# 1/λ ≈ 