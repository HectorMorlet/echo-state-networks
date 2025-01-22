mat = empty(10)
Threads.@threads for i in 1:10
    println("$(length(mat))")
    push!(mat, Threads.threadid())
end
mat