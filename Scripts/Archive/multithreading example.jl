# using Threads
using JSON

function periodic_saver(tests, seconds)
    while true
        sleep(seconds)  # Save every 5 seconds
        println("Saving...")
        open("mat.json", "w") do io
            JSON.print(io, tests)
        end
    end
end

mat = fill(0, 10)
@async periodic_saver(mat, 5)

Threads.@threads for i in 1:100
    println(string(i) * " started")
    # println("$(length(mat))")
    sleep(10)
    mat[i] = Threads.threadid()
    println(string(i) * " finished")
end
