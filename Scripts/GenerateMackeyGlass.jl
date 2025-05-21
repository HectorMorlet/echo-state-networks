using DelimitedFiles

total_time = 500*50*5
time_delta = 2.5

# Following taken from: https://github.com/andferrari/MackeyGlass.jl/blob/master/src/MackeyGlass.jl

################################ MACKEYGLASS ###################################
#   Author : J.Courtois                                                        #
#   This script generates a Mackey-Glass time series using the 4th order       #
#   Runge-Kutta method. The code is a straighforward translation in Julia      #
#   of Matlab code, available here :                                           #
#   https://ww2.mathworks.cn/matlabcentral/fileexchange/24390-mackey-glass-    #
#   time-series-generator?s_tid=prof_contriblnk                                #
################################################################################

"""
    mackeyglass_eq(x_t, x_t_minus_tau, a , b)

"""
mackeyglass_eq(x_t, x_t_minus_tau, a, b) =
    -b*x_t + a*x_t_minus_tau/(1 + x_t_minus_tau^10.0);


"""
    mackeyglass_rk4(x_t, x_t_minus_tau, deltat a , b)

"""
function mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b)
    k1 = deltat*mackeyglass_eq(x_t,          x_t_minus_tau, a, b);
    k2 = deltat*mackeyglass_eq(x_t+0.5*k1,   x_t_minus_tau, a, b);
    k3 = deltat*mackeyglass_eq(x_t+0.5*k2,   x_t_minus_tau, a, b);
    k4 = deltat*mackeyglass_eq(x_t+k3,       x_t_minus_tau, a, b);
    return (x_t + k1/6 + k2/3 + k3/3 + k4/6);
end

"""
    T,X = MGGenerator()

Advanced : T,X = MGGenerator(;sample_n, x0, deltat, tau, a, b)

By default :\n
`a = 0.2`     # value for a in eq (1)\n
`b = 0.1`     # value for b in eq (1)\n
`tau = 17`		# delay constant in eq (1)\n
`x0 = 1.2`		# initial condition: x(t=0)=x0\n
`deltat = 0.1`	    # time step size (which coincides with the integration step)\n
`sample_n = 12000`	# total no. of samples, excluding the given initial condition\n
"""
function MGGenerator(;sample_n = 12000, x0 = 1.2, deltat = 0.1, tau = 17,
    timeStep= 0, a = 0.2, b = 0.1)
    X, T, x0, deltat, tau, x_history, x_t, index = MGInit(;sample_n = sample_n,
        timeStep = timeStep, x0 = x0, deltat = deltat, tau = tau,
        a = a, b = b, xwidth = 1)
    for i = 2:sample_n+1
        X[i], x_t, x_history, index, T[i] = MGStep(x_t,x_history, T[i-1],
            index; deltat = deltat, tau = tau, a = a, b = b)
    end
    return T,X
end

"""
    X[i], x_t, x_history, index, T[i] = MGStep(x_t,x_history, T[i-1],index)

Advanced : `X[i], x_t, x_history, index, T[i] = MGStep(x_t,x_history, T[i-1], index; deltat, tau, a, b)`

By default :\n
`a = 0.2`     # value for a in eq (1)\n
`b = 0.1`     # value for b in eq (1)\n
`tau = 17`		# delay constant in eq (1)\n
`deltat = 0.1`	    # time step size (which coincides with the integration step)\n
"""
function MGStep(x_t, x_history, timeStep, index; deltat = 0.1,
                tau = 17, a = 0.2, b = 0.1)
    if tau == 0
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history[index]
    end
    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b);

    if (tau != 0)
        x_history[index] = x_t_plus_deltat;
        nextIndex = mod(index, length(x_history))+1;
    end
    nextTime = timeStep + deltat;

    return x_t, x_t_plus_deltat, x_history, nextIndex, nextTime
end

"""
    X, T, x0, deltat, tau, x_history, x_t, index = MGInit()

Fixe all variables to run an `MGStep(...)`

Advanced : `MGInit(; sample_n = 12000, timeStep = 0,
        x0 = 1.2, deltat = 0.1, tau = 17, a = 0.2, b = 0.1)
"""
function MGInit(; sample_n = 12000, timeStep = 0,
        x0 = 1.2, deltat = 0.1, tau = 17, a = 0.2, b = 0.1,
        xwidth = 1, X = zeros(sample_n+1, xwidth) .* NaN)
    history_length = Int64(floor(tau/deltat))
    x_history = zeros(history_length) # here we assume x(t)=0 for -tau <= t < 0
    x_t = x0;
    #X = zeros(sample_n+1, xwidth) .* NaN; # vector of all generated x samples
    # there is two
    T = zeros(sample_n+1, 1); # vector of time samples
    T[1] = timeStep
    [X[1,i] = x0 for i=1:xwidth]
    return X, T, x0, deltat, tau, x_history, x_t, 1
end

T, X = MGGenerator(sample_n=Int64(total_time/time_delta), deltat=time_delta)

function SplitSet(set, ratio)
    split_point = trunc(Int, ratio*length(set))
    train_set = set[1:split_point]
    test_set = set[split_point+1:length(set)]
    
    return(train_set, test_set)
end

mg_train, mg_test = SplitSet(vec(X), 0.8)

writedlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_train_2_5.txt"), mg_train)
writedlm(joinpath(@__DIR__, "..", "Data", "mackey_glass_test_2_5.txt"), mg_test)
