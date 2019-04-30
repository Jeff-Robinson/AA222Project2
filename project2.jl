## Jeff Robinson - jbrobin@stanford.edu ##

using Random
using Distributions

## Basis Vector Generator ("Algorithms For Optimization" Algorithm 7.1, Kochenderfer & Wheeler) ##
basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

## Quadratic Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.41, Kochenderfer & Wheeler) ##
function P_quad(constraint_values)
    P_quadratic = 0.0
    for i = 1:length(constraint_values)
        P_quadratic += max(constraint_values[i], 0.0)^2
    end
    return P_quadratic
end

## Modified Quadratic Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.41, Kochenderfer & Wheeler) ##
# Addition of 1000 to constraint evaluation to enable minimization of constraint to center of feasible set
function P_quad_mod(constraint_values)
    P_quadratic = 0.0
    for i = 1:length(constraint_values)
        floor_hgt = 10.0^3
        P_quadratic += (constraint_values[i] + floor_hgt)^2
    end
    return P_quadratic
end

## Count Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.39, Kochenderfer & Wheeler) ##
function P_count(constraint_values)
    P_count = 0.0
    for i = 1:length(constraint_values)
        if constraint_values[i] > 0.0
            P_count += 1.0
        end
    end
    return P_count
end

## Log Barrier Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.46, Kochenderfer & Wheeler) ##
function P_log_barrier(constraint_values)
    P_barrier = 0.0
    for i = 1:length(constraint_values)
        if constraint_values[i] >= -1.0 && constraint_values[i] <= 0.0
            P_barrier -= log(-constraint_values[i])
        elseif constraint_values[i] > 0.0
            P_barrier = Inf
        end
    end
    return P_barrier
end

## Hard Barrier Penalty, Inequality <= 0 constraints only ##
function P_hard_barrier(constraint_values)
    if any(constraint_values .> 0.0)
        return Inf
    else
        return 0.0
    end
end

## Penalty Calculator
function Penalties(constraints, x, which="log")
    constraint_values = constraints(x)
    if which == "quad"
        penalties = P_quad(constraint_values)
    elseif which == "quad_mod"
        penalties = P_quad_mod(constraint_values)
    elseif which == "quad+count"
        penalties = P_quad(constraint_values) + P_count(constraint_values)
    elseif which == "log"
        penalties = P_log_barrier(constraint_values)
    elseif which == "hard"
        penalties = P_hard_barrier(constraint_values)
    end
    return penalties
end



## Generalized Pattern Search ("Algorithms For Optimization" Algorithm 7.6, Kochenderfer & Wheeler) ##
# Opportunistic/Dynamically ordered Hooke-Jeeves on quadratic penalty constraint function to find point within feasible set
function generalized_pattern_search(constraints, x, α, max_n_evals, tolerance=5e-1, γ=1/MathConstants.golden)
    D_plus = [basis(i, length(x)) for i = 1:length(x)]
    D_minus = [-basis(i, length(x)) for i = 1:length(x)]
    D = vcat(D_plus, D_minus)
    y = P_quad_mod(constraints(x))

    n_constraint_evals = 0

    feasible_x = []
    while true
        improved = false
        for (i,d) in enumerate(D)

            n_constraint_evals += 1
            if n_constraint_evals >= max_n_evals || α <= tolerance
                return x, feasible_x, n_constraint_evals
            end

            xp = x + α*d
            constraints_xp = constraints(xp)
            yp_act = P_quad(constraints_xp)
            if yp_act == 0.0
                push!(feasible_x, xp)
            end
            yp = P_quad_mod(constraints_xp)
            if yp < y
                x, y, improved = xp, yp, true
                D = pushfirst!(deleteat!(D, i), d)
                break
            end
        end
        if !improved
            α *= γ
        end
    end
end



## Cross-Entropy Method ("Algorithms For Optimization" Algorithm 8.7, Kochenderfer & Wheeler) ##
# using Distributions
# function cross_entropy_method(f, constraints, P, max_n_evals, num_evals_used, m=100, m_elite=10, ρ=1.0)
#     n_evals = num_evals_used
#     while true
#         samples = [rand(P) for i=1:m]
#         sample_y_vals = []
#         for i in 1:m
#             penalty = Penalties(constraints, samples[i], "hard")
#             push!(sample_y_vals, f(samples[i]) + ρ*penalty)

#             n_evals += 1
#             if n_evals >= max_n_evals
#                 break
#             end
#         end
#         infeasible_samples = sample_y_vals .== Inf
#         if any(infeasible_samples)
#             deleteat!(samples, findall(infeasible_samples))
#             deleteat!(sample_y_vals, findall(infeasible_samples))
#         end
#         if length(samples) >= 4
#             order = sortperm(sample_y_vals)
#             flattened_samples = Array{Float64,2}(undef,length(samples[1]),length(samples))
#             for i = 1:length(samples)
#                 flattened_samples[:,i] = samples[i]
#             end
#             if length(order) < m_elite
#                 P = fit(typeof(P), flattened_samples[:,order])
#             else
#                 P = fit(typeof(P), flattened_samples[:,order[1:m_elite]])
#             end
#         end

#         if n_evals >= max_n_evals
#             return mean(P)
#         end
#     end
# end

    

## Simulated Annealing ("Algorithms For Optimization" Algorithm 8.4, Kochenderfer & Wheeler) ##
function simulated_annealing(f, cnstrnts, x, univar_dist, max_n_evals, n_evals_used; 
    t0 = 100.0, 
    γ = 0.98,
    ρ_quad = 1000.0,
    ρ_count = 1000.0
    )

    n_dims = length(x)
    y = f(x)
    n_evals = n_evals_used + 1
    x_best = x
    y_best = y
    while true
        xp = x + rand(univar_dist, n_dims)
        cnstrnt_vals = cnstrnts(xp)
        # penalties = ρ_quad*P_quad(cnstrnt_vals) + ρ_count*P_count(cnstrnt_vals)
        penalties = P_quad_mod(cnstrnt_vals)
        yp = f(xp) + penalties
        dy = yp - y
        # t = t0 * log(2)/log(n_evals) # logarithmic schedule
        t = t0*γ^(n_evals-1) # exponential schedule
        if dy == 0 || rand() < exp(-dy/t)
            x = xp
            y = yp
        end
        if yp < y_best
            x_best = x
            y_best = y
        end
        n_evals += 1
        if n_evals >= max_n_evals - 1
            return x_best
        end
    end
end



## Simulated Annealing ("Algorithms For Optimization" Algorithm 8.6, Kochenderfer & Wheeler) ##
function adaptive_simulated_annealing(f, cnstrnts, x, step_sizes, max_n_evals;
    step_interval = 20,  # num cycles before Corana step adjustment
    temperature = 1000,
    temp_interval = 5,  # num step adjustments before temp adjustment
    γ = 0.85,  # temp reduction coeff
    c = fill(2.0, length(x)),  # Corana step scaling factors
    ρ_quad = 1.0,
    ρ_count = 1.0,
    γp = 2.0 # penalty update coeff
    )

    y = f(x)
    n_evals = 1
    # x_best = x
    # y_best = y
    x_prev = [x]
    y_prev = [y]
    n_evals_prev = [n_evals]
    n_dims = length(x)
    rng = RandomDevice()
    accepted_count = zeros(n_dims)
    counts_cycles = 0
    counts_resets = 0

    while true
        for i in 1:n_dims
            rand_n1_p1 = sign(randn(rng))*rand(rng)
            xp = x + basis(i, n_dims)*rand_n1_p1*step_sizes[i]
            cnstrnt_vals = cnstrnts(xp)
            penalties = ρ_quad*P_quad(cnstrnt_vals) + 
                        ρ_count*P_count(cnstrnt_vals)
            yp = f(xp) + penalties
            n_evals += 1
            dy = yp - y
            # Metropolis Criterion
            if dy <= 0 || rand(rng) < exp(-dy/temperature)
                x = xp
                y = yp
                accepted_count[i] += 1
                # if yp < y_best
                #     x_best = xp
                #     y_best = yp
                # end
                push!(x_prev, x)
                push!(y_prev, y)
                push!(n_evals_prev, n_evals)
            end
            if n_evals >= max_n_evals - 1
                return x_prev[findmin(y_prev)[2]], x_prev, y_prev, n_evals_prev
            end
        end

        ρ_quad *= γp
        ρ_count *= γp
        counts_cycles += 1
        counts_cycles >= step_interval || continue

        counts_cycles = 0
        corana_update!(step_sizes, accepted_count, c, step_interval)
        fill!(accepted_count, 0)
        counts_resets += 1
        counts_resets >= temp_interval || continue

        temperature *= γ
        counts_resets = 0
    end
end

## Corana Update for Simulated Annealing ("Algorithms For Optimization" Algorithm 8.5, Kochenderfer & Wheeler) ##
function corana_update!(v, a, c, ns)
    for i in 1:length(v)
        ai, ci = a[i], c[i]
        if ai > 0.6*ns
            v[i] *= (1 + ci * (ai/ns - 0.6)/0.4)
        elseif ai < 0.4*ns
            v[i] /= (1 + ci * (0.4 - ai/ns)/0.4)
        end
    end
    return v
end



"""

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraints function. Evaluates all the constraint equations and return a Vector of values
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem
"""
function optimize(f, g, c, x0, n, prob)
    num_dims = length(x0)

    ## Find Initial Feasible Point with GPS
    step_size = 1
    feasible_eval_max = 30
    # if prob == "simple_2"
    #     feasible_eval_max = 2
    # end
    start_point, feasible_points, n_evals_used = generalized_pattern_search(c, x0, step_size, feasible_eval_max)
    # println(start_point, ",  ",f(start_point))
    # n_evals_used += 1
    # dist_init_scale = MathConstants.golden
    # init_dist = MvNormal(start_point, dist_init_scale*sqrt(sum([start_point[i]^2 for i=1:length(start_point)])))
    # init_dist = MvNormal(start_point, 1)
    # x_best = cross_entropy_method(f, c, init_dist, n, n_evals_used)


    # println(start_point)
    # x_best = start_point


    ## Simulated Annealing with Corana Update
    # sim_anneal_step_size = 2.0
    # start_point = x0
    # n_evals_used = 0
    # update_interval = Integer(round(100/length(x0)))
    # x_best = simulated_annealing(f, c, start_point, sim_anneal_step_size, update_interval, n_evals_used, n)

    # step_sizes = fill(sim_anneal_step_size, num_dims)
    # x_best, x_prev, y_prev, n_evals_prev = adaptive_simulated_annealing(f, c, start_point, step_sizes, n)


    # univar_dist = Chi(3)
    # univar_dist = Cauchy()
    univar_dist = Normal()
    # n_evals_used = 0
    x_best = simulated_annealing(f, c, start_point, univar_dist, n, n_evals_used)



    # PyPlot.plot(n_evals_prev, y_prev)
    return x_best
end


# using PyPlot