## Jeff Robinson - jbrobin@stanford.edu ##

using Random

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
        P_quadratic += max(constraint_values[i]+1000, 0.0)^2
    end
    return P_quadratic
end

## Count Penalty, Inequality <= 0 constraints only ("Algorithms For Optimization" Equation 10.39, Kochenderfer & Wheeler) ##
function P_count(constraint_values)
    P_count = 0.0
    for i = 1:length(constraint_values)
        if constraint_values[i] > 0
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
            # P_barrier = Float64(typemax(Int32))
            P_barrier = Inf
        end
    end
    return P_barrier
end

## Barrier Penalty, Inequality <= 0 constraints only ##
function P_hard_barrier(constraint_values)
    P_barrier = 0.0
    for i = 1:length(constraint_values)
        if constraint_values[i] > 0.0
            # P_barrier = Float64(typemax(Int32))
            P_barrier = Inf
        end
    end
    return P_barrier
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



## Particle Swarm Optimization ("Algorithms For Optimization" Algorithm 9.11-12, Kochenderfer & Wheeler) ##
# mutable struct Particle
#     x
#     v
#     x_best
# end
# function particle_swarm(f, constraints, x0, pop_size::Integer, max_n_evals; w=1, c1=1, c2=1, ρ=1, γ=2)
#     num_dims = length(x0)
#     population = [Particle(x0, zeros(num_dims), x0)]
#     for i=2:pop_size # create a population from initial point
#         particle_loc = x0.*randn(num_dims)
#         push!(population, Particle(particle_loc, zeros(num_dims), particle_loc))
#     end
#     x_best, y_best = x0, Inf
#     n_evals = 0
#     for P in population # Initialize "best" values by searching population
#         y = f(P.x) + ρ*Penalties(constraints, P.x)
#         n_evals += 1
#         if y < y_best
#             x_best[:], y_best = P.x, y
#         end
#     end
#     while true
#         for P in population
#             r1, r2 = rand(num_dims), rand(num_dims)
#             P.x += P.v
#             P.v = w*P.v + c1*r1.*(P.x_best - P.x) + c2*r2.*(x_best - P.x)
#             y = f(P.x) + ρ*Penalties(constraints, P.x)
#             if y < y_best
#                 x_best[:], y_best = P.x, y
#             end
#             if y < f(P.x_best) + ρ*Penalties(constraints, P.x_best)
#                 P.x_best[:] = P.x
#             end

#             n_evals += 2
#             if n_evals >= max_n_evals - 2
#                 return x_best
#             end
#         end
#         # ρ *= γ
#     end
#     return x_best
# end



## Full-Factorial Sampling Plan ("Algorithms For Optimization" Algorithm 13.1, Kochenderfer & Wheeler) ##
# function samples_full_factorial(lower_bounds, upper_bounds, sample_counts)
#     ranges = [range(lower_bounds[i], stop=upper_bounds[i], length=sample_counts[i]) for i = 1:length(lower_bounds)]
#     collect.(collect(Iterators.product(ranges...)))
# end



## Cross-Entropy Method ("Algorithms For Optimization" Algorithm 8.7, Kochenderfer & Wheeler) ##
using Distributions
function cross_entropy_method(f, constraints, P, max_n_evals, num_evals_used, m=100, m_elite=10, ρ=1.0)
    n_evals = num_evals_used
    while true
        # if n_evals == 0 # initialize with full factorial sampling
        #     num_dims = length(rand(P))
        #     sample_limit = 3
        #     num_init_samples = 200
        #     full_factorial_samples = samples_full_factorial(
        #         fill(-sample_limit, num_dims), 
        #         fill(sample_limit, num_dims), 
        #         fill(Int(round(num_init_samples^(1/num_dims))), num_dims))
        #     samples = []
        #     for i=1:length(full_factorial_samples)
        #         push!(samples, full_factorial_samples[i])
        #     end
        # else
            samples = [rand(P) for i=1:m]
        # end

        sample_y_vals = []
        for i in 1:m
            penalty = Penalties(constraints, samples[i], "hard")
            push!(sample_y_vals, f(samples[i]) + ρ*penalty)

            n_evals += 1
            if n_evals >= max_n_evals
                break
            end
        end
        infeasible_samples = sample_y_vals .== Inf
        if any(infeasible_samples)
            deleteat!(samples, findall(infeasible_samples))
            deleteat!(sample_y_vals, findall(infeasible_samples))
        end
        if length(samples) >= 4
            order = sortperm(sample_y_vals)
            flattened_samples = Array{Float64,2}(undef,length(samples[1]),length(samples))
            for i = 1:length(samples)
                flattened_samples[:,i] = samples[i]
            end
            if length(order) < m_elite
                P = fit(typeof(P), flattened_samples[:,order])
            else
                P = fit(typeof(P), flattened_samples[:,order[1:m_elite]])
            end
        end

        if n_evals >= max_n_evals
            return mean(P)
        end
    end
end

    

## Simulated Annealing ("Algorithms For Optimization" Algorithm 8.4, Kochenderfer & Wheeler) ##
# function simulated_annealing(f, constraints, x, step_size, update_interval, num_evals_used, max_n_evals)
#     num_dims = length(x)
#     pos_basis_vectors = [basis(i, num_dims) for i = 1:num_dims]
#     rng = RandomDevice()

#     y = f(x)
#     n_evals = num_evals_used + 1
#     x_best, y_best = x, y
#     k = 0

#     v = fill(step_size, num_dims)
#     a = zeros(num_dims)
#     while true
#         k += 1
#         # xp = x + rand(T)
#         for i = 1:num_dims
#             xp = x + sign(randn(rng))*rand(rng)*v[i]*pos_basis_vectors[i]
#             penalty = Penalties(constraints, xp, "hard")
#             yp = f(xp) + penalty
#             n_evals += 1
#             dy = yp - y
#             if dy <= 0 || rand() < exp(-dy/t_fast(k))
#                 x, y = xp, yp
#                 a[i] += 1
#             end
#             if yp < y_best
#                 x_best, y_best = xp, yp
#             end
#             if n_evals >= max_n_evals - 1
#                 return x_best
#             end
#         end
#         if k >= update_interval
#             c = fill(2.0, num_dims)
#             v = corana_update!(v, a, c, update_interval)
#             a = zeros(num_dims)
#         end
#     end
# end

## Fast Annealing Schedule for Simulated Annealing ("Algorithms For Optimization" Equation 8.7, Kochenderfer & Wheeler) ##
# function t_fast(k)
#     t = 10.0
#     return t/k
# end

## Corana Update for Simulated Annealing ("Algorithms For Optimization" Algorithm 8.5, Kochenderfer & Wheeler) ##
# function corana_update!(v, a, c, ns)
#     for i in 1:length(v)
#         ai, ci = a[i], c[i]
#         if ai > 0.6*ns
#             v[i] *= (1 + ci * (ai/ns - 0.6)/0.4)
#         elseif ai < 0.4*ns
#             v[i] /= (1 + ci * (0.4 - ai/ns)/0.4)
#         end
#     end
#     return v
# end



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
    ## Find Initial Feasible Point with GPS
    step_size = 1
    feasible_eval_max = 25
    if prob == "simple_2"
        feasible_eval_max = 2
    end
    start_point, feasible_points, n_evals_used = generalized_pattern_search(c, x0, step_size, feasible_eval_max)
    dist_init_scale = MathConstants.golden
    init_dist = MvNormal(start_point, dist_init_scale*sqrt(sum([start_point[i]^2 for i=1:length(start_point)])))
    # init_dist = MvNormal(start_point, 1)
    x_best = cross_entropy_method(f, c, init_dist, n, n_evals_used)


    ## Initialize with Full Factorial Sampling Plan
    # init_dist = MvNormal(x0, 1)
    # x_best = cross_entropy_method(f, c, init_dist, n, 0)


    # println(start_point)
    # x_best = start_point
    
    # pop_size = num_dims * 20
    # x_best = particle_swarm(f, c, x0, pop_size, n)


    ## Simulated Annealing with Corana Update
    # sim_anneal_step_size = 1.0
    # update_interval = 100/length(x0)
    # x_best = simulated_annealing(f, c, start_point, sim_anneal_step_size, update_interval, n_evals_used, n)

    return x_best
end


