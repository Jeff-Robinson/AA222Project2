## Jeff Robinson - jbrobin@stanford.edu ##

using Random
using Distributions
using LinearAlgebra

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
    elseif which == "quad+count"
        penalties = P_quad(constraint_values) + P_count(constraint_values)
    elseif which == "hard"
        penalties = P_hard_barrier(constraint_values)
    end
    return penalties
end



## Covariance Matrix Adaptation ("Algorithms For Optimization" Algorithm 8.9, Kochenderfer & Wheeler) ##
function covariance_matrix_adaptation(f, c, x, max_n_evals, n_evals_used;
    σ = 0.8,
    m = 19, #10 + floor(Int, 3*log(length(x))),
    m_elite = 5, #div(m, 2),
    p_type = "quad+count",
    ρ = 1.0,
    γ = 1.0
    )
    k_max = round(Integer, (max_n_evals - n_evals_used)/(m+1), RoundDown)
    x_best = x
    y_best = Inf

    μ, n_dims = copy(x), length(x)
    ws = normalize!(vcat(log((m+1)/2) .- log.(1:m_elite), 
                    zeros(m - m_elite)), 
                    1)
    μ_eff = 1 / sum(ws.^2)
    cσ = (μ_eff + 2)/(n_dims + μ_eff + 5)
    dσ = 1 + 2*max(0, sqrt((μ_eff-1)/(n_dims+1))-1) + cσ
    cΣ = (4 + μ_eff/n_dims)/(n_dims + 4 + 2*μ_eff/n_dims)
    c1 = 2/((n_dims+1.3)^2 + μ_eff)
    cμ = min(1-c1, 2*(μ_eff-2+1/μ_eff)/((n_dims+2)^2 + μ_eff))
    E = n_dims^0.5*(1-1/(4*n_dims)+1/(21*n_dims^2))
    pσ, pΣ, Σ = zeros(n_dims), zeros(n_dims), Matrix(1.0I, n_dims, n_dims)

    for k in 1:k_max
        if mod(k, 100) == 0
            ρ *= γ
        end
        P = MvNormal(μ, σ^2*Σ)
        xs = [rand(P) for i in 1:m]
        ps = [Penalties(c, x, p_type) for x in xs]
        ys = [f(xs[i])+ρ*ps[i] for i in 1:m]
        is = sortperm(ys) # best to worst

        infeas_xs = ps .> 0.0
        infeas_xs_locs = findall(infeas_xs)
        feas_xs = copy(xs)
        feas_ys = copy(ys)
        if any(infeas_xs)
            deleteat!(feas_xs, infeas_xs_locs)
            deleteat!(feas_ys, infeas_xs_locs)
        end
        feas_is = sortperm(feas_ys)

        # selection and mean update
        δs = [(x - μ)/σ for x in xs]
        δw = sum(ws[i]*δs[is[i]] for i in 1:m_elite)
        μ += σ*δw

        # step size control
        C = Σ^(-0.5)
        pσ = (1-cσ)*pσ + sqrt(cσ*(2-cσ)*μ_eff)*C*δw
        σ *= exp(cσ/dσ * (norm(pσ)/E - 1))

        # covariance Adaptation
        hσ = Int(norm(pσ)/sqrt(1-(1-cσ)^(2*k)) < (1.4+2/(n_dims+1))*E)
        pΣ = (1-cΣ)*pΣ + hσ*sqrt(cΣ*(2-cΣ)*μ_eff)*δw
        w0 = [ws[i]>=0 ? ws[i] : n_dims*ws[i]/norm(C*δs[is[i]])^2 for i in 1:m]
        Σ = (1-c1-cμ) * Σ + c1*(pΣ*pΣ' + (1-hσ) * cΣ*(2-cΣ) * Σ) + cμ*sum(w0[i]*δs[is[i]]*δs[is[i]]' for i in 1:m)
        Σ = triu(Σ) + triu(Σ,1)' # enforce symmetry

        if length(feas_xs) != 0
            x_best_potential = feas_xs[feas_is[1]]
            y_best_potential = feas_ys[feas_is[1]]
            if y_best_potential < y_best
                x_best = x_best_potential
                y_best = y_best_potential
            end
        end
        x_best_potential = μ
        y_best_potential = f(μ) + Penalties(c, μ, "hard")
        if y_best_potential != Inf && y_best_potential < y_best
            x_best = x_best_potential
            y_best = y_best_potential
        end
    end
    return x_best
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
    start_point = x0
    n_used = 0
    if prob == "simple_1"
        x_best = covariance_matrix_adaptation(f, c, start_point, n, n_used;
            σ = 0.8,
            m = 19, #10 + floor(Int, 3*log(length(x))),
            m_elite = 5, #div(m, 2),
            p_type = "quad+count",
            ρ = 1.0,
            γ = 1.0
            )
    elseif prob == "simple_2"
        x_best = covariance_matrix_adaptation(f, c, start_point, n, n_used;
            σ = 0.9,
            m = 24, #10 + floor(Int, 3*log(length(x))),
            m_elite = 5, #div(m, 2),
            p_type = "quad",
            ρ = 1.0,
            γ = 1.0
            )
    elseif prob == "simple_3"
        x_best = covariance_matrix_adaptation(f, c, start_point, n, n_used;
            σ = 1.0,
            m = 19, #10 + floor(Int, 3*log(length(x))),
            m_elite = 5, #div(m, 2),
            p_type = "quad+count",
            ρ = 1.0,
            γ = 1.0
            )
    elseif prob == "secret_1" || prob == "secret_2"
        x_best = covariance_matrix_adaptation(f, c, start_point, n, n_used;
            σ = 1.0,
            m = 4 + floor(Int, 3*log(length(start_point))),
            m_elite = div((4 + floor(Int, 3*log(length(start_point)))), 2),
            p_type = "quad",
            ρ = 1.0e5,
            γ = 2.0
            )
    end
    return x_best
end