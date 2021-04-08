n_arm, horizon, true_value = try
    parse.([Int, Int, Float64], ARGS)
catch
    println("Usage: julia main.jl [N_ARM] [HORIZON] [TRUE_VALUE]")
    exit(1)
end
@show n_arm horizon true_value

using JSON
using StatsBase
using Revise
includet("bernoulli_bandits.jl")

#=
n_arm, max_obs, true_value = 2, 20, 0.9
m = Bandits(;n_arm, max_obs)
V = ValueFunction(m)
V(Belief(m))
=#

# %% --------
m = Bandits(;n_arm, max_obs=horizon)

V = ValueFunction(m)
print("Running dynamic programming...")
@time V(Belief(m))
println("Number of (symmetry-reduced) states: ", length(V.cache))

# %% --------
println("Checking that DP value matches empirical performance...")
pol = OptimalPolicy(V)
N = 100_000
empirical_values = map(1:100_000) do i
    rollout(pol).reward
end

if abs(V(Belief(m)) - mean(empirical_values)) < 3sem(empirical_values)
    println("...success!")
else
    println("Uh oh...")
    println("DP says:        $(V(Belief(m)))")
    println("Empirical says: $(mean(empirical_values)) ± $(sem(empirical_values))")
end

# %% --------

true_state = true_value * ones(n_arm)
println("Simulating...")
N_sim = 10000
sims = map(1:N_sim) do i
    sim = simulate(pol, true_state)
    beliefs = [b.counts for b in sim.beliefs]
    (;beliefs, sim.choices)
end

write("optimal_simulations.json", JSON.json(sims))
println("Wrote simulations.json")
