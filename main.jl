n_arm, max_obs = try
    parse.(Int, ARGS)
catch
    println("Usage: julia main.jl [N_ARM] [HORIZON]")
    exit(1)
end

using JSON
using StatsBase
using Revise
includet("bernoulli_bandits.jl")

# # %% --------
# n_arm, max_obs = 4, 30
# m = Bandits(;n_arm, max_obs)
# @time ValueFunction(m)(Belief(m))
# @time ValueFunction(m, hash)(Belief(m))


# %% --------
m = Bandits(;n_arm, max_obs)
println("$n_arm arms, $max_obs steps")

V = ValueFunction(m)
print("Running dynamic programming")
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

true_state = 0.9 * ones(4)
simulate(pol, true_state).beliefs[end]
N_sim = 10000
sims = map(1:N_sim) do i
    sim = simulate(pol, true_state)
    beliefs = [b.counts for b in sim.beliefs]
    (;beliefs, sim.choices)
end

write("optimal_simulations.json", JSON.json(sims))
println("Wrote simulations.json")