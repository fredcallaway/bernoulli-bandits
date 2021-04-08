# Bernoulli Bandits

Julia code to solve bernoulli bandit problems by dynamic programming.

Running `julia main.jl 4 40 0.9` will solve a 4-arm 40-step bandit problem and then generate simulations from the optimal policy assuming all the arms have value 0.9. The result of that is in optimal_simulations.json

That file is a JSON list where each entry is an object describing one simulation: a list of 41 belief states (including the initial and final) and a list of the 40 arms chosen. The format of a belief state is a list of alpha and beta pairs for each arm.
