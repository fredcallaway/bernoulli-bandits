# Bernoulli Bandits

Julia code to solve bernoulli bandit problems by dynamic programming.

Running `julia main.jl 4 40 0.9` will solve a 4-arm 40-step bandit problem and then generate simulatoins from the optimal policy assuming all the arms have value 0.9. The result of that is in optimal_simulations.json