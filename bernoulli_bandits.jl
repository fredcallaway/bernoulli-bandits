"""
This module defines belief updating for bandis with Bernoulli-distributed rewards
and an optimal policy computed by dynamic programming.
"""

using Distributions
import Random
using Parameters
import Base
using Printf
# using SplitApplyCombine
# using StaticArrays
using Memoize


@isdefined(⊥) || const ⊥ = 0

@with_kw struct Bandits
    n_arm::Int = 3
    # sample_cost::Float64 = 0.001
    # switch_cost::Float64 = 1
    max_obs::Int = 20
end

struct Belief
    counts::Vector{Tuple{Int, Int}}  # number of heads and tails for each item
    focused::Int
end

Belief(m::Bandits) = begin
    counts = [(1,1) for _ in 1:m.n_arm]
    Belief(counts, 0)
end

Base.:(==)(b1::Belief, b2::Belief) = b1.focused == b2.focused && b1.counts == b2.counts
Base.hash(b::Belief) = hash(b.focused, hash(b.counts))
Base.getindex(b::Belief, idx) = b.counts[idx]
Base.length(b::Belief) = length(b.counts)
Base.iterate(b::Belief) = iterate(b.counts)
Base.iterate(b::Belief, i) = iterate(b.counts, i)
n_obs(b::Belief) = sum(map(sum, b)) - 2length(b)
# actions(m::Bandits, b::Belief) = n_obs(b) >= m.max_obs ? (0:0) : (0:length(b))
actions(m::Bandits, b::Belief) = n_obs(b) >= m.max_obs ? (0:0) : (1:length(b))

function Base.show(io::IO, b::Belief)
    print(io, "[ ")
    counts = map(1:length(b.counts)) do i
        h, t = b.counts[i]
        i == b.focused ? @sprintf("<%02d %02d>", h, t) : @sprintf(" %02d %02d ", h, t)
    end
    print(io, join(counts, " "))
    print(io, " ]")
end

function update(b::Belief, arm::Int, heads::Bool)::Belief
    counts = copy(b.counts)
    h, t = counts[arm]
    counts[arm] = heads ? (h+1, t) : (h, t+1)
    Belief(counts, arm)
end

# function cost(m::Bandits, b::Belief, c::Int)::Float64
#     m.sample_cost * (c != b.focused ? m.switch_cost : 1)
# end

p_heads(counts) = counts[1] / (counts[1] + counts[2])

term_reward(b::Belief)::Float64 = 0
# term_reward(b::Belief)::Float64 = maximum(p_heads.(b.counts))
is_terminal(b::Belief) = b.focused == -1
terminate(b::Belief) = Belief(b.counts, -1)

Result = Tuple{Float64, Belief, Float64}
function results(m::Bandits, b::Belief, c::Int)::Vector{Result}
    is_terminal(b) && error("Belief is terminal.")
    if c == ⊥
        return [(1., terminate(b), term_reward(b))]
    end
    p1 = p_heads(b.counts[c])
    p0 = 1 - p1
    # r = -cost(m, b, c)
    [(p0, update(b, c, false), 0),
     (p1, update(b, c, true), 1)]
end


# # %% ==================== Value Function ====================

function symmetry_breaking_hash(b::Belief)
    return hash(sort(b.counts))
    # key = UInt64(0)
    # for i in 1:length(b.counts)
    #     # key += (hash(b.counts[i]) << 3 * (i == b.focused))
    #     key += hash(b.counts[i])
    # end
    # key
end

struct ValueFunction{F}
    m::Bandits
    hasher::F
    cache::Dict{UInt64, Float64}
end
ValueFunction(m::Bandits, hasher=symmetry_breaking_hash, cache=Dict{UInt64, Float64}()) = ValueFunction(m, hasher, cache)

function Q(V::ValueFunction, b::Belief, c::Int)::Float64
    c == ⊥ && return term_reward(b)
    sum(p * (r + V(s1)) for (p, s1, r) in results(V.m, b, c))
end

function (V::ValueFunction)(b::Belief)::Float64
    key = V.hasher(b)
    # key = hash(b)
    haskey(V.cache, key) && return V.cache[key]
    return V.cache[key] = maximum(Q(V, b, c) for c in actions(V.m, b))
end


# %% ==================== Policy ====================

function argmaxes(f, x::AbstractArray{T})::Set{T} where T
    r = Set{T}()
    fx = f.(x)
    mfx = maximum(fx)
    for i in eachindex(x)
        if fx[i] == mfx
            push!(r, x[i])
        end
    end
    r
end

abstract type Policy end
function act(pol::Policy, b::Belief)
    rand(actions(pol, b))
end

struct OptimalPolicy <: Policy
    m::Bandits
    V::ValueFunction
end
OptimalPolicy(m::Bandits) = OptimalPolicy(m, ValueFunction(m))
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)
(pol::OptimalPolicy)(b::Belief) = act(pol, b)

function actions(pol::OptimalPolicy, b::Belief)
    argmaxes(c->Q(pol.V, b, c), actions(pol.m, b))
end


# %% ==================== Simulation ====================

function update!(b::Belief, c::Int, heads::Bool)
    h, t = b.counts[c]
    b.counts[c] = heads ? (h+1, t) : (h, t+1)
    b
end

State = Vector{Float64}
function sample_outcome(b::Belief, s::State, c::Int)
    rand() < s[c]
end

function sample_outcome(b::Belief, s::Nothing, c::Int)
    rand() < p_heads(b.counts[c])
end

function rollout(policy; s=nothing, b=nothing, max_steps=1000, callback=(b, c)->nothing)
    m = policy.m
    if b == nothing
        b = Belief(m)
    end
    reward = 0
    # print('x')
    max_steps = min(max_steps, m.max_obs + 1)
    for step in 1:m.max_obs+1
        c = (step == max_steps) ? ⊥ : policy(b)
        callback(b, c)
        if c == ⊥
            reward += term_reward(b)
            return (reward=reward, steps=step, belief=b)
        else
            heads = sample_outcome(b, s, c)
            reward += heads
            b = update(b, c, heads)
            # update!(b, c, heads)
        end
    end
end

function simulate(policy, s=nothing)
    beliefs = Belief[]
    choices = Int[]
    roll = rollout(policy; s) do b, c
        push!(beliefs, b)
        push!(choices, c)
    end
    # pop!(sim.choices)  # don't count termination
    (;beliefs, choices, roll.reward)
end


rollout(callback::Function, policy; kws...) = rollout(policy; kws..., callback=callback)

