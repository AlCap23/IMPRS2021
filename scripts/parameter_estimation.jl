using LinearAlgebra
using Plots

using OrdinaryDiffEq
using Flux
using DiffEqFlux

using Optim

# Simple example of fine tuning the parameters of a Lotka Volterra Model

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[1]*u[2]
    du[2] = γ*u[1]*u[2] - δ*u[2]
end

u0 = ones(Float32, 2)
p = Float32[1.5, 1.0, 3.0, 1.0]
tspan = (0.0f0, 10.0f0)
Δt = 0.1f0

prob = ODEProblem(lotka!, u0, tspan, p)
solution = solve(prob, Tsit5(), saveat = Δt)

# Extract solution and add noise
X̂ = Array(solution)
X̂ .+= 0.1*sum(X̂, dims = 2)/size(X̂, 2) .* randn(eltype(X̂) ,size(X̂))
t = solution.t

plot(solution)
scatter!(t, X̂', label = nothing)

savefig(joinpath(pwd(), "figures", "lotka_volterra_baseline.png"))

# Setup the estimation problem
p_init = Float32[1.0, 0.8, 2.0, 0.9]

function loss(p)
    s_ = solve(prob, Tsit5(), p = p, saveat = Δt)
    l = sum(abs2, s_ .- X̂) / size(X̂, 2)
    return l, s_
end

ps = typeof(solution)[]
ls = eltype(p_init)[]

callback = function (p, l, pred)
    #global ps, ls
    display(l)
    #plt = plot(pred, ylim = (0, 6))
    #plot!(solution, ylim = (0,6))
    #display(plt)
    push!(ps, pred)
    push!(ls, l)
    return false
end

res = DiffEqFlux.sciml_train(loss, p_init, cb = callback, maxiters = 100)

anm = @animate for i in 1:length(ps)
    _p = plot(ps[i], ylim = (0, 4), title = "Iteration $i : $(ls[i])", label = ["Estimate" nothing], color = :black)
    plot!(solution, ylim = (0,4), label = ["Groundtruth" nothing], color = :blue)
    _p
end

gif(anm, joinpath(pwd(), "figures", "lotka_volterra_estimate.gif"), fps = 4)

plot(ls, title = "Sum-of-Squares Error", xlabel = "Iterations", yaxis = :log10, label = nothing)
savefig(joinpath(pwd(), "figures", "lotka_volterra_loss.png"))
