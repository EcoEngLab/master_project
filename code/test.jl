using Pkg
Pkg.activate("/home/ruizao/Documents/MiCRM")
    using Distributions

    #set system size and leakage
    N,M,leakage = 10,10,0.3

    #uptake
    du = Distributions.Dirichlet(N,1.0)
    u = copy(rand(du, M)')

    #cost term
    m = ones(N)

    #inflow + outflow
    ρ,ω = ones(M),ones(M)

    #leakage
    l = copy(rand(du,M)' .* leakage)

    param = (N = N, M = M, u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)

    using MiCRM

    #set system size and leakage
    N,M,leakage = 10,10,0.3

    #generate community parameters
    param = MiCRM.Parameters.generate_params(N, M; λ=0.3)

    #inital state
    x0 = ones(N+M)
    #time span
    tspan = (0.0, 10.0)
    
    #define problem
    using DifferentialEquations
    prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
    sol = solve(prob, Tsit5())

