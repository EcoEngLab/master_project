using Pkg
Pkg.activate("/home/ruizao/Documents/MiCRM")
using MiCRM
using Distributions
   #set system size and leakage
   N,M,leakage = 16,4,0.3

   #uptake
   du = Distributions.Dirichlet(N,1.0)
   u = copy(rand(du, M)')
   u_m =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 2, s_ratio = 10.0)
   u = u.*u_m

   #cost term
   m = ones(N)

   #inflow + outflow
   ρ,ω = ones(M),ones(M)

   #leakage
   l = copy(rand(du,M)' .* leakage)
   l_m = MiCRM.Parameters.modular_leakage(M; N_modules = 2, s_ratio = 10.0, λ = 0.5)
   l = l_m*l
   param = MiCRM.Parameters.generate_params(N, M;  u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)
   x0 = ones(N+M)
   #time span
   tspan = (0.0, 10.0)
   
   #define problem
   using DifferentialEquations
   prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
   sol = solve(prob, Tsit5())
   #visualize
   using Plots
   plot(sol, title="Consumer-Resource Dynamics", legend=:right, xlabel="Time", ylabel="Population/Resource")
   # jacobian of system from the solution object
   J = MiCRM.Analysis.get_jac(sol)
   # calculate key properties
   # generate purbulence matrix
   u = rand(size(J, 1))
   #  the instantaneous rate of growth of the perturbation u at time t.
   MiCRM.Analysis.get_Rins(J, u, 1)
   MiCRM.Analysis.get_stability(J)
   MiCRM.Analysis.get_reactivity(J,u)
   MiCRM.Analysis.get_return_rate(J)