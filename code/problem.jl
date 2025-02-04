using Pkg
Pkg.activate("packages")
using MiCRM
using Distributions

   N,M,leakage = 100,50,0.3
  
   #can we make it uneven? 30 % prefer one module and 70% other module of resource
   u =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 2, s_ratio = 10.0)
   #why set to 1
   m = ones(N)
   #rate or amount? 
   ρ,ω = ones(M),ones(M)

   #how exactly the s_ratio will affect the leakage structure
   l = MiCRM.Parameters.modular_leakage(M; N_modules = 5, s_ratio = 10.0, λ = leakage)

   #why do we need λ
   param = MiCRM.Parameters.generate_params(N, M;  u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)
   
   #why set all population and resource size to one
   x0 = ones(N+M)
   #
   tspan = (0.0, 10.0)
   
   #
   using DifferentialEquations
   prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
   sol = solve(prob, Tsit5())

   #
   using Plots
   plot(sol, title="Consumer-Resource Dynamics", legend=:right, xlabel="Time", ylabel="Population/Resource")
   
   #
   J = MiCRM.Analysis.get_jac(sol)
   pur = rand(size(J, 1))
   # it is different from the usage on website
   t = 5
   MiCRM.Analysis.get_Rins(J, pur, t)
   # 
   MiCRM.Analysis.get_stability(J)
   #
   #I am not so clear about this
   MiCRM.Analysis.get_reactivity(J,pur)
   #and this
   MiCRM.Analysis.get_return_rate(J)