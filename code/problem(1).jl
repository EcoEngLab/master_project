using Pkg
Pkg.activate("packages")
using MiCRM
using Distributions

   N,M,leakage = 100,50,0.3
  
   #can we make it uneven? 30 % prefer one module and 70% other module of resource
   #####yes, we can, but this is not nessary or critical for the model.
   u =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 2, s_ratio = 10.0)



   #why set to 1
   ##### to simplify the model. In "dx.jl" file, Line 8, m seems to be the mortality rate, if so, it should be < 1. 
   m = ones(N)



   #rate or amount?
   ##### ρ is amount, while ω is rate.
   ρ,ω = ones(M),ones(M)



   #how exactly the s_ratio will affect the leakage structure
   ##### When s_ratio = 1: Resources have a uniform leakage probability；
   ##### When s_ratio > 1: Increases leakage probability within the same module, Increases leakage probability between adjacent modules.
   l = MiCRM.Parameters.modular_leakage(M; N_modules = 5, s_ratio = 10.0, λ = leakage)




   #why do we need λ
   #####The function aims to pack all parameters into a tuple, thus no calculatipn here.
   param = MiCRM.Parameters.generate_params(N, M; u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)
   






   #why set all population and resource size to one
   ##### to simplify the initial condition.
   x0 = ones(N+M)
   #
   tspan = (0.0, 10.0)
   


   #
   using DifferentialEquations
   prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
   sol = solve(prob, Tsit5())



   #
   using Plots
   p = plot(sol, title="Consumer-Resource Dynamics", legend=:right, xlabel="Time", ylabel="Population/Resource")
   display(p)



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
   ##### this determines if a system is "reactive" to a random perturbation `pur`. 
   ##### It returns a boolean:
                   ##### `true`: Initial perturbation amplifies, causing system deviation from equilibrium
                   ##### `false`: Initial perturbation is damped, system remains stable
   MiCRM.Analysis.get_reactivity(J,pur)





   #and this
   MiCRM.Analysis.get_return_rate(J)