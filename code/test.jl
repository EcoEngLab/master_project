using Pkg
Pkg.activate("packages")
using MiCRM
using Distributions
   #set system size and leakage
   N,M,leakage = 50,50,0.3
  
   #make uptake matrix out of dirichlet distribution and make it modular
   u =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 2, s_ratio = 10.0)
   #cost term
   m = 0.06 * ones(N)
   #resource inflow + outflow
   ρ = 10 .* ones(M) 
   ω = 0.05 .* ones(M)
   

   #make leakage matrix out of dirichlet distribution and make it modular
   l = MiCRM.Parameters.modular_leakage(M; N_modules = 5, s_ratio = 5.0, λ = leakage)

   #define parameters
   param = MiCRM.Parameters.generate_params(N, M;  u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)
   
   #original state
   x0 = vcat(0.1 .+ 0.2 .* rand(N),
   1.0 .+ 0.5 .* rand(M)) 

   #time span
   tspan = (0.0, 10.0)
   
   #define problem
   using DifferentialEquations
   prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
   sol = solve(prob, Tsit5())

   #visualize
   dx_test = zeros(length(x0))
   MiCRM.Simulations.dx!(dx_test, x0, param, 0.0)
   println("dx values: ", dx_test)
   
   using Plots

   sol_u = reduce(hcat, sol.u) 


   plot(sol.t, sol_u[N+1:end, :]', color=:blue, label=false)
   plot!(sol.t, sol_u[1:N, :]', color=:red, label=false) 
   

   plot!([NaN], color=:blue, label="Resource")
   plot!([NaN], color=:red, label="Species")
   
   
   # jacobian of system from the solution object
   J = MiCRM.Analysis.get_jac(sol)
   #test whether it is stiff
   using LinearAlgebra
   eigenvalues = eigvals(J) 
   λ_max = maximum(abs.(eigenvalues)) 
   λ_min = minimum(abs.(eigenvalues[eigenvalues .!= 0])) 
   stiffness_ratio = λ_max / λ_min

   # calculate key properties
   # generate purbulence matrix
   pur = rand(size(J, 1))
   # the instantaneous rate of growth of the perturbation pur at time t.
   t = 5
   MiCRM.Analysis.get_Rins(J, pur, t)
   # Determine the stability a system given its jaccobian by testing if the real part of the leading eigenvalue is positive.
   MiCRM.Analysis.get_stability(J)
   #test whether a system is reactive to the perturbation pur
   #A reactive system is one where an initial perturbation amplifies in the short term before potentially stabilizing or decaying.
   MiCRM.Analysis.get_reactivity(J,pur)
   #get the rate of return of the system from perturbation
   MiCRM.Analysis.get_return_rate(J)