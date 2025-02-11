using Pkg
Pkg.activate("packages")
using MiCRM
using Distributions
   #set system size and leakage
   N,M,leakage = 16,4,0.3
  
   #make uptake matrix out of dirichlet distribution and make it modular
   du = Distributions.Dirichlet(N,1.0)
   u = copy(rand(du, M)')
   u_m =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 2, s_ratio = 10.0)
   u = u.*u_m

   #cost term
   m = ones(N)

   #resource inflow + outflow
   ρ,ω = ones(M),ones(M)

   #make leakage matrix out of dirichlet distribution and make it modular
   l = copy(rand(du,M)' .* leakage)
   l_m = MiCRM.Parameters.modular_leakage(M; N_modules = 2, s_ratio = 10.0, λ = 0.5)
   l = l_m*l

   #define parameters
   param = MiCRM.Parameters.generate_params(N, M;  u = u, m = m, ρ = ρ, ω = ω, l = l, λ = leakage)
   
   #original state
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
   # generate purbulence matrix u to 
   u = rand(size(J, 1))
   # the instantaneous rate of growth of the perturbation u at time t.
   MiCRM.Analysis.get_Rins(J, u, 1)
   # Determine the stability a system given its jaccobian by testing if the real part of the leading eigenvalue is positive.
   MiCRM.Analysis.get_stability(J)
   #test whether a system is reactive to the perturbation 
   #we need to analyze how the system responds to small disturbances. A reactive system is one where an initial perturbation amplifies in the short term before potentially stabilizing or decaying.
   MiCRM.Analysis.get_reactivity(J,u)
   #get the rate of return of the system from perturbation
   MiCRM.Analysis.get_return_rate(J)