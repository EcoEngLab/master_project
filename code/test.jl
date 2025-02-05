using Pkg
Pkg.activate("packages")
using MiCRM
using Distributions
   #set system size and leakage
   N,M,leakage = 100,50,0.3
  
   #make uptake matrix out of dirichlet distribution and make it modular
   u =  MiCRM.Parameters.modular_uptake(M,N; N_modules = 10, s_ratio = 3.0)
   #cost term
   m = 0.05 * ones(N)
   #resource inflow + outflow
   ρ = ones(N)
   ω = 0.03 * ones(N)
   #make leakage matrix for each consumer and restore
    leakage_matrices = []

    for _ in 1:100
        l = MiCRM.Parameters.modular_leakage(M; N_modules=5, s_ratio=2.0, λ=leakage)
        push!(leakage_matrices, l)  
    end

   #define parameters
   param = MiCRM.Parameters.generate_params(N, M;  u = u, m = m, ρ = ρ, ω = ω, l = leakage_matrices, λ = leakage)
   
   #original state
   x0 = ones(N+M) 

   #time span
   tspan = (0.0, 10.0)
   
   #define problem
   using DifferentialEquations
   prob = ODEProblem(MiCRM.Simulations.dx!, x0, tspan, param)
   sol = solve(prob, Tsit5())

   #test
   dx_test = zeros(length(x0))
   MiCRM.Simulations.dx!(dx_test, x0, param, 0.0)
   println("dx values: ", dx_test)

   #visualize
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

#get a functional system
#resource inflow and outflow
using Plots
# 计算 total_inflow 和 total_outflow
t_values = 0:0.1:10  
total_inflow = [sum(resource_inflow(t)) for t in t_values]
total_outflow = [sum(ω) + sum(l) for t in t_values]

# 绘制结果
plot(t_values, total_inflow, label="Total Resource Inflow", xlabel="Time", ylabel="Resource Flux", title="Resource Inflow and Outflow over Time")
plot!(t_values, total_outflow, label="Total Resource Outflow", linestyle=:dash)
# check maintenance
println("Total metabolic cost: ", sum(m))
println("Total uptake capacity: ", sum(u))

using JSON
data = Dict(
    "u" => u,
    "m" => m,
    "ρ" => ρ,
    "l" =>leakage_matrices,
    "ω" =>ω
)

# 保存 JSON 文件
open("data.json", "w") do f
    JSON.print(f, data)
end