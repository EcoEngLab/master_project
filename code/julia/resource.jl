using LinearAlgebra, Random, DifferentialEquations, PlotlyJS, LsqFit

# ------------------------- 参数定义 -------------------------

N = 5 # Number of consumers
M = 5 # Number of resources
λ = 0.3  # Total leakage rate
kaf0 = 0.04

# 生成 epsilon: 物种特异性的能量摄取修正
epsilon = rand(Normal(0, 0.1), N)

# 每个消费者的摄取缩放因子
λ_u = rand(Uniform(0.2, 0.6), N) 

# 摄取分布标准差
σ = rand(Uniform(0.1, 1), N)

N_modules = 5  # Number of modules connecting consumers to resources
s_ratio = 10.0  # Strength of modularity

# 生成摄取矩阵 u
function modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)
    u = rand(N, M) .* λ_u .+ σ  # 生成随机摄取矩阵
    return clamp.(u, 0.001, 0.6)  # 限制 u 的范围
end

u = modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)

# 资源泄漏率
lambda_alpha = fill(λ, M)  

# 资源输入率 & 衰减率
rho = fill(1.0, M)  
omega = fill(0.05, M)  

# 生成泄漏张量 l (N × M × M)
function generate_l_tensor(N, M, N_modules, s_ratio, λ)
    return rand(N, M, M) .* λ
end

l = generate_l_tensor(N, M, N_modules, s_ratio, λ)

# 计算维护成本 m
function compute_m(N, kaf0, epsilon, λ, u)
    u_sum = sum(u, dims=2)  # 计算每个物种的总摄取量
    return kaf0 .* (1 .+ epsilon) .* (1 .- λ) .* u_sum
end

m = compute_m(N, kaf0, epsilon, λ, u)

# ------------------------- ODE 模型 -------------------------
function dCdt_Rdt!(du, u, p, t)
    C = u[1:N]  # 消费者种群
    R = u[N+1:end]  # 资源浓度
    dCdt = zeros(N)
    dRdt = zeros(M)
    
    # 消费者增长方程
    for i in 1:N
        dCdt[i] = sum(C[i] * R[α] * u[i, α] * (1 - lambda_alpha[α]) for α in 1:M) - C[i] * m[i]
    end
    
    # 资源衰减 & 泄漏方程
    for α in 1:M
        dRdt[α] = rho[α] - R[α] * omega[α]  # 资源输入 & 衰减
        dRdt[α] -= sum(C[i] * R[α] * u[i, α] for i in 1:N)  # 资源摄取
        dRdt[α] += sum(sum(C[i] * R[β] * u[i, β] * l[i, β, α] for β in 1:M) for i in 1:N)  # 资源泄漏
    end

    du .= vcat(dCdt, dRdt)  # 连接消费者 & 资源动态变化
end

# 初始条件
C0 = fill(1.0, N)
R0 = fill(1.0, M)
Y0 = vcat(C0, R0)

# 设定求解时间范围
tspan = (0.0, 500.0)
t_eval = range(0, stop=500, length=300)

# 解决 ODE
prob = ODEProblem(dCdt_Rdt!, Y0, tspan)
sol = solve(prob, Tsit5(), saveat=t_eval)

# ------------------------- 计算 CUE -------------------------

C_eq = sol.u[end][1:N]  # 平衡状态下的消费者种群
R_eq = sol.u[end][N+1:end]  # 平衡状态下的资源浓度

# 计算总摄取量
total_uptake_eq = u * (R0 - R_eq)  

# 计算平衡态的 m
l_sum_eq = sum(l, dims=2)  
u_sum_eq = sum(u, dims=2)
m_eq = kaf0 .* (1 .+ epsilon) .* (1 .- λ) .* u_sum_eq  

# 计算净摄取量
net_uptake_eq = total_uptake_eq .* (1 .- λ) - m_eq

# 计算 CUE
CUE_eq = net_uptake_eq ./ total_uptake_eq
CUE_eq[isnan.(CUE_eq)] .= 0  # 避免除零错误

println("CUE at equilibrium: ", CUE_eq)

# ------------------------- 3D 可视化 -------------------------

u_variance = var(u, dims=2)  # 计算摄取方差
u_mean = mean(u, dims=2)  # 计算摄取均值

# 创建 3D 散点图
plt = PlotlyJS.plot(PlotlyJS.scatter3d(
    x=u_mean[:], y=u_variance[:], z=CUE_eq[:], mode="markers",
    marker=attr(size=5, color=CUE_eq[:], colorscale="Viridis", opacity=0.8)
))

PlotlyJS.savefig(plt, "results/interactive_3D_plot.html")
display(plt)

# ------------------------- 对数拟合 -------------------------

using LsqFit

function log_func(x, a, b)
    return a .+ b .* log.(x)
end

x_data = u_mean[:]
y_data = CUE_eq[:]
p0 = [0.0, 1.0]  # 初始参数
fit = curve_fit(log_func, x_data, y_data, p0)

a, b = fit.param

# 生成拟合曲线
x_fit = range(minimum(x_data), maximum(x_data), length=100)
y_fit = log_func(x_fit, a, b)

# 绘制拟合曲线
using Plots

plot(x_data, y_data, seriestype=:scatter, label="Data", color=:blue)
plot!(x_fit, y_fit, label="Fitted Curve", color=:red)
title!("Logarithmic Fit: Uptake Variance vs. Final CUE")
xlabel!("Uptake variance")
ylabel!("Final CUE")
savefig("results/logarithmic_fit.png")
