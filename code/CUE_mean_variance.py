import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import param_3D

# parameters
N = 20 # consumer number
M = 10  # resource number
λ = 0.3  # total leakage rate
λ_u = np.random.uniform(0.2, 0.8, N)  # 每个 consumer 的 uptake scaling factor
λ_l = np.random.uniform(0.2, 0.4, N)
σ = np.random.uniform(1, 5, N)
N_modules = 5  # module number of consumer to resource
s_ratio = 10.0  

# Generate uptake matrix
u = param_3D.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  # uptake matrix
lambda_alpha = np.full(M, λ)  # total leakage rate for each resource

m = np.full(N, 0.2)  # mortality rate of N consumers
rho = np.full(M, 1)  # input of M resources
omega = np.full(M, 0.05)  # decay rate of M resources

l = param_3D.generate_l_tensor(N, M, N_modules, s_ratio, λ)  # a tensor for all consumers' leakage matrices

# ODE system
def dCdt_Rdt(t, y):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]
    
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))
    
    return np.concatenate([dCdt, dRdt])
    
# Initial conditions
C0 = np.full(N, 1)
R0 = np.full(M, 1)
Y0 = np.concatenate([C0, R0])
    
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
    
CUE = np.zeros((N, len(sol.t)))                                                                
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  
    R = sol.y[N:, i]  
    total_uptake = u @ (R0-R)  
    net_uptake = total_uptake * (1 - λ) - m  
    CUE[:, i] = net_uptake / total_uptake  
    
final_CUE = CUE[:, -1]
u_variance = np.var(u, axis=1, ddof=0)
u_mean = np.mean(u, axis=1)
# Scatter plot: CUE vs. u_mean
plt.figure()
plt.scatter(u_mean, final_CUE, label='Data', color='blue', alpha=0.6)
plt.xlabel("Uptake Mean")
plt.ylabel("Final CUE")
plt.title("CUE vs. Uptake Mean")
plt.legend()
plt.show()

# Scatter plot: CUE vs. u_variance
plt.figure()
plt.scatter(u_variance, final_CUE, label='Data', color='green', alpha=0.6)
plt.xlabel("Uptake Variance")
plt.ylabel("Final CUE")
plt.title("CUE vs. Uptake Variance")
plt.legend()
plt.show()

# Create 3D scatter plot
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(
    x=u_mean,
    y=u_variance,
    z=final_CUE,
    mode='markers',
    marker=dict(
        size=5,
        color=final_CUE,  # 颜色映射 CUE 值
        colorscale='viridis',
        opacity=0.8
    )
)])

# 设置坐标轴标签和标题
fig.update_layout(
    scene=dict(
        xaxis_title="Uptake Mean",
        yaxis_title="Uptake Variance",
        zaxis_title="CUE"
    ),
    title="3D Interactive Scatter Plot of CUE vs. Uptake Mean & Variance"
)

# **保存为 HTML 文件**
fig.write_html("../results/interactive_3D_plot.html")

# 显示图像
fig.show()
