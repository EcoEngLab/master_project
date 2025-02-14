import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import param_uσ


# parameters
N = 20 # consumer number
M = 20  # resource number
λ = 0.3  # total leakage rate
λ_u = np.full(N, 0.5)
σ = np.random.uniform(2, 5, N)

N_modules = 5  # module number of consumer to resource
s_ratio = 10.0  

# Generate uptake matrix
u = param_uσ.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  # uptake matrix
u_mean = np.mean(u, axis=1)
print(u_mean)
u_upper = np.triu(u)
print("原始 u 方差:", np.var(u, axis=1))
print("上三角 u 方差:", np.var(np.triu(u), axis=1))

lambda_alpha = np.full(M, λ)  # total leakage rate for each resource
m = np.full(N, 0.2)  # mortality rate of N consumers
rho = np.full(M, 1)  # input of M resources
omega = np.full(M, 0.05)  # decay rate of M resources

l = param_uσ.generate_l_tensor(N, M, N_modules, s_ratio, λ)  # a tensor for all consumers' leakage matrices


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

# Initial values
C0 = np.full(N, 1)  # consumer
R0 = np.full(M, 1)  # resource
Y0 = np.concatenate([C0, R0])

# Time scale
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)

# Solve ODE
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  
    R = sol.y[N:, i]  
    total_uptake = u @ (R0-R)  
    net_uptake = total_uptake * (1 - λ) - m  
    CUE[:, i] = net_uptake / total_uptake  

# 计算最终 CUE 值
final_CUE = CUE[:, -1]

# calculate uptake variance
u_variance = np.var(u, axis=1, ddof=0)

# 画 u_variance vs. CUE 关系
def plot_regression(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label='Data points', color='b')
    
    # 计算线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='r', linestyle='dashed', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

# 画 u_variance vs. CUE 关系
plot_regression(u_variance, final_CUE, "Uptake variance", "Final CUE", "Uptake variance vs. CUE")