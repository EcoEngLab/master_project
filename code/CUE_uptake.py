import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import sys
import os

# Manually add the directory where param_3D.py is located
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param_3D

np.random.seed(30) 
# Parameters
N = 20 # Number of consumers
M = 12 # Number of resources
λ = 0.3  # Total leakage rate

# Each consumer's uptake scaling factor
λ_u = np.random.uniform(2, 2.5, N)  
σ = np.random.uniform(0.05 * λ_u, 0.2 * λ_u)
N_modules = 5  # Number of modules connecting consumers to resources
s_ratio = 4.0  # Strength of modularity

# Generate uptake matrix, defining consumer-resource interaction strengths
u = param_3D.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  
print(u)
# Total leakage rate for each resource
lambda_alpha = np.full(M, λ)  

# Mortality rate for each consumer
m = np.full(N, 0.2)  
# Input rate of each resource
rho = np.full(M, 1)  
# Decay rate for each resource
omega = np.full(M, 0.01)  

# Generate leakage tensor representing leakage flow between consumers and resources
l = param_3D.generate_l_tensor(N, M, N_modules, s_ratio, λ)  

# ODE system describing the dynamics of consumers (C) and resources (R)
def dCdt_Rdt(t, y):
    C = y[:N]  # Consumer populations
    R = y[N:]  # Resource concentrations
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    # Consumer growth equation
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]
    
    # Resource depletion and leakage equation
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]  # Input and decay
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))  # Uptake by consumers
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))  # Leakage contributions
    
    return np.concatenate([dCdt, dRdt])
    
# Initial conditions: assume all consumers and resources start at concentration 1
C0 = np.full(N, 1)
R0 = np.full(M, 1)
Y0 = np.concatenate([C0, R0])
    
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
    
# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  
    R = sol.y[N:, i]  
    total_uptake = u @ (R0-R)  
    net_uptake = total_uptake * (1 - λ) - m  
    CUE[:, i] = net_uptake / total_uptake  

# Final CUE value 
final_CUE = CUE[:, -1]
u_variance = np.var(u, axis=1, ddof=0)  # Variance of uptake per consumer
u_mean = np.mean(u, axis=1)  # Mean uptake per consumer
print(np.corrcoef(u_mean, u_variance))

# community CUE
initial_C = sol.y[:N, 0] 
final_C = sol.y[:N, -1]  
total_C_change = np.sum(final_C - initial_C)

initial_R = sol.y[N:, 0]  # 初始资源丰度
final_R = sol.y[N:, -1]  # 平衡态资源丰度
total_R_change = np.sum(rho*t - final_R)
total_R_used = total_R_change + np.sum(rho)


CUE_community = total_C_change / total_R_used if total_R_used > 0 else 0
print("Community CUE:", CUE_community)


# plot dynamics of Consumers and Resources
plt.figure(figsize=(12, 6))

for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Consumer {i+1}')
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], label=f'Resource {alpha+1}', linestyle='dashed')

plt.xlabel('Time')
plt.ylabel('Consumer / Resource')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, fontsize=10, columnspacing=1.5)
plt.title('Dynamics of Consumers and Resources')
plt.tight_layout()
plt.savefig("results/dynamics_of_consumers_resources.png", dpi=300, bbox_inches='tight')
plt.show()

# Create an interactive 3D scatter plot
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(
    x=u_mean,
    y=u_variance,
    z=final_CUE,
    mode='markers',
    marker=dict(
        size=5,
        color=final_CUE,  # Color mapped to CUE values
        colorscale='viridis',
        opacity=0.8
    )
)])

# Configure axis labels and title
fig.update_layout(
    scene=dict(
        xaxis_title="Uptake Mean",
        yaxis_title="Uptake Variance",
        zaxis_title="CUE"
    ),
    title="3D Interactive Scatter Plot of CUE vs. Uptake Mean & Variance"
)

# Save the plot as an HTML file
fig.write_html("results/CUE_uptake.html")

# Show the interactive plot
fig.show()
# linear regression
from scipy.stats import linregress
def plot_regression(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label='Data points', color='b')
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = slope * x_fit + intercept
    
    plt.plot(x_fit, y_fit, color='r', linestyle='dashed', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
plot_regression(
    u_mean, final_CUE,
    xlabel="Uptake Mean",
    ylabel="Final CUE",
    title="Linear Regression of Final CUE vs. Uptake Mean"
)
plt.savefig("results/linear_regression_of_consumers_resources.png", dpi=300, bbox_inches='tight')