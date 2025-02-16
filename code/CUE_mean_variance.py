import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import sys
import os

# Manually add the directory where param_3D.py is located
sys.path.append(os.path.abspath("~/Documents/MiCRM/code"))
import param_3D

# Parameters
N = 20  # Number of consumers
M = 10  # Number of resources
λ = 0.3  # Total leakage rate

# Each consumer's uptake scaling factor, randomly chosen between 0.2 and 0.8
λ_u = np.random.uniform(0.2, 0.8, N)  
# Each consumer's leakage scaling factor, randomly chosen between 0.2 and 0.4
λ_l = np.random.uniform(0.2, 0.4, N)  
# Standard deviation for uptake distribution per consumer
σ = np.random.uniform(1, 5, N)

N_modules = 5  # Number of modules connecting consumers to resources
s_ratio = 10.0  # Strength of modularity

# Generate uptake matrix, defining consumer-resource interaction strengths
u = param_3D.modular_uptake(N, M, N_modules, s_ratio, λ_u, σ)  

# Total leakage rate for each resource
lambda_alpha = np.full(M, λ)  

# Mortality rate for each consumer
m = np.full(N, 0.2)  
# Input rate of each resource
rho = np.full(M, 1)  
# Decay rate for each resource
omega = np.full(M, 0.05)  

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
    
# Compute Carbon Use Efficiency (CUE) over time
CUE = np.zeros((N, len(sol.t)))                                                                
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  # Consumer populations
    R = sol.y[N:, i]  # Resource concentrations
    total_uptake = u @ (R0 - R)  # Total resource uptake by each consumer
    net_uptake = total_uptake * (1 - λ) - m  # Net uptake considering leakage and mortality
    CUE[:, i] = net_uptake / total_uptake  # CUE definition
    
final_CUE = CUE[:, -1]  # Final CUE values
u_variance = np.var(u, axis=1, ddof=0)  # Variance of uptake per consumer
u_mean = np.mean(u, axis=1)  # Mean uptake per consumer

# Scatter plot: Final CUE vs. Uptake Mean
plt.figure()
plt.scatter(u_mean, final_CUE, label='Data', color='blue', alpha=0.6)
plt.xlabel("Uptake Mean")
plt.ylabel("Final CUE")
plt.title("CUE vs. Uptake Mean")
plt.legend()
plt.show()

# Scatter plot: Final CUE vs. Uptake Variance
plt.figure()
plt.scatter(u_variance, final_CUE, label='Data', color='green', alpha=0.6)
plt.xlabel("Uptake Variance")
plt.ylabel("Final CUE")
plt.title("CUE vs. Uptake Variance")
plt.legend()
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
fig.write_html("../results/interactive_3D_plot.html")

# Show the interactive plot
fig.show()
