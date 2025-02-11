import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import param

# Parameters
N = 5  # Number of consumers
M = 5  # Number of resources
λ = 0.3  # Total leakage rate
N_modules = 2  # Number of consumer-resource modules
s_ratio = 10.0  # Modular scaling ratio

# Initial conditions
C0 = np.full(N, 1)  # Initial consumer abundances
R0 = np.full(M, 1)  # Initial resource concentrations
Y0 = np.concatenate([C0, R0])

# Time scale
t_span = (0, 200)
t_eval = np.linspace(*t_span, 300)

# Define colors for different λ_u values
colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Generate 10 different colors

# Create figure(1) and figure(2)
fig1, ax1 = plt.subplots(figsize=(10, 5))  # Community Structure
fig2, ax2 = plt.subplots(figsize=(10, 5))  # CUE Dynamics

# Iterate over different λ_u values
for idx, λ_u in enumerate(np.linspace(0.1, 1, 10)):
    # Update uptake matrix for current λ_u
    u = param.modular_uptake(N, M, N_modules, s_ratio, λ_u)
    
    # Leakage tensor remains unchanged
    l = param.generate_l_tensor(N, M, N_modules, s_ratio, λ)
    
    lambda_alpha = np.full(M, λ)  # Total leakage rate for each resource
    m = np.full(N, 0.1)  # Mortality rate of consumers
    rho = np.full(M, 1)  # Resource inflow rate
    omega = np.full(M, 1)  # Resource decay rate

    # ODE system
    def dCdt_Rdt(t, y):
        C = y[:N]  # Consumer abundances
        R = y[N:]  # Resource concentrations
        dCdt = np.zeros(N)
        dRdt = np.zeros(M)

        # Compute dC/dt (Consumer Growth)
        for i in range(N):
            dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - lambda_alpha[alpha]) for alpha in range(M)) - C[i] * m[i]

        # Compute dR/dt (Resource Dynamics)
        for alpha in range(M):
            dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
            dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
            dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))

        return np.concatenate([dCdt, dRdt])

    # Solve ODE
    sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

    # Compute CUE at each time step
    CUE = np.zeros((N, len(sol.t)))

    for i, t in enumerate(sol.t):
        C = sol.y[:N, i]  # Consumer abundances at time t
        R = sol.y[N:, i]  # Resource concentrations at time t
        total_uptake = u @ R 
        net_uptake = total_uptake * (1 - λ) - m 
        CUE[:, i] = net_uptake / total_uptake  # Compute CUE per consumer

    # Plot Consumer & Resource Dynamics
    for i in range(N):
        ax1.plot(sol.t, sol.y[i], color=colors[idx], alpha=0.7, label=f'Consumer {i+1}' if idx == 0 else None)
    for alpha in range(M):
        ax1.plot(sol.t, sol.y[N + alpha], color=colors[idx], linestyle='dashed', alpha=0.7, label=f'Resource {alpha+1}' if idx == 0 else None)

    # Plot CUE dynamics
    for i in range(N):
        ax2.plot(sol.t, CUE[i], color=colors[idx], alpha=0.7, label=f'λ_u={λ_u:.1f}' if i == 0 else None)



# calculate uptake variance of each species



# Set plot properties for Community Structure
ax1.set_xlabel('Time')
ax1.set_ylabel('Consumer & Resource Abundances')
ax1.set_title('Community Structure Dynamics for Different λ_u')
ax1.legend()

# Set plot properties for CUE
ax2.set_xlabel('Time')
ax2.set_ylabel('CUE')
ax2.set_title('CUE for Different λ_u')
ax2.legend()

# Save CUE plot before displaying
fig2.savefig('../cue_plot.png', dpi=300)
plt.show()
