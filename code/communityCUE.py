import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
λ_u = np.random.uniform(2.4, 2.6, N)  
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
C0 = np.full(N, 0.1)
R0 = np.full(M, 1)
Y0 = np.concatenate([C0, R0])
    
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)
sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)
    
# CUE
# Compute CUE at each time step
CUE = np.zeros((N, len(sol.t)))
for i, t in enumerate(sol.t):
    C = sol.y[:N, i]  # Consumer abundances at time t
    R = sol.y[N:, i]  # Resource concentrations at time t
    total_uptake = u @ R# (N × M) @ (M,) -> (N,)
    net_uptake = total_uptake * (1 - λ) - m  # Adjusted for leakage and metabolism
    CUE[:, i] = net_uptake / total_uptake  # Compute CUE per consumer
# Final CUE value 
final_CUE = CUE[:, -1]
print(final_CUE)
#cue second method
CUE2 = (u @ (R0+rho*t-omega*t-R) * (1 - λ) - m) / (u @ (R0+rho*t-omega*t-R))
print(CUE2)




# Example data
t_eval = np.linspace(0, 100, 1000)  # Time points (from 0 to 100, with 1000 steps)
C_traj = 1 - np.exp(-0.1 * t_eval)  # Simulated species growth towards steady state at 1

# Compute the time to reach steady state
steady_time = get_steady_state_time(C_traj, t_eval)

u_variance = np.var(u, axis=1, ddof=0)  # Variance of uptake per consumer
u_mean = np.mean(u, axis=1)  # Mean uptake per consumer

# Calculate community energy efficiency based on net energy intake
total_input = np.sum(rho)  # Total resource input
# Compute net energy intake trajectory
C_traj = sol.y[:N, :]  # Consumer biomass over time
R_traj = sol.y[N:, :]  # Resource concentration over time
net_intake_traj = np.zeros(len(t_eval))
for t in range(len(t_eval)):
    net_intake = 0
    for i in range(N):
        for alpha in range(M):
            net_intake += C_traj[i, t] * R_traj[alpha, t] * u[i, alpha] * (1 - lambda_alpha[alpha])
    net_intake_traj[t] = net_intake
efficiency_traj = net_intake_traj / total_input  # Efficiency trajectory
efficiency_final = efficiency_traj[-1]  # Steady-state efficiency
efficiency_avg = np.mean(efficiency_traj)  # Time-averaged efficiency

# community CUE 2
total_input = np.sum(R0+rho*t-omega*t-R)
def get_steady_state_time(C_traj, t_eval, epsilon=1e-4):
    """
    Compute the time required for each species in C_traj to reach steady state.

    Parameters:
    - C_traj (array, shape: N x T): Biomass trajectories for N species over T time points
    - t_eval (array, shape: T): Time points corresponding to C_traj
    - epsilon (float): Threshold for steady state detection (default: 1e-4)

    Returns:
    - steady_times (array, shape: N): Time at which each species reaches steady state
    """
    N, T = C_traj.shape  # Number of species (N) and time steps (T)
    steady_times = np.full(N, t_eval[-1])  # Default to the last time point

    # Compute the absolute rate of change for each species
    dC_dt = np.abs(np.diff(C_traj, axis=1) / np.diff(t_eval))

    # Find steady state time for each species
    for i in range(N):
        for t_idx, rate in enumerate(dC_dt[i, :]):
            if rate < epsilon:
                steady_times[i] = t_eval[t_idx]  # First time the species stabilizes
                break  # Stop checking once steady state is found

    return steady_times

# Compute the time to reach steady state
steady_time = get_steady_state_time(C_traj, t_eval)
print(steady_time)
def compute_biomass_integral(C_traj, t_eval, steady_times):
    """
    Compute the integral of biomass change over time for each species until steady state.

    Parameters:
    - C_traj (array, shape: N x T): Biomass trajectories for N species over T time points
    - t_eval (array, shape: T): Time points corresponding to C_traj
    - steady_times (array, shape: N): Steady-state time for each species

    Returns:
    - biomass_integrals (array, shape: N): Integral of biomass over time until steady state
    """
    N, T = C_traj.shape
    biomass_integrals = np.zeros(N)

    for i in range(N):
        # Find the index where time reaches the steady state time
        t_s_index = np.searchsorted(t_eval, steady_times[i])

        # Compute the integral from t=0 to t_s using the trapezoidal rule
        biomass_integrals[i] = np.trapz(C_traj[i, :t_s_index], t_eval[:t_s_index])

    return biomass_integrals
compute_biomass_integral(C_traj, t_eval, steady_time)

# Output efficiency results
print(f"Steady-state CUE: {efficiency_final:.4f}")
print(f"Time-averaged CUE: {efficiency_avg:.4f}")
print(f"Maximum CUE: {np.max(efficiency_traj):.4f}")  # Check if exceeds 1
# Plot efficiency over time
plt.figure(figsize=(10, 5))
plt.plot(sol.t, efficiency_traj, label='Efficiency', linewidth=2)
plt.xlabel('Time', fontsize=20)
plt.ylabel('CUE', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.title('CUE Over Time', fontsize=20)
plt.show()