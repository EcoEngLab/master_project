import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
from scipy.integrate import solve_ivp
import CUE
# Parameter settings
N_pool = 1000  # Number of species in the species pool
M_pool = 20    # Number of resources in the resource pool

# Leakage parameters
λ = 0.2        # Total leakage rate
N_modules = 5  # Number of functional modules
s_ratio = 10.0 # Modularity parameter

# Sizes of two communities
N1 = 10        # Number of species in Community 1
M1 = 5         # Number of resources in Community 1

N2 = 15        # Number of species in Community 2
M2 = 5         # Number of resources in Community 2

# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)

# Randomly select species and resources for Community 1
species_indices1 = np.random.choice(N_pool, N1, replace=False)
resource_indices1 = np.random.choice(M_pool, M1, replace=False)
u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

# Select species and resources for Community 2 from the remaining pool
remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
species_indices2 = np.random.choice(remaining_species, N2, replace=False)
resource_indices2 = np.random.choice(remaining_resources, M2, replace=False)
u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

# Define the system of differential equations
def dCdt_Rdt(t, y, u, l, N, M, m, rho, omega):
    C = y[:N]  # Consumer abundances
    R = y[N:]  # Resource abundances
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    # Compute changes in consumer abundances
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - λ) for alpha in range(M)) - C[i] * m[i]
    
    # Compute changes in resource abundances
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]  # Input and decay
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))  # Resource consumption
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))  # Leakage contribution
    
    return np.concatenate([dCdt, dRdt])

# Simulation time range
t_span = (0, 300)
t_eval = np.linspace(*t_span, 300)

# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
R0_1 = np.full(M1, 1)     # Initial resource abundance
Y0_1 = np.concatenate([C0_1, R0_1])
m1 = np.full(N1, 0.3)     # Maintenance cost per species
rho1 = np.full(M1, 0.5)   # Resource input rate
omega1 = np.full(M1, 0.5) # Resource decay rate
sol1 = solve_ivp(dCdt_Rdt, t_span, Y0_1, t_eval=t_eval, args=(u1, l1, N1, M1, m1, rho1, omega1))
ce1 = sol1.y[:N1, -1]     # Equilibrium consumer abundances
re1 = sol1.y[N1:, -1]     # Equilibrium resource abundances

# Simulate Community 2
C0_2 = np.full(N2, 0.01)
R0_2 = np.full(M2, 1)
Y0_2 = np.concatenate([C0_2, R0_2])
m2 = np.full(N2, 0.2)
rho2 = np.full(M2, 0.9)
omega2 = np.full(M2, 0.5)
sol2 = solve_ivp(dCdt_Rdt, t_span, Y0_2, t_eval=t_eval, args=(u2, l2, N2, M2, m2, rho2, omega2))
ce2 = sol2.y[:N2, -1]
re2 = sol2.y[N2:, -1]

# Merge into Community 3
N3 = N1 + N2  # Total number of species
M3 = M1 + M2  # Total number of resources
species_indices3 = np.concatenate([species_indices1, species_indices2])
resource_indices3 = np.concatenate([resource_indices1, resource_indices2])

# Extract uptake matrix and leakage tensor for the merged community
u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

# Set parameters for Community 3
m3 = np.concatenate([m1, m2])
rho3 = np.concatenate([rho1, rho2])  # Resource input rates from both communities
omega3 = np.concatenate([omega1, omega2])  # Resource decay rates from both communities

# Initial conditions for Community 3
C0_3 = np.concatenate([ce1, ce2])
R0_3 = np.concatenate([re1, re2])
Y0_3 = np.concatenate([C0_3, R0_3])

# Simulate Community 3
sol3 = solve_ivp(dCdt_Rdt, t_span, Y0_3, t_eval=t_eval, args=(u3, l3, N3, M3, m3, rho3, omega3))
# ---- Compute Intergral Community CUE for Multiple Communities ----
sol_list = [sol1, sol2, sol3]  # List of community solutions
N_list = [N1, N2, N3]  # Number of consumers for each community
rho_list = [rho1, rho2, rho3]  # Resource input rates for each community
num_communities = len(sol_list)  # Number of communities

# Initialize figure for visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))

# Loop through each community to compute and plot CUE
for i in range(num_communities):
    time_indices, community_cue, total_integral = CUE.compute_community_CUE1(sol_list[i], N_list[i], rho_list[i], return_all= True)

    # Plot Community CUE over time
    plt.plot(sol_list[i].t[time_indices[1:]], community_cue, marker='o', linestyle='-', label=f'Community {i+1} CUE')

    # Print total numerical integral of CUE
    print(f"Total numerical integral of community {i+1} CUE: {total_integral:.4f}")

# Set plot labels and title
plt.xlabel('Time')
plt.ylabel('Community CUE')
plt.title('Instantaneous Community CUE Over Time')
plt.legend()
plt.grid()
plt.show()
### Second CUE method
# ---- Compute Intergral Community CUE for Multiple Communities ----
sol_list = [sol1, sol2, sol3]  # List of community solutions
N_list = [N1, N2, N3]  # Number of consumers for each community
u_list = [u1, u2, u3]
R0_list = [R0_1, R0_2, R0_3]
l_list = [l1, l2, l3]
m_list = [m1, m2, m3]
num_communities = len(sol_list)  # Number of communities
# Loop through each community to compute and plot CUE
for i in range(num_communities):
    community_CUE, species_CUE = CUE.compute_community_CUE2(sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i])

    # Plot Community CUE over time
    plt.plot(sol_list[i].t[time_indices[1:]], community_cue, marker='o', linestyle='-', label=f'Community {i+1} CUE')

    # Print total numerical integral of CUE
    print(f"Average community {i+1} CUE: {community_CUE:.4f}")