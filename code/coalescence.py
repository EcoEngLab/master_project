import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameter settings
np.random.seed(37)
N_pool = 1000  # Species pool size
M_pool = 20     # Resource pool size
λ = 0.2        # Total leakage rate
N_modules = 5  # Number of modules
s_ratio = 10.0 # Modularity ratio
N1 = 10
M1 = 5
m1 = np.full(N1, 0.2)  # maintaining cost rate
N2 = 10
M2 = 5
m2 = np.full(N2, 0.2)
# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)
# Set rho and omega for the resource pool
rho_pool = np.full(M_pool, 0.6)
omega_pool = np.full(M_pool, 0.1)
# Community 1
species_indices1 = np.random.choice(N_pool, N1, replace=False)
resource_indices1 = np.random.choice(M_pool, M1, replace=False)
u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]
lambda_alpha1 = np.full(M1, λ)
rho1 = rho_pool[resource_indices1]
omega1 = omega_pool[resource_indices1]
# Community 2
if M1 > M2:
    resource_indices2 = np.random.choice(resource_indices1, M2, replace=False)
elif M1 < M2:
    remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
    additional_resources = np.random.choice(remaining_resources, M2 - M1, replace=False)
    resource_indices2 = np.concatenate([resource_indices1, additional_resources])
else:
    resource_indices2 = resource_indices1.copy()
remaining_species = np.setdiff1d(np.arange(N_pool), species_indices1)
species_indices2 = np.random.choice(remaining_species, N2, replace=False)
u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]
lambda_alpha2 = np.full(M2, λ)
rho2 = rho_pool[resource_indices2]
omega2 = omega_pool[resource_indices2]


# Define the differential equations
def dCdt_Rdt(t, y, u, l, N, M, m, rho, omega):
    C = y[:N]
    R = y[N:]
    dCdt = np.zeros(N)
    dRdt = np.zeros(M)
    
    for i in range(N):
        dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - λ) for alpha in range(M)) - C[i] * m[i]
    
    for alpha in range(M):
        dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
        dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
        dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))
    
    return np.concatenate([dCdt, dRdt])

# Time span for simulation
t_span = (0, 500)
t_eval = np.linspace(*t_span, 300)

# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
R0 = np.full(M1, 1)        # Initial resource abundance
Y0_1 = np.concatenate([C0_1, R0])
sol1 = solve_ivp(dCdt_Rdt, t_span, Y0_1, t_eval=t_eval, args=(u1, l1, N1, M1, m1, rho1, omega1))
ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
re1 = sol1.y[N1:, -1]  # Resource abundance at equilibrium

# Simulate Community 2
C0_2 = np.full(N2, 0.01)
Y0_2 = np.concatenate([C0_2, R0])
sol2 = solve_ivp(dCdt_Rdt, t_span, Y0_2, t_eval=t_eval, args=(u2, l2, N2, M2, m2, rho2, omega2))
ce2 = sol2.y[:N2, -1]
re2 = sol2.y[N2:, -1]


# Merge into Community 3
species_indices3 = np.concatenate([species_indices1, species_indices2])
resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2
uu = u_pool[np.ix_(species_indices3, resource_indices3)]
ll = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
m3 = np.concatenate([m1, m2])
lambda_alpha3 = np.full(len(resource_indices3), λ)
rho3 = rho_pool[resource_indices3]
omega3 = omega_pool[resource_indices3]
N3 = N1 + N2
M3 = len(resource_indices3)

C0_3 = np.concatenate([ce1, ce2])
R0_3 = re1 + re2
Y0_3 = np.concatenate([C0_3, R0_3])
sol3 = solve_ivp(dCdt_Rdt, t_span, Y0_3, t_eval=t_eval, args=(uu, ll, N3, M3, m3, rho3, omega3))
#############################################
# Plot biomass change over time
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot for Community 1
cmap1 = plt.get_cmap("Blues")
for i, idx in enumerate(species_indices1):
    axes[0].plot(sol1.t, sol1.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
axes[0].set_title('Community 1 Dynamics')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Consumer Abundance')
axes[0].grid(True)
axes[0].legend(loc='upper right', fontsize='small')
# Plot for Community 2
cmap2 = plt.get_cmap("Reds")
for i, idx in enumerate(species_indices2):
    axes[1].plot(sol2.t, sol2.y[i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[1].set_title('Community 2 Dynamics ')
axes[1].set_xlabel('Time')
axes[1].grid(True)
axes[1].legend(loc='upper right', fontsize='small')
# Plot for Community 3 (merged)
for i, idx in enumerate(species_indices1):
    axes[2].plot(sol3.t, sol3.y[i], color=cmap1((i + 1) / (N1 + 1)), label=f"c{idx}")
for i, idx in enumerate(species_indices2):
    axes[2].plot(sol3.t, sol3.y[N1 + i], color=cmap2((i + 1) / (N2 + 1)), label=f"c{idx}")
axes[2].set_title('Coalescence Dynamics')
axes[2].set_xlabel('Time')
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
###### compare the community CUE between survival and extinction######
sol_list = [sol1, sol2, sol3]  # List of community solutions
N_list = [N1, N2, N3]  # Number of consumers for each community
u_list = [u1, u2, uu]
R0_list = [R0, R0, R0_3]
l_list = [l1, l2, ll]
m_list = [m1, m2, m3]
M_list = [M1, M2, M3]
num_communities = len(sol_list)  # Number of communities
data_to_save = [] 
for i in range(num_communities):
    C_final = np.array(sol_list[i].y[:, -1])  # shape: (num_species,)
    
    community_CUE, species_CUE = CUE.compute_community_CUE2(
        sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
    )
    species_CUE = np.array(species_CUE, dtype=float)  # shape: (num_species_filtered,)

    surviving_CUE = []
    extinct_CUE = []
    # Separate surviving and extinct species based on CUE threshold
    surviving_CUE = [species_CUE[j] for j in range(len(species_CUE)) if C_final[j] >= 0.01]
    extinct_CUE = [species_CUE[j] for j in range(len(species_CUE)) if C_final[j] < 0.01]
    surviving_species_count = len(surviving_CUE)


    mean_surviving_CUE = np.mean(surviving_CUE) if surviving_CUE else np.nan
    mean_extinct_CUE = np.mean(extinct_CUE) if extinct_CUE else np.nan

    print(f"Community {i+1}:")
    print(f"  Surviving species count: {surviving_species_count}")
    print(f"  Mean CUE (Surviving): {mean_surviving_CUE:.4f}")
    print(f"  Mean CUE (Extinct):   {mean_extinct_CUE:.4f}")
    print("-" * 50)


    for val in surviving_CUE:
        data_to_save.append({
            "Community": i + 1,
            "Status": "Survival",
            "CUE": val
        })
    for val in extinct_CUE:
        data_to_save.append({
            "Community": i + 1,
            "Status": "Extinction",
            "CUE": val
        })

# After the loop: create a DataFrame and save to CSV
df_out = pd.DataFrame(data_to_save)
df_out.to_csv("data/CUE_distribution.csv", index=False)

############### Control R0_3 value ######################
R0_3_values = np.linspace(0, 5, 50)  # 50 different R0_3 values
cue_community = []

# # Iterate over different R0_3 values
# for R0_3_val in R0_3_values:
#     R0_3 = np.full(M, R0_3_val)  # Ensure R0_3 is an array of size (M,)
#     Y0_3 = np.concatenate([C0_3, R0_3])  # Ensure Y0_3 is correct shape

#     sol3 = solve_ivp(dCdt_Rdt, t_span, Y0_3, t_eval=t_eval, args=(uu, ll, N3, M, m3, rho3, omega3))

#     # Compute total CUE integral
#     average_CUE, _ = CUE.compute_community_CUE2(sol3, N3, uu, R0_3, ll, m3)  # Unpack if function returns tuple

#     # Store integral value
#     cue_community.append(average_CUE)

# # Plotting outside the loop
# plt.figure(figsize=(8, 5))
# plt.plot(R0_3_values, cue_community, marker='o', linestyle='-', color='b')
# plt.xlabel("R0_3 Value (Resource Input Rate)")
# plt.ylabel("Total Numerical Integral of CUE")
# plt.title("Effect of R0_3 on Community CUE")
# plt.grid()
# plt.show()

