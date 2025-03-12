import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Parameters
N_pool = 1000  # Number of species in the pool
M = 5          # Number of resources

# Leakage for the species pool
λ = 0.3        # Total leakage rate
N_modules = 4  # Number of modules
s_ratio = 10.0 # Modularity parameter

# Size of the two communities
N1 = 20        # Number of species in Community 1
N2 = 20        # Number of species in Community 2

# Parameters for Community 1
m1 = np.full(N1, 0.3)     # Maintenance cost rate for Community 1
rho1 = np.full(M, 0.4)    # Resource input rate for Community 1
omega1 = np.full(M, 0.5)  # Decay rate of resource for Community 1

# Parameters for Community 2
m2 = np.full(N2, 0.35)     # Maintenance cost rate for Community 2
rho2 = np.full(M, 0.7)    # Resource input rate for Community 2
omega2 = np.full(M, 0.5)  # Decay rate of resource Community 2

# Parameters for merged Community
m3 = np.concatenate([m1, m2])  # Maintenance cost rate for Community 3
rho3 = np.full(M, 0.5)         # Resource input rate for Community 3
omega3 = np.full(M, 0.5)       # Decay rate of resource for Community 3

# Generate uptake matrix and leakage tensor for the species pool
u_pool = param.modular_uptake(N_pool, M, N_modules, s_ratio)
l_pool = param.generate_l_tensor(N_pool, M, N_modules, s_ratio, λ)

# Randomly select indices for Community 1
indices1 = np.random.choice(N_pool, N1, replace=False)
u1 = u_pool[indices1, :]
l1 = l_pool[indices1, :, :]

# Select indices for Community 2 from remaining species
remaining_indices = np.setdiff1d(np.arange(N_pool), indices1)
indices2 = np.random.choice(remaining_indices, N2, replace=False)
u2 = u_pool[indices2, :]
l2 = l_pool[indices2, :, :]

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
t_span = (0, 50)
t_eval = np.linspace(*t_span, 300)

# Simulate Community 1
C0_1 = np.full(N1, 0.01)  # Initial consumer abundance
R0 = np.full(M, 1)        # Initial resource abundance
Y0_1 = np.concatenate([C0_1, R0])
sol1 = solve_ivp(dCdt_Rdt, t_span, Y0_1, t_eval=t_eval, args=(u1, l1, N1, M, m1, rho1, omega1))
ce1 = sol1.y[:N1, -1]  # Consumer abundance at equilibrium
re1 = sol1.y[N1:, -1]  # Resource abundance at equilibrium

# Simulate Community 2
C0_2 = np.full(N2, 0.01)
Y0_2 = np.concatenate([C0_2, R0])
sol2 = solve_ivp(dCdt_Rdt, t_span, Y0_2, t_eval=t_eval, args=(u2, l2, N2, M, m2, rho2, omega2))
ce2 = sol2.y[:N2, -1]
re2 = sol2.y[N2:, -1]

# Merge into Community 3
N3 = N1 + N2
uu = np.vstack([u1, u2])            # Combined uptake matrix (25×3)
ll = np.vstack([l1, l2])            # Combined leakage tensor (25×5×5)
C0_3 = np.concatenate([ce1, ce2])   # Initial consumer abundance for 25 species
R0_3 = re1+re2          # total resource from two communities
Y0_3 = np.concatenate([C0_3, R0_3])
sol3 = solve_ivp(dCdt_Rdt, t_span, Y0_3, t_eval=t_eval, args=(uu, ll, N3, M, m3, rho3, omega3))

sol_list = [sol1, sol2, sol3]  # List of community solutions
N_list = [N1, N2, N3]  # Number of consumers for each community
u_list = [u1, u2, uu]
R0_list = [R0, R0, R0_3]
l_list = [l1, l2, ll]
m_list = [m1, m2, m3]
M_list = [M, M, M]
num_communities = len(sol_list)  # Number of communities
#############################################
# Plot biomass change over time
for i in range(num_communities):
    sol = sol_list[i]
    N = N_list[i]
    C_final = sol.y[:, -1]  # Final abundance
    # Assign colors
    colors = (
        ['red'] * N if i == 0 else
        ['blue'] * N if i == 1 else
        ['red'] * N_list[0] + ['blue'] * (N - N_list[0])
    )
    # Plot biomass change
    plt.figure(figsize=(8, 5))
    for j in range(N):
        plt.plot(sol.t, sol.y[j], label=f'Species {j+1}', color=colors[j])

    plt.xlabel("Time")
    plt.ylabel("Biomass (C)")
    plt.title(f"Change of Biomass Over Time (Community {i+1})")
    plt.legend()
    plt.show()
###### visualization for the seperation between survival and extinction######
species_CUE_list = []
C_final_list = []
survival_counts = []

for i in range(num_communities):
    # Compute CUE for each community
    community_CUE, species_CUE = CUE.compute_community_CUE2(
        sol_list[i], N_list[i], u_list[i], R0_list[i], l_list[i], m_list[i]
    )
    species_CUE_list.append(species_CUE)
    # Store final biomass values
    C_final = sol_list[i].y[:, -1]
    C_final_list.append(C_final)

    # Count richness using the correct C_final reference
    richness = sum(1 for j in range(len(C_final)) if C_final[j] >= 0.1)
    survival_counts.append(richness)

    # Convert to NumPy array for calculations
    species_CUE = np.array(species_CUE, dtype=float)

    # Compute statistics
    species_var = np.var(species_CUE, ddof=1)  # Variance (unbiased estimator)
    # Create scatter plot
    plt.figure(figsize=(8, 5))
    C_final = C_final_list[i]
    species_CUE = species_CUE_list[i]

    # Separate surviving and extinct species based on CUE threshold
    surviving_CUE = [species_CUE[j] for j in range(len(species_CUE)) if C_final[j] >= 0.1]
    extinct_CUE = [species_CUE[j] for j in range(len(species_CUE)) if C_final[j] < 0.1]

    # X-axis positions (community index)
    x_surviving = [i + 1] * len(surviving_CUE)
    x_extinct = [i + 1] * len(extinct_CUE)

    # Dominance
    total_community1 = np.sum(C_final[:N_list[0]])
    total_community2 = np.sum(C_final[N_list[0]:])

    if total_community1 > total_community2:
        dominant = "Community 1"
    else:
        dominant = "Community 2"
    # Print results
    print(f"\nCommunity {i+1}:")
    # Print survival counts
    print(f"{survival_counts[i]} species richness")
    print(f"Average community {i+1} CUE: {community_CUE:.4f}")
    print(f"Species {i+1} CUE: {[f'{cue:.4f}' for cue in species_CUE]}")
    print(f"  Variance: {species_var:.4f}")
    print("-" * 50)
    # Plot scatter points
    plt.scatter(x_surviving, surviving_CUE, color='green', label="Survival" if i == 0 else "")
    plt.scatter(x_extinct, extinct_CUE, color='gray', label="Extinction" if i == 0 else "")

# Set labels and title
plt.xticks(range(1, num_communities + 1), [f"Community {i+1}" for i in range(num_communities)])
plt.xlabel("Community")
plt.ylabel("CUE")
plt.title("CUE Distribution Across Communities")

# Add legend
plt.legend()
plt.show()

############# compute aij ################
def compute_aij(R, u, l, N, M):
    dR_dC = np.zeros((M, N))

    for alpha in range(M):
        for j in range(N):
            dR_dC[alpha, j] = - R[alpha] * u[j, alpha] + np.sum(R * u[j, :] * l[j, :, alpha])
    aij = np.dot(uu* (1 - λ), dR_dC)
    return aij

different_aij = []
for i in range(num_communities):
    aij = compute_aij( R0_list[i], u_list[i], l_list[i], N_list[i], M_list[i])
    different_aij.append(aij)
    # Print mean of interaction coefficients (avoid printing entire list)
    print(f"Mean interaction coefficient {i+1}: {np.mean(aij):.4f}")

############### Control R0_3 value ######################
R0_3_values = np.linspace(0, 5, 50)  # 50 different R0_3 values
cue_community = []

# Iterate over different R0_3 values
for R0_3_val in R0_3_values:
    R0_3 = np.full(M, R0_3_val)  # Ensure R0_3 is an array of size (M,)
    Y0_3 = np.concatenate([C0_3, R0_3])  # Ensure Y0_3 is correct shape

    sol3 = solve_ivp(dCdt_Rdt, t_span, Y0_3, t_eval=t_eval, args=(uu, ll, N3, M, m3, rho3, omega3))

    # Compute total CUE integral
    average_CUE, _ = CUE.compute_community_CUE2(sol3, N3, uu, R0_3, ll, m3)  # Unpack if function returns tuple

    # Store integral value
    cue_community.append(average_CUE)

# Plotting outside the loop
plt.figure(figsize=(8, 5))
plt.plot(R0_3_values, cue_community, marker='o', linestyle='-', color='b')
plt.xlabel("R0_3 Value (Resource Input Rate)")
plt.ylabel("Total Numerical Integral of CUE")
plt.title("Effect of R0_3 on Community CUE")
plt.grid()
plt.show()
