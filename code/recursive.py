import numpy as np
import os
import sys
sys.path.append(os.path.expanduser("~/Documents/MiCRM/code"))
import param
import CUE
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

# Parameters
num_simulations = 20  # Number of repeated simulations with different random seeds
N_pool = 1000  # Number of species in the pool
M = 5  # Number of resources
λ = 0.3  # Total leakage rate
N_modules = 4  # Number of modules
s_ratio = 10.0  # Modularity parameter
N1, N2 = 20, 20  # Number of species in Community 1 and 2

# Storage for results
results = []
C_final_list = []
# Loop over multiple simulations with different random seeds
for seed in range(num_simulations):
    np.random.seed(seed)  # Change random seed
    # Generate uptake matrix and leakage tensor for the species pool
    u_pool = param.modular_uptake(N_pool, M, N_modules, s_ratio)
    l_pool = param.generate_l_tensor(N_pool, M, N_modules, s_ratio, λ)

    # Select indices for Community 1 and Community 2
    indices1 = np.random.choice(N_pool, N1, replace=False)
    remaining_indices = np.setdiff1d(np.arange(N_pool), indices1)
    indices2 = np.random.choice(remaining_indices, N2, replace=False)

    # Extract uptake and leakage matrices for each community
    u1, u2 = u_pool[indices1, :], u_pool[indices2, :]
    l1, l2 = l_pool[indices1, :, :], l_pool[indices2, :, :]

    # Define parameters
    m1, m2 = np.full(N1, 0.3), np.full(N2, 0.35)
    rho1, rho2 = np.full(M, 0.4), np.full(M, 0.7)
    omega1, omega2 = np.full(M, 0.5), np.full(M, 0.5)

    # Function for ODE system
    def dCdt_Rdt(t, y, u, l, N, M, m, rho, omega):
        C, R = y[:N], y[N:]
        dCdt, dRdt = np.zeros(N), np.zeros(M)

        for i in range(N):
            dCdt[i] = sum(C[i] * R[alpha] * u[i, alpha] * (1 - λ) for alpha in range(M)) - C[i] * m[i]

        for alpha in range(M):
            dRdt[alpha] = rho[alpha] - R[alpha] * omega[alpha]
            dRdt[alpha] -= sum(C[i] * R[alpha] * u[i, alpha] for i in range(N))
            dRdt[alpha] += sum(sum(C[i] * R[beta] * u[i, beta] * l[i, beta, alpha] for beta in range(M)) for i in range(N))

        return np.concatenate([dCdt, dRdt])

    # Solve ODE for each community
    t_span, t_eval = (0, 50), np.linspace(0, 50, 300)
    C0, R0 = np.full(N1, 0.01), np.full(M, 1)

    sol1 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0, R0]), t_eval=t_eval, args=(u1, l1, N1, M, m1, rho1, omega1))
    sol2 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0, R0]), t_eval=t_eval, args=(u2, l2, N2, M, m2, rho2, omega2))

    # Merge into Community 3
    N3, m3 = N1 + N2, np.concatenate([m1, m2])
    uu, ll = np.vstack([u1, u2]), np.vstack([l1, l2])
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])  # Final abundance as initial for coalescence
    R0_3 = sol1.y[N1:, -1] + sol2.y[N2:, -1]
    sol3 = solve_ivp(dCdt_Rdt, t_span, np.concatenate([C0_3, R0_3]), t_eval=t_eval, args=(uu, ll, N3, M, m3, np.full(M, 0.5), np.full(M, 0.5)))
    sol_list = [sol1, sol2, sol3]
    N_list = [N1, N2, N3]
    # Compute Community CUE for each community
    community_CUE1, species_CUE1 = CUE.compute_community_CUE2(sol1, N1, u1, R0, l1, m1)
    community_CUE2, species_CUE2 = CUE.compute_community_CUE2(sol2, N2, u2, R0, l2, m2)
    community_CUE3, species_CUE3 = CUE.compute_community_CUE2(sol3, N3, uu, R0_3, ll, m3)
    for sol in sol_list:
        C_final = sol.y[:, -1]
    # Compute dominance in Community 3
    total_community1, total_community2 = np.sum(C_final[:N_list[0]]), np.sum(C_final[N_list[0]:])
    dominant = "Community 1" if total_community1 > total_community2 else "Community 2"

    # Store results
    results.append({
        "Seed": seed,
        "Community CUE 1": community_CUE1,
        "Community CUE 2": community_CUE2,
        "Community CUE 3": community_CUE3,
        "Total Abundance 1": total_community1,
        "Total Abundance 2": total_community2,
        "Dominant Community": dominant
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Assign Dominance values
df_results["Dominance Community 1"] = df_results["Dominant Community"].apply(lambda x: 1 if x == "Community 1" else 0)
df_results["Dominance Community 2"] = df_results["Dominant Community"].apply(lambda x: 1 if x == "Community 2" else 0)

# Logistics regression and scatter plot for Community CUE vs. Dominance
from sklearn.linear_model import LinearRegression
# Merge df_results into a unified format
df_c1 = df_results[["Community CUE 1", "Dominance Community 1"]].rename(
    columns={"Community CUE 1": "CUE", "Dominance Community 1": "Dominance"}
)

# Merge "Community CUE 2" and "Dominance Community 2" into a unified format
df_c2 = df_results[["Community CUE 2", "Dominance Community 2"]].rename(
    columns={"Community CUE 2": "CUE", "Dominance Community 2": "Dominance"}
)

# Concatenate both DataFrames
df_combined = pd.concat([df_c1, df_c2], ignore_index=True)

X = df_combined["CUE"].values.reshape(-1, 1)
y = df_combined["Dominance"].values
# Save df_combined to a CSV file without the index
df_combined.to_csv("results/df_combined.csv", index=False)
