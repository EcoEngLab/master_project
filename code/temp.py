import numpy as np
import matplotlib.pyplot as plt

N = 20
k = 0.0000862 # Boltzman constant
Tref = 273.15 + 10 # Reference temperature Kelvin, 0 degrees C
T = 273.15 + np.linspace(0,60,61) # Temperatures
Ea_D = 3.5
lf = 0.4

rho_R = -0.75
rho_U = -0.75

B_R0 = np.log(1.70 * np.exp((-0.67/k) * ((1/Tref)-(1/273.15)))/(1 + (0.67/(Ea_D - 0.67)) * np.exp(Ea_D/k * (1/311.15 - 1/Tref)))) # Using CUE0 = 0.22, mean growth rate = 0.48
B_R0_var = 0.05* B_R0
Ea_R_mean = 0.67; Ea_R_var = 0.04*Ea_R_mean
cov_xy_R = rho_R * B_R0_var**0.5 * Ea_R_var ** 0.5
mean_R = [B_R0, Ea_R_mean]
cov_R = [[B_R0_var, cov_xy_R], [cov_xy_R, Ea_R_var]]  

B_U0 = np.log((1.70/(1 - lf - 0.22)) * np.exp((-0.82/k) * ((1/Tref)-(1/273.15)))/(1 + (0.82/(Ea_D - 0.82)) * np.exp(Ea_D/k * (1/308.15 - 1/Tref))))
B_U0_var = 0.05* B_U0
Ea_U_mean = 0.82; Ea_U_var = (0.04*Ea_U_mean)
cov_xy_U = rho_U * B_U0_var**0.5 * Ea_U_var ** 0.5
mean_U = [B_U0, Ea_U_mean]
cov_U = [[B_U0_var, cov_xy_U], [cov_xy_U, Ea_U_var]]  


np.random.seed(0)
T_pk_U = 273.15 + np.random.normal(35, 5, size = N); T_pk_R = T_pk_U + 3
B_R_log, Ea_R = np.random.multivariate_normal(mean_R, cov_R, N).T
B_U_log, Ea_U = np.random.multivariate_normal(mean_U, cov_U, N).T
B_R = np.exp(B_R_log); B_U = np.exp(B_U_log)


for i in range(N):
    U_Sharpe = B_U[i] * np.exp((-Ea_U[i]/k) * ((1/T)-(1/Tref)))/(1 + (Ea_U[i]/(Ea_D - Ea_U[i])) * np.exp(Ea_D/k * (1/T_pk_U[i] - 1/T))) 
    plt.plot(T - 273.15, np.log(U_Sharpe), color = 'darkorange', linewidth=0.7)

plt.xlabel('Temperature ($^\circ$C)', fontsize = 12) 
plt.ylabel('Uptake Rate', fontsize = 12)
plt.show()

for i in range(N):
    R_Sharpe = B_R[i] * np.exp((-Ea_R[i]/k) * ((1/T)-(1/Tref)))/(1 + (Ea_R[i]/(Ea_D - Ea_R[i])) * np.exp(Ea_D/k * (1/T_pk_R[i] - 1/T))) 
    plt.plot(T - 273.15, np.log(R_Sharpe), 'darkgreen', linewidth=0.7)

plt.xlabel('Temperature ($^\circ$C)', fontsize = 12) 
plt.ylabel('Repiration Rate', fontsize = 12)
plt.show()

T = T[0:26]
for i in range(N):
    U_Sharpe = B_U[i] * np.exp((-Ea_U[i]/k) * ((1/T)-(1/Tref)))/(1 + (Ea_U[i]/(Ea_D - Ea_U[i])) * np.exp(Ea_D/k * (1/T_pk_U[i] - 1/T))) 
    R_Sharpe = B_R[i] * np.exp((-Ea_R[i]/k) * ((1/T)-(1/Tref)))/(1 + (Ea_R[i]/(Ea_D - Ea_R[i])) * np.exp(Ea_D/k * (1/T_pk_R[i] - 1/T))) 
    plt.plot(T - 273.15, (U_Sharpe*(1-lf) - R_Sharpe)/U_Sharpe, color = 'maroon', linewidth=0.7)

plt.xlabel('Temperature ($^\circ$C)', fontsize = 12) 
plt.ylabel('CUE', fontsize = 12)
plt.show()
