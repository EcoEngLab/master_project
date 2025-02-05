import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


N = 2  # 消费者数量
M = 4  # 资源数量


u = np.array([[ 0.491501,0.404814,0.0547006,0.0489846], [0.0723922,0.0287308,0.156474,0.7424035]])  
lambda_alpha = np.array([0.01, 0.01, 0.01, 0.01])  
m = np.array([0.3, 0.3])  
rho = np.array([1, 1, 1, 1])  
omega = np.array([1, 1, 1, 1])  



l = np.array([
    [[ 0.00315368, 0.00104629, 0.00557898, 0.000221047], [0.000420775, 0.00445454, 0.00338405, 0.001740632], [0.00245671, 0.0002242, 0.000637847, 0.00668125], [0.000313903, 0.00028416, 0.00530849, 0.00409345]],
   [[0.00222667, 0.00141503, 0.00186738, 0.00449091], [0.000615695, 0.00216138, 0.00343717, 0.00378576], [0.000386634, 0.000251224, 0.00284617, 0.00651597], [0.000662907, 0.000814148, 0.00486366, 0.00365928]]
]) 



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


C0 = np.array([1.0, 1.0])  
R0 = np.array([1.0, 1.0, 1.0, 1.0])  
Y0 = np.concatenate([C0, R0])


t_span = (0, 10)
t_eval = np.linspace(*t_span, 300)


sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)


plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Consumer {i+1}')
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], label=f'Resource {alpha+1}', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Population / Resource')
plt.legend()
plt.title('Dynamics of Consumers and Resources')
plt.show()