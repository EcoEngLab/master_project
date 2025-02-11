import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import json

# read JSON
with open("data.json", "r") as f:
    data = json.load(f)

# convert to NumPy
import numpy as np
data_np = {key: np.array(value) for key, value in data.items()}


N = 100 
M = 50


u = data_np["u"]  
lambda_alpha = np.full(50, 0.3)
m = data_np["m"] 
rho = data_np["ρ"] 
omega = data_np["ω"]



l = data_np["l"]


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


C0 = np.full(100, 1) 
R0 = np.full(50, 1)
Y0 = np.concatenate([C0, R0])


t_span = (0, 10)
t_eval = np.linspace(*t_span, 300)


sol = solve_ivp(dCdt_Rdt, t_span, Y0, t_eval=t_eval)

# visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# consumer
for i in range(N):
    plt.plot(sol.t, sol.y[i], color='blue', alpha=0.7)

# resource
for alpha in range(M):
    plt.plot(sol.t, sol.y[N + alpha], color='red', linestyle='dashed', alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Population / Resource')

# change legend
consumer_legend = plt.Line2D([0], [0], color='blue', lw=2, label='Consumers')
resource_legend = plt.Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='Resources')
plt.legend(handles=[consumer_legend, resource_legend])

plt.title('Dynamics of Consumers and Resources')
plt.show()
