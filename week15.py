

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numdifftools as nd
def pred_prey_eq(X, pars):
  
    x = X[0]
    y = X[1]
    a, b, d = pars[0], pars[1], pars[2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])

## simulating pred/prey equations uysing scipy solve_ivp. Should be able to switch this out for one of my functions eventually.

a = 1
d = 0.1
bs = [0.1, 0.5]

for b in bs:
    pars = [a, b, d]
    X0 = [0.5, 0.5]
    t_eval = np.linspace(0, 10, 100)
    sol = solve_ivp(lambda t, X: pred_prey_eq(X, pars), [0, 10], X0, t_eval=t_eval)


    plt.plot(sol.t, sol.y[0], label='Predator population')
    plt.plot(sol.t, sol.y[1], label='Prey population')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()





