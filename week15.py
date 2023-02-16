

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
    t_eval = np.linspace(0, 200, 1000)
    sol = solve_ivp(lambda t, X: pred_prey_eq(X, pars), [0, 200], X0, t_eval=t_eval)


    plt.plot(sol.t, sol.y[0], label='Predator population')
    plt.plot(sol.t, sol.y[1], label='Prey population')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

#To isolate a periodic orbit, we need to look for a closed trajectory in the phase space of the predator-prey system. One way to do this is to plot the phase portrait of the system, which shows the direction of motion of the system's trajectories. We can then identify any closed trajectories, which correspond to periodic orbits.

a = 1
d = 0.1
b= 0.5
pars = [a, b, d]

x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)
DX, DY = pred_prey_eq([X, Y], pars)
fig, ax = plt.subplots()
ax.streamplot(X, Y, DX, DY, color='red')
ax.set_xlabel('Prey population')
ax.set_ylabel('Predator population')

plt.show()


#We can see that there is a closed trajectory, or periodic orbit, in the phase space of the system. To find the starting conditions and period of this orbit, we can integrate the system starting from different initial conditions and look for orbits that close in on themselves.

X0 = [0.3, 0.3]
t_eval = np.linspace(0, 200, 1000)
sol = solve_ivp(lambda t, X: pred_prey_eq(X, pars), [0, 200], X0, t_eval=t_eval)

fig, ax = plt.subplots()
ax.plot(sol.y[0], sol.y[1], color='red')
ax.streamplot(X, Y, DX, DY, color='gray')
ax.set_xlabel('Prey population')
ax.set_ylabel('Predator population')

plt.show()



#%%New attempt at week15

