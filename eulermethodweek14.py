import numpy as np
import math
import matplotlib.pyplot as plt

x0 = 1
t0 = 0
t1 = 1
delta_max = 0.1

#%% defining Euler method

def euler_step(x, t, dt, f):
    x_n_1 = x + dt * f(x, t)
    return x_n_1
#%% making solve_to function to solve ODE between t0 and t1, with initial condition x0.



def rk4_step(x, t, dt, f):
    k1 = f(x, t)
    k2 = f(x + dt/2 * k1, t + dt/2)
    k3 = f(x + dt/2 * k2, t + dt/2)
    k4 = f(x + dt * k3, t + dt)
    x_i_1 = x + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_i_1

def solve_to(x0, t0, t1, delta_max, f, method='Euler'):
    x = x0
    t = t0  
    times = [t0]
    values = [x0]

    #defining timesteps and then time loop with update rule.
    while t < t1:
        dt = min(delta_max, t1 - t)
        if method == 'Euler':
            x = euler_step(x, t, dt, f)
        elif method == 'RK4':
            x = rk4_step(x, t, dt, f)
        t += dt
        times.append(t)
        values.append(x)
    return times, values

def x_dot(x, t):
    return x

def main(method='Euler'):
    
    if method == 'Euler':
        times, values = solve_to(x0, t0, t1, delta_max, x_dot, method='Euler')
    elif method == 'RK4':
        times, values = solve_to(x0, t0, t1, delta_max, x_dot, method='RK4')
        
    print("Values of the solution:")
    for i, value in enumerate(values):
        print("x({:.2f}) = {:.6f}".format(times[i], value))

# Call the main function with either 'euler' or 'rk4'
main(method='RK4')


# Analytical solution of the ODE x_dot = x
def true_solution(x0, t):
    return x0 * np.exp(t)


# Compute the error for different values of delta_max
delta_max_values = np.logspace(-2, 0, 10)
errors = []
for delta_max in delta_max_values:
    
    times, values = solve_to(x0, t0, t1, delta_max, x_dot)
    true_values = true_solution(x0, times)
    error = np.abs(true_values - values)
    errors.append(error)

# Plot the error as a function of delta_max
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time step')
plt.ylabel('Error')
for i, delta_max in enumerate(delta_max_values):
    plt.plot(delta_max * np.ones(len(errors[i])), errors[i], 'o', label=f'delta_max = {delta_max}')
plt.legend()
plt.show()