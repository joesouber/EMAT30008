def euler_step(x, t, dt, f):
    return x + dt * f(x, t)

def solve_to(x0, t0, t1, delta_max, f):
    x = x0
    t = t0  
    times = [t0]
    values = [x0]
    while t < t1:
        dt = min(delta_max, t1 - t)
        x = euler_step(x, t, dt, f)
        t += dt
        times.append(t)
        values.append(x)
    return times, values

def x_dot(x, t):
    return x

def main():
    x0 = 1
    t0 = 0
    t1 = 1
    delta_max = 0.1
    times, values = solve_to(x0, t0, t1, delta_max, x_dot)
    print("Values of the solution:")
    for i, value in enumerate(values):
        print("x({:.2f}) = {:.6f}".format(times[i], value))
    
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(times, values, 'r', label='Euler')
    plt.plot(times, np.exp(times), 'k', label='True')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xscale('log')  
    plt.yscale('log')
    plt.legend()
    plt.show()


