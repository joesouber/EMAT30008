import numpy as np
import matplotlib.pyplot as plt

def finite_difference_grid(N, a, b):
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    return x, dx

def Dirichlet_bcs(gamma1, gamma2):
    return [gamma1, gamma2] 

def construct_A_and_b(grid,bc_left,bc_right):
    N = len(grid) - 1
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    for i in range(1, N):
        A[i, i-1] = -1
        A[i, i] = 2
        A[i, i+1] = -1
    A[0, 0] = 1
    A[N, N] = 1
    b[0] = bc_left[0]
    b[N] = bc_right[N]
    return A, b

def q(x):
    return np.ones(np.size(x))

bc_left= Dirichlet_bcs(0,0)
bc_right = Dirichlet_bcs(0,0)

def solve_poisson(grid, bc_left, bc_right):
    N = len(grid) - 1
    A, b = construct_A_and_b(grid, bc_left, bc_right)
    u = np.linalg.solve(A, (-b)[:,np.newaxis] - dx**2 * q(x[1:-1]))
    return u


grid = finite_difference_grid(10, 0, 1)
dx = grid[1]
x = grid[0]
u = solve_poisson(grid, bc_left, bc_right)
u_exact = 1/2 * x * (1-x)

plt.plot(x, u,'o' ,label="Numerical solution")
plt.plot(x, u_exact,'k', label="Exact solution")
plt.show()