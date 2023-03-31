import unittest
from ODE import solve_ode
from ODE_PDE_funcs import f_true_solution, f
import numpy as np

class TestODESolver(unittest.TestCase):
    
    def test_euler_solver(self):
        # Test the Euler method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'euler', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-3)
    
    def test_RK4_solver(self):
        # Test the Runge-Kutta 4 method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'RK4', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-8)
    
    def test_heun_solver(self):
        # Test the Heun method solver on a simple ODE
        x0 = 1
        t0 = 0
        t1 = 1
        h = 0.1
        X, T = solve_ode(f, x0, t0, t1, 'heun', h, False)
        self.assertTrue(np.abs(X[-1]-f_true_solution(T)) < 10**-8)
        

if __name__ == '__main__':
    unittest.main()

# %%
