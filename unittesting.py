#%%
import unittest
import numpy as np
from ODE import solve_ode

class TestSolveODE(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.ODE = lambda x, t: -x
        self.x0 = 1.0
        self.t0 = 0.0
        self.t1 = 1.0
        self.h = 0.1
        self.system = False
        self.args = ()
        
    def test_euler(self):
        # Test using Euler's method
        method_name = 'euler'
        X, T = solve_ode(self.ODE, self.x0, self.t0, self.t1, method_name, self.h, self.system, *self.args)
        X_exact = np.exp(-T)
        self.assertTrue(np.allclose(X, X_exact, atol=1e-4))
        
    def test_RK4(self):
        # Test using RK4 method
        method_name = 'RK4'
        X, T = solve_ode(self.ODE, self.x0, self.t0, self.t1, method_name, self.h, self.system, *self.args)
        X_exact = np.exp(-T)
        self.assertTrue(np.allclose(X, X_exact, atol=1e-6))
        
    def test_heun(self):
        # Test using Heun's method
        method_name = 'heun'
        X, T = solve_ode(self.ODE, self.x0, self.t0, self.t1, method_name, self.h, self.system, *self.args)
        X_exact = np.exp(-T)
        self.assertTrue(np.allclose(X, X_exact, atol=1e-5))
        
    def test_wrong_method_name(self):
        # Test using an invalid method name
        method_name = 'invalid'
        with self.assertRaises(ValueError):
            X, T = solve_ode(self.ODE, self.x0, self.t0, self.t1, method_name, self.h, self.system, *self.args)
            
if __name__ == '__main__':
    unittest.main(argv=['--verbose'])


# %%
