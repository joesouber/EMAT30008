#Shooting Tests.
import numpy as np
from week15 import shooting, pc_predator_prey

def shooting_tester(shooting_func, ODE, true_ODE, u0, pc, system, *args):
    """
    Finds the accuracy of a shooting solution compared to the true solution
        Parameters:
            shooting (function):   this is the shooting function, but to avoid circular imports, the function could not be imported into this code
            ODE (function):             the ODE to test the solutions on
            true_ODE (function):        the true solution function of the ODE
            u0:                         initial conditions x0 and t
            pc (function):              the phase condition
            system (bool):              True if the ODE is a system of equations, False otherwise
            *args:                      any additional arguments that the ODE function defined above expects
        Returns:
            The accuracy of the true solution, as a power of 10.
    """
    empirical_sol = shooting_func(ODE, u0, pc, system, False, *args)
    true_sol = true_ODE(u0[-1], *args)

    tolerance_values = [10**x for x in list(range(-10, 1, 1))]

    proximity = False
    i = 0

    while not proximity:
        if system:
            proximity = np.allclose(empirical_sol[0][-1, :], true_sol, rtol=tolerance_values[i])

        else:
            proximity = np.isclose(empirical_sol[0][-1], true_sol, rtol=tolerance_values[i])

        i += 1

        if i == 11:
            raise ValueError(f"The empirical solution was never close to the true solution, to a tolerance of 1. Please try again with a different ODE or check your functions.")

    return tolerance_values[i - 1]




#%% Hopf Bifurcation tests for shooting


def test():

    def Hopf_bif(U, t, args):
        beta = args[0]
        sigma = args[1]

        u1, u2 = U
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        dudt = np.array([du1dt, du2dt])

        return dudt

    def true_Hopf_bif(t, args):
        beta = args[0]
        phase = args[1]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)

        return np.array([u1, u2])

    shooting_tester(shooting, Hopf_bif, true_Hopf_bif, [1.2, 1.2, 8], pc_predator_prey, True, [1, -1])


if __name__ == "__main__":
    # numerical_shooting_tests()
    test()