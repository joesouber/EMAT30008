
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve
from PDE import *
from matplotlib.animation import FuncAnimation

def main():
    '''
    This function is used to animate the solutions of the Bratu equation, 
    the Allen-Cahn equation and the simple diffusion equation.
    '''

    def animate_solution(u, t, L, Title=''):
        fig, ax = plt.subplots()
        x = np.linspace(0, L, u.shape[0])
        line, = ax.plot(x, u[:, 0])
        ax.set_xlim(0, L)
        ax.set_ylim(np.min(u), np.max(u))
        ax.set_xlabel('x')
        ax.set_ylabel('u')

        def update(frame):
            line.set_ydata(u[:, frame])
            ax.set_title('{}\nSolution at t = {:.3f}'.format(Title, t[frame]))

        animation = FuncAnimation(fig, update, frames=u.shape[1], interval=10, blit=False)
        plt.show()

    def Bratu_animate():
        # Dynamic Bratu
        L = 1
        T = 30
        mx = 100
        mt = 1000

        def dirichlet_0(x,t):
            '''
            Function containing the homogeneous boundary conditions.
            '''
            return 0

        def Initial_Conds(x, t):

            return 0

        def nonlinear_source(u, x, t):
            mu = np.linspace(0,0.1,101)
            y = np.exp(mu*u)
            return y

        u,t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Conds, discretisation='implicit', source_term = nonlinear_source, D=0.1, linearity='nonlinear')

        animate_solution(u,t,L,'Dynamic Bratu')

    def Allen_cahn_animate():

        import random
        
        L = 0.5
        T = 500
        mx = 1500
        mt = 10000

        def dirichlet_0(x,t):
            '''
            Function containing the homogeneous boundary conditions.
            '''
            return 0

        def Initial_Conds(x, t):
            random.seed(123)
            return np.array([random.uniform(-0.001, 0.001)]) 


        def nonlinear_source(u, x, t):
            
            y = u - u**3
            return y

        u,t = finite_difference(L, T, mx, mt, 'neumann', dirichlet_0, Initial_Conds, discretisation='implicit', source_term = nonlinear_source, D=0.0001, linearity='nonlinear')
       

        animate_solution(u,t,L,'Allen Cahn')
    def simple_animate():

        L = 1 # length of the domain
        T = 0.5 # total time
        mx = 100 # number of grid points in space
        mt = 1000 # number of grid points in time

        def u_exact(x, t,D,L):  
            y = np.exp(-D*(np.pi**2/L**2)*t)*np.sin(np.pi*x/L)
            return y

        # Dirichlet boundary conditions, 0 on both ends.
        def dirichlet_0(x, t):
            return 0

        # Initial condition, sin(pi*x/L).
        def Initial_Condition(x, L):
            y = (np.sin(np.pi*x/L))
            return y

        # Diffusion coefficient,needs to be a function of x, to be compatible with the finite difference method.
        def D(x):
            return x / (x * 10)

        # Source term, needs to be a function of x and t, to be compatible with the finite difference method. 0 for this demonstration.
        def source_term(x,t):
            return 0

        u, t = finite_difference(L, T, mx, mt, 'dirichlet', dirichlet_0, Initial_Condition, discretisation='explicit',source_term = source_term, D = D, linearity='linear')

        animate_solution(u,t,L,'Simple Diffusion')
    
    Bratu_animate()
    Allen_cahn_animate()
    simple_animate()


if __name__ == '__main__':
    main()
