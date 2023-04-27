# EMAT30008

This repository contains numerical software for solving ODEs, carrying out numerical shooting and numerical contnuation and solving PDEs in the form of the diffusion equation. The test_solutions file provides input and output testing for the body of the code.

# ODEs
Euler's method, Runge-Kuttas 4th order method and Heun's method are implemented to solve ODE systems with arbitary dimensions.
Extensive Error testing has been implemented showing the difference in converegence between methods.

# Numerical Shooting
Shooting can be implemented alongside a solver such as fsolve to discretise an IVP and reduce it to a BVP.

# Numerical Continuation
Two methods have been implemented: Natural Paramter Continuation and Pseudo-Arclength.

#PDEs
Can use finite_differences function to solve a range of PDEs in the form of the Diffusion equation. Explicit and Implicit Euler mthods have been implemented alongside the Crank-Nicholson method. I have also included an animation file to demonstrate the development of the solutions of some examples over time.



Assumes use of Python 3.9.12
