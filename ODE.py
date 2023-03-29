import matplotlib.pyplot as plt
import math
import numpy as np

def euler_step(f, x0, t0, h, *args):

    x1 = x0 + h * f(x0, t0, *args)
    t1 = t0 + h

    return x1, t1

def RK4_step(f, x0, t0, h, *args):
   
    k1 = f(x0, t0, *args)
    k2 = f(x0 + h * 0.5 * k1, t0 + 0.5 * h, *args)
    k3 = f(x0 + h * 0.5 * k2, t0 + 0.5 * h, *args)
    k4 = f(x0 + h * k3, t0 + h, *args)

    k = 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def heun_step(f, x0, t0, h, *args):
    k1 = f(x0, t0, *args)
    x1_tilde = x0 + h * k1
    k2 = f(x1_tilde, t0 + h, *args)

    k = 1 / 2 * h * (k1 + k2)
    x1 = x0 + k
    t1 = t0 + h

    return x1, t1
