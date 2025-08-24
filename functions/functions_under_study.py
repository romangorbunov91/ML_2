# version 0.1 by romangorbunov91
# 24-Aug-2025

import numpy as np

# Functions.
def f_2D(x):
    return -0.5*x**3 - 3*x**2 - 5*x + 3

def f_3D(x):
    return 2 * x[0]**2 + 205 * x[1]**2 - 10 * x[0]*x[1] + 20 * x[0] + 30 * x[1] + 23

def f_4D(x):
    return 2 * x[0]**2 + 205 * x[1]**2 - 10 * x[0]*x[1] + 20 * x[0] + 30 * x[1] + 23 + 110 * x[2]**2

def f_5D(x):
    return 2 * x[0]**2 + 205 * x[1]**2 - 10 * x[0]*x[1] + 20 * x[0] + 30 * x[1] + 23 + 110 * x[2]**2 - 4 * x[3]**3

# Gradients.
def grad_2D(x):
    return -1.5*x**2 - 6*x - 5

def grad_3D(x):
    return np.array([4*x[0] - 10*x[1] + 20,
                     -10*x[0] + 410*x[1] + 30])

def grad_4D(x):
    return np.array([4*x[0] - 10*x[1] + 20,
                     -10*x[0] + 410*x[1] + 30,
                     220*x[2]])

def grad_5D(x):
    return np.array([4*x[0] - 10*x[1] + 20,
                     -10*x[0] + 410*x[1] + 30,
                     220*x[2],
                     -12*x[3]**2])