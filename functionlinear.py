import numpy as np
from numpy import linalg as la
import sympy as sp
import random

t = sp.symbols('t')


def func_linear(B, b, T):
    """
    return the solution of a linear functional system with the minial norm, its uniqueness and error
    :param B: the function matrix of the system
    :param b: the function vector of the system
    :param T: the upper limit of integration
    """
    # preliminary computations
    P20 = np.transpose(B) @ B
    aux = np.shape(P20)[0]
    P2 = np.array([[sp.integrate(P20[i][j], (t, 0, T)) for j in range(aux)] for i in range(aux)], dtype=float)
    Bb0 = np.transpose(B) @ b
    Bb = np.array([sp.integrate(Bb0[i], (t, 0, T)) for i in range(aux)], dtype=float)
    Pp = la.pinv(P2)
    # solution
    x0 = Pp @ Bb
    # uniqueness
    un = []
    if la.det(P2) > 0:
        phrase = "It\'s the unique solution"
    else:
        phrase = "There\'re another solutions to the system"
    pert = np.array([random.uniform(-1, 1) for i in range(aux)])
    x = x0 + pert - Pp @ P2 @ pert
    un = [phrase, pert, x]
    # error
    sqer = sp.integrate(np.transpose(b) @ b, (t,0,T)) - np.transpose(Bb) @ Pp @ Bb
    return [x0, un, sqer]


