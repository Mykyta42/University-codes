import numpy as np
from numpy import linalg as la
import random


def pseudo(A, b):
    """
    return the solution of a linear algebraic system with the minial norm, its uniqueness and error
    :param A: the matrix of the system
    :param b: the constant vector of the system
    """
    Ap = la.pinv(A)
    # solution
    x0 = Ap @ b
    # uniqueness
    un = []
    if la.det(np.transpose(A) @ A) > 0:
        phrase = "It\'s the unique solution"
    else:
        phrase = "There\'re another solutions to the system"
    pert = np.array([random.uniform(-1, 1) for i in range(n)])
    x = x0 + pert - Ap @ A @ pert
    un = [phrase, pert, x]
    # error
    sqer = np.transpose(b) @ b - np.transpose(b) @ A @ Ap @ b
    return [x0, un, sqer]

try:
    # obtain parameters
    m = int(input("Enter the amount of rows: "))
    n = int(input("Enter the amount of columns: "))
    A = []
    print("Enter a matrix:", '\n')
    for i in range(m):
        l = input()
        buf1 = list(map(float, l.split(' ')))
        buf2 = []
        for j in range(n):
            buf2.append(buf1[j])
        A.append(np.array(buf2))
    A = np.array(A)
    print("Enter a vector:", '\n')
    l = input()
    buf = list(map(float, l.split(' ')))
    b = []
    for i in range(m):
        b.append(buf[i])
    b = np.array(b)
    # compute and obtain results
    res = pseudo(A, b)
    s1, s2, s3 = res[0], res[1], res[2]
    print("The solution with the minimal norm: ", s1)
    print(s2[0])
    print("For example, with a perturbation equal ", s2[1])
    print("The perturbed solution is ", s2[2])
    print("The square of the error: ", s3)
# Just in case if something would go wrong
except BaseException as e:
    print(e)