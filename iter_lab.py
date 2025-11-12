import numpy as np

def richardson(A, b):
    eps = 1e-4
    n = np.size(b)
    x0 = np.zeros(n)
    G = np.identity(n) - A
    x1 = G @ x0 + b
    a = 1
    while np.linalg.norm(x0 - x1, np.inf) >= eps and a <= 30:
        x0, x1 = x1, G @ x1 + b
        a = a + 1
        print(x1)
    return x1, a + 1

A = np.asarray([[0.5, 0.2, 0.2], [0.2, 0.5, 0.2], [0.2, 0.2, 0.5]])
x = np.asarray([-10, 20, 50])
b = A @ x
xn = richardson(A, b)
print("Obtained solution: ", xn[0])
print("The amount of iterations: ", xn[1])
print("The error equals: ", np.linalg.norm(x - xn[0], np.inf))