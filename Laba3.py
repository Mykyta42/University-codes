import math
import numpy as np
from matplotlib import pyplot as plt


def Euler(f, y0, x0, N, h):
    y = [y0]
    for i in range(N):
        y.append(y[i] + h * f(x0 + i * h, y[i]))
    return np.array(y)


def Runge_Kutta(f, y0, x0, N, h):
    m = 4
    a = [0, 1/2, 1/2, 1]
    p = [1/6, 1/3, 1/3, 1/6]
    b = [[], [1/2], [0, 1/2], [1/2, 0, 1]]
    y = [y0]
    for i in range(N):
        yi1 = y[i]
        k = []
        for j in range(m):
            xi = x0 + i*h + a[j] * h
            eta = y[i] + np.array(sum([b[j][ii] * k[ii] for ii in range(j)]))
            ki = h * f(xi, eta)
            k.append(ki)
            yi1 = yi1 + p[j] * ki
        y.append(yi1)
    return np.array(y)


def f1(x, y):
    return np.array([y[1], y[2], x + y[0] - 3 * y[1] + 3 * y[2]])

def f2(x, y):
    return (1 + y)**2 / (x * (y + 1) - x**2)


# First problem
y0 = [1, 1, 1]
x0 = 0
h = 0.01
N = 500
ex_sol = np.vectorize(lambda x: 1/2 * math.exp(x) * (x**2 - 4*x + 8) - x - 3)
res = Euler(f1, y0, x0, N, h)
ts = np.linspace(x0, x0 + h * N, N + 1)
apr_sol = res[:, 0]
fig, ax = plt.subplots()
ax.plot(ts, ex_sol(ts), label="exact solution")
ax.plot(ts, apr_sol, label="Euler")
ax.legend()
plt.show()
y0 = [4]
x0 = 2
# Second problem
h = 0.01
N = 500
res = Runge_Kutta(f2, y0, x0, N, h)
ts = np.linspace(x0, x0 + h * N, N + 1)
apr_sol = res[:, 0]
fig, ax = plt.subplots()
ax.plot(ts, apr_sol, label="Runge Kutta")
ax.legend()
plt.show()
