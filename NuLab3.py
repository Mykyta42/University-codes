import math
import numpy as np
from matplotlib import pyplot as plt
import random

def f(x):
    return x*math.sin(6*x)

A = -1
B = 1
n = 20
eps = 1.0e-4
xs1 = [A + (B - A) / (n - 1) * i for i in range(n)]
xs2 = [(A+B)/2 + (B - A) / 2 * math.cos((2 * i + 1)*math.pi / (2*n)) for i in range(n)]
xs2.sort()
xs3 = [A + (B - A) * random.random() for i in range(n)]
xs3.sort()
xs = xs1


def L(x):
    p = 0
    for i in range(n):
        l1 = f(xs[i])
        for j in range(n):
            if j == i:
                continue
            l1 *= (x - xs[j]) / (xs[i] - xs[j])
        p += l1
    return p


def N(x):
    p = 0
    dif = [f(xs[i]) for i in range(n)]
    for i in range(n):
        n1 = dif[0]
        for j in range(i):
            n1 *= (x - xs[j])
        for j in range(n - 1 - i):
            dif[j] = (dif[j + 1] - dif[j]) / (xs[j + i + 1] - xs[j])
        p += n1
    return p

def S(x):
    fs = [f(xs[i]) for i in range(n)]
    A = np.zeros((n - 2, n - 2))
    b = np.zeros(n - 2)
    for i in range(n - 2):
        A[i][i] = (xs[i + 2] - xs[i]) / 3
        b[i] = (fs[i + 2] - fs[i + 1]) / (xs[i + 2] - xs[i + 1]) - (fs[i + 1] - fs[i]) / (xs[i + 1] - xs[i])
    for i in range(n - 3):
        A[i][i + 1] = (xs[i + 2] - xs[i + 1]) / 6
        A[i + 1][i] = (xs[i + 2] - xs[i + 1]) / 6
    d = []
    e = []
    d.append(-A[0][1] / A[0][0])
    e.append(b[0] / A[0][0])
    for i in range(1, n - 3):
        d.append(-A[i][i + 1] / (A[i][i] + d[i - 1] * A[i][i - 1]))
        e.append(-(-b[i] + A[i][i - 1] * e[i - 1]) / (A[i][i] + d[i - 1] * A[i][i - 1]))
    m0 = np.zeros(n - 2)
    m0[n - 3] = -(-b[n - 3] + A[n - 3][n - 4] * e[n - 4]) / (A[n - 3][n - 3] + d[n - 4] * A[n - 3][n - 4])
    for i in range(n - 4, -1, -1):
        m0[i] = d[i] * m0[i + 1] + e[i]
    m = [0]
    for i in range(n - 2):
        m.append(m0[i])
    m.append(0)
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            return m[i] * (xs[i + 1] - x) ** 3 / (6 * (xs[i + 1] - xs[i])) + m[i + 1] * (x - xs[i]) ** 3 / (6 * (xs[i + 1] - xs[i])) + (fs[i] - m[i] * (xs[i + 1] - xs[i]) ** 2 / 6) * (xs[i + 1] - x) / (xs[i + 1] - xs[i]) + (fs[i + 1] - m[i + 1] * (xs[i + 1] - xs[i]) ** 2 / 6) * (x - xs[i]) / (xs[i + 1] - xs[i])


x0 = np.arange(A, B, eps)
f0 = np.vectorize(f)
L0 = np.vectorize(L)
N0 = np.vectorize(N)
S0 = np.vectorize(S)
fig, ax = plt.subplots()
ax.scatter(xs, f0(xs), color='purple', marker='o')
ax.plot(x0, f0(x0), color='black', label='function')
ax.plot(x0, L0(x0), color='red', label='Lagrange')
ax.plot(x0, N0(x0), color='green', label='Newton')
ax.plot(x0, S0(x0), color='blue', label='Spline')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()
