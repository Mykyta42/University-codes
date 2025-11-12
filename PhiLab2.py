import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

# Variables
x = sp.symbols('x')
q1, q2, q3 = 1, 1, 1
k1, k2, k3 = 1, 1, 1
p1, p2, p3 = 1, 1, 1
m1, m2, m3 = 1, 1, 1
a = 0
b = 4
a1, a2 = 1, 1
k = k1 * sp.sin(k2 * x) + k3
p = p1*x**p2 + p3
q = q1 * sp.cos(q2 * x) + q3
u = m1 * sp.cos(m2 * x) + m3
n1 = float((-k * sp.diff(u, x) + a1 * u).subs(x, a))
n2 = float((k * sp.diff(u, x) + a2 * u).subs(x, b))
x_dr = np.linspace(a, b, 101)


def A_op(v):
    return -sp.diff(k * sp.diff(v, x), x) + p * sp.diff(v, x) + q * v


f = A_op(u)

# Methods


def IIAM(n):
    """
    return an approximated solution to the given differential equation
    using integral identity approximation method
    :param n: the amount of knots
    """
    N = n - 1
    d = (b - a) / N
    G = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            G[i][i] = float(k.subs(x, a + 1/2*d) / (d * d) - p.subs(x, a + 1/2*d) / (2 * d) + q.subs(x, a)/2 + a1 / d)
        elif i == n - 1:
            G[i][i] = float(k.subs(x, b - 1/2*d) / (d * d) + p.subs(x, b - 1/2*d) / (2 * d) + q.subs(x, b)/2 + a2 / d)
        else:
            G[i][i] = float((k.subs(x, a + (i - 1/2)*d) + k.subs(x, a + (i + 1/2)*d)) / (d * d) + (p.subs(x, a + (i - 1/2)*d) - p.subs(x, a + (i + 1/2)*d)) / (2 * d) + q.subs(x, a + i*d))
    for i in range(n - 1):
        G[i][i + 1] = float(-k.subs(x, a + (i + 1/2)*d) / (d*d) + p.subs(x, a + (i + 1/2)*d) / (2 * d))
        G[i + 1][i] = float(-k.subs(x, a + (i + 1/2)*d) / (d*d) - p.subs(x, a + (i + 1/2)*d) / (2 * d))
    h = np.zeros(n)
    for i in range(n):
        if i == 0:
            h[i] = float(f.subs(x, a) / 2 + n1 / d)
        elif i == n - 1:
            h[i] = float(f.subs(x, b) / 2 + n2 / d)
        else:
            h[i] = float(f.subs(x, a + i*d))
    c = np.linalg.inv(G) @ h
    u_dr = np.vectorize(sp.lambdify(x, u))
    fig, ax = plt.subplots()
    net = np.linspace(a, b, n)
    ax.scatter(net, c)
    ax.plot(x_dr, u_dr(x_dr), color='red')
    plt.show()
    for i in range(n):
        print(net[i], float(u.subs(x, net[i])), c[i], c[i] - float(u.subs(x, net[i])))



try:
    n = int(input("Choose the amount of knots: "))
    IIAM(n)
except BaseException as e:
    print(e)
