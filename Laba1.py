import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

x = sp.symbols("x")
f = x * sp.sin(6 * x)
a = -1
b = 1
n = 4
m = 10
dots = [float(f.subs(x, a + (b - a) / (m - 1) * i)) for i in range(m)]
N = 100
x0 = np.arange(a, b, (b-a) / N)


def conti():
    basis = [1] + [sp.sin(i*x) for i in range(1, n + 1)] + [sp.cos(i*x) for i in range(1, n + 1)]
    size = len(basis)
    C = np.array([[float(sp.integrate(basis[i] * basis[j], (x, a, b))) for j in range(size)] for i in range(size)])
    h = np.array([float(sp.integrate(f * basis[i], (x, a, b))) for i in range(size)])
    c = np.linalg.inv(C) @ h
    apr = basis @ c
    apr_plt = np.vectorize(sp.lambdify(x, apr))
    f_plt = np.vectorize(sp.lambdify(x, f))
    fig, ax = plt.subplots()
    ax.plot(x0, f_plt(x0), color='red', label="f(x)")
    ax.plot(x0, apr_plt(x0), color='blue', label="T(x)")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.show()
    print(float(sp.sqrt(sp.integrate(sp.expand((f - apr)**2), (x, a, b)))))


def disc():
    basis = [x**i for i in range(n+1)]
    size = len(basis)
    C = np.array([[sum([float(basis[i].subs(x, xi) * basis[j].subs(x, xi)) for xi in dots]) for j in range(size)] for i in range(size)])
    h = np.array([sum([float(basis[i].subs(x, xi) * f.subs(x, xi)) for xi in dots]) for i in range(size)])
    c = np.linalg.inv(C) @ h
    apr = basis @ c
    apr_plt = np.vectorize(sp.lambdify(x, apr))
    f_plt = np.vectorize(sp.lambdify(x, f))
    fig, ax = plt.subplots()
    ax.plot(x0, f_plt(x0), color='red', label="f(x)")
    ax.plot(x0, apr_plt(x0), color='blue', label="T(x)")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.show()
    print(float(sp.sqrt(sp.integrate(sp.expand((f - apr)**2), (x, a, b)))))


conti()
disc()
