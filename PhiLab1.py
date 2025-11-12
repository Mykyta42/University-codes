import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

# Variables
x = sp.symbols('x')
q1, q2, q3 = 1, 1, 1
k1, k2, k3 = 1, 1, 1
p1, p2, p3 = 1, 1, 1
m1, m2, m3 = 1, 3, 2
a = 0
b = 4
a1, a2 = 1, 1
k = k1 * sp.sin(k2 * x) + k3
q = q1 * sp.cos(q2 * x) + q3
u = m1 * sp.cos(m2 * x) + m3
n1 = float((-k * sp.diff(u, x) + a1 * u).subs(x, a))
n2 = float((k * sp.diff(u, x) + a2 * u).subs(x, b))
x_dr = np.arange(a, b, (b - a) * 0.01)


def A_op(v):
    return -sp.diff(k * sp.diff(v, x), x) + p * sp.diff(v, x) + q * v


# Methods


def C64(n):
    """
    return an approximated solution to the given differential equation using collocation method
    :param n: the amount of basis functions
    The basis functions: (x-a)**2*(x-A), (x-B)*(b-x)**2 and (x-a)**i*(b-x)**2
    The bias has form of C*x+D
    the points of collocation are chosen uniformly on the interval
    """
    C = float((a1 * n2 - a2 * n1)/((b - a)*a1*a2 + k.subs(x, b)*a1 + k.subs(x, a)*a2))
    D = float((n1 + C * (k.subs(x, a) - a * a1))/a1)
    f0 = C * x + D
    base_func = []
    if n >= 1:
        A = float(b + (k.subs(x, b) * (b - a))/(2 * k.subs(x, b) + a2 * (b - a)))
        base_func.append((x - a)**2*(x - A))
    if n >= 2:
        B = float(a - (k.subs(x, a) * (b - a))/(2 * k.subs(x, a) + a1 * (b - a)))
        base_func.append((b - x)**2*(x - B))
    for i in range(2, n):
        base_func.append((x - a)**i*(b - x)**2)
    dots = [a + (b - a)*(i + 1)/(n + 1) for i in range(n)]
    G = np.array([[float(A_op(base_func[j]).subs(x, dots[i])) for j in range(n)] for i in range(n)])
    h = np.array([float((f - A_op(f0)).subs(x, dots[i])) for i in range(n)])
    c = np.linalg.inv(G) @ h
    u_apr = f0 + sum([c[i] * base_func[i] for i in range(n)])
    u_apr_dr = np.vectorize(sp.lambdify(x, u_apr))
    u_dr = np.vectorize(sp.lambdify(x, u))
    fig, ax = plt.subplots()
    ax.plot(x_dr, u_apr_dr(x_dr), color='blue')
    ax.plot(x_dr, u_dr(x_dr), color='red')
    plt.show()
    print("Error: ", float(sp.sqrt(abs(sp.integrate(sp.expand((u - sp.expand(u_apr))**2), (x, a, b))))))


def R64(n):
    """
    return an approximated solution to the given differential equation using Ritz method
    :param n: the amount of basis functions
    The basis functions: jump functions
    """
    d = (b - a) / (n - 1)
    net = [a + i*d for i in range(n)]
    base_func = [sp.Piecewise(((net[1] - x)/ d, x <= net[1]), (0, x >= net[1]))] + [sp.Piecewise(((x - net[i - 1])/d, (x >= net[i - 1]) & (x <= net[i])), ((net[i + 1] - x)/d, (x <= net[i + 1]) & (x >= net[i])), (0, True)) for i in range(1, n - 1)] + [sp.Piecewise((0, x <= net[n - 2]), ((x-net[n - 2])/d, x >= net[n - 2]))]
    f0 = 0
    G = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            G[i][i] = float(sp.integrate(k / d**2 +
                q * (net[1] - x)**2/ d**2 , (x, net[0], net[1])) + a1)
        elif i == n - 1:
            G[i][i] = float(sp.integrate(k / d ** 2 +
                                         q * (x - net[n - 2]) ** 2 / d ** 2, (x, net[n - 2], net[n - 1])) + a2)
        else:
            G[i][i] = float(sp.integrate(k / d ** 2 +
                                         q * (net[i - 1] - x) ** 2 / d ** 2, (x, net[i - 1], net[i])) +
                            sp.integrate(k / d ** 2 +
                                         q * (x - net[i + 1]) ** 2 / d ** 2, (x, net[i], net[i + 1])))
    for i in range(n - 1):
        G[i][i + 1] = float(sp.integrate(-k / d**2 +
                q * (net[i + 1] - x)*(x - net[i])/ d**2 ,(x, net[i], net[i + 1])))
        G[i + 1][i] = float(sp.integrate(-k / d**2 +
                q * (net[i + 1] - x)*(x - net[i])/ d**2 ,(x, net[i], net[i + 1])))
    h = np.array([float(sp.integrate(sp.expand(f * base_func[i]), (x, a, b)) +
                        n1 * base_func[i].subs(x, a) + n2 * base_func[i].subs(x, b))
                  for i in range(n)])
    c = np.linalg.inv(G) @ h
    u_apr = f0 + sum([c[i] * base_func[i] for i in range(n)])
    u_apr_dr = np.vectorize(sp.lambdify(x, u_apr))
    u_dr = np.vectorize(sp.lambdify(x, u))
    fig, ax = plt.subplots()
    ax.plot(x_dr, u_apr_dr(x_dr), color='blue')
    ax.plot(x_dr, u_dr(x_dr), color='red')
    plt.show()
    print("Error: ", float(sp.sqrt(abs(sum([sp.integrate((u - c[i] * (net[i + 1] - x) / d - c[i + 1] * (x - net[i]) / d)**2, (x, net[i], net[i + 1])) for i in range(n-1)])))))


try:
    code = int(input("Choose a method: "))
    n = int(input("Choose the dimension of basis: "))
    if code == 1:
        p = p1 * x**p2 + p3
    if code == 2:
        p = 0
    f = A_op(u)
    if code == 1:
        C64(n)
    if code == 2:
        R64(n)
except BaseException as e:
    print(e)
