import math

p = 2  # accuracy of the integration formula
eps = 1e-5  # desired error


def InTegr(f, a, b):
    return (b - a) * f((a+b)/2)  # middle rectangles formula, the accuracy degree is 2


def f1(x):
    return 1 / (x * math.sqrt(math.log(x)))


a1 = 1.1
b1 = math.e


def f2(x):
    return math.atan(x) / (1 + x**2)**(3/2)


M2 = 2.5  # estimation of the second derivative
a2 = 0
b2 = math.inf
A = 397  # replacement for b2
eps2 = math.pi / (4 * A**2)  # estimation of the improper part


def runge(f, a, b):
    n = 1
    h = b - a
    h2 = h / 2
    Ih = InTegr(f, a, b)
    Ih2 = InTegr(f, a, a + h2) + InTegr(f, a + h2, b)
    eps1 = abs(Ih - Ih2) / (2**p - 1)
    while eps1 > eps:
        n *= 2
        h, h2 = h / 2, h2 / 2
        Ih = sum([InTegr(f, a + i*h, a + (i+1)*h) for i in range(n)])
        Ih2 = sum([InTegr(f, a + i*h2, a + (i+1)*h2) for i in range(2 * n)])
        eps1 = abs(Ih - Ih2) / (2**p - 1)
    return Ih2, eps1, h2


def formula(f, a, b):
    h0 = math.sqrt(24 * eps / (2 * M2 * (b-a)))
    n = math.ceil((b - a) / h0)
    h = (b - a) / n
    return sum([InTegr(f, a + i*h, a + (i+1)*h) for i in range(n)]), M2 * (b - a) * h**2 / 24, h


res1 = runge(f1, a1, b1)
res2 = formula(f2, a2, A)
print("Task A")
print("Integral: ", res1[0], "Error: ", res1[1], "Step: ", res1[2])
print("Task B")
print("Integral: ", res2[0], "Error: ", res2[1] + eps2, "Step: ", res2[2])
