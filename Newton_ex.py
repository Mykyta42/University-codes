import math


def f(x: float) -> float:
    return 3*x+math.cos(x)+1  # Here can be any strictly monotonic and differentiable function


def df(x: float) -> float:
    return -math.sin(x) + 3  # With function conditions it is defined and never zero

def gextrdf():
    return 4  # maximal value for positive derivative and vise versa

def lextrdf():
    return 2  # maximal value for negative derivative and vise versa


A = -0.7  # Left non root border
B = 0  # Right non root border
eps = 1e-6  # Halt condition


def bsrch():
    """
    Find the root using binary search
    return approximated root
    """
    n = 0
    x1 = 0.5*(A+B)
    x0 = A
    a = A
    b = B
    while abs(x1-x0) >= eps and f(x1) != 0:
        if (f(a) > 0 and f(x1) < 0) or (f(a) < 0 and f(x1) > 0):
            x1, x0, b = (a+x1) / 2, x1, x1
        else:
            x1, x0, a = (b + x1) / 2, x1, x1
        n += 1
        if n == 2: print("After 2 iterations: ", x1)
    print()
    print("x = ", x1)
    print("Amount of iterations: ", n)
    print("Error: ", abs(f(x1)))


def contr():
    """
    Find the root using contracting mapping
    return approximated root
    """
    n = 0
    beta = - 2 / (gextrdf() + lextrdf())
    x1 = 0.5*(A+B)
    x0 = A
    while abs(x1-x0) >= eps:
        x1, x0 = x1 + beta * f(x1), x1
        n += 1
        if n == 2: print("After 2 iterations: ", x1)
    print()
    print("x = ", x1)
    print("Amount of iterations: ", n)
    print("Error: ", abs(f(x1)))

def nwt():
    """
    Find the root using Newton method
    return approximated root
    """
    n = 0
    x1 = 0.5*(A+B)
    x0 = A
    while abs(x1-x0) >= eps:
        x1, x0 = x1 - f(x1) / df(x1), x1
        n += 1
        if n == 2: print("After 2 iterations: ", x1)
    print()
    print("x = ", x1)
    print("Amount of iterations: ", n)
    print("Error: ", abs(f(x1)))

disp = {1: bsrch, 2: contr, 3: nwt}


active = -1
while active:
    print()
    active = int(input("Choose a method or quit: "))
    if active not in {1, 2, 3}:
        continue
    disp[active]()

