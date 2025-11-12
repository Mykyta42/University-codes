import numpy as np
import random
import time
import matplotlib.pyplot as plt

low1 = 0.1
upper1 = 1
p = 1
dp = 8
P = p + p + dp  # P is at least 2 times p
tests = 1000
a = 0
b = 0
y = 0
ps = np.array([])
k = 0
m = 0
s = 0
As = []
Bs = []
test_sizes = np.array([])
x0 = np.array([])
real_res = np.array([])
A = None
Pc = None
L = None
past1 = None
past2 = None

def Pc1(xs: np.array):
    """
    return projection onto simplex
    """
    n = len(xs)
    ind = [i for i in range(n)]
    ind.sort(reverse=True, key=lambda i: xs[i])
    sum = 0
    l = 0
    r = 0
    for i in range(n):
        a = xs[ind[i]]
        sum += a
        l1 = (y - sum) / (r+1)
        if a + l1 <= 0:
            break
        else:
            r += 1
            l = l1
    pr = [0 for i in range(n)]
    for i in range(r):
        pr[ind[i]] = l + xs[ind[i]]
    return np.array(pr)


def seq(n):
    return 1/(n+1)**0.75


def Pc2(xs: np.array):
    """
    return projection onto n-dimensional rectangle
    """
    return np.array([p if xs[i] <= p else xs[i] if p <= xs[i] <= P else P for i in range(len(xs))])


def A1(xs: np.array):
    """
    function for the first problem
    """
    return np.array([ps[i] - max(0, a - 2 * b * xs[i]) for i in range(len(xs))])


def A2(xs: np.array):
    """
    function for the second problem
    """
    d = [k + m * sum(xs) - (m + s) * xs[i] for i in range(len(xs))]
    return np.array([-s * (2 * As[i] * d[i] + Bs[i]) + xs[i] * s - d[i] for i in range(len(xs))])


def alg1(x0, A, PC, L):
    """
    extragradient method
    """
    N = 0
    xn = x0
    lam = 1 / (2 * L)
    while N < it_max:
        N += 1
        yn = PC(xn - lam * A(xn))
        xn1 = PC(xn - lam * A(yn))
        if np.linalg.norm(xn1 - xn, 2) < eps:
            break
        else:
            xn = xn1
    return [N, xn]


def alg2(x0, A, PC, L, remember=False):
    """
    extrapolation from the past
    """
    global past1
    N = 0
    xn = x0
    yn = x0
    if remember and past1 is not None:
        yn = past1
    lam = 1 / (6 * L)
    while N < it_max:
        N += 1
        aux = A(xn)
        yn = PC(yn - lam * aux)
        xn1 = PC(yn - lam * aux)
        if np.linalg.norm(xn1 - xn, 2) < eps:
            break
        else:
            xn = xn1
    if remember:
        past1 = yn
    return [N, xn]


def alg3(x0, A, PC, seq, remember=False):
    """
    Cesaro means
    """
    global past2
    N = 0
    n = 0
    xn = x0
    yn = x0
    sig = seq(n)
    if remember and past2 is not None:
        yn = past2[0]
        n = past2[1]
        sig = past2[2]
    lam = seq(n)
    while N < it_max:
        N += 1
        n += 1
        yn = PC(yn - lam * A(yn))
        lam = seq(n)
        sig += lam
        xn1 = (1 - lam / sig) * xn + lam / sig * yn
        if np.linalg.norm(xn1 - xn, 2) < eps:
            break
        else:
            xn = xn1
    if remember:
        past2 = [yn, n, sig]
    return [N, xn]


def construct1():
    """
    first problem construction
    """
    global A, L, Pc, x0, real_res, a, b, y, ps
    a = random.uniform((P - p) * n, 2 * (P - p) * n)
    b = random.uniform(low1, upper1)
    y = random.uniform((P - p) * n / (2 * b), a / (2 * b))
    # Here the objective function is quadratic
    ps = [random.uniform(p, P) for i in range(n)]
    x0 = np.array([y / n for i in range(n)])
    real_res = np.array([1 / (2 * b) * (sum(ps) / n - ps[i]) + y / n for i in range(n)])
    L = 2 * b
    A = A1
    Pc = Pc1


def construct2():
    """
    second problem construction
    """
    global A, L, Pc, x0, real_res, As, Bs, k, m, n, s
    real_res = np.array([random.uniform((2 / 3) * P, P) for i in range(n)])
    # next variables are generated to guarantee real_res is the solution
    m = random.uniform(low1, upper1)
    s = random.uniform(19 * m * (n - 1) * (P - p) / P, 20 * m * (n - 1) * (P - p) / P)
    # here s >= 9.5*m*(m-1)
    k = 1.1 * s * P - (n - 1) * m * p
    d0 = np.array([k + m * sum(real_res) - (m + s) * real_res[i] for i in range(n)])
    # Here 0.1*s*P <= d[i] <= 0.5*s*P
    As = np.array([random.uniform(float((1 / (4 * d0[i])) * (real_res[i] - d0[i] / s)),
                                  float((1 / (2 * d0[i])) * (real_res[i] - d0[i] / s))) for i in range(n)])
    # Here 0.25 <= As[i] <= 4.5
    # It guarantees the matrix (M + M_t)/2 is symmetric and diagonal dominant
    Bs = np.array([real_res[i] - d0[i] / s - 2 * As[i] * d0[i] for i in range(n)])
    x0 = np.array([(p + P) / 2 for i in range(n)])
    L = (sum([(2 * As[i] * s * s + 2 * s + m * (n - 1) ** 0.5 + (2 * As[i] * s + 1)) ** 2 for i in
              range(n)])) ** (1 / 2)
    A = A2
    Pc = Pc2


action = input('Choose an action: ')
number = int(input('Choose a problem: '))  # choose a problem: 1 or 2
if action == 'solve':
    eps = 1e-4
    it_max = 10 ** 9
    if number == 1:
        test_sizes = [10, 50, 100, 500]
    elif number == 2:
        test_sizes = [10, 50]
    for n in test_sizes:
        iters = []
        errors = []
        for attempt in range(tests):
            if number == 1:
                construct1()
            elif number == 2:
                construct2()
            res1 = alg1(x0, A, Pc, L)
            res2 = alg2(x0, A, Pc, L)
            res3 = alg3(x0, A, Pc, seq)
            err1 = np.linalg.norm(res1[1] - real_res, 2)
            err2 = np.linalg.norm(res2[1] - real_res, 2)
            err3 = np.linalg.norm(res3[1] - real_res, 2)
            iters.append([res1[0], res2[0], res3[0]])
            errors.append([err1, err2, err3])
        print('size: ', n)
        print('=========================')
        print('First')
        print('mean iters: ', sum([iters[i][0] for i in range(tests)]) / tests)
        print('mean error: ', sum([errors[i][0] for i in range(tests)]) / tests)
        print('=========================')
        print('Second')
        print('mean iters: ', sum([iters[i][1] for i in range(tests)]) / tests)
        print('mean error: ', sum([errors[i][1] for i in range(tests)]) / tests)
        print('=========================')
        print('Third')
        print('mean iters: ', sum([iters[i][2] for i in range(tests)]) / tests)
        print('mean error: ', sum([errors[i][2] for i in range(tests)]) / tests)
        print('=========================')
elif action == 'time':
    eps = -1
    it_max = 1
    if number == 1:
        test_sizes = [10, 50, 100, 500]
    elif number == 2:
        test_sizes = [10, 50]
    for n in test_sizes:
        secs = []
        for attempt in range(tests):
            if number == 1:
                construct1()
            elif number == 2:
                construct2()
                A = A2
                Pc = Pc2
            start = time.time()
            alg1(x0, A, Pc, L)
            end = time.time()
            time1 = end - start
            start = time.time()
            alg2(x0, A, Pc, L)
            end = time.time()
            time2 = end - start
            start = time.time()
            alg3(x0, A, Pc, seq)
            end = time.time()
            time3 = end - start
            secs.append([time1, time2, time3])
        print('size: ', n)
        print('=========================')
        print('First')
        print('mean time for one iteration: ', sum([secs[i][0] for i in range(tests)]) / tests)
        print('=========================')
        print('Second')
        print('mean time for one iteration: ', sum([secs[i][1] for i in range(tests)]) / tests)
        print('=========================')
        print('Third')
        print('mean time for one iteration: ', sum([secs[i][2] for i in range(tests)]) / tests)
        print('=========================')
elif action == 'converge':
    eps = -1
    it_max = 1
    if number == 1:
        iters = 200
    else:
        iters = 500
    n = 10
    errors = []
    for attempt in range(tests):
        err_aux = []
        if number == 1:
            construct1()
        elif number == 2:
            construct2()
        x1 = x0
        x2 = x0
        x3 = x0
        for it in range(iters):
            x1 = alg1(x1, A, Pc, L)[1]
            x2 = alg2(x2, A, Pc, L, remember=True)[1]
            x3 = alg3(x3, A, Pc, seq, remember=True)[1]
            err1 = np.linalg.norm(x1 - real_res, 2)
            err2 = np.linalg.norm(x2 - real_res, 2)
            err3 = np.linalg.norm(x3 - real_res, 2)
            err_aux.append([err1, err2, err3])
        errors.append(err_aux)
        past1, past2 = None, None
    ax = plt.subplot()
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean error')
    Ns = np.arange(1, iters + 1, 1)
    ers1 = [sum([errors[j][i][0] for j in range(tests)]) / tests for i in range(iters)]
    ers2 = [sum([errors[j][i][1] for j in range(tests)]) / tests for i in range(iters)]
    ers3 = [sum([errors[j][i][2] for j in range(tests)]) / tests for i in range(iters)]
    ax.plot(Ns, ers1, color='red', label='Algorithm1')
    ax.plot(Ns, ers2, color='green', label='Algorithm2')
    ax.plot(Ns, ers3, color='blue', label='Algorithm3')
    legend = ax.legend(loc='upper right')
    plt.show()
