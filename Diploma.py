import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

class RBFN:
    def __init__(self, in_size, hide_size, rbf, rbf_grad):
        r = 5
        R = 1
        self.__isz = in_size
        self.__hsz = hide_size
        self.__ws = np.random.uniform(-r, r, hide_size)
        self.__cs = np.random.uniform(-R, R, (hide_size, in_size))
        self.__rbf = rbf
        self.__der = rbf_grad


    def assign(self, x):
        self.__ws = x[self.__hsz * self.__isz:]
        self.__cs = np.reshape(x[:self.__hsz * self.__isz], (self.__hsz, self.__isz))

    def compute(self, arg):
        return self.__ws @ (np.asarray([self.__rbf((arg - self.__cs[i]) @ (arg - self.__cs[i])) for i in range(self.__hsz)]))

    def test(self, test_set):
        d1, d2 = np.asarray([el[0] for el in test_set]), np.asarray([el[1] for el in test_set])
        return np.linalg.norm(np.asarray([self.compute(d1[i]) - d2[i] for i in range(len(test_set))]), np.inf)

    def __aux_alg(self, x0, A):
        N = 1000
        steps = 0
        eps = 1e-6
        A0 = A()
        if np.linalg.norm(A0) == 0:
            return x0
        l = 1e-10
        L = 1e-10
        x = x0
        x1 = x - l * A0
        y = x1
        self.assign(x1)
        A1 = A()
        l1 = np.linalg.norm(x1 - x0) / (2 * np.linalg.norm(A1 - A0))
        L1 = np.linalg.norm(A1 - A0) / (2 * np.linalg.norm(x1 - x0))
        b = (1 - math.sqrt(l1 * L1)) / (1 + math.sqrt(l1 * L1))
        y1 = x1 - l1 * A1
        x, x1 = x1, y1 + b * (y1 - y)
        self.assign(x1)
        A0, A1 = A1, A()
        s = l1 / l
        S = L1 / L
        while np.linalg.norm(A1) > eps and steps < N and np.linalg.norm(A1 - A0) != 0 and np.linalg.norm(x1 - x0) != 0:
            l, l1 = l1, min(math.sqrt(1 + s / 2)*l1, np.linalg.norm(x1 - x) / (2 * np.linalg.norm(A1 - A0)))
            L, L1 = L1, min(math.sqrt(1 + S / 2)*L1, np.linalg.norm(A1 - A0) / (2 * np.linalg.norm(x1 - x)))
            b = (1 - math.sqrt(l1 * L1)) / (1 + math.sqrt(l1 * L1))
            y, y1 = y1, x1 - l1 * A1
            x, x1 = x1, y1 + b * (y1 - y)
            self.assign(x1)
            A0, A1 = A1, A()
            s = l1 / l
            S = L1 / L
            steps = steps + 1
        return x1

    def __adam(self, x0, A, N):
        x = x0
        steps = 0
        eps1 = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        b1 = 1
        b2 = 1
        alpha = 0.001
        m = np.zeros(len(x0))
        v = np.zeros(len(x0))
        errs = []
        secs = []
        gr = []
        while steps < N:
            start = time.time()
            random.shuffle(A)
            for i in range(len(A)):
                g = A[i]()
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g * g
                b1 = b1 * beta1
                b2 = b2 * beta2
                mn = m / (1 - b1)
                vn = v / (1 - b2)
                x = x - alpha * mn / (np.sqrt(vn) + eps1)
                self.assign(x)
            steps = steps + 1
            end = time.time()
            secs.append(end - start)
            errs.append(self.test(data_set))
            gr.append(np.linalg.norm(sum([A[i]() for i in range(len(A))]) / len(A)))
        return errs, secs, gr

    def __bdca(self, x0, f, g, h, N):
        alpha = 0.4
        beta = 0.5
        l0 = 5
        x = x0
        steps = 0
        errs = []
        secs = []
        gr = []
        while steps < N:
            start = time.time()
            h0 = h()
            obj = lambda : g() - h0
            y = self.__aux_alg(x, obj)
            d = y - x
            l = l0
            f0 = f()
            sn = d @ d
            self.assign(y + l * d)
            while f() > f0 - alpha * l * sn:
                l = l * beta
                self.assign(y + l * d)
            x = y + l * d
            steps = steps + 1
            end = time.time()
            errs.append(self.test(data_set))
            secs.append(end - start)
            gr.append(np.linalg.norm(g() - h()))
        return errs, secs, gr

    def adam_train(self, data_set):
        d1, d2 = np.asarray([el[0] for el in data_set]), np.asarray([el[1] for el in data_set])
        lam = 1e-14
        A = []
        for i in range(len(data_set)):
            def Ai(i = i):
                difs = np.asarray([self.__cs[j] - d1[i] for j in range(self.__hsz)])
                rbfs = np.asarray([self.__rbf(difs[j] @ difs[j]) for j in range(self.__hsz)])
                mul1 = rbfs @ self.__ws - d2[i]
                mul = mul1 / math.sqrt(mul1**2 + 1)
                cgrad = 2 * difs * np.asarray([[self.__ws[j] * self.__der(difs[j] @ difs[j])] for j in range(self.__hsz)]) * mul
                wgrad = rbfs * mul + 2 * lam * self.__ws
                return np.concatenate((np.reshape(cgrad, self.__isz * self.__hsz), wgrad))
            A.append(Ai)
        x0 = np.concatenate((np.reshape(self.__cs, self.__isz * self.__hsz), self.__ws))
        r1 = self.__adam(x0, A, N)
        return r1

    def bdca_train(self, data_set):
        d1, d2 = np.asarray([el[0] for el in data_set]), np.asarray([el[1] for el in data_set])
        lam = 1e-14
        rho = 100
        f = lambda : sum(np.asarray([(self.compute(d1[i]) - d2[i])**2 for i in range(len(data_set))]))/(2*len(data_set)) + lam * self.__ws @ self.__ws
        def g():
            gs = []
            A = self.__der(0)
            B = max(0, -self.__rbf(0))
            for i in range(len(data_set)):
                difs = np.asarray([self.__cs[j] - d1[i] for j in range(self.__hsz)])
                hj1 = np.asarray([self.__rbf(difs[j]@difs[j]) - A * difs[j]@difs[j] + B for j in range(self.__hsz)])
                hj2 = np.asarray([-A * difs[j]@difs[j] + B for j in range(self.__hsz)])
                hdj1 = np.asarray([self.__der(difs[j] @ difs[j]) - A for j in range(self.__hsz)])
                hdj2 = np.asarray([-A for j in range(self.__hsz)])
                cd1 = np.asarray([hdj1[j]*(math.sqrt(self.__ws[j]**2 + 1) + self.__ws[j] + hj1[j]) + hdj2[j]*(math.sqrt(self.__ws[j]**2 + 1) + hj2[j]) for j in range(self.__hsz)])
                cd2 = np.asarray([hdj1[j]*(math.sqrt(self.__ws[j]**2 + 1) + hj1[j]) + hdj2[j]*(math.sqrt(self.__ws[j]**2 + 1) + self.__ws[j] + hj2[j]) for j in range(self.__hsz)])
                wd1 = np.asarray([(self.__ws[j] / math.sqrt(self.__ws[j]**2 + 1) + 1)*(math.sqrt(self.__ws[j]**2 + 1) + self.__ws[j] + hj1[j]) + (self.__ws[j] / math.sqrt(self.__ws[j]**2 + 1))*(math.sqrt(self.__ws[j]**2 + 1) + hj2[j]) for j in range(self.__hsz)])
                wd2 = np.asarray([(self.__ws[j] / math.sqrt(self.__ws[j]**2 + 1))*(math.sqrt(self.__ws[j]**2 + 1) + hj1[j]) + (self.__ws[j] / math.sqrt(self.__ws[j]**2 + 1) + 1)*(math.sqrt(self.__ws[j]**2 + 1) + self.__ws[j] + hj2[j]) for j in range(self.__hsz)])
                mul1 = (hj1 - hj2) @ self.__ws - d2[i]
                mul = mul1 / math.sqrt(mul1**2 + 1)
                cgradi = (2 * difs * np.asarray([[cd1[j] + cd2[j]] for j in range(self.__hsz)])) / len(data_set)
                cgradi = cgradi + 2 * difs * np.asarray([[self.__ws[j] * (hdj1[j] - hdj2[j])] for j in range(self.__hsz)]) * mul / len(data_set)
                wgradi = (wd1 + wd2) / len(data_set)
                wgradi = wgradi + np.asarray([hj1[j] - hj2[j] for j in range(self.__hsz)]) * mul / len(data_set)
                gs.append(np.concatenate((np.reshape(cgradi, self.__isz * self.__hsz), wgradi)))
            return sum(gs) + np.concatenate((np.reshape(rho * self.__cs, self.__isz * self.__hsz), (2 * lam + rho) * self.__ws))

        def h():
            gs = []
            A = self.__der(0)
            B = max(0, -self.__rbf(0))
            for i in range(len(data_set)):
                difs = np.asarray([self.__cs[j] - d1[i] for j in range(self.__hsz)])
                hj1 = np.asarray([self.__rbf(difs[j] @ difs[j]) - A * difs[j] @ difs[j] + B for j in range(self.__hsz)])
                hj2 = np.asarray([-A * difs[j] @ difs[j] + B for j in range(self.__hsz)])
                hdj1 = np.asarray([self.__der(difs[j] @ difs[j]) - A for j in range(self.__hsz)])
                hdj2 = np.asarray([-A for j in range(self.__hsz)])
                cd1 = np.asarray([hdj1[j] * (math.sqrt(self.__ws[j] ** 2 + 1) + self.__ws[j] + hj1[j]) + hdj2[j] * (math.sqrt(self.__ws[j] ** 2 + 1) + hj2[j]) for j in range(self.__hsz)])
                cd2 = np.asarray([hdj1[j] * (math.sqrt(self.__ws[j] ** 2 + 1) + hj1[j]) + hdj2[j] * (math.sqrt(self.__ws[j] ** 2 + 1) + self.__ws[j] + hj2[j]) for j in range(self.__hsz)])
                wd1 = np.asarray([(self.__ws[j] / math.sqrt(self.__ws[j] ** 2 + 1) + 1) * (math.sqrt(self.__ws[j] ** 2 + 1) + self.__ws[j] + hj1[j]) + (self.__ws[j] / math.sqrt(self.__ws[j] ** 2 + 1)) * (math.sqrt(self.__ws[j] ** 2 + 1) + hj2[j]) for j in range(self.__hsz)])
                wd2 = np.asarray([(self.__ws[j] / math.sqrt(self.__ws[j] ** 2 + 1)) * (math.sqrt(self.__ws[j] ** 2 + 1) + hj1[j]) + (self.__ws[j] / math.sqrt(self.__ws[j] ** 2 + 1) + 1) * (math.sqrt(self.__ws[j] ** 2 + 1) + self.__ws[j] + hj2[j]) for j in range(self.__hsz)])
                cgradi = (2 * difs * np.asarray([[cd1[j] + cd2[j]] for j in range(self.__hsz)])) / len(data_set)
                wgradi = (wd1 + wd2) / len(data_set)
                gs.append(np.concatenate((np.reshape(cgradi, self.__isz * self.__hsz), wgradi)))
            return sum(gs) + np.concatenate((np.reshape(rho * self.__cs, self.__isz * self.__hsz), rho * self.__ws))
        x0 = np.concatenate((np.reshape(self.__cs, self.__isz * self.__hsz), self.__ws))
        r1 = self.__bdca(x0, f, g, h, N)
        return r1


h = 10
i = 3
n = 50
N = 1000
k = 100
cd = 1
wd = 10
xd = 1
aders = []
bders = []
adime = []
bdime = []
agrad = []
bgrad = []
for attempt in range(k):
    test1 = RBFN(i, h, lambda x : math.exp(-x), lambda x : -math.exp(-x))
    test2 = RBFN(i, h, lambda x : math.exp(-x), lambda x : -math.exp(-x))
    etalon = RBFN(i, h, lambda x :math.exp(-x), lambda x : -math.exp(-x))
    c1 = np.random.uniform(-cd, cd, (h, i))
    w1 = np.random.uniform(-wd, wd, h)
    test1.assign(np.concatenate((np.reshape(c1, h*i), w1)))
    test2.assign(np.concatenate((np.reshape(c1, h*i), w1)))
    xs = [np.random.uniform(-xd, xd, i) for j in range(n)]
    ys = [etalon.compute(xs[j]) for j in range(n)]
    data_set = [[xs[j], ys[j]] for j in range(n)]
    r1 = test1.adam_train(data_set)
    r2 = test2.bdca_train(data_set)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set(xlabel='Number of iterations', ylabel='Error')
    ax2.set(xlabel='Number of iterations', ylabel='Iteration time, seconds')
    ax3.set(xlabel='Number of iterations', ylabel='Gradient')
    Ns = np.arange(1, N + 1, 1)
    ax1.plot(Ns, r1[0], color='red', label='Adam')
    ax1.plot(Ns, r2[0], color='green', label='BDCA')
    ax2.plot(Ns, r1[1], color='red', label='Adam')
    ax2.plot(Ns, r2[1], color='green', label='BDCA')
    ax3.plot(Ns, r1[2], color='red', label='Adam')
    ax3.plot(Ns, r2[2], color='green', label='BDCA')
    legend1 = ax1.legend(loc='upper right', prop={'size': 8})
    legend2 = ax2.legend(loc='upper right', prop={'size': 8})
    legend3 = ax3.legend(loc='upper right', prop={'size': 8})
    plt.tight_layout()
    plt.show()
    aders.append(r1[0])
    bders.append(r2[0])
    adime.append(r1[1])
    bdime.append(r2[1])
    agrad.append(r1[2])
    bgrad.append(r2[2])
print("maximal ultimate error for Adam is: ", max([el[-1] for el in aders]))
print("maximal ultimate error for BDCA is: ", max([el[-1] for el in bders]))
print("minimal ultimate error for Adam is: ", min([el[-1] for el in aders]))
print("minimal ultimate error for BDCA is: ", min([el[-1] for el in bders]))
print("average ultimate error for Adam is: ", sum([el[-1] for el in aders]) / k)
print("average ultimate error for BDCA is: ", sum([el[-1] for el in bders]) / k)
print("maximal iteration time for Adam is: ", max([max(el) for el in adime]))
print("maximal iteration time for BDCA is: ", max([max(el) for el in bdime]))
print("minimal iteration time for Adam is: ", min([min(el) for el in adime]))
print("minimal iteration time for BDCA is: ", min([min(el) for el in bdime]))
print("average iteration time for Adam is: ", sum([sum(el) / N for el in adime]) / k)
print("average iteration time for BDCA is: ", sum([sum(el) / N for el in bdime]) / k)
print("maximal ultimate gradient for Adam is: ", max([el[-1] for el in agrad]))
print("maximal ultimate gradient for BDCA is: ", max([el[-1] for el in bgrad]))
print("minimal ultimate gradient for Adam is: ", min([el[-1] for el in agrad]))
print("minimal ultimate gradient for BDCA is: ", min([el[-1] for el in bgrad]))
print("average ultimate gradient for Adam is: ", sum([el[-1] for el in agrad]) / k)
print("average ultimate gradient for BDCA is: ", sum([el[-1] for el in bgrad]) / k)
Ns = np.arange(1, N + 1, 1)
fig1, avers = plt.subplots(1, 1)
avers.set(xlabel='Number of iterations', ylabel='Mean Error')
avers.plot(Ns, sum([np.asarray(el) for el in aders]) / k, color='red', label='Adam')
avers.plot(Ns, sum([np.asarray(el) for el in bders]) / k, color='green', label='BDCA')
legend1 = avers.legend(loc='upper right', prop={'size': 8})
plt.show()
fig2, avime = plt.subplots(1, 1)
avime.set(xlabel='Number of iterations', ylabel='Mean Iteration time, seconds')
avime.plot(Ns, sum([np.asarray(el) for el in adime]) / k, color='red', label='Adam')
avime.plot(Ns, sum([np.asarray(el) for el in bdime]) / k, color='green', label='BDCA')
legend2 = avime.legend(loc='upper right', prop={'size': 8})
plt.show()
fig3, avrad = plt.subplots(1, 1)
avrad.set(xlabel='Number of iterations', ylabel='Mean Gradient')
avrad.plot(Ns, sum([np.asarray(el) for el in agrad]) / k, color='red', label='Adam')
avrad.plot(Ns, sum([np.asarray(el) for el in bgrad]) / k, color='green', label='BDCA')
legend3 = avrad.legend(loc='upper right', prop={'size': 8})
plt.show()