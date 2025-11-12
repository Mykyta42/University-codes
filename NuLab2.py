import numpy as np

a = 5
eps = 1.0e-10

def gauss():
    try:
        n = int(input("type size of the matrix: "))
        t = input("Choose type of the matrix: ")
        if t == 'H':
            A = np.asarray([[1/(i+j+1) for i in range(n)] for j in range(n)])
        else:
            A = 2*a*(np.random.random((n, n))-1/2)
        x = 2*a*(np.random.random(n)-1/2)
        b = A @ x
        print("Matrix", '\n', A)
        print("constant vector", '\n', b)
        print("Expected solution", '\n', x)
        for i in range(n):
            m = abs(A[i][i])
            j0 = i
            for j in range(i, n):
                if abs(A[j][i]) > m:
                    j0 = j
                    m = abs(A[j][i])
            A[[i, j0]] = A[[j0, i]]
            b[[i, j0]] = b[[j0, i]]
            M = np.identity(n)
            M[i][i] = 1/A[i][i]
            for j in range(i+1, n):
                M[j][i] = -A[j][i]/A[i][i]
            A = M @ A
            b = M @ b
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                b[i] = b[i] - A[i][j]*b[j]
        print("Obtained solution", '\n', b)
        print("Error: ", max(abs(x-b)))
    except:
        print("Something went wrong. Better luck next time!")



def tma():
    try:
        n = int(input("type size of the matrix: "))
        A = np.zeros((n, n))
        for i in range(n):
            A[i][i]= 2 * a * (np.random.random() - 1 / 2)
        for i in range(n-1):
            A[i][i+1]= 2 * a * (np.random.random() - 1 / 2)
        for i in range(n-1):
            A[i+1][i]= 2 * a * (np.random.random() - 1 / 2)
        x = 2*a*(np.random.random(n)-1/2)
        b = A @ x
        print("Matrix", '\n', A)
        print("constant vector", '\n', b)
        print("Expected solution", '\n', x)
        d = []
        e = []
        d.append(-A[0][1]/A[0][0])
        e.append(b[0]/A[0][0])
        for i in range(1, n-1):
            d.append(-A[i][i+1] / (A[i][i] + d[i-1]*A[i][i-1]))
            e.append(-(-b[i] + A[i][i-1]*e[i-1]) / (A[i][i] + d[i-1]*A[i][i-1]))
        x0 = np.zeros(n)
        x0[n-1] = -(-b[n-1] + A[n-1][n-2]*e[n-2]) / (A[n-1][n-1] + d[n-2]*A[n-1][n-2])
        for i in range(n-2, -1, -1):
            x0[i] = d[i]*x0[i+1] + e[i]
        print("Obtained solution", '\n', x0)
        print("Error: ", max(abs(x-x0)))
    except:
        print("Something went wrong. Better luck next time!")

def jacobi():
    try:
        n = int(input("type size of the matrix: "))
        A = 2 * a * (np.random.random((n, n)) - 1 / 2)
        for i in range(n):
            A[i][i] = (sum(abs(A[i])) - abs(A[i][i]))*(2*np.random.random()*a/(n-1)+1)*(2*np.random.randint(2)-1)
        x = 2 * a * (np.random.random(n) - 1 / 2)
        b = A @ x
        print("Matrix", '\n', A)
        print("constant vector", '\n', b)
        print("Expected solution", '\n', x)
        N = 0
        D = np.diag(np.diag(A))
        Di = np.diag(1/np.diag(A))
        x0 = np.zeros(n)
        x1 = Di @ ( - (A - D) @ x0 + b)
        while(max(abs(x0-x1)) >= eps):
            x0, x1 = x1, Di @ ( - (A - D) @ x1 + b)
            N += 1
        print("Obtained solution", '\n', x1)
        print("Amount of iterations: ", N)
        print("Error: ", max(abs(x - x1)))
    except:
        print("Something went wrong. Better luck next time!")

def seidel():
    try:
        n = int(input("type size of the matrix: "))
        t = input("Choose type of the matrix: ")
        A = 2 * a * (np.random.random((n, n)) - 1 / 2)
        if t == 'H':
            A = np.asarray([[1 / (i + j + 1) for i in range(n)] for j in range(n)])
        else:
            A = A @ A.transpose()
        x = 2 * a * (np.random.random(n) - 1 / 2)
        b = A @ x
        print("Matrix", '\n', A)
        print("constant vector", '\n', b)
        print("Expected solution", '\n', x)
        N = 0
        D = np.diag(np.diag(A))
        A1 = np.tril(A, -1)
        x0 = np.zeros(n)
        x1 =  x0 - np.linalg.inv(A1 + D) @ A @ x0 + np.linalg.inv(A1 + D) @ b
        while (max(abs(x0 - x1)) >= eps):
            x0, x1 = x1,  x1 - np.linalg.inv(A1 + D) @ A @ x1 + np.linalg.inv(A1 + D) @ b
            N += 1
        print("Obtained solution", '\n', x1)
        print("Amount of iterations: ", N)
        print("Error: ", max(abs(x - x1)))
    except:
        print("Something went wrong. Better luck next time!")





disp = {1: gauss, 2: tma, 3: jacobi, 4: seidel}


active = -1
while active:
    print()
    active = int(input("Choose a method or quit: "))
    if active not in {1, 2, 3, 4}:
        continue
    disp[active]()
