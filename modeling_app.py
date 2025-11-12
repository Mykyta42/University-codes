import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sympy as sp
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings("ignore")

x1, x2, t, x1p, x2p, tp = sp.symbols("x1 x2 t x1p x2p tp")
f = sp.Function('f')(x1, x2, t)


class FunctionPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dialog Box")
        # Variables
        self.function_str = tk.StringVar(value="cos(x1)**2 + sin(x2)**2 + t")
        self.x1_min = tk.DoubleVar(value=0)
        self.x1_max = tk.DoubleVar(value=2)
        self.x2_min = tk.DoubleVar(value=0)
        self.x2_max = tk.DoubleVar(value=2)
        self.T_value = tk.DoubleVar(value=2)
        self.num_boundary_operators = tk.IntVar(value=1)
        self.num_initial_operators = tk.IntVar(value=1)
        self.k_value = tk.DoubleVar(value=1)
        self.boundary_points_count = tk.IntVar(value=1)
        self.initial_points_count = tk.IntVar(value=1)
        self.boundary_operators = []
        self.initial_operators = []
        self.boundary_points = []
        self.initial_points = []


        # UI Layout
        self.create_widgets()

    def create_widgets(self):
        # Entry for function
        ttk.Label(self.root, text="Y function:").grid(row=0, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.function_str, width=30).grid(row=0, column=1, columnspan=2, sticky='we')

        # Range inputs
        ttk.Label(self.root, text="Minimum value of x1:").grid(row=1, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.x1_min).grid(row=1, column=1, sticky='we')

        ttk.Label(self.root, text="Maximum value of x1:").grid(row=2, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.x1_max).grid(row=2, column=1, sticky='we')

        ttk.Label(self.root, text="Minimum value of x2:").grid(row=3, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.x2_min).grid(row=3, column=1, sticky='we')

        ttk.Label(self.root, text="Maximum value of x2:").grid(row=4, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.x2_max).grid(row=4, column=1, sticky='we')

        # T and k values
        ttk.Label(self.root, text="Value of T:").grid(row=5, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.T_value).grid(row=5, column=1, sticky='we')

        ttk.Label(self.root, text="Value of k:").grid(row=6, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.k_value).grid(row=6, column=1, sticky='we')

        # Number of boundary and initial operators
        ttk.Label(self.root, text="Number of Boundary Operators:").grid(row=7, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.num_boundary_operators).grid(row=7, column=1, sticky='we')

        ttk.Label(self.root, text="Number of Initial Operators:").grid(row=8, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.num_initial_operators).grid(row=8, column=1, sticky='we')

        # Number of boundary and interface points
        ttk.Label(self.root, text="Number of Points of Boundary Operator:").grid(row=9, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.boundary_points_count).grid(row=9, column=1, sticky='we')

        ttk.Label(self.root, text="Number of Points of Initial Operator:").grid(row=10, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.initial_points_count).grid(row=10, column=1, sticky='we')

        # Buttons for updating operator and point fields
        ttk.Button(self.root, text="Enter Operators", command=self.update_operators).grid(row=11, column=0, sticky='we')
        ttk.Button(self.root, text="Enter Points", command=self.update_points).grid(row=11, column=1, sticky='we')
        ttk.Button(self.root, text="Clear All Fields", command=self.clear_conditions).grid(row=11, column=2, sticky='we')

        # Plot Button
        ttk.Button(self.root, text="Get Values, Solve Problem, Plot Graph", command=self.GSP).grid(row=12, column=0, columnspan=3, sticky='we')

        # Frame for Operators and Points
        self.conditions_frame = ttk.Frame(self.root)
        self.conditions_frame.grid(row=13, column=0, columnspan=3, sticky='we')


    def update_operators(self):
        # Clear existing operator entries
        for widget in self.conditions_frame.winfo_children():
            widget.destroy()
        self.boundary_operators.clear()
        self.initial_operators.clear()

        # Create boundary operator entries
        for i in range(self.num_boundary_operators.get()):
            ttk.Label(self.conditions_frame, text=f"Boundary Operator {i + 1}:").grid(row=i, column=0, sticky='w')
            operator = tk.StringVar()
            self.boundary_operators.append(operator)
            ttk.Entry(self.conditions_frame, textvariable=operator).grid(row=i, column=1, ipadx=90,  sticky='we')

        # Create initial operator entries
        offset = self.num_boundary_operators.get()
        for i in range(self.num_initial_operators.get()):
            ttk.Label(self.conditions_frame, text=f"Initial Operator {i + 1}:").grid(row=offset + i, column=0, sticky='w')
            operator = tk.StringVar()
            self.initial_operators.append(operator)
            ttk.Entry(self.conditions_frame, textvariable=operator).grid(row=offset + i, column=1, ipadx=90, sticky='we')

    def update_points(self):
        # Clear existing point entries
        for widget in self.conditions_frame.winfo_children():
            widget.destroy()
        self.boundary_points.clear()
        self.initial_points.clear()

        # Create boundary points entries based on boundary_points_count
        for i in range(self.boundary_points_count.get()):
            ttk.Label(self.conditions_frame, text=f"Boundary Operator Points {i + 1}:").grid(row=i, column=2, sticky='w')
            point = tk.StringVar(value="t, x1, x2")
            self.boundary_points.append(point)
            ttk.Entry(self.conditions_frame, textvariable=point).grid(row=i, column=3, sticky='we')

        # Create initial points entries based on initial_points_count
        offset = self.boundary_points_count.get()
        for i in range(self.initial_points_count.get()):
            ttk.Label(self.conditions_frame, text=f"Initial Operator Points {i + 1}:").grid(row=offset + i, column=2, sticky='w')
            point = tk.StringVar(value="t, x1, x2")
            self.initial_points.append(point)
            ttk.Entry(self.conditions_frame, textvariable=point).grid(row=offset + i, column=3, sticky='we')

    def clear_conditions(self):
        for widget in self.conditions_frame.winfo_children():
            widget.destroy()

    def __get_data(self):
        # receive and convert data
        y = self.function_str.get()
        y = sp.sympify(y)
        a1 = self.x1_min.get()
        b1 = self.x1_max.get()
        a2 = self.x2_min.get()
        b2 = self.x2_max.get()
        T = self.T_value.get()
        k = self.k_value.get()
        l0 = [sp.sympify(op.get()) for op in self.initial_operators]
        lb = [sp.sympify(op.get()) for op in self.boundary_operators]
        p0 = [[float(el) for el in po.get().strip().split(',')] for po in self.initial_points]
        pb = [[float(el) for el in po.get().strip().split(',')] for po in self.boundary_points]
        n0 = self.initial_points_count.get()
        nb = self.boundary_points_count.get()
        m0 = self.num_initial_operators.get()
        mb = self.num_boundary_operators.get()
        data = [y, a1, b1, a2, b2, T, k, l0, lb, p0, pb, n0, nb, m0, mb]
        return data

    def __apprint(self, f, x11, x12, x21, x22, t1, t2):
        # approximated integration
        N = 10
        x1n = np.linspace(x11, x12, N + 1)
        x2n = np.linspace(x21, x22, N + 1)
        tn = np.linspace(t1, t2, N + 1)
        h1 = (x12 - x11) / N
        h2 = (x22 - x21) / N
        h3 = (t2 - t1) / N
        S = 0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    S += h1 * h2 * h3 * float(f.subs({x1: (x1n[i1] + x1n[i1 + 1]) / 2, x2: (x2n[i2] + x2n[i2 + 1]) / 2,
                                                      t: (tn[i3] + tn[i3 + 1]) / 2}))
        return S


    def __aprint_f(self, f, x11, x12, x21, x22, t1, t2):
        # function as approximated integral
        N = 5
        x1n = np.linspace(x11, x12, N + 1)
        x2n = np.linspace(x21, x22, N + 1)
        tn = np.linspace(t1, t2, N + 1)
        h1 = (x12 - x11) / N
        h2 = (x22 - x21) / N
        h3 = (t2 - t1) / N
        S = 0
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    S += h1 * h2 * h3 * f.subs(
                        {x1p: (x1n[i1] + x1n[i1 + 1]) / 2, x2p: (x2n[i2] + x2n[i2 + 1]) / 2,
                         tp: (tn[i3] + tn[i3 + 1]) / 2})
        return S

    def __solver(self, data):
        # find solution, determine its uniqueness and error
        y, a1, b1, a2, b2, T, k, l0, lb, p0, pb, n0, nb, m0, mb = data
        G = sp.Piecewise((1 / sp.sqrt(4 * sp.pi * k * t) * sp.exp(- (x1 ** 2 + x2 ** 2) / (4 * k * t)), t > 0.001), (0, t <= 0.001))
        u = sp.diff(y, t) - k * (sp.diff(y, x1, x1) + sp.diff(y, x2, x2))
        Y0 = [[float(l0[i].replace(f, y).doit().subs({t: p0[j][0], x1: p0[j][1], x2: p0[j][2]})) for j in range(n0)] for i in
              range(m0)]
        Yb = [[float(lb[i].replace(f, y).doit().subs({t: p0[j][0], x1: p0[j][1], x2: p0[j][2]})) for j in range(nb)] for i in
              range(mb)]
        Mx1 = 100
        Mx2 = 100
        Mt = 100
        Goo = G.subs({t: t - tp, x1: x1 - x1p, x2: x2 - x2p})
        up = u.subs({t: tp, x1: x1p, x2: x2p})
        yinf = self.__aprint_f(Goo * up, a1, b1, a2, b2, 0, T)
        func_0 = np.zeros((m0, n0), dtype=sp.Symbol)
        func_b = np.zeros((mb, nb), dtype=sp.Symbol)
        yf0 = np.zeros((m0, n0))
        yfb = np.zeros((mb, nb))
        for l in range(m0):
            for r in range(n0):
                Glr = G.subs({t: t - tp, x1: x1 - x1p, x2: x2 - x2p})
                func_0[l][r] = (l0[l].replace(f, Glr).doit()).subs(
                    {t: p0[r][0], x1: p0[r][1], x2: p0[r][2], tp: t, x1p: x1, x2p: x2})
                yf0[l][r] = float(Y0[l][r] - l0[l].replace(f, yinf).doit().subs({t: 0, x1: p0[r][0], x2: p0[r][1]}))
        for l in range(mb):
            for r in range(nb):
                Glr = G.subs({t: t - tp, x1: x1 - x1p, x2: x2 - x2p})
                func_b[l][r] = (lb[l].replace(f, Glr).doit()).subs(
                    {t: pb[r][0], x1: pb[r][1], x2: pb[r][2], tp: t, x1p: x1, x2p: x2})
                yfb[l][r] = float(
                    Yb[l][r] - lb[l].replace(f, yinf).doit().subs({t: pb[r][0], x1: pb[r][1], x2: pb[r][1]}))
        func_0 = np.array(func_0).reshape(m0 * n0)
        func_b = np.array(func_b).reshape(mb * nb)
        F0 = np.concat((func_0, func_b))
        yf0 = yf0.reshape(m0 * n0)
        yfb = yfb.reshape(mb * nb)
        Y = np.concat((yf0, yfb))
        n = m0 * n0 + mb * nb
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                F = F0[i] * F0[j]
                I1 = self.__apprint(F, a1, b1, a2, b2, -Mt, 0)
                I21 = self.__apprint(F, -Mx1, Mx1, b2, Mx2, 0, T)
                I22 = self.__apprint(F, -Mx1, a1, a2, b2, 0, T)
                I23 = self.__apprint(F, b1, Mx1, a2, b2, 0, T)
                I24 = self.__apprint(F, -Mx1, Mx1, -Mx2, a2, 0, T)
                P[i][j] = I1 + I21 + I22 + I23 + I24
        Pp = la.pinv(P)
        u0 = F0 @ Pp @ Y
        ub = F0 @ Pp @ Y
        N = 21
        test = np.zeros((2 * N, 2 * N))
        blocks = []
        ts1 = np.linspace(-Mt, 0, N)
        x1s1 = np.linspace(a1, b1, N)
        x2s1 = np.linspace(a2, b2, N)
        ts2 = np.linspace(0, T, N)
        x1s2 = np.linspace(-Mx1, Mx1, N)
        x2s2 = np.linspace(-Mx2, Mx2, N)
        for k in range(N):
            block = np.zeros((n, 2))
            for i in range(n):
                block[i][0] = float(F0[i].subs({x1: x1s1[i], x2: x2s1[i], t: ts1[i]}))
                block[i][1] = float(F0[i].subs({x1: x1s2[i], x2: x2s2[i], t: ts2[i]}))
            blocks.append(block)
        for i in range(N):
            for j in range(N):
                test[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = np.transpose(blocks[i]) @ blocks[j]
        if la.det(test) > 0:
            phrase = "This solution is unique"
        else:
            phrase = "This solution is not unique"
        sqer = np.transpose(Y) @ Y - np.transpose(Y) @ np.transpose(P) @ Pp @ Y
        u0p = u0.subs({x1: x1p, x2: x2p, t: tp})
        ubp = ub.subs({x1: x1p, x2: x2p, t: tp})
        int1 = self.__aprint_f(Goo * u0p, a1, b1, a2, b2, -Mt, 0)
        int2 = self.__aprint_f(Goo * ubp, -Mx1, Mx1, b2, Mx2, 0, T)
        int3 = self.__aprint_f(Goo * ubp, -Mx1, a1, a2, b2, 0, T)
        int4 = self.__aprint_f(Goo * ubp, b1, Mx1, a2, b2, 0, T)
        int5 = self.__aprint_f(Goo * ubp, -Mx1, Mx1, -Mx2, a2, 0, T)
        x0 = yinf + int1 + int2 + int3 + int4 + int5
        return [x0, phrase, sqer]


    def __plotter(self, data, res):
        # plt graphics
        y, a1, b1, a2, b2, T, k, l0, lb, p0, pb, n0, nb, m0, mb = data
        ya, text, eps = res
        N = 41
        Nt = 17
        x1r = np.linspace(a1, b1, N)
        x2r = np.linspace(a2, b2, N)
        tr = np.linspace(0, T, Nt)
        x1r, x2r = np.meshgrid(x1r, x2r)
        f0 = np.vectorize(sp.lambdify((x1, x2, t), y))
        f0a = np.vectorize(sp.lambdify((x1, x2, t), ya))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        Z = 5
        ax.set_zlim(-Z, Z)
        frames = {ti: f0(x1r, x2r, ti) for ti in tr}
        framesa = {ti: f0a(x1r, x2r, ti) for ti in tr}
        ur1 = frames[tr[0]]
        ur2 = framesa[tr[0]]
        ax.plot_surface(x1r, x2r, ur1, label='original')
        ax.plot_surface(x1r, x2r, ur2, label='obtained')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        fig.subplots_adjust(left=0.25, bottom=0.25)
        legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
        my_text = f'{text}.\nThe square of the error is {eps:.5f}'
        props = dict(boxstyle='round', facecolor='white')
        ax.text2D(0.85, 1.00, my_text, transform=ax.transAxes, fontsize=6, verticalalignment='top', bbox=props)
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        t_slider = Slider(
            ax=axtime,
            label='Time',
            valmin=0,
            valmax=T,
            valinit=0,
            valstep=tr
        )

        def update(val):
            ax.cla()
            ax.set_zlim(-Z, Z)
            ax.plot_surface(x1r, x2r, frames[t_slider.val], label='original')
            ax.plot_surface(x1r, x2r, framesa[t_slider.val], label='obtained')
            legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
            ax.text2D(0.85, 1.00, my_text, transform=ax.transAxes, fontsize=6, verticalalignment='top', bbox=props)

        t_slider.on_changed(update)
        plt.show()

    def GSP(self):
        data = self.__get_data()
        res = self.__solver(data)
        self.__plotter(data, res)


root = tk.Tk()
app = FunctionPlotterApp(root)
root.mainloop()
