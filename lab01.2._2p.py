import random
import matplotlib.pyplot as plt
import numpy as np
import math  # For the second part


# The second part. Based on real events
a = 0
b = 100
fav_num = 5
aux_list = [i for i in range(a, b+1) if i % fav_num == 0]
print("For better experience you can relaunch the program. \n")
print(f"""Your mathematical analysis lector decided to give you a random mark from {a} to {b}.
So your mark is {random.randint(a, b)}.""")


print(f"""But also he prefers numbers, which are divisible on {fav_num}.
So in this case your mark is {random.choice(aux_list)}""")


min_size = 2
max_size = 12
k = random.randint(min_size, max_size)
hint_list = random.sample(aux_list, random.randint(min_size, max_size))
print(f"""He wants to give you a hint about possibles marks.
So here are some potential marks: {hint_list}""")


x = random.random()  # Some number 0<=x<1
answer = math.sin(x) <= x  # It's well-known this inequality is always true
print(f'for {x =: } the inequality sin(x) <= x is {answer}')


def f(a, b, c, d, x):
    return x**4 + a*x**3 + b*x**2 + c*x + d


mini = float(input('Enter left bound here '))
maxi = float(input('Enter right bound here '))
a = float(input('Enter x^3 coefficient here '))
b = float(input('Enter x^2 coefficient here '))
c = float(input('Enter x coefficient here '))
d = float(input('Enter coefficient here '))
x = np.array(sorted([random.uniform(mini, maxi) for i in range(50)]))
y = f(a, b, c, d, x)
y_min = y[0]
y_max = y[0]
i = 1
while i < 50:
    z = x[i]
    if y_min > z:
        y_min = z
    if y_max < z:
        y_max = z
    i = i + 1
if y_min*y_max > 0:
    print(f'Probably the equation x^4 + {a}x^3 + {b}x^2 + {c}x + {d} = 0 doesn\'t have real solutions')
else:
    print(f'The equation x^4 + {a}x^3 + {b}x^2 + {c}x + {d} = 0 has a real solution')
plt.plot(x, y, color='green', marker='o')
plt.show()
print("In conclusion one can say: considered library describes methods for work with pseudo random values.")
print("It can be applied to such values as integers, float and lists.")
print("Also some methods let you put restrictions onto some variables, but they will be chosen random")
print("But they will be chosen randomly anyway.")
print("The end of the second part.")
