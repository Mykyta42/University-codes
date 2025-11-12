import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt


def act(u):  # activation function
    return 1 if u >= 0 else -1


def dig(ar):  # transform pixel into number
    return 0 if ar == white else 1


def rec(t):  # transform number into pixel
    return black if t == 1 else white


height = 5
width = 5
deep = 3
S1 = height * width
zeta = 2
ro = 0.6
S2 = 3
W21 = np.ones((S1, S2))
W12 = np.ones((S2, S1)) * (zeta / (zeta + S1 - 1))
W = np.zeros((S1, S1))
black, white = 0x00, 0xFF
dv = np.vectorize(dig)
fv = np.vectorize(act)
cv = np.vectorize(rec)
# get training set
train = [np.asarray([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]).reshape(S1, 1), np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]).reshape(S1, 1), np.asarray([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(S1, 1), np.asarray([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]).reshape(S1, 1), np.asarray([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(S1, 1)]
# model learning
for k in range(3):
      for p in train:
            # step 1 - competition with checking
            no_win = True
            y =np.zeros(S1)
            j = 0
            disq = []
            while(no_win):
                  m = -1
                  for i in range(S2):
                        if m < (W12[i].T @ p)[0] and i not in disq:
                              j, m = i, (W12[i].T @ p)[0]
                  y = W21[:, j].reshape(S1, 1) * p
                  if la.norm(y,1) / la.norm(p, 1) >= ro:
                        no_win = False
                  else:
                        disq.append(j)
            # step 2 - upgrade
            y = y.reshape(S1)
            W12[j] = (zeta / (zeta + la.norm(y,1) - 1)) * y
            W21[:, j] = y
            # step 3 - illustration
            for i in range(S2):
                  vector = (cv(W21[:, i])).reshape(height, width)
                  pic = np.zeros((height, width, deep), dtype=int)
                  for i in range(deep):
                        pic[:, :, i] = vector
                  plt.imshow(pic)
                  plt.axis('off')
                  plt.show()
