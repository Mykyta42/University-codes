import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt


def act(u):  # activation function
    return 1 if u >= 0 else -1


def dig(ar):  # transform pixel into number
    return -1 if ar == white else 1


def rec(t):  # transform number into pixel
    return black if t == 1 else white


height = 6
width = 5
deep = 3
mn = height * width
W = np.zeros((mn, mn))
black, white = 0, 255
train = []
test1 = []
test2 = []
dv = np.vectorize(dig)
fv = np.vectorize(act)
cv = np.vectorize(rec)
# get training set
for pict in os.listdir('train'):
      p = f = os.path.join('train',pict)
      image = Image.open(p)
      vector = (dv(cv(dv(np.asarray(image)[:,:,1])))).reshape(mn, 1)
      train.append(vector)
# get halves
for pict in os.listdir('halves'):
      p = f = os.path.join('halves',pict)
      image = Image.open(p)
      vector = (dv(cv(dv(np.asarray(image)[:,:,1])))).reshape(mn, 1)
      test1.append(vector)
# get thirds
for pict in os.listdir('thirds'):
      p = f = os.path.join('thirds',pict)
      image = Image.open(p)
      vector = (dv(cv(dv(np.asarray(image)[:,:,1])))).reshape(mn, 1)
      test2.append(vector)
# learn by Hebb's principle
for el in train:
      W += el @ el.T
# try to recognize halves
for el in test1:
      vector = (cv(fv(W @ el))).reshape(height, width)
      pic = np.zeros((height, width, deep), dtype=int)
      for i in range(deep):
            pic[:, :, i] = vector
      plt.imshow(pic)
      plt.axis('off')
      plt.show()
# try to recognize thirds
for el in test2:
      vector = (cv(fv(W @ el))).reshape(height, width)
      pic = np.zeros((height, width, deep), dtype = int)
      for i in range(deep):
            pic[:, :, i] = vector
      plt.imshow(pic)
      plt.axis('off')
      plt.show()
