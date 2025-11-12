import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from math import sqrt

# initialization
x1 = 10
x2 = 50
C1 = 1.0
C2 = 1.0
f1 = np.vectorize(lambda x: 0 if x <= x1 else C2 * (x - x1) / (x2 - x1) if x1 <= x <= x2 else C2)
# get picture
p = f = os.path.join('image', 'cameraman.tif')
image = Image.open(p)
matrix = np.asarray(image)
#histogram
histo = np.zeros(256)
xs = np.linspace(0, 255, 256)
for pixel in matrix:
     histo[pixel] += 1
plt.plot(xs, histo)
plt.show()
# picture proceeding and showing
plt.imshow(matrix, cmap='gray')
plt.axis('off')
plt.show()
matrix1 = f1(matrix)
plt.imshow(matrix1, cmap='gray')
plt.axis('off')
plt.show()
length, width = np.shape(matrix)
N = 3
Gx = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Gy = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
matrix2 = np.zeros((length - (N // 2) * 2, width - (N // 2) * 2))
for i in range(N // 2, length - N // 2):
    for j in range(N // 2, width - N // 2):
        sub = matrix1[np.ix_([i - 1, i, i + 1], [j - 1, j, j + 1])]
        dx = (Gx * sub).sum()
        dy = (Gy * sub).sum()
        matrix2[i - 1][j - 1] = sqrt(dx * dx + dy * dy)
plt.imshow(matrix2, cmap='gray')
plt.axis('off')
plt.show()
for i in range(length - 2 * (N // 2)):
    for j in range(width - 2 * (N // 2)):
        matrix2[i][j] = 255 - matrix2[i][j]
plt.imshow(matrix2, cmap='gray')
plt.axis('off')
plt.show()