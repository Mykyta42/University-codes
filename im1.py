import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

# initialization
x1 = 0
x2 = 30
C1 = 1.0
C2 = 1.0
f1 = np.vectorize(lambda x: C1 * np.log2(1.0 + x))
f2 = np.vectorize(lambda x: 0 if x <= x1 else C2 * (x - x1) / (x2 - x1) if x1 <= x <= x2 else C2)
# get picture
p = f = os.path.join('image', 'cameraman.tif')
image = Image.open(p)
matrix = np.asarray(image)
# picture proceeding and showing
plt.imshow(matrix, cmap='gray')
plt.axis('off')
plt.show()
matrix1 = f1(matrix)
plt.imshow(matrix1, cmap='gray')
plt.axis('off')
plt.show()
matrix2 = f2(matrix)
plt.imshow(matrix2, cmap='gray')
plt.axis('off')
plt.show()
histo = np.zeros(256)
xs = np.linspace(0, 255, 256)
for pixel in matrix:
     histo[pixel] += 1
plt.plot(xs, histo)
plt.show()