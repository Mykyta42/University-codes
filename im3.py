import numpy as np
from PIL import Image
import os
from re import search


def inv(im):
    length, width = np.shape(im)
    m00 = im.sum()
    m10 = (im * np.asarray([[i for j in range(width)] for i in range(length)])).sum()
    m01 = (im * np.asarray([[j for j in range(width)] for i in range(length)])).sum()
    xm = m10 / m00
    ym = m01 / m00
    n20 = (im * np.asarray(
        [[(i - xm) ** 2 for j in range(width)] for i in range(length)])).sum() / m00 ** 2
    n02 = (im * np.asarray(
        [[(j - ym) ** 2 for j in range(width)] for i in range(length)])).sum() / m00 ** 2
    Phi1 = n20 + n02
    n30 = (im * np.asarray(
        [[(i - xm) ** 3 for j in range(width)] for i in range(length)])).sum() / m00 ** 2.5
    n12 = (im * np.asarray(
        [[(i - xm) * (j - ym) ** 2 for j in range(width)] for i in range(length)])).sum() / m00 ** 2.5
    n03 = (im * np.asarray(
        [[(j - ym) ** 3 for j in range(width)] for i in range(length)])).sum() / m00 ** 2.5
    n21 = (im * np.asarray(
        [[(j - ym) * (i - xm) ** 2 for j in range(width)] for i in range(length)])).sum() / m00 ** 2.5
    Phi2 = (n30 + n12) ** 2 + (n03 + n21) ** 2
    return np.asarray([Phi1, Phi2])

# initialization
train = {}
test = {}
# get picture and compute invariants
for name in os.listdir('images'):
    p = f = os.path.join('images', name)
    image = Image.open(p)
    image = image.convert("P")
    matrix = np.asarray(image)
    vals = inv(matrix)
    if not search(r"\d", name):
        train[name] = vals
    else:
        test[name] = vals
for name in test.keys():
    letter_name = ([key for key in train.keys() if np.linalg.norm(train[key] - test[name]) == min([np.linalg.norm(train[key1] - test[name]) for key1 in train.keys()])])[0]
    print(f"The picture {name} depicts a letter {letter_name[:letter_name.find('.')].upper()}")
