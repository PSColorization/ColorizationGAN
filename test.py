import numpy as np


def preparePalet(palet, img_size):
    layers = list()
    for p in palet:
        layer = [[p for _ in range(img_size)] for _ in range(img_size)]
        layers.append(layer)
    resultArr = np.array(layers)

    return resultArr


def prepareImage(img, palet_size):
    cube = np.array(list(img) * palet_size)
    cube.resize((palet_size, len(img), len(img)))

    return cube


palet = np.array([1, 2, 3, 4, 5])

grayScale = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [15, 25, 35, 45],
    [55, 65, 75, 85]
])

preparePalet(palet=palet, img_size=4)
prepareImage(img=grayScale, palet_size=5)
