import tensorflow as tf
from glob import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy import moveaxis
import os
import PIL
from tensorflow.keras import layers
import time
import cv2
import random
from tqdm import tqdm


def normalizeHSV(x):
        x[:, 0] /= 180
        x[:, 1] /= 255
        x[:, 2] /= 255
        return x

def get_X_y(npzFolderPath):
    npzFiles = list(glob(f"{os.path.join(npzFolderPath,'**/*.npz')}", recursive=True))
    random.shuffle(npzFiles)
    npzFiles = npzFiles[:500]
    # print(npzFiles)

    X_gray, X_hue, y = [], [], []

    for npzFile in tqdm(npzFiles):
        loaded_npz = np.load(npzFile)

        grayPic = loaded_npz['gray']
        grayPic = prepareImage(grayPic, 10)
        rgbPic = loaded_npz['hsv']
        rgbPic = moveaxis(rgbPic, 2, 0)
        # print(rgbPic.shape)
        # print(grayPic.shape)
        # rgbPic = cv2.cvtColor(rgbPic, cv2.COLOR_HSV2RGB)
        # hue = np.array(list(loaded_npz['palette'][:, :1].reshape((10,)))*256*256)
        hue = preparePalet(loaded_npz['palette'][:, :1].reshape((10,)), 256)
        # hue = normalizeHSV(hue)
        # hue.resize((256, 256, 10))

        X_gray.append(grayPic)
        X_hue.append(hue)
        y.append(rgbPic)
        # break

    X_gray = np.array(X_gray)
    X_hue = np.array(X_hue)
    y = np.array(y)

    return (X_gray, X_hue), y
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


if __name__ == '__main__':
    get_X_y("./TEST/")