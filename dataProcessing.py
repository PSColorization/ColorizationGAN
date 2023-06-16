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
    npzFiles = list(glob(f"{os.path.join(npzFolderPath, '**/*.npz')}", recursive=True))
    # random.shuffle(npzFiles)
    npzFiles = npzFiles[:500]
    # print(npzFiles)

    X_gray, X_hue, y = [], [], []

    for npzFile in tqdm(npzFiles):
        loaded_npz = np.load(npzFile)

        grayPic = loaded_npz['gray']
        rgbPic = loaded_npz['hsv']
        rgbPic = cv2.cvtColor(rgbPic, cv2.COLOR_HSV2RGB)
        # hue = np.array(list(loaded_npz['palette'][:, :1].reshape((10,)))*256*256)
        hue = np.array(list(loaded_npz['palette'][:, :1].reshape((10,))) * 256 * 256)
        # hue = normalizeHSV(hue)
        hue.resize((256, 256, 10))
        rgbPic = cv2.resize(rgbPic, (128, 128))

        grayPic = cv2.resize(grayPic, (128, 128))

        X_gray.append(grayPic)
        X_hue.append(hue)
        y.append(rgbPic)

    X_gray = np.array(X_gray)
    X_gray = np.expand_dims(X_gray, axis=-1)
    X_gray = np.broadcast_to(X_gray, (500, 128, 128, 3))

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