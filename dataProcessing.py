import tensorflow as tf
from glob import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import cv2


def normalizeHSV(x):
        x[:, 0] /= 180
        x[:, 1] /= 255
        x[:, 2] /= 255
        return x

def get_X_y(npzFolderPath):
    npzFiles = sorted(glob(f"{os.path.join(npzFolderPath, '*.npz')}"))[:500]

    X_gray, X_hue, y = [], [], []

    for npzFile in npzFiles:
        loaded_npz = np.load(npzFile)

        grayPic = loaded_npz['gray']
        rgbPic = loaded_npz['hsv']
        rgbPic = cv2.cvtColor(rgbPic, cv2.COLOR_HSV2RGB)
        hue = loaded_npz['palette']#[:, :2]#.reshape((10,1))
        hue = normalizeHSV(hue)

        X_gray.append(grayPic)
        X_hue.append(hue)
        y.append(rgbPic)

    X_gray = np.array(X_gray)
    X_hue = np.array(X_hue)
    y = np.array(y)

    return (X_gray, X_hue), y
