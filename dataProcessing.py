from glob import glob

import numpy as np
import os



def get_X_y(npzFolderPath):
    npzFiles = glob(f"{os.path.join(npzFolderPath, '*.npz')}")[:10]

    X_gray, X_hue, y = [], [], []

    for npzFile in npzFiles:
        loaded_npz = np.load(npzFile)

        grayPic = loaded_npz['gray']
        rgbPic = loaded_npz['hsv']
        hue = loaded_npz['palette'][:, 0]#.reshape((10,1))

        X_gray.append(grayPic)
        X_hue.append(hue)
        y.append(rgbPic)

    X_gray = np.array(X_gray)
    X_hue = np.array(X_hue)
    y = np.array(y)

    return (X_gray, X_hue), y
