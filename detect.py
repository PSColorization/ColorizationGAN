import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model

loaded_model = load_model("generatorTrained.h5", compile=True)

loaded_npz = np.load("TEST/sky/00000037.npz")

grayPic = loaded_npz['gray'].reshape((1, 256, 256, 1))
rgbPic = loaded_npz['rgb']
hue = loaded_npz['hsv'][:, 0].reshape((1, 10))

print(grayPic.shape, hue.shape, type(grayPic))
prediction = loaded_model.predict([grayPic, hue])


print("HELLO")
