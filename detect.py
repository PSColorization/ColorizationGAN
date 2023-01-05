
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow
print(tensorflow.__version__)
loaded_model = load_model("models/sky_600e_300i_generatorTrained.h5", compile=False)

# loaded_npz = np.load("/content/drive/MyDrive/ColorizationGAN/TEST/sky_onlynpz/00000050.npz")

# grayPic = loaded_npz['gray'].reshape((-1, 256, 256, 1))
# hsvPic = loaded_npz['hsv']

grayPic = cv2.imread("DemoTestImages/sky/00000012.png")
if grayPic.shape[2] == 3:
    grayPic, _, _ = cv2.split(grayPic)

prediction = loaded_model.predict([grayPic.reshape((-1, 256, 256, 1))])

# hsvPic = cv2.cvtColor(hsvPic, cv2.COLOR_HSV2RGB)

OUTPUT = cv2.cvtColor(prediction[0], cv2.COLOR_RGB2BGR)
# cv2.imwrite("/content/drive/MyDrive/ColorizationGAN/output.jpg", OUTPUT)

cv2.imshow('generated', OUTPUT)
cv2.imshow('input', grayPic)

