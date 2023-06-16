# %pwd

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib
from PIL import Image

loaded_model = load_model("generatorTrained.h5", compile=False)

loaded_npz = np.load("/content/drive/MyDrive/ColorizationGAN/TEST/onlynpz/00000011.npz")

grayPic = loaded_npz['gray'].reshape((-1, 256, 256, 1))
hsvPic = loaded_npz['hsv']


prediction = loaded_model.predict([grayPic])


#prediction = Image.fromarray((prediction[0]).astype('uint8')).resize((256, 256))
#prediction = np.asarray(prediction)

#prediction = prediction.reshape((3, 256, 256))
#print("TRANSPOSED: ", prediction.shape)
#prediction = cv2.cvtColor(prediction, cv2.COLOR_HSV2RGB)
hsvPic = cv2.cvtColor(hsvPic, cv2.COLOR_HSV2RGB)




OUTPUT = cv2.cvtColor(prediction[0], cv2.COLOR_BGR2RGB)
cv2.imwrite("/content/drive/MyDrive/ColorizationGAN/output.jpg", OUTPUT)



# %pylab inline
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# print(prediction.shape)
# #imgplot = plt.imshow(prediction[0])
# #plt.show()
# imgplot = plt.imshow(hsvPic)
# plt.show()