import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

import matplotlib.pyplot as plt


def detect(modelName, photoPath, npzName):
    loaded_model = load_model(modelName, compile=False)

    loaded_npz = np.load(os.path.join(photoPath, npzName) + '.npz')

    grayPic = loaded_npz['gray'].reshape((-1, 256, 256, 1))

    hsvPic = loaded_npz['hsv']
    rgbPic = cv2.cvtColor(hsvPic, cv2.COLOR_HSV2RGB)

    prediction = loaded_model.predict([grayPic])

    plt.figure(figsize=(10, 10))
    grayInput = plt.subplot(1, 3, 1)
    grayInput.set_title('Grayscale Input', fontsize=16)
    plt.imshow(loaded_npz['gray'].reshape((256, 256)), cmap='gray')

    generatedImage = plt.subplot(1, 3, 2)
    generated_image = Image.fromarray((prediction[0]).astype('uint8')).resize((256, 256))
    generated_image = np.asarray(generated_image)
    generatedImage.set_title('Colorized Output', fontsize=16)
    plt.imshow(generated_image)

    # cv2.imwrite(f"TEST/test_results_200/soccer_field/{npzName}.jpg", cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))

    groundTruth = plt.subplot(1, 3, 3)
    ground_truth = Image.fromarray((rgbPic).astype('uint8')).resize((256, 256))
    groundTruth.set_title('Ground Truth', fontsize=16)
    plt.imshow(ground_truth)
    plt.savefig(f"TestResults_sky/{npzName}.jpg")

    plt.show()


modelName = "/Users/enesguler/Downloads/soccer_field.h5"
photoPath = "/Users/enesguler/PycharmProjects/GraduationProject/ColorizationGAN/TEST/soccer_field_onlynpz"
npzName = "00000100"

for i in range(1, 300):
    print(i)
    # try:
    detect(modelName, photoPath, str(i).zfill(8))
    # except:
    #     pass
