import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

%pylab inline
import matplotlib.pyplot as plt

SCENE_NAME = "soccer_field"
def darken_image(grayscale_img, constant, mask_threshold):
    mask = grayscale_img > mask_threshold
    grayscale_img[mask] -= constant
    return grayscale_img

def detect(modelName, photoPath, npzName):
    loaded_model = load_model(modelName, compile=False)
    print(loaded_model.layers)

    loaded_npz = np.load(os.path.join(photoPath, npzName) + '.npz')

    #grayPic = loaded_npz['gray'].reshape((-1, 128, 128, 1))
    grayPic = loaded_npz['gray']
    grayPic = cv2.resize(grayPic, (128,128))
    grayPic = np.expand_dims(grayPic, axis=-1)
    grayPic = darken_image(grayPic, 40, 230)
    grayPic = np.broadcast_to(grayPic, (1, 128, 128, 3))
    print(grayPic.shape)
    #hue = np.array(list(loaded_npz['palette'][:, :1].reshape((10,)))*256*256)
    hue = np.array(list(loaded_npz['palette'][:, :1].reshape((10,)))*256*256)
    hue.resize((256, 256, 10))
    hue = np.broadcast_to(hue, (1, 256, 256, 10))
    print(hue.shape)
    #hue = hue.reshape((-1, 256, 256, 10))
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

    cv2.imwrite(f"TEST/final_outputs/{SCENE_NAME}/{npzName}_generated.jpg", cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))

    groundTruth = plt.subplot(1, 3, 3)
    ground_truth = Image.fromarray((rgbPic).astype('uint8')).resize((256, 256))
    groundTruth.set_title('Ground Truth', fontsize=16)
    plt.imshow(ground_truth)
    cv2.imwrite(f"TEST/final_outputs//{SCENE_NAME}/{npzName}_gt.jpg", cv2.cvtColor(rgbPic, cv2.COLOR_RGB2BGR))
    #plt.savefig(f"TEST/deneme/{npzName}.jpg")

    plt.show()


modelName = "/content/drive/MyDrive/ColorizationGAN/aeModel/model5unet_soccer_field_500_light300e.h5" #f"./nonFreezedTrainOutputs/a_{SCENE_NAME}_generatorTrained_999e.h5"
photoPath = f"/content/drive/MyDrive/ColorizationGAN/TEST/only_npz/{SCENE_NAME}"
npzName = "00000100"

for i in range(1, 100):
  try:
    print(i)
    detect(modelName, photoPath, str(i).zfill(8))
  except:
    pass




