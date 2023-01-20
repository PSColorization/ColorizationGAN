from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate(gt_path, gen_path, file_name):
    loaded_npz = np.load(os.path.join(gt_path, file_name) + '.npz')
    hsvPic = loaded_npz['hsv']
    gt = cv.cvtColor(hsvPic, cv.COLOR_HSV2RGB)

    gen = cv.imread(os.path.join(gen_path, file_name) + '.jpg')

    accuracy = ssim(gt, gen, multichannel=True)
    return accuracy


if __name__ == '__main__':
    scenes = "sky", "soccer_field", "forest_road"

    for scene_name in scenes:
        gen_path = f"/Users/enesguler/Downloads/test_results_200/{scene_name}"
        gt_path = f"/Users/enesguler/PycharmProjects/GraduationProject/ColorizationGAN/TEST/{scene_name}_onlynpz"

        accuracies = []
        for i in range(1, 300):
            try:
                accuracy = evaluate(gt_path, gen_path, str(i).zfill(8))
                print(f"{str(i).zfill(8)}: {accuracy}")
                accuracies.append(accuracy)
            except:
                pass

        print("Mean Acuracy: ", sum(accuracies)[:200] / 200)

        plt.hist(accuracies, bins=50)
        plt.show()
