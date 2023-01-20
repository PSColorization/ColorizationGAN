import glob

from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate(gt_path, gen_path, others_path, file_name, count):
    loaded_npz = np.load(os.path.join(gt_path, file_name) + '.npz')
    hsvPic = loaded_npz['hsv']
    gt = cv.cvtColor(hsvPic, cv.COLOR_HSV2RGB)

    gen = cv.imread(os.path.join(gen_path, file_name) + '.jpg')

    eccv16_gen = cv.imread(os.path.join(others_path, f'{count}_eccv16') + '.png')
    siggraph17_gen = cv.imread(os.path.join(others_path, f'{count}_siggraph17') + '.png')

    accuracy_our = ssim(gt, gen, multichannel=True)
    accuracy_eccv16 = ssim(gt, eccv16_gen, multichannel=True)
    accuracy_siggraph17 = ssim(gt, siggraph17_gen, multichannel=True)

    return accuracy_our, accuracy_eccv16, accuracy_siggraph17


if __name__ == '__main__':
    scenes = ("sky", "soccer_field", "forest_road")

    for scene_name in scenes:
        gen_path = f"/Users/enesguler/Downloads/test_results_200/{scene_name}"
        gt_path = f"/Users/enesguler/PycharmProjects/GraduationProject/ColorizationGAN/TEST/{scene_name}_onlynpz"
        others_path = f"/Users/enesguler/Downloads/colorization/{scene_name}_results"
        accuracies_our = []
        accuracies_eccv16 = []
        accuracies_siggraph17 = []

        gt_names = sorted(glob.glob(f'{gt_path}/**/*.npz', recursive=True))[:200]
        count = 0
        if scene_name == 'soccer_field':
            count -= 1
        for index, gt_name in enumerate(gt_names):
            filename = gt_name.split('/')[-1].split('.')[0]
            try:

                count += 1
                accuracy_our, accuracy_eccv16, accuracy_siggraph17 = evaluate(gt_path, gen_path, others_path, filename, count)
                # print(f"{str(i).zfill(8)}: {accuracy}")
                accuracies_our.append(accuracy_our)
                accuracies_eccv16.append(accuracy_eccv16)
                accuracies_siggraph17.append(accuracy_siggraph17)
            except:
                pass
        print(f"\n{scene_name}: ")
        print("Mean Acuracy accuracies_our: ", sum(accuracies_our) / len(accuracies_our))
        print("Mean Acuracy accuracies_eccv16: ", sum(accuracies_eccv16) / len(accuracies_eccv16))
        print("Mean Acuracy accuracies_siggraph17: ", sum(accuracies_siggraph17) / len(accuracies_siggraph17))


        fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)


        axs[0].hist(accuracies_our, bins=50)
        axs[1].hist(accuracies_eccv16, bins=50)
        axs[2].hist(accuracies_siggraph17, bins=50)

        axs[0].title.set_text("accuracies_our")
        axs[1].title.set_text("accuracies_eccv16")
        axs[2].title.set_text("accuracies_siggraph17")
        plt.show()
