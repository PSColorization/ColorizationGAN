import os
from skimage import io, metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_similarity_measures.evaluate import evaluation
# !python3 -m pip install image-similarity-measures pyfftw

# Create empty lists to store the PSNR and SSIM values for each model
psnr_model1_list = []
psnr_model2_list = []
psnr_model3_list = []
ssim_model1_list = []
ssim_model2_list = []
ssim_model3_list = []

# Define the paths to the ground truth and model image folders
num_of_img_per_scene = 200
scene_names = "army_base", "bazaar-indoor", "bazaar-outdoor", "classroom", "forest_road", "kitchen", "ski_slope", "sky", "soccer_field", "wave"
for scene in scene_names:

    gt_path = f'/content/drive/MyDrive/ColorizationGAN/TEST/final_outputs/{scene}'
    model1_path = f'/content/drive/MyDrive/ColorizationGAN/TEST/final_outputs/{scene}'
    model2_path = f'/content/drive/MyDrive/ColorizationGAN/TEST/final_outputs/{scene}_out'
    model3_path = f'/content/drive/MyDrive/ColorizationGAN/TEST/final_outputs/{scene}_out'

    # Loop through each image in the folder
    for filename in tqdm(os.listdir(model2_path)[:num_of_img_per_scene]):
        if "ipynb" in filename:
          continue
        # if "eccv16" in model2_path:
        #   continue
        num = filename.split('_')[0]

        # Load the ground truth image
        gt_img = io.imread(os.path.join(gt_path, f"{num}_gt.jpg"))

        eccv_postname = "_gt_eccv16.png"
        siggraph_postname = "_gt_siggraph17.png"

        # Load the three model images
        model1_img = io.imread(os.path.join(model1_path, f"{num}_generated.jpg"))
        model2_img = io.imread(os.path.join(model2_path, f"{num}{eccv_postname}"))
        model3_img = io.imread(os.path.join(model3_path, f"{num}{siggraph_postname}"))

        # Compute PSNR for each model image
        psnr_model1 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model1_path, f"{num}_generated.jpg"), metrics=["psnr"])["psnr"]
        psnr_model2 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model2_path, f"{num}{eccv_postname}"), metrics=["psnr"])["psnr"]
        psnr_model3 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model3_path, f"{num}{siggraph_postname}"), metrics=["psnr"])["psnr"]

        # Compute SSIM for each model image
        ssim_model1 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model1_path, f"{num}_generated.jpg"), metrics=["ssim"])["ssim"]
        ssim_model2 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model2_path, f"{num}{eccv_postname}"), metrics=["ssim"])["ssim"]
        ssim_model3 = evaluation(org_img_path=os.path.join(gt_path, f"{num}_gt.jpg"), pred_img_path= os.path.join(model3_path, f"{num}{siggraph_postname}"), metrics=["ssim"])["ssim"]

        # Append the PSNR and SSIM values to the lists
        psnr_model1_list.append(psnr_model1)
        psnr_model2_list.append(psnr_model2)
        psnr_model3_list.append(psnr_model3)
        ssim_model1_list.append(ssim_model1)
        ssim_model2_list.append(ssim_model2)
        ssim_model3_list.append(ssim_model3)



# Create a plot for PSNR
plt.figure().set_figwidth(20)
plt.plot(psnr_model1_list, label='Our Model')
plt.plot(psnr_model2_list, label='ECCV16')
plt.plot(psnr_model3_list, label='SIGGRAPH17')
plt.title('PSNR')
plt.legend()
plt.show()

# Create a plot for SSIM
plt.figure().set_figwidth(20)
plt.plot(ssim_model1_list, label='Our Model')
plt.plot(ssim_model2_list, label='ECCV16')
plt.plot(ssim_model3_list, label='SIGGRAPH17')
plt.title('SSIM')
plt.legend()
plt.show()

print("psnr_model1_list", sum(psnr_model1_list)/ len(psnr_model1_list))
print("psnr_model2_list", sum(psnr_model2_list)/ len(psnr_model2_list))
print("psnr_model3_list", sum(psnr_model3_list)/ len(psnr_model3_list))
print("ssim_model1_list", sum(ssim_model1_list)/ len(ssim_model1_list))
print("ssim_model2_list", sum(ssim_model2_list)/ len(ssim_model2_list))
print("ssim_model3_list", sum(ssim_model3_list)/ len(ssim_model3_list))
