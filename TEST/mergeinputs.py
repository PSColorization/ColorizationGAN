import cv2 as cv
import numpy as np
import pandas as pd
import os
import sys
import extcolors
import os
from glob import glob
import matplotlib as plt
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from colormap import rgb2hex, rgb2hsv


def color_to_df(input):
    colors_pre_list = str(input).replace('([(', '').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')', '') for i in colors_pre_list]

    # convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(", "")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")", ""))) for i in df_rgb]

    df = pd.DataFrame(list(zip(df_color_up, df_percent)), columns=['c_code', 'occurence'])
    return df


def exact_color(input_image, resize, tolerance, zoom):
    # background
    bg = 'bg.png'
    fig, ax = plt.subplots(figsize=(192, 108), dpi=10)
    fig.set_facecolor('white')
    plt.savefig(bg)
    plt.close(fig)

    # resize
    output_width = resize
    img = Image.open(input_image)
    if img.size[0] >= resize:
        wpercent = (output_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((output_width, hsize), Image.ANTIALIAS)
        resize_name = 'resize_' + input_image
        # img.save(resize_name)
    else:
        resize_name = input_image

    # crate dataframe
    img_url = resize_name
    colors_x = extcolors.extract_from_image(img, tolerance=tolerance, limit=zoom)

    # rgb_list = [colors_x[0][i][0] for i in range(len(colors_x[0]))]

    hsvlist = [tuple(rgb2hsv(colors_x[0][i][0][0] / 255, colors_x[0][i][0][1] / 255, colors_x[0][i][0][2] / 255)) for i
               in range(len(colors_x[0]))]

    # df_color = color_to_df(colors_x)
    # df_dict = df_color.to_dict()
    return hsvlist


sys.path.append(os.getcwd())
os.chdir("sky/")

image_paths = [dir for dir in os.listdir(".")]

rgblist = list()
graylist = list()
hsvpalette = list()
npz_paths = list()
print(image_paths)
for imgdir in tqdm(image_paths):
    try:
        img = cv.imread(imgdir)
        rgbimg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hsvlist = np.array(exact_color(imgdir, 256, 17, 10))
        rgblist.append(rgbimg)
        graylist.append(grayimg)
        hsvpalette.append(hsvlist)
        npz_path = imgdir.split('.')[0] + '.npz'
        npz_paths.append(npz_path)

        if hsvlist.shape[0] == 10:
            np.savez_compressed(npz_path, hsv=rgbimg, gray=grayimg, palette=hsvlist)

    except:
        pass

image_df = pd.DataFrame(
    {"ImagePath": image_paths, 'RGB': rgblist, "Gray": graylist, "HSV": hsvpalette, "NPZ": npz_paths})

print(image_df.head())
