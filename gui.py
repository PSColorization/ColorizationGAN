import streamlit as st
#import tkinter as tk
#from tkinter import filedialog
import os
#from PIL import Image
#import tensorflow as tf
from tensorflow.keras.models import load_model
#import numpy
import cv2
import subprocess

#root = tk.Tk()
#root.withdraw()

#root.wm_attributes('-topmost', 1)

st.title("Image Colorization")

option = st.sidebar.selectbox("Select a scene to colorize",("Sky", "Soccer Field", "Forest Road"))
if option == "Sky":
    model = load_model("/Users/enesguler/Downloads/sky.h5", compile=False)
if option == "Soccer Field":
    model = load_model("/Users/enesguler/Downloads/soccer_field.h5", compile=False)
if option == "Forest Road":
    model = load_model("/Users/enesguler/Downloads/forest_road.h5", compile=False)


def colorize(img,model):
    loaded_model = model
    grayPic = img.reshape((-1, 256, 256, 1))
    prediction = loaded_model.predict([grayPic])
    print("HELLO")
    OUTPUT = cv2.cvtColor(prediction[0], cv2.COLOR_RGB2BGR)
    #cv2.imwrite("/content/drive/MyDrive/ColorizationGAN-main/ColorizationGAN-mainoutput.jpg", OUTPUT)
    return OUTPUT

if st.button("Select Image"):
    dirname = subprocess.run(["python3", "/Users/enesguler/PycharmProjects/GraduationProject/ColorizationGAN/guiSelectbox.py"], capture_output=True) # easygui.fileopenbox(default=".")  # filedialog.askopenfile(mode="r", master=root)
    dirname = str(dirname.stdout)
    # print(dirname.split("'")[1][:-2])
    dirname = dirname.split("'")[1][:-2]
    st.text_input('Selected image:', os.path.abspath(dirname))
    picked_img = cv2.imread(dirname)
    picked_img = cv2.cvtColor(picked_img, cv2.COLOR_BGR2RGB)
    st.image(picked_img)
    # if st.button("Colorize!"):

    outimagepath = "/Users/enesguler/Downloads/places365_standard_rgb/val/soccer_field/Places365_val_00000201.jpg"
    outimage = cv2.imread(outimagepath)

    st.image(cv2.cvtColor(outimage, cv2.COLOR_BGR2RGB))


