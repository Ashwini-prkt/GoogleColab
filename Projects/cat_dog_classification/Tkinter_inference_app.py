from PIL import Image
import numpy as np
from tensorflow import keras
import os
import tkinter as tk
from PIL import ImageTk

Image_Width = Image_Height = 128
Image_Channels = 3
Image_size = (Image_Width, Image_Height)

input_img_dir = "./dataset/test"
ip_img = "./dataset/test/dog.4853.jpg"

model = keras.models.load_model('./model_catsVSdogs_10epoch.h5')

label_map = {0: 'cat', 1: 'dog'}

def classify(ip_img, top):
	test_img = Image.open(ip_img)
	resized_img = test_img.resize(Image_size)
	single_img = np.expand_dims(resized_img, axis=0)
	single_img = np.array(single_img)
	single_img = single_img/255
	pred = model.predict_classes(single_img)[0]
	return label_map[pred]

for i in os.listdir(input_img_dir):
	top = tk.Tk()

	### increasing the size:
	top.geometry("500x500")

	### reading image and showcasing:
	img = ImageTk.PhotoImage(Image.open(os.path.join(input_img_dir, i)))
	tk.Label(top, image=img).pack()

	pred_res = classify(os.path.join(input_img_dir, i), top)
	top.title(pred_res)

	top.mainloop()



