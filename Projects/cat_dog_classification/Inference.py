from PIL import Image
import numpy as np
from tensorflow import keras
import os

Image_Width = Image_Height = 128
Image_Channels = 3
Image_size = (Image_Width, Image_Height)

input_img_dir = "./dataset/test"

label_map = {0:'cat', 1: 'dog'}
### Inferencing on a single image:
model = keras.models.load_model('./model_catsVSdogs_10epoch.h5')

for i in os.listdir(input_img_dir):
	test_img = Image.open(os.path.join(input_img_dir, i))
	resized_img = test_img.resize(Image_size)
	single_img = np.expand_dims(resized_img, axis=0)
	single_img = np.array(single_img)
	single_img = single_img/255

	pred = model.predict_classes(single_img)[0]
	print("\n", i, "----->" "\tPredicted: ", label_map[pred], "\tActual: ", i.split('.')[0])

