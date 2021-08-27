import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,load_img

Image_Width = Image_Height = 128
Image_Channels = 3
Image_Size = (128,128)
batch_size=15
epochs = 10

filenames = os.listdir("./dataset/training_set/images_cat_dog")

categories=[]
for f_name in filenames:
        categories.append(f_name.split('.')[0])

df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})

#print(df.head())

### Creating the neural network CNN model:
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3,3),activation='relu',input_shape=(Image_Width, Image_Height, Image_Channels)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3,3),activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, (3,3),activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(2,activation='softmax'))

### defining the loss, optimizer and metrics for training:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
#model.summary()

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

### Augmenting the train, validation and test images:
train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )

train_data_iter = train_datagen.flow_from_dataframe(train_df,
                                                 "./dataset/training_set/images_cat_dog/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=1./255)

val_data_iter = val_datagen.flow_from_dataframe(validate_df,
                                                 "./dataset/training_set/images_cat_dog/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)

test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)

test_data_iter = test_datagen.flow_from_dataframe(train_df,
                                                 "./dataset/test_set/images_cat_dog/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
### Training starts:
history = model.fit_generator(
    train_data_iter, 
    epochs=epochs,
    validation_data=val_data_iter,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
)

### saving the weights of the trained model:
model.save("model_catsVSdogs_10epoch.h5")


