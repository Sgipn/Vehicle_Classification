#Import Libraries:
##NOTE: long strings of hashtags indicate new chunk.

#######################################################
import numpy as np
import scipy
import tensorflow as fl
import tensorflow_io as tfio
from keras.preprocessing.image import load_img,ImageDataGenerator,img_to_array
import math
import os
from tensorflow import keras
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from os.path import join
import keras.models
import pandas as pd

from kerastuner.tuners import Hyperband
from kerastuner.tuners import BayesianOptimization
from kerastuner import HyperModel
from keras_tuner.applications import HyperResNet
from keras_tuner.applications import HyperXception
import keras_tuner

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
#######################################################################



######################################################################
#Setup directories:

#NOTE: This section and the next section may be different since we are using NPZ files. These were set up to import images from a specific directory structure on my PC.

dat = os.listdir("INSERT FILE PATH TO IMAGES HERE!")
print(dat)

correct_plots = os.listdir("INSERT FILE PATH TO IMAGES HERE!")
incorrect_plots = os.listdir("INSERT FILE PATH TO IMAGES HERE!")


#######################################################################



###################################################################
#obtain image size:

#NOTE: This section and the previous section may be different since we are using NPZ files. These were set up to import images from a specific directory structure on my PC.

plot_size = load_img(join("INSERT FILE PATH TO IMAGES HERE!",correct_plots[0]))
img_size = plot_size.size
print(img_size)

######################################################################
#process data and create training set and validation set:

data_gen = ImageDataGenerator(
    rotation_range=45,
    #width_shift_range=0.5,
    #height_shift_range=0.5,
    #horizontal_flip=True,
    #vertical_flip=True,
    rescale=1./255,
    #zoom_range=0.1,
    validation_split=0.35,
)

test_data_gen = ImageDataGenerator(
    rescale=1/255,
)

target = (128,128)
target2 = (128,128,1)

train = data_gen.flow_from_directory(directory="INSERT FILE PATH TO IMAGES HERE!", target_size=target, color_mode= "grayscale", class_mode="categorical", subset="training",shuffle=True)

test = data_gen.flow_from_directory(directory="INSERT FILE PATH TO IMAGES HERE!", target_size=target, color_mode= "grayscale", class_mode="categorical", subset="validation", shuffle=True )

eval = test_data_gen.flow_from_directory(directory = "INSERT FILE PATH TO IMAGES HERE!"", target_size=target, color_mode="grayscale", class_mode="categorical", shuffle=True)
print(test.classes, test.class_indices, test.image_shape)


######################################################################
###define and compile CNN model:
target2 = (128,128,1)
model = Sequential()
model.add(Conv2D(36,3, activation="relu", input_shape=target2))
model.add(layers.MaxPool2D())
model.add(layers.BatchNormalization())
model.add(Conv2D(18, 3, activation="relu"))
model.add(layers.MaxPool2D())
model.add(layers.BatchNormalization())
model.add(Conv2D(9, 3, activation="relu"))
#model.add(layers.MaxPool2D())
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(2, activation="sigmoid"))

model.summary()

######################################################################
model.compile(optimizer=fl.keras.optimizers.Adam(),loss="categorical_crossentropy" ,metrics=['Accuracy'])

early_stp = EarlyStopping(monitor = "val_loss", mode = "min",  patience=6, restore_best_weights=True)

history = model.fit(train,validation_data=test,epochs=500,callbacks=[early_stp])
print(history)
#######################################################################
