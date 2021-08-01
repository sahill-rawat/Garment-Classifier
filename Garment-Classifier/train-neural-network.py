import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import keras.layers as layers
from keras.models import Sequential
import cv2

dataset = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = dataset.load_data()

train_img = train_img / 255
test_img = test_img / 255

model = Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_img, train_labels, epochs=5)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
