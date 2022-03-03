
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers, regularizers


# Inputs Import

dataPathTrain = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/FER2013/train/'
dataPathTest = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/FER2013/test/'

emotion_labels = sorted(os.listdir(dataPathTrain))
print(emotion_labels)

# Data Generator

BatchSize = 64
TargetSize = (48, 48)

TrainDataGen = ImageDataGenerator(rescale=1./255)
TestDataGen = ImageDataGenerator(rescale=1./255)

TrainGenerator = TrainDataGen.flow_from_directory(
    dataPathTrain,
    target_size=TargetSize,
    batch_size=BatchSize,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True)

TestGenerator = TestDataGen.flow_from_directory(
    dataPathTest,
    target_size=TargetSize,
    batch_size=BatchSize,
    color_mode="grayscale",
    class_mode='categorical')

# Build Model
InputShape = (48, 48, 1)  # img_rows, img_colums, color_channels
NumClasses = 7

model = models.Sequential()


model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
          padding='same', input_shape=InputShape))
model.add(layers.Conv2D(32, kernel_size=(3, 3),
          activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(64, kernel_size=(3, 3),
          activation='relu', padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3),
          activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Conv2D(128, kernel_size=(3, 3),
          activation='relu', padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3),
          activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(NumClasses, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train Model
NumEpochs = 100
STEP_SIZE_TRAIN = TrainGenerator.n//TrainGenerator.batch_size
STEP_SIZE_VAL = TestGenerator.n//TestGenerator.batch_size

model.fit(TrainGenerator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=NumEpochs,
          verbose=1, validation_data=TestGenerator, validation_steps=STEP_SIZE_VAL)


models.save_model(model, 'fer2013_cnn.h5') 

score = model.evaluate_generator(TestGenerator, steps=STEP_SIZE_VAL) 
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
