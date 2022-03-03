
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print('Ingrese el modo de analisis de las etiquetas')

Mode=input()

dataPathTrain = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/Data/Train/' + Mode
ExpressionListTrain = os.listdir(dataPathTrain)
print('Lista de Emociones de Train: ', ExpressionListTrain)
dataPathTest = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/Data/Test/' + Mode
ExpressionListTest = os.listdir(dataPathTest)
print('Lista de Emociones de Test: ', ExpressionListTest)
labelsTrain = []
ExpressionDataTrain = []
labelsTest = []
ExpressionDataTest = []

label1 = 0
for nameDir in ExpressionListTrain:
    ExpressionPath = dataPathTrain + '/' + nameDir
    print('Leyendo las imágenes...')
    for fileName in os.listdir(ExpressionPath):
        #print('Expresiones: ', nameDir + '/' + fileName)
        labelsTrain.append(label1)
        ExpressionDataTrain.append(cv2.imread(ExpressionPath+'/'+fileName))
        #image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image',image)   
    label1 = label1 + 1
label2 = 0
for nameDir in ExpressionListTest:
    ExpressionPath = dataPathTest + '/' + nameDir
    print('Leyendo las imágenes...')
    for fileName in os.listdir(ExpressionPath):
        #print('Expresiones: ', nameDir + '/' + fileName)
        labelsTest.append(label2)
        ExpressionDataTest.append(cv2.imread(ExpressionPath+'/'+fileName))
        #image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image',image)
    label2 = label2 + 1

print('Imagenes correctamente etiquetadas')



X_train = np.array(ExpressionDataTrain).astype(float) / 255
Y_train= np.array(labelsTrain)

X_test = np.array(ExpressionDataTest).astype(float) / 255
Y_test= np.array(labelsTest)
print(X_train.shape)

#Alteración de datos

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X_train)

#creacion de modelos

# modeloCNN_AD = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),

#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(100, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])

modeloCNN2_AD = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN2_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#Separar los datos de entrenamiento y los datos de pruebas en variables diferentes



#Usar la funcion flow del generador para crear un iterador que podamos enviar como entrenamiento a la funcion FIT del modelo
data_gen_entrenamiento = datagen.flow(X_train, Y_train, batch_size=32)
tensorboardCNN2_AD = TensorBoard(log_dir='logs/cnn2_AD')

modeloCNN2_AD.fit(
    data_gen_entrenamiento,
    epochs=100, batch_size=32,
    validation_data=(X_test, Y_test),
    steps_per_epoch=int(np.ceil(len(X_train) / float(32))),
    validation_steps=int(np.ceil(len(X_test) / float(32))),
    callbacks=[tensorboardCNN2_AD]
)

modeloCNN2_AD.save('felicidad-tristeza-cnn-ad-'+Mode+'.h5')