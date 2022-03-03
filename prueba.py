import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# print('Ingrese el modo de analisis de las etiquetas')

# Mode=input()

dataPathTrain = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/FER2013/train/'
ExpressionListTrain = os.listdir(dataPathTrain)
print('Lista de Emociones de Train: ', ExpressionListTrain)
labelsTrain = []
ExpressionDataTrain = []


label1 = 0
for nameDir in ExpressionListTrain:
    ExpressionPath = dataPathTrain + '/' + nameDir
    print('Leyendo las imágenes de ', nameDir, 'codigo', label1)
    for fileName in os.listdir(ExpressionPath):
        #print('Expresiones: ', nameDir + '/' + fileName)
        labelsTrain.append(label1)
        # ExpressionDataTrain.append(cv2.cvtColor(cv2.imread(ExpressionPath+'/'+fileName), cv2.COLOR_RGB2GRAY))
        gray=cv2.cvtColor(cv2.imread(ExpressionPath+'/'+fileName), cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        image = cv2.imread(ExpressionPath+'/'+fileName,1)
        print('gray forma', gray.shape, 'image forma', image.shape)
        #cv2.imshow('image',image)   
    label1 = label1 + 1