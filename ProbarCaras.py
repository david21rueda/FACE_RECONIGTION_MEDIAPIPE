import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import mediapipe as mp
from tkinter import *

tf.keras.backend.clear_session()

os.system("cls")

FelicidadTristeza = keras.models.load_model('felicidad-tristeza-cnn-ad-Normal.h5')


mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
mp_facedetector = mp.solutions.face_detection


# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('Videos/Test/DAVID.mp4')

# def make_720p():
#     cap.set(3, 1280)
#     cap.set(4, 720)


# def make_480p():
#     cap.set(3, 640)
#     cap.set(4, 480)


# make_720p()
count = 0

Root=Tk()


while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    position_landmark = np.zeros((h, w, 3), np.uint8)
    results = FaceMesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks1 in results.multi_face_landmarks:
            mp_draw.draw_landmarks(position_landmark, face_landmarks1, mp_face_mesh.FACEMESH_CONTOURS, mp_draw.DrawingSpec((0, 255, 0), 1, 1))
            cx_min = w
            cy_min = h
            cx_max = cy_max = 0
            for id, lm in enumerate(face_landmarks1.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx < cx_min:
                    cx_min = cx
                if cy < cy_min:
                    cy_min = cy
                if cx > cx_max:
                    cx_max = cx
                if cy > cy_max:
                    cy_max = cy
            #cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

            f_crop=frame[cy_min:cy_max,cx_min:cx_max]
            PL_crop=position_landmark[cy_min:cy_max,cx_min:cx_max]
            f_expression = cv2.resize(f_crop,(100,100),interpolation=cv2.INTER_CUBIC)
            PL_expression = cv2.resize(PL_crop,(100,100),interpolation=cv2.INTER_CUBIC)
            f_shape =np.asarray(f_expression).astype(float) / 255 
            X_test = np.expand_dims(f_shape, axis=0)
            # Verifique que el estado esté preservado
            new_predictions = FelicidadTristeza.predict(X_test) 
            print (new_predictions)

    
    cv2.imshow("FRAME", f_expression)
    cv2.imshow("puntos",PL_expression)
    
    # k =  cv2.waitKey(1)
    # if k == 27 or count >= 600:
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()





