from logging import root
from tkinter import font
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import mediapipe as mp
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

tf.keras.backend.clear_session()

os.system("cls")

Emotions = keras.models.load_model('fer2013_cnn.h5')


mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
mp_facedetector = mp.solutions.face_detection




label_dict = {0:'IRA',1:'DISGUSTO',2:'MIEDO',3:'FELICIDAD',4:'NEUTRAL',5:'TRISTEZA',6:'SORPRESA'}


def Start(): 
    global cap
    cap = cv2.VideoCapture(0)
    def make_720p():
        cap.set(3, 1280)
        cap.set(4, 720)

    def make_480p():
        cap.set(3, 640)
        cap.set(4, 480)

    make_720p()
    Viewer()
    
def Viewer():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            PredictionEmotion, f_show = Process(frame)
            f_show = cv2.cvtColor(f_show, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(f_show)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(30,Viewer)
            lblPrediction["text"]=PredictionEmotion
            
        


def Finish():
    global cap
    cap.release()
    
def Process(frame):
    h, w, c = frame.shape
    position_landmark = np.zeros((h, w, 3), np.uint8)
    results = FaceMesh.process(frame)
    f_show=cv2.resize(frame,(854,480))
    PredictionEmotion = "No hay rostro"
    if results.multi_face_landmarks:
        for face_landmarks1 in results.multi_face_landmarks:
            mp_draw.draw_landmarks(position_landmark, face_landmarks1, 
                                   mp_face_mesh.FACEMESH_CONTOURS, 
                                   mp_draw.DrawingSpec((0, 255, 0), 1, 1))
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
            f_show=cv2.resize(f_crop,(395,480))
            f_expression = cv2.resize(f_crop,(48,48))
            f_pgray=cv2.cvtColor(f_expression, cv2.COLOR_RGB2GRAY)
            f_gray = np.expand_dims(f_pgray, axis=-1)
            f_shape =np.asarray(f_gray).astype(float) / 255 
            X_test = np.expand_dims(f_shape, axis=0)
            new_predictions = Emotions.predict(X_test)
            new_predictions = list(new_predictions[0])
            img_index = new_predictions.index(max(new_predictions))  
            PredictionEmotion = label_dict[img_index]
            
    return PredictionEmotion, f_show

    
    

cap=None
root = Tk()
root.title(string = "HERRAMIENTA DE TERAPIA")
root.iconbitmap('Escudo_Universidad_de_Pamplona.ico')
lblTitle1= Label(root, text="David Leonardo Rueda Rivera")
lblTitle2= Label(root, text="Ingenieria en Telecomunicaciones")

lblTitle1.grid(column=0, row=0, columnspan=3)
lblTitle1.config(font=("Verdana",14))

lblTitle2.grid(column=0, row=1, columnspan=3)
lblTitle2.config(font=("Verdana",14))

btnAnalizar = Button(root, text="Analizar", width=45, command=Start, font=("Verdana",8))
btnAnalizar.grid(column=0, row=2, padx=5, pady=5)
btnFinalizar = Button(root, text="Finalizar", width=45, command=Finish, font=("Verdana",8))
btnFinalizar.grid(column=1, row=2, padx=5, pady=5)

lblVideo = Label(root)
lblVideo.grid(column=0, row=3, columnspan=2)

lblPrediction=Label(root)
lblPrediction.grid(column=0, row=4, columnspan=3)
lblPrediction.config(font=("Verdana",12))
root.mainloop()

