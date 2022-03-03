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
import mediapipe as mp
import imutils

mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils
mp_facedetector = mp.solutions.face_detection


os.system("cls")

Emotions = keras.models.load_model('fer2013_cnn.h5')


label_dict = {0: 'IRA', 1: 'DISGUSTO', 2: 'MIEDO',
              3: 'FELICIDAD', 4: 'NEUTRAL', 5: 'TRISTEZA', 6: 'SORPRESA'}

ejemplo_dir = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/FOTOS'
with os.scandir(ejemplo_dir) as ficheros:
    ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]


def Start():
    global cap
    global number
    cap = cv2.VideoCapture(ejemplo_dir+'/'+ficheros[number])

    # def make_720p():
    #     cap.set(3, 1280)
    #     cap.set(4, 720)

    # def make_480p():
    #     cap.set(3, 640)
    #     cap.set(4, 480)

    # make_480p()
    Viewer()

def Viewer():
    global cap
    if cap != None:
        ret, frame = cap.read()
        if ret == True:
            PredictionEmotion, f_show, f_pl = Process(frame)
            if PredictionEmotion != "No hay rostro":
                f_show = cv2.resize(f_show, (395, 480))
                f_pl = cv2.resize(f_pl, (459, 480))
                f_con = cv2.hconcat([f_pl,f_show])
                im = Image.fromarray(f_con)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(30, Viewer)
                lblPrediction["text"] = PredictionEmotion
            else:
                im = Image.fromarray(f_show)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(30, Viewer)
                lblPrediction["text"] = PredictionEmotion

def Finish():
    global cap
    cap.release()
    root.quit

def Process(frame):
    h, w, c = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(frame)
    f_show = cv2.resize(frame, (854, 480))
    f_pl = cv2.resize(frame, (854, 480))
    PredictionEmotion = "No hay rostro"
    if results.multi_face_landmarks:
        for face_landmarks1 in results.multi_face_landmarks:
            mp_draw.draw_landmarks(f_pl, face_landmarks1,
                                   mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_draw.DrawingSpec((0, 255, 0), 1, 1))
            BBx_min = w
            BBy_min = h
            BBx_max = BBy_max = 0
            for id, lm in enumerate(face_landmarks1.landmark):
                BBx, BBy = int(lm.x * w), int(lm.y * h)
                if BBx < BBx_min:
                    BBx_min = BBx
                if BBy < BBy_min:
                    BBy_min = BBy
                if BBx > BBx_max:
                    BBx_max = BBx
                if BBy > BBy_max:
                    BBy_max = BBy
            PredictionEmotion = "No se ve el rostro con claridad"
            if BBy_max <= h and BBx_max <= w and BBy_min >= 0 and BBx_min >= 0 and BBy_max >= 0 and BBx_max >= 0 and BBy_min <= w and BBx_min <= h:
                f_crop = frame[BBy_min:BBy_max, BBx_min:BBx_max]
                f_show = cv2.resize(f_crop, (395, 480))
                f_expression = cv2.resize(f_crop, (48, 48))
                f_pgray = cv2.cvtColor(f_expression, cv2.COLOR_RGB2GRAY)
                f_gray = np.expand_dims(f_pgray, axis=-1)
                f_shape = np.asarray(f_gray).astype(float) / 255
                X_test = np.expand_dims(f_shape, axis=0)
                new_predictions = Emotions.predict(X_test)
                new_predictions = list(new_predictions[0])
                img_index = new_predictions.index(max(new_predictions))
                PredictionEmotion = label_dict[img_index]

    return PredictionEmotion, f_show, f_pl
    
def Next():
    global number
    number=number+1
    Start()
def Back():
    global number
    number=number-1
    Start()
    
number=0
cap = None
root = Tk()
root.title(string="HERRAMIENTA DE TERAPIA")
root.iconbitmap('Escudo_Universidad_de_Pamplona.ico')
root.geometry("980x600")
lblTitle1 = Label(root, text="David Leonardo Rueda Rivera")
lblTitle2 = Label(root, text="Ingenieria en Telecomunicaciones")

lblTitle1.grid(column=0, row=0, columnspan=3)
lblTitle1.config(font=("Verdana", 14))

lblTitle2.grid(column=0, row=1, columnspan=3)
lblTitle2.config(font=("Verdana", 14))

btnAnalizar = Button(root, text="Iniciar", width=45,
                     command=Start, font=("Verdana", 8))
btnAnalizar.grid(column=0, row=2, padx=5, pady=5)

btnFinalizar = Button(root, text="Siguiente", width=45,
                      command=Next, font=("Verdana", 8))
btnFinalizar.grid(column=1, row=2, padx=5, pady=5)

btnVolver = Button(root, text="Volver", width=45,
                      command=Back, font=("Verdana", 8))
btnVolver.grid(column=2, row=2, padx=5, pady=5)

lblVideo = Label(root)
lblVideo.grid(column=0, row=3, columnspan=3)

lblPrediction = Label(root)
lblPrediction.grid(column=0, row=4, columnspan=3)
lblPrediction.config(font=("Verdana", 12))

root.mainloop()
