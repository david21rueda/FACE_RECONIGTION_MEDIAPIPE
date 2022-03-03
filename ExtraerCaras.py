from operator import pos
import cv2
import os
import dlib
import matplotlib as plt
import numpy as np
from PIL import Image
import mediapipe as mp
os.system("cls")


ExpressionNameF = 'HappinessF'
ExpressionNamePL = 'HappinessPL'
dataPath = 'D:/Universidad/10 semestre/Trabajo de Grado/Reconocimiento facial/Data'
ExpressionPathF = dataPath + '/' + ExpressionNameF
ExpressionPathPL = dataPath + '/' + ExpressionNamePL
if not os.path.exists(ExpressionPathF):
    print('Carpeta creada: ', ExpressionPathF)
    os.makedirs(ExpressionPathF)

if not os.path.exists(ExpressionPathPL):
    print('Carpeta creada: ', ExpressionPathPL)
    os.makedirs(ExpressionPathPL)

mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
mp_facedetector = mp.solutions.face_detection


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Videos/Test/DAVID_ SONRIENDO.mp4')

# def make_720p():
#     cap.set(3, 1280)
#     cap.set(4, 720)


# def make_480p():
#     cap.set(3, 640)
#     cap.set(4, 480)


# make_720p()
count = 0

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
            f_expression = cv2.resize(f_crop,(150,150),interpolation=cv2.INTER_CUBIC)
            PL_expression=cv2.resize(PL_crop,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(ExpressionPathF + '/f_expression_{}.jpg'.format(count),f_expression)
            cv2.imwrite(ExpressionPathPL + '/PL_expression_{}.jpg'.format(count),PL_expression)
            count = count + 1
        
    cv2.imshow("Face Landmarks", f_crop)
    cv2.imshow("puntos",PL_crop)
    
    k =  cv2.waitKey(1)
    if k == 27 or count >= 600:
        break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


cap.release()
cv2.destroyAllWindows()
