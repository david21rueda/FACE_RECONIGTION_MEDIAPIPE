import cv2
import sys
import numpy as np

cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smileClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eyeClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
while True:
   ret, frame = cap.read()
   #rame=cv2.imread("imagen.jpg")
   # scale_percent = 30
   # width = int(frame.shape[1] * scale_percent / 100)
   # height = int(frame.shape[0] * scale_percent / 100)
   # dim = (width, height)
   # frame = cv2.resize(frame, dim)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   cara = faceClassif.detectMultiScale(gray, 1.3, 5)

   for (x, y, z, h) in cara:
      cv2.rectangle(frame, (x, y), (x+z, y+h), (255, 0, 0), 2)
      fgcara=gray[y:y+h,x:x+z]
      ffcara=frame[y:y+h,x:x+z]
      sonrisa = smileClassif.detectMultiScale(fgcara,1.1,25)
      ojos = eyeClassif.detectMultiScale(fgcara,1.1,25)

      for (xs,ys,zs,hs) in sonrisa:
         cv2.rectangle(ffcara,(xs, ys), (xs+zs, ys+hs), (0, 0, 255), 2)
      for (xo,yo,zo,ho) in ojos:
         cv2.rectangle(ffcara,(xo, yo), (xo+zo, yo+ho), (0, 255, 0), 2)
         


   cv2.imshow('Camara', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

#cap.release()
cv2.destroyAllWindows()
