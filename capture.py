
import cv2 as cv
import os
import sys

cap = cv.VideoCapture(0)
leido, frame = cap.read()

if leido == True:
	cv.imwrite('foto_' + sys.argv[1] + '.png' , frame)
	os.system('say "ok"')
else:
	print("Error al acceder a la cámara")

cap.release()