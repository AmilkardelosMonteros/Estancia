
import cv2 as cv
import os
import sys

vid = cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv.CAP_GSTREAMER)


leido, frame = vid.read()

if leido == True:
	cv.imwrite('images/foto_' + sys.argv[1] + '.png' , frame)
	os.system('say "ok"')
else:
	print("Error al acceder a la cámara")

vid.release()
