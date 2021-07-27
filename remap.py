import json
import cv2 as cv
import numpy as np
from time import time
  

f = open('dic_of_calibration.json','r')
data = json.load(f)
mtx = np.array(data['mtx'])
dist = np.array(data['dist'])
vid = cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv.CAP_GSTREAMER)
_, image = vid.read()
h,  w = image.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)


def compute_rate(funcion,n):
    rate = list()
    i = 0
    while(True):
        t1 = time()
        _, image = vid.read()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if funcion == 'remap':
            dst = cv.remap(gray, mapx, mapy, cv.INTER_LINEAR)
        if funcion == 'undistort':
            dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imshow('Compute rate of ' + funcion, dst)
        t2 = time()
        rate.append(t2-t1)
        i += 1
        if cv.waitKey(1) & 0xFF == ord('q') or i > n:
            break
    mean_rate = np.mean(rate)
    cv.destroyAllWindows()
    print('El frame rate usando ' + funcion + ' es de ' + str(mean_rate))

compute_rate('undistort',1000)
