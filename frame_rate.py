import numpy as np
from time import time
import cv2 as cv
from progress.bar import Bar  

def compute_rate(n=1000):
    bar = Bar('Computing frame rate', max = n)
    vid = cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv.CAP_GSTREAMER)
    rates = np.zeros(n)
    i = 0
    #Check if camera was opened correctly
    if not (vid.isOpened()):
        print("Could not open video device")
    
    while(True):
        t1 = time()
        bar.next()
        _, frame = vid.read()
        if not(_):
            print('Algo anda mal con la camara')
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        t2 = time()
        rates[i]=(t2-t1)
        i+=1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        elif i>n-1:
            break
    bar.finish()
    vid.release()
    cv.destroyAllWindows()
    return np.mean(rates)

if __name__ == '__main__':
   rate = compute_rate(100)
   print(rate)
