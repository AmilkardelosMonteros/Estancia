
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from time import time
from progress.bar import Bar
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
img1 = cv.imread('akira.png',cv.IMREAD_GRAYSCALE)
dst = cv.cornerHarris(img1,2,3,0.04)
img1[dst>0.01*dst.max()]=255
kp1, des1 = orb.detectAndCompute(img1,None)
vid = cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv.CAP_GSTREAMER)

def compute_rate(n):
    bar = Bar('Computing frame rate...', max = n)
    i = 0
    rates = np.ones(n)
    while(True):
        bar.next()
        t1 = time()
        _, frame = vid.read()
        img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dst = cv.cornerHarris(img2,2,3,0.04)
        img2[dst>0.01*dst.max()]=255
        kp2, des2 = orb.detectAndCompute(img2,None)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:30],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('frame', img3)
        t2 = time()
        rates[i] = t2-t1
        i+=1
        if cv.waitKey(1) & 0xFF == ord('q'):
           break
        elif i > n-1:
           break
    for _ in range(10):print('')
    print('Frame rate de Harris + ORB = ', np.mean(rates))
    vid.release()
    cv.destroyAllWindows()
    bar.finish()
    return rates
if __name__ == '__main__':
    rate = compute_rate(1000)
