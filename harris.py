
import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt
from time import time
from progress.bar import Bar
import json 

f = open('dic_of_calibration.json','r')
data = json.load(f)
mtx = np.array(data['mtx'])
dist = np.array(data['dist'])
import platform


def harris(img):
    '''
    Harris detector
    :param img: an color image
    :return: keypoint, image with feature marked corner
    '''
    img = np.float32(img)
    dst = cv.cornerHarris(img, 2, 3, 0.04)

    # for each dst larger than threshold, make a keypoint out of it
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]

    return keypoints

orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
img1 = cv.imread('akira.png',cv.IMREAD_GRAYSCALE)
kp1 = harris(img1)

kp1, des1 = orb.compute(img1, kp1)


vid = cv.VideoCapture(0) if platform.system() == 'Darwin' else cv.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv.CAP_GSTREAMER)
_, image = vid.read()
h,  w = image.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

def compute_rate(n):
    bar = Bar('Computing frame rate...', max = n)
    i = 0
    rates = np.ones(n)
    while(True):
        bar.next()
        t1 = time()
        _, frame = vid.read()
        img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img2 = cv.remap(img2, mapx, mapy, cv.INTER_LINEAR)
        x, y, w, h = roi
        img2 = img2[y:y+h, x:x+w]
        kp2 = harris(img2)
        kp2, des2 = orb.compute(img2, kp2)
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
    print('Frame rate de Harris + ORB = ', np.mean(rates[0:i]))
    vid.release()
    cv.destroyAllWindows()
    bar.finish()
    return rates
if __name__ == '__main__':
    rate = compute_rate(500)
