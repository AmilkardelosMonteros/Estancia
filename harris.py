
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    gray[dst>0.01*dst.max()]=255

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(gray,(x,y),3,255,-1)
'''


orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
img1 = cv.imread('akira.png',cv.IMREAD_GRAYSCALE)
dst = cv.cornerHarris(img1,2,3,0.04)
img1[dst>0.01*dst.max()]=255
kp1, des1 = orb.detectAndCompute(img1,None)

vid = cv.VideoCapture(0)
while(True):
    _, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray,2,3,0.04)
    gray[dst>0.01*dst.max()]=255
    kp2, des2 = orb.detectAndCompute(gray,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    if matches != []:
    # Draw first 10 matches.
        img3 = cv.drawMatches(img1,kp1,gray,kp2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('frame', img3)
        cv.waitKey(1)
        cv.destroyAllWindows()
    elif cv.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        print('Mejora tu imagen')

vid.release()
cv.destroyAllWindows()
