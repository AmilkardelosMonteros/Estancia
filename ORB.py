import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
img1 = cv.imread('akira.png',cv.IMREAD_GRAYSCALE)          # queryImage
kp1, des1 = orb.detectAndCompute(img1,None)

vid = cv.VideoCapture(0)
while(True):
    _, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    if matches != []:
    # Draw first 10 matches.
        img3 = cv.drawMatches(img1,kp1,gray,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('frame', img3)
        cv.waitKey(1)
        cv.destroyAllWindows()
    elif cv.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        print('Mejora tu imagen')

vid.release()
cv.destroyAllWindows()



