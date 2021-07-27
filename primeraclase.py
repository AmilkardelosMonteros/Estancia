import numpy as np

import cv2
  

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
    _, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    gray[dst>0.01*dst.max()]=255
    '''

    '''
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(gray,(x,y),3,255,-1)
    '''
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img = cv2.drawKeypoints(gray,kp,frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()
