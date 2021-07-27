import numpy as np
import cv2 as cv
import json
path = 'images'


def corners():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(14):
        name  = path + '/foto_' + str(i) + '.png'
        img = cv.imread(name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6,9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (6,9), corners2, ret)
            #cv.imshow(name, img)
            #cv.waitKey(500)
    return objpoints,imgpoints,gray
        
def save_dic(dic):
    name = 'dic_of_calibration.json'
    fp = open(name, 'w') 
    json.dump(dic, fp,  indent=4)
    fp.close()


'''
# undistort

img = cv.imread('images/foto_12.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('images/calibresult.png', dst)


vid = cv.VideoCapture(0)s
while(True):
    _, image = vid.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(gray, mapx, mapy, cv.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('imagen', dst)
    cv.waitKey(500)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
'''
def main():
    objpoints,imgpoints,gray = corners()
    names                    = ['ret', 'mtx', 'dist']
    dic = {}
    calibrate = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    for name,value in zip(names,calibrate[0:3]):
        if isinstance(value,np.ndarray):
            dic[name] = value.tolist()
        else:
            dic[name] = value  
    save_dic(dic)

if __name__ == '__main__':
    main()