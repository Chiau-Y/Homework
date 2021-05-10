from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import time
import cv2

def Findcorners():   #Q1.1
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)        #8*11 points     
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    # etx = []

    for i in range(1, 16):
        #img = cv2.imread("D:/Master (2019-2021)/Homework/DeepLearning/Homework/HW1/images/CameraCalibration/%d.bmp"%(i))
        img = cv2.imread("images/CameraCalibration/%d.bmp" %(i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
        
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8,11), corners2,ret)   #8*11 points
            #cv2.imwrite("D:/Master (2019-2021)/Homework/DeepLearning/hw1/1/1.1/result_%d.bmp"%(i), img)
            cv2.imwrite("1.1/1.1_%d.bmp"%(i), img)
            cv2.namedWindow('1.1_%d (press any key to leave)'%(i),0)
            cv2.resizeWindow('1.1_%d (press any key to leave)'%(i), 700, 700)
            cv2.imshow('1.1_%d (press any key to leave)'%(i),img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Findintrinsicmatrix():   #Q1.2
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)        #8*11 points     
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(1, 16):
        img = cv2.imread("images/CameraCalibration/%d.bmp" %(i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:],None,None)   #mtx = intrinsic matrix, dist = Distortion Matrix
    print("Intrinsic Matrix : ","\n",mtx,"\n")

def Findextrinsicmatrix (x):   #Q1.3

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)        #8*11 points     
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    etx = []
    
    for i in range(1, 16):
        img = cv2.imread("images/CameraCalibration/%d.bmp" %(i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:],None,None)   #mtx = intrinsic matrix, dist = Distortion Matrix
    
    rtx,jmx = cv2.Rodrigues(rvecs[x-1])   #3 by 3 matrix
    etx.append(np.hstack((rtx,tvecs[x-1])))   #Extrinsic Matrix
    #etx.append(np.c_[rtx,tvecs[x]])
    print("Entrinsic Matrix of %d.bmp : " %(x),"\n",etx[0],"\n")

def Finddistortionmatrix():   #Q1.4
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)        #8*11 points     
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for i in range(1, 16):
        img = cv2.imread("images/CameraCalibration/%d.bmp" %(i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)   #mtx = intrinsic matrix, dist = Distortion Matrix
    print("Distortion Matrix  : ","\n",dist,"\n")

def Augmentedreality():   #Q2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((8*11,3), np.float32)        #8*11 points
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    axis = np.float32([[1,1,0], [1,-1,0], [-1,-1,0], [-1,1,0], [0,0,-2], [0,0,-2], [0,0,-2], [0,0,-2]])
    
    for i in range(1, 6):
        img = cv2.imread("images/CameraCalibration/%d.bmp" %(i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
    
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            _ ,rvecs, tvecs, inliers= cv2.solvePnPRansac(objp, corners2, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1,2)
    
            cv2.line(img, tuple(imgpts[4]), tuple(imgpts[0]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[4]), tuple(imgpts[1]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[4]), tuple(imgpts[2]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[4]), tuple(imgpts[3]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[1]), tuple(imgpts[2]),[0,0,255],4)
            cv2.line(img, tuple(imgpts[2]), tuple(imgpts[3]),[0,0,255],4)
    
            cv2.imwrite("2/2_%d.bmp"%(i), img)
            cv2.namedWindow("2_%d.bmp"%(i),0)
            cv2.resizeWindow("2_%d.bmp"%(i), 700, 700)
            cv2.imshow("2_%d.bmp"%(i),img)
            time.sleep(0.5)
            cv2.waitKey(1000)
            cv2.destroyWindow("2_%d.bmp"%(i))

def ImageOriginaltransform():   #Q3.1
    img = cv2.imread("images/OriginalTransform.png")
    cv2.imshow('OriginalTransform', img)
def Rotationscalingtranslation(r,s,tx,ty):   #Q3.1
    img = cv2.imread("images/OriginalTransform.png")
    h,w = img.shape[:2]
    R = cv2.getRotationMatrix2D((130,125), r, s)
    rotated = cv2.warpAffine(img, R, (w, h))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(rotated, M, (w, h))

    cv2.imshow('Transforms (press any key to leave)', shifted)
    cv2.imwrite("3/3.1.png", shifted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.destroyWindow('Transforms')

def Perspectivetransformation():   #Q3.2
    global i

    def draw_point(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_c,(x,y),10,(0,0,255),-1)
            pts1.append([float(x),float(y)])  #int turns into float
            global i
            i +=1

    img = cv2.imread("images/OriginalPerspective.png")
    img_c = np.copy(img)
    cv2.namedWindow('OriginalPerspective')
    cv2.setMouseCallback('OriginalPerspective',draw_point)

    pts1 = []
    pts1_n = []
    i = 0

    pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])

    while(1):
        cv2.imshow('OriginalPerspective',img_c)
        if i == 4:   #press Enter to show Perspective Result Image
            pts1_n = np.vstack((pts1[0],pts1[1]))  #list turns into array
            pts1_n = np.vstack((pts1_n,pts1[2]))
            pts1_n = np.vstack((pts1_n,pts1[3]))
            pts1_n = pts1_n.astype('float32')      #float64 turns into float32
            # 用這八個點獲得3*3的變換矩陣
            M = cv2.getPerspectiveTransform(pts1_n,pts2)
            dst = cv2.warpPerspective(img,M,(470,470))
            cv2.imshow('Perspective Result Image(press Esc to leave)',dst)
            cv2.imwrite("3/3_Original.png", img_c)
            cv2.imwrite("3/3_Result.png", dst)
            i = 0
        if cv2.waitKey(1) & 0xFF == 27:   #press Esc to leave
            break
    cv2.destroyAllWindows()

    #原理 : https://blog.csdn.net/guduruyu/article/details/72518340

def Findcontour():   #Q4
    img = cv2.imread("images/Contour.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
    cv2.imwrite("4/1.png", img)
    cv2.imshow('FindContour (press any key to leave)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
