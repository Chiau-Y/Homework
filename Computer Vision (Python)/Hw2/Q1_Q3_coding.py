import matplotlib.pyplot as plt
import numpy as np
import cv2


def Disparity_Q1():
    imgL = cv2.imread('image/imL.png',0)
    imgR = cv2.imread('image/imR.png',0)
    stereo = cv2.StereoBM_create(64, 9)
    disparity = stereo.compute(imgL,imgR)

    plt.figure()
    plt.gcf().canvas.set_window_title('Result')
    plt.imshow(disparity,'gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('image_result/1.png')
    plt.show()

def NCC_Q2():
    img = cv2.imread('image/ncc_img.jpg')
    template = cv2.imread('image/ncc_template.jpg',0)
    w, h = template.shape[::-1]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply template Matching (Normalized Cross-Correlation Matching Method (cv::TM_CCORR_NORMED)，最不匹配為 0，越大越匹配)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)
    
    threshold = 0.999
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,0), 5)
    
    
    plt.figure(1)
    plt.imshow(res,cmap = 'gray')
    plt.title('Template Matching Feature'), plt.xticks([]), plt.yticks([])
    plt.savefig('image_result/2_1.jpg')
    
    plt.figure(2)
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    plt.imshow(img2)
    plt.title('5 Detected Template Images'), plt.xticks([]), plt.yticks([])
    plt.savefig('image_result/2_2.jpg')
    plt.show()
    
    
    # # 兩張圖合起來
    # plt.subplot(121)
    # plt.imshow(res,cmap='gray')
    # plt.title('Template Matching Feature'), plt.xticks([]), plt.yticks([])
    
    # plt.subplot(122)
    # b,g,r = cv2.split(img)
    # img2 = cv2.merge([r,g,b])
    # plt.imshow(img2)
    # plt.title('5 Detected Template Images'), plt.xticks([]), plt.yticks([])
    # plt.savefig('image_result/2.jpg')
    
    # plt.show()

def Keypoint_Q3():
    img = cv2.imread('image/Aerial1.jpg',0)
    img2 = cv2.imread('image/Aerial2.jpg',0)

    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    feature1 = []
    feature2 = []
    
    for i in range(len(kp1)):
        if kp1[i].size > 50:
            feature1.append(kp1[i])
    for i in range(len(kp2)):
        if kp2[i].size > 45:
            feature2.append(kp2[i])
    
    img = cv2.drawKeypoints(img, feature1, img, (0,255,0))
    img2 = cv2.drawKeypoints(img2, feature2, img2, (0,255,0))
    
    
    cv2.imwrite("image_result/FeatureAerial1.jpg",img)
    cv2.imwrite("image_result/FeatureAerial2.jpg",img2)
    cv2.imshow('FeatureAerial1 (press any key to leave)',img)
    cv2.imshow('FeatureAerial2 (press any key to leave)',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def MatchedKeypoints_Q3():
    img = cv2.imread('image/Aerial1.jpg',0)
    img2 = cv2.imread('image/Aerial2.jpg',0)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    feature1 = []
    feature2 = []
    des11 = []
    des22 = []
    

    for i in range(len(kp1)):
        if kp1[i].size > 50:
            feature1.append(kp1[i])
            des11.append(des1[i])
    for i in range(len(kp2)):
        if kp2[i].size > 45:
            feature2.append(kp2[i])
            des22.append(des2[i])
    
    des1_array = np.array(des11)
    des2_array = np.array(des22)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    # 如果k等於2，就會為每個關鍵點繪製兩條最佳匹配直線
    matches = bf.knnMatch(des1_array,des2_array, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # 比值測試的意思是首先獲取與A距離最近的點B（最近）和C（次近），只有當B/C小於閾值（0.75）才被認為是匹配。因為假設匹配是一一對應，真正匹配的理想距離為0。
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img,feature1,img2,feature2,good,None,(0,255,0),(0,255,0),flags=2)
    
    cv2.imwrite("image_result/Feature points and their corresponding points.jpg", img3)
    cv2.imshow('Feature points and their corresponding points (press any key to leave)',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
