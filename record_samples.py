import numpy as np
import cv2 as cv
import glob

camera = cv.VideoCapture(0)
camera.set(3, 2560)
camera.set(4, 960)
camera.set(cv.CAP_PROP_FPS, 60)

counter = 1

vig = cv.imread('./480.png')

CHECKERBOARD = (6,9)

while True:

    pathLeft = ('./left/' + str(counter) + '.png')
    pathRight = ('./right/' + str(counter) + '.png')
    ret, img = camera.read()



    #cv.imshow('img',img)

    h, w, channels = img.shape

    half = w//2

    if ret:
        frameL = img[:,:half]
        frameR = img[:,half:]

        #frameL = cv.bitwise_and(frameL,vig)
        #frameR = cv.bitwise_and(frameR,vig)

        retL, corners = cv.findChessboardCorners(frameL, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        retR, corners = cv.findChessboardCorners(frameR, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        if retL and retR:
            cv.imwrite(pathLeft,frameL)
            cv.imwrite(pathRight,frameR)
            cv.drawChessboardCorners(frameL, CHECKERBOARD, corners, ret)
            cv.drawChessboardCorners(frameR, CHECKERBOARD, corners, ret)
            cv.imshow("left",frameL)
            cv.imshow("right",frameR)

            counter = counter + 1
            k = cv.waitKey(100)

