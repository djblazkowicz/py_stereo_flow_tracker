import cv2 as cv

import numpy as np
import glob

CHECKERBOARD = (6,9)

subpix_criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW

_img_shape = None

leftImages  = glob.glob('./left/*.png')
rightImages = glob.glob('./right/*.png')

left_objpoints = []
left_imgpoints = []
right_objpoints = []
right_imgpoints = []

for fname in leftImages:
    img = cv.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        left_objpoints.append(objp)
        cv.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        left_imgpoints.append(corners)       
left_N_OK = len(left_objpoints)

for fname in rightImages:
    img = cv.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        right_objpoints.append(objp)
        cv.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        right_imgpoints.append(corners)       
right_N_OK = len(right_objpoints)

left_K = np.zeros((3, 3))
left_D = np.zeros((4, 1))
left_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(left_N_OK)]
left_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(left_N_OK)]
R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)

right_K = np.zeros((3, 3))
right_D = np.zeros((4, 1))
right_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(right_N_OK)]
right_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(right_N_OK)]

objpoints = np.array([objp]*len(left_imgpoints), dtype=np.float64)
imgpoints_left = np.asarray(left_imgpoints, dtype=np.float64)
imgpoints_right = np.asarray(right_imgpoints, dtype=np.float64)


objpoints = np.reshape(objpoints, (left_N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
imgpoints_left = np.reshape(imgpoints_left, (left_N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))
imgpoints_right = np.reshape(imgpoints_right, (right_N_OK, 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))	

(rms, left_K, left_D, right_K, right_D, R, T) = \
    cv.fisheye.stereoCalibrate( 
        objpoints,
        imgpoints_left,
        imgpoints_right,
        left_K,
        left_D,
        right_K,
        right_D,
        (1280,960),
        R,
        T,
        calibration_flags
    )


R1 = np.zeros([3,3])
R2 = np.zeros([3,3])
P1 = np.zeros([3,4])
P2 = np.zeros([3,4])
Q = np.zeros([4,4])
# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection,dispartityToDepthMap) = cv.fisheye.stereoRectify(
                    left_K, left_D,
                    right_K, right_D,
                    (1280,960), R, T,
                    0, R2, P1, P2, Q,
                    cv.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)

leftMapX, leftMapY = cv.fisheye.initUndistortRectifyMap(
    left_K, left_D, leftRectification,
    leftProjection, (1280,960), cv.CV_16SC2)
rightMapX, rightMapY = cv.fisheye.initUndistortRectifyMap(
    right_K, right_D, rightRectification,
    rightProjection, (1280,960), cv.CV_16SC2)


leftImage = cv.imread(leftImages[0])
rightImage = cv.imread(rightImages[0])

undistorted_left = cv.remap(leftImage,leftMapX,leftMapY,interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
undistorted_right = cv.remap(rightImage,rightMapX,rightMapY,interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)


#DISPARITY SETTINGS
block = 8
P1 = block * block * 8
P2 = block * block * 32
left_matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=block, P1=P1, P2=P2)
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
# Create the filter instance

#DISP FILTER SETTINGS
wsize=31
max_disp = 128
sigma = 1.5
lmbda = 8000.0
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

left_disp = left_matcher.compute(undistorted_left, undistorted_right)
right_disp = right_matcher.compute(undistorted_right,undistorted_left)
filtered_disp = wls_filter.filter(left_disp, undistorted_left, disparity_map_right=right_disp)

left_disp = cv.normalize(left_disp, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
right_disp = cv.normalize(left_disp, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
filtered_disp = cv.normalize(left_disp, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

print('leftK:')
print(left_K)
print('leftD:')
print(left_D)
print('rightK:')
print(right_K)
print('rightD:')
print(right_D)
print('R:')
print(R)
print('T')
print(T)
print('leftRectification')
print(leftRectification)
print('leftProjection')
print(leftProjection)
print('rightRectification')
print(rightRectification)
print('rightProjection')
print(rightProjection)
print('dispartityToDepthMap')
print(dispartityToDepthMap)
print('R1')
print(R1)
print('R2')
print(R2)
print('P1')
print(P1)
print('P2')
print(P2)
print('Q')
print(Q)
print('leftMapX')
print(leftMapX)
print('leftMapY')
print(leftMapY)
print('rightMapX')
print(rightMapX)
print('rightMapY')
print(rightMapY)

while True:
    cv.imshow('left',undistorted_left)
    cv.imshow('right',undistorted_right)
    cv.imshow('left disparity',left_disp )
    cv.imshow('right disparity',right_disp)
    cv.imshow('filtered disparity',filtered_disp)
    cv.waitKey(16)