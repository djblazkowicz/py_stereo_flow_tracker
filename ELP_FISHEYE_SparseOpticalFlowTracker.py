#importing packages
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image


#camera settings
cap = cv.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 960)
cap.set(cv.CAP_PROP_FPS, 30)
cap.set(cv.CAP_PROP_MONOCHROME, 1)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv.CAP_PROP_EXPOSURE, -8)

#BLOB DETECTOR
detector = cv.SimpleBlobDetector()

#DISPARITY SETTINGS
block = 3
P1 = block * block * 2
P2 = block * block * 8
left_matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=block, P1=P1, P2=P2)
#left_matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=block)
#left_matcher =cv.StereoBM_create(numDisparities=96, blockSize=15)
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

#disparity = cv.StereoBM_create(numDisparities=158, blockSize=21)
#disparity.setPreFilterType(1)
#disparity.setPreFilterSize(5)
#disparity.setPreFilterCap(63)
#disparity.setMinDisparity(8)
#disparity.setNumDisparities(160)
#disparity.setTextureThreshold(998)
#disparity.setUniquenessRatio(20)
#disparity.setSpeckleRange(4)
#disparity.setSpeckleWindowSize(104)

# PARAMETERS for Lucas-Kanade optical flow tracking
lk_params = dict( winSize  = (21, 21),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                  flags = (cv.OPTFLOW_LK_GET_MIN_EIGENVALS),
                  minEigThreshold = 0.0085)

#max number of concurrent features to keep
MAXFEATURES = 2000


#calibrated camera intrinsics
#leftK = np.array([[312.5253575140719, 0.0, 719.081633375778], [0.0, 312.32632059717497, 484.18338866902945], [0.0, 0.0, 1.0]])
#rightK = np.array([[307.5979750282324, 0.0, 724.2644503718968], [0.0, 307.64174740700685, 505.90926036352687], [0.0, 0.0, 1.0]])

def loadCalib(filepath):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    filepath (str): The file path to the camera file
    Returns
    -------
    K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
    P_l (ndarray): Projection matrix for left camera. Shape (3,4)
    K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
    P_r (ndarray): Projection matrix for right camera. Shape (3,4)
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_r = np.reshape(params, (3, 4))
        K_r = P_r[0:3, 0:3]
    return K_l, P_l, K_r, P_r

def loadStereoCalib(filepath):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    filepath (str): The file path to the camera file
    Returns
    -------
    K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
    P_l (ndarray): Projection matrix for left camera. Shape (3,4)
    K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
    P_r (ndarray): Projection matrix for right camera. Shape (3,4)
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        D_l = np.reshape(params, (4, 1))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P_r = np.reshape(params, (3, 4))
        K_r = P_l[0:3, 0:3]
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        D_r = np.reshape(params, (4, 1))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        R = np.reshape(params, (3, 3))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        T = np.reshape(params, (3, 1))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        leftRectification = np.reshape(params, (3, 3))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        leftProjection = np.reshape(params, (3, 4))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        rightRectification = np.reshape(params, (3, 3))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        rightProjection = np.reshape(params, (3, 4))
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        disparityToDepthMap = np.reshape(params, (4, 4))
    return P_l,K_l,D_l,P_r,K_r,D_r,R,T,leftRectification,leftProjection,rightRectification,rightProjection,disparityToDepthMap


#K_l,P_l,K_r,P_r = loadCalib('./calib.txt')
P_l,K_l,D_l,P_r,K_r,D_r,R,T,leftRectification,leftProjection,rightRectification,rightProjection,disparityToDepthMap = loadStereoCalib('./STEREO_calib2.txt')

leftMapX, leftMapY = cv.fisheye.initUndistortRectifyMap(
    K_l, D_l, leftRectification,
    leftProjection, (1280,960), cv.CV_16SC2)
rightMapX, rightMapY = cv.fisheye.initUndistortRectifyMap(
    K_r, D_r, rightRectification,
    rightProjection, (1280,960), cv.CV_16SC2)

DIM = (1280,960)
D_l = np.array([[-0.06368196408307585], [0.012889005452688151], [0.003132339223866623], [-0.005726470317750285]])
D_r = np.array([[-0.05808067301819261], [0.038969508131868766], [-0.04389979720360535], [0.01798604746516268]])

def undistort(img,mapX,mapY):

    undistorted_img = cv.remap(img, mapX, mapY, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return undistorted_img

#get frame from camera, split it into left/right images
def getFrames(videocapture):
   
    ret, frame = videocapture.read()

    h, w, channels = frame.shape
    half = w//2
    decimatew = w//4
    decimateh = h//4

    dim_quart = (decimatew,decimateh)
    dim_eight = (decimatew//2,decimateh//2)
    dim = (w,h) 

    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame,dim_quart)
    frame = cv.resize(frame,dim)
    frame = cv.GaussianBlur(frame,(5,5),0)
    frame = cv.equalizeHist(frame)
    frameL = frame[:,:half]
    frameR = frame[:,half:]
    frameL = undistort(frameL,leftMapX,leftMapY)
    frameR = undistort(frameR,rightMapX,rightMapY)
    dimh,dimw =  frameL.shape
    dim_depth = (dimw,dimh)
    #depth_scaled_left = cv.resize(frameL,dim_quart)
    #depth_scaled_right = cv.resize(frameR,dim_quart)

    #left_disp = left_matcher.compute(frameL, frameR)
    #right_disp = right_matcher.compute(frameR,frameL)
    #filtered_disp = wls_filter.filter(left_disp, frameL, disparity_map_right=right_disp)
    #filtered_disp = cv.resize(filtered_disp,dim_depth)

    return frame,frameL,frameR#,filtered_disp

#get features for tracking
def getFeatures(img):
    try:
        features = cv.goodFeaturesToTrack(img, mask = None, maxCorners = MAXFEATURES, qualityLevel = 0.01, minDistance = 50, blockSize = 30, useHarrisDetector = True, k = 0.05)
        return features
    except:
        return None



def _form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector

    Parameters
    ----------
    R (ndarray): The rotation matrix. Shape (3,3)
    t (list): The translation vector. Shape (3)
    Returns
    -------
    T (ndarray): The transformation matrix. Shape (4,4)
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def reprojection_residuals(dof, q1, q2, Q1, Q2):
    """
    Calculate the residuals

    Parameters
    ----------
    dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
    q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
    q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
    Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
    Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

    Returns
    -------
    residuals (ndarray): The residuals. In shape (2 * n_points * 2)
    """
    # Get the rotation vector
    r = dof[:3]
    # Create the rotation matrix from the rotation vector
    R, _ = cv.Rodrigues(r)
    # Get the translation vector
    t = dof[3:]
    # Create the transformation matrix from the rotation matrix and translation vector
    transf = _form_transf(R, t)

    # Create the projection matrix for the i-1'th image and i'th image
    f_projection = np.matmul(P_l, transf)
    b_projection = np.matmul(P_l, np.linalg.inv(transf))

    # Make the 3D points homogenize
    ones = np.ones((q1.shape[0], 1))
    Q1 = np.hstack([Q1, ones])
    Q2 = np.hstack([Q2, ones])

    # Project 3D points from i'th image to i-1'th image
    q1_pred = Q2.dot(f_projection.T)
    # Un-homogenize
    q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

    # Project 3D points from i-1'th image to i'th image
    q2_pred = Q1.dot(b_projection.T)
    # Un-homogenize
    q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

    # Calculate the residuals
    residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
    return residuals

def matchStereoFeatures(oldpoints,newpoints,old_disp,new_disp, min_disp,max_disp):
    def getIndexes(points,disp):
        #this is rubbish but the sub-pixel precision corners can somehow have coords bigger than the image dimension
        points_i = points.astype(int)
        for point in points_i:
            if point[0] >= disp.shape[1]:
                point[0] = disp.shape[1] - 1
            if point[1] >= disp.shape[0]:
                point[1] = disp.shape[0] - 1

        disp_points = disp[points_i[:,1], points_i[:,0]]
        disp_mask = np.where(np.logical_and(min_disp < disp_points, disp_points < max_disp), True, False)
        return disp_points, disp_mask

    disp_old_l,mask_old_l = getIndexes(oldpoints,old_disp)
    disp_new_l,mask_new_l = getIndexes(newpoints,new_disp)

    in_bounds = np.logical_and(mask_old_l, mask_new_l)

    goodQold_l, goodQnew_l, disp_old_l, disp_new_l = oldpoints[in_bounds], newpoints[in_bounds], disp_old_l[in_bounds], disp_new_l[in_bounds]


    #for some reason we have more features in the new image, likely due to how they are picked up between frames.
    #this is rubbish but will work better than nothing
    if len(goodQold_l) > len(goodQnew_l):
        goodQold_l = goodQold_l[0:len(goodQnew_l)]  
    if len(goodQnew_l) > len(goodQold_l):
        goodQnew_l = goodQnew_l[0:len(goodQold_l)]

    goodQold_r = np.copy(goodQold_l)
    goodQnew_r = np.copy(goodQnew_l)
    goodQold_r[:, 0] -= disp_old_l
    goodQnew_r[:, 0] -= disp_new_l

    return goodQold_l, goodQnew_l, goodQold_r, goodQnew_r

def calc_3d(q1_l, q1_r, q2_l, q2_r):
    """
    Triangulate points from both images 
        
    Parameters
    ----------
    q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
    q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
    q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
    q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

    Returns
    -------
    Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
    Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
    """

    # Triangulate points from i-1'th image
    Q1 = cv.triangulatePoints(leftProjection, rightProjection, q1_l.T, q1_r.T).T

    # Un-homogenize
    #Q1 = np.transpose(Q1[:3] / Q1[3])
    Q1 /= Q1[:,3:]
    
    # Triangulate points from i'th image
    Q2 = cv.triangulatePoints(leftProjection, rightProjection, q2_l.T, q2_r.T).T

    # Un-homogenize
    #Q2 = np.transpose(Q2[:3] / Q2[3])
    Q2 /= Q2[:,3:]

    outQ1 = np.delete(Q1, 3, 1)
    outQ2 = np.delete(Q2, 3, 1)

    return outQ1, outQ2

def estimate_pose(q1, q2, Q1, Q2, max_iter=100):
    """
    Estimates the transformation matrix
    
    Parameters
    ----------
    q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
    q2 (ndarray): Feature points in i'th image. Shape (n, 2)
    Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
    Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
    max_iter (int): The maximum number of iterations

    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
    """
    early_termination_threshold = 5

    # Initialize the min_error and early_termination counter
    min_error = float('inf')
    early_termination = 0

    for _ in range(max_iter):
        # Choose 6 random feature points
        sample_idx = np.random.choice(range(q1.shape[0]), 6)
        sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

        # Make the start guess
        in_guess = np.zeros(6)
        # Perform least squares optimization
        opt_res = least_squares(reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

        # Calculate the error for the optimized transformation
        error = reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
        error = error.reshape((Q1.shape[0] * 2, 2))
        error = np.sum(np.linalg.norm(error, axis=1))

        # Check if the error is less the the current min error. Save the result if it is
        if error < min_error:
            min_error = error
            out_pose = opt_res.x
            early_termination = 0
        else:
            early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = _form_transf(R, t)
        return transformation_matrix

# color used for feature visualization
color = (0,125,255)


#INITIAL FRAME
framecounter = 0
#oldframe,oldframeL,oldframeR,olddisparity = getFrames(cap)
oldframe,oldframeL,oldframeR = getFrames(cap)
oldfeatures = getFeatures(oldframeL)
oldfeaturesR = getFeatures(oldframeR)

# Create a mask image for drawing purposes
mask = np.zeros_like(oldframeL)
mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
cur_pose =np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.,],[0.,0.,0.,1.]])

while cap.isOpened():

    
    framecounter += 1

    start = time.perf_counter()

    #take frame and future points
    #frame,frameL,frameR,newdisparity = getFrames(cap)
    frame,frameL,frameR = getFrames(cap)
    future_points = getFeatures(frameL)
    future_pointsR = getFeatures(frameR)

    #track past features onto new frame
    newfeatures,st,err = cv.calcOpticalFlowPyrLK(oldframeL,frameL,oldfeatures, None, **lk_params)
    newfeaturesR,stR,errR = cv.calcOpticalFlowPyrLK(oldframeR,frameR,oldfeaturesR, None, **lk_params)
    #print('ERROR:')
    #print(err)
    
    #do the thing if we have acceptable optical flow
    if (newfeatures is not None and len(newfeatures) > 50) and (newfeaturesR is not None and len(newfeaturesR) > 50):

        # Select good points
        good_new = newfeatures[st==1]
        good_old = oldfeatures[st==1]

        good_newR = newfeaturesR[stR==1]
        good_oldR = oldfeaturesR[stR==1]

        #goodOldQleft,goodOldQright,goodQleft,goodQright = matchStereoFeatures(good_old,good_new,olddisparity,newdisparity,-1,400)


        #Q1calc,Q2calc = calc_3d(goodOldQleft,goodOldQright,goodQleft,goodQright)

        #disp_lofasz = cv.normalize(olddisparity, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)#

        #transformation_matrix = estimate_pose(goodOldQleft,goodQleft,Q1calc,Q2calc)
        #cur_pose = np.matmul(cur_pose,transformation_matrix)
        #print(cur_pose)

        #update reference frames
        oldframe = frame
        oldframeL = frameL
        oldframeR = frameR
        #olddisparity = newdisparity

        #update reference features
        oldfeatures = good_new.reshape(-1,1,2)
        oldfeaturesR = good_newR.reshape(-1,1,2)
          
        #visualize tracked points
        bgr = cv.cvtColor(frameL,cv.COLOR_GRAY2BGR)
        bgr_raw = bgr.copy()
        bgrR = cv.cvtColor(frameR,cv.COLOR_GRAY2BGR)

        #bgrDisp = cv.cvtColor(disp_lofasz,cv.COLOR_GRAY2BGR)
        for i, (new, old) in enumerate(zip(good_old, good_new)):
            a, b = new.ravel()
            c, d = old.ravel()
            bgr = cv.circle(bgr, (int(a), int(b)), 5, color, -1)

        for i, (new, old) in enumerate(zip(good_newR, good_oldR)):
            a, b = new.ravel()
            c, d = old.ravel()
            bgrR = cv.circle(bgrR, (int(a), int(b)), 5, color, -1)

        blob = bgr.copy()
        blob[np.where((blob!=[0,125,255]).all(axis=2))] = [0,0,0]
        blobR = bgrR.copy()
        blobR[np.where((blobR!=[0,125,255]).all(axis=2))] = [0,0,0]
        #left_disp = left_matcher.compute(blob, blobR)
        #right_disp = right_matcher.compute(blobR,blob)
        #filtered_disp = wls_filter.filter(left_disp, blob, disparity_map_right=right_disp)
        #filtered_disp = cv.resize(filtered_disp,(1280,960))
        #disp_lofasz = cv.normalize(filtered_disp, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)#
        #bgrDisp = cv.cvtColor(disp_lofasz,cv.COLOR_GRAY2BGR)
        grayblob = cv.cvtColor(blob,cv.COLOR_BGR2GRAY)
        #keypoints = detector.detect(grayblob)

        #blob = cv.drawKeypoints(blob, keypoints, 0, (0, 0, 255),
        #                         flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


        #add future points to reference features to next iteration so we don't run out of tracked features
        oldfeatures = np.concatenate((oldfeatures,future_points))
        oldfeaturesR = np.concatenate((oldfeaturesR,future_pointsR))


    #reset if the number of tracked features is too low
    else:
        print('RAN OUT OF GOOD FEATURES, RESETTING ON FRAME: ',str(framecounter))
        oldfeatures = getFeatures(frameL)
        oldfeaturesR = getFeatures(frameR)

        if newfeatures is not None:
            oldfeatures = np.concatenate((newfeatures,oldfeatures))
        if newfeaturesR is not None:
            oldfeaturesR = np.concatenate((newfeaturesR,oldfeatures))

        oldframe = frame
        oldframeL = frameL
        oldframeR = frameR
        #olddisparity = newdisparity
        cv.waitKey(16)
        continue
    #cull features if we have too many collected
    if len(oldfeatures) > MAXFEATURES:
        oldfeatures = oldfeatures[0:MAXFEATURES]
    if len(oldfeaturesR) > MAXFEATURES:
        oldfeaturesR = oldfeaturesR[0:MAXFEATURES]

    end = time.perf_counter()
    
    total_time = end - start
    fps = 1 / total_time
    cv.putText(bgr, f'FPS: {int(fps)}', (20,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv.putText(bgr, f'X: {cur_pose[0,3]}', (20,200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv.putText(bgr, f'Z: {cur_pose[1,3]}', (20,350), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv.putText(bgr, f'Y: {cur_pose[2,3]}', (20,500), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #draw visualization and sleep for 1ms
    cv.imshow('frame', bgr)
    cv.imshow('frameR', bgrR)
    cv.imshow('frameblob', blob)
    #cv.imshow('disp',bgrDisp)
    cv.waitKey(1)