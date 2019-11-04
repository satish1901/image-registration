#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:00:49 2019

@author: bisque
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:29:39 2019

@author: satish
"""

import os
import cv2
import numpy as np
import argparse
from sklearn.neighbors.nearest_centroid import NearestCentroid

#%%
'''
DIRECTORY = "/home/bisque/projects/DSP-278A/image_registration"
#FILES = []
IMAGES = f'{DIRECTORY}/images_final_test'
RESULTS = f'{DIRECTORY}/test_image_descrip'
#for x in os.listdir(IMAGES):
#    FILES.append(x)
#print(FILES)

'''
#%%

def homography_user_defined(p1_all,p2_all):
    '''Computing the homography'''
    for i in range(len(p1_all)):
        x = p1_all[i][0]
        y = p1_all[i][1]
        w = 1
        x_p = p2_all[i][0]
        y_p = p2_all[i][1]
        w_p = 1 
        A_temp = np.array([[0, 0, 0, -w_p * x, -w_p * y, -w_p * w, y_p * x, y_p * y, y_p * w ],
                           [w_p * x, w_p * y, w_p * w, 0, 0, 0, -x_p * x, -x_p * y, -x_p * w]])
        if i == 0:
           A = A_temp
        else:    
           A = np.append(A, A_temp, axis = 0)
    
    
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    
    # Homegenous Matrix 
    H = L.reshape(3,3)
    
    # Calculating Homogenous Matrix using inbuilt function
    M, mask = cv2.findHomography(p1_all, p2_all, 0)
    print(H, M)   
 
    return H


def Normalization(point):
    ''' Normalization of Coordinates (centroid to the origin and mean distance of sqrt(2))'''
    
    #point = np.reshape(point,(point.shape[0],point.shape[2]))
    m, s = np.mean(point,0), np.std(point)
    Tr = np.array([[s, 0, m[0]],[0, s, m[1]],[0, 0, 1]])
    Tr = np.linalg.inv(Tr)
    trans_point = np.dot(Tr, np.concatenate((point.T, np.ones((1,point.shape[0])))))
    trans_point = trans_point[0:2,:].T
  
    # To check average distance is sqrt(2) 
    g  = np.sum((trans_point**2), axis = 1)
    g1 = np.mean(np.sqrt(g))
    print(g1)
    return Tr, trans_point

def denormalization(T1, T2, H_norm):
    H_denorm = np.dot(np.dot( np.linalg.inv(T2.T), H_norm), T1)
    return H_denorm
  
#%%
if __name__  == '__main__' :
#    
#    ap = argparse.ArgumentParser()
#
#    #give input image path
#    ap.add_argument("-i", "--input", required=True, nargs='+', help="path toimages to register separated by space", type=str) 
#    args = vars(ap.parse_args())

    args = {}
    args["input"] = ['/home/bisque/projects/DSP-278A/image_registration/images_final_test/kitp_1.jpg', '/home/bisque/projects/DSP-278A/image_registration/images_final_test/kitp_2.jpg']

    if(len(args["input"])<2):
        print("Minimum 2 images required to register")
        exit
    else:
        print(args["input"])
        img1 = cv2.imread(args["input"][0],  cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args["input"][1],  cv2.IMREAD_GRAYSCALE)
        name = args["input"][0].split('/')[-1]
        img_name = name.split('_')[0]
        print(img_name)
        
        #ORB method for feature computation
        orb = cv2.ORB_create()
        kp1, desp1 = orb.detectAndCompute(img1,None)
        kp2, desp2 = orb.detectAndCompute(img2,None)
        
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        matches = bf.match(desp1,desp2)
        print(matches)
        print(len(matches))
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)

        '''Need to draw only good matches, so create a mask'''
#        good = []
#        for m,n in matches:
#            if m.distance < 0.75*n.distance:
#                good.append([m])
      
        # Homography Using RANSAC
#        h_ransac, status = cv2.findHomography(p1_all, p2_all, cv2.RANSAC, 5.0)
        
#        print("h_ransac", h_ransac)
        #print("status", status)
    
#        img_wrap = cv2.warpPerspective(img1, h_mat, (1000, 1000))
#        cv2.imwrite("img_wrap.jpg", img_wrap)
#        cv2.imshow("Warped Source Image", img_wrap)
#        cv2.waitKey(0)      
 
#        draw_params = dict(matchColor = (0,0,255),
#                   singlePointColor = (255,0,0),
#                   matchesMask = matchesMask,
#                   flags = 0)
        
#        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
       # PATH = f'{RESULTS}/{img_name}_12.png'
       # print(PATH)
        cv2.imwrite("img3.png", img3)