#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:00:23 2019

@author: satish, devendra
"""
import cv2
import numpy as np
import argparse

class image_registration:
    
    def homography_user_defined(self, p1_all,p2_all):
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
        return H
        
    def Normalization(self, point):
        ''' Normalization of Coordinates (centroid to the origin and mean distance of sqrt(2))'''
    
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
        
    def denormalization(self, T1, T2, H_norm):
        H_denorm = np.dot(np.dot( np.linalg.inv(T2), H_norm), T1)
        return H_denorm
        
#%%        
def main():   
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, nargs='+', help="path toimages to register separated by space", type=str)
    ap.add_argument("-a", "--algorithm", help="h_mat, h_mat_denorm, h_ransac", type=str)
    args = vars(ap.parse_args())

    #image_registration class object
    imgReg = image_registration()
    if(len(args["input"])<2):
        print("Minimum 2 images required to register")
        exit
    else:
        if(args["algorithm"] is None):
            print("Enter algorithm type, h_mat, h_mat_denorm, h_ransac")
            print("Running RANSAC by default, h_ransac")
            print(args["input"])
        img1 = cv2.imread(args["input"][0])
        img2 = cv2.imread(args["input"][1])
        
        dim = (640, 480)
        img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
        
        #detect and compute features using SIFT
        surf = cv2.xfeatures2d.SIFT_create()
        kp1, desp1 = surf.detectAndCompute(img1, None)
        kp2, desp2 = surf.detectAndCompute(img2, None)
        print("Done computing features")    
        '''FLANN parameters'''
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)        
        print("Running Flann Matcher..")
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desp1,desp2,k=2)
        
        '''Need to draw only good matches, so create a mask'''
        matchesMask = [[0,0] for i in range(len(matches))]
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.55*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
                
        MIN_MATCH_COUNT = 10        
        if len(good) > MIN_MATCH_COUNT:
           src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
           dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        # taking 1000 keypoints to calculate homography    
        if(len(src_pts)>1000):
            num_of_points = 1000
        else:
            num_of_points = len(src_pts)
        p1_all = src_pts[0:num_of_points,:,:]
        p2_all = dst_pts[0:num_of_points,:,:]
        # Reshaping from 3D to 2D 
        p1_all = np.reshape(p1_all,(p1_all.shape[0],p1_all.shape[2]))
        p2_all = np.reshape(p2_all, (p2_all.shape[0], p2_all.shape[2]))
        # homograpy matrix 
        if (args["algorithm"] == 'h_mat'):
            h_mat = imgReg.homography_user_defined(p1_all,p2_all)
            img_wrap = cv2.warpPerspective(img1, h_mat, (img1.shape[1], img1.shape[0]))
            print("Done Computing homography", h_mat)
        elif (args["algorithm"] == 'h_mat_denorm'):
            # Normalization 
            Trans_matrix_1, p1_all_norm = imgReg.Normalization(p1_all)
            Trans_matrix_2, p2_all_norm = imgReg.Normalization(p2_all)
            h_mat_norm = imgReg.homography_user_defined(p1_all_norm, p2_all_norm)
            h_mat_denorm = imgReg.denormalization(Trans_matrix_1, Trans_matrix_2, h_mat_norm)
            img_wrap = cv2.warpPerspective(img1, h_mat_denorm, (img1.shape[1], img1.shape[0]))
            print("Computed after Normalization", h_mat_denorm)
        else:
            # Homography Using RANSAC
            h_ransac, status = cv2.findHomography(p1_all, p2_all, cv2.RANSAC, 5.0)       
            print("Computed homography using RANSAC", h_ransac)
            img_wrap = cv2.warpPerspective(img1, h_ransac, (img1.shape[1], img1.shape[0]))
        
        results = cv2.addWeighted(img2, 0.4, img_wrap, 0.7 , 0)
        cv2.imwrite("Transformed.png", img_wrap)
        cv2.imwrite("regImg.png", results)
        
        draw_params = dict(matchColor = (0,0,255),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
        
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        cv2.imwrite("keypoint.png", img3)
        
if __name__ == "__main__":
    main()
