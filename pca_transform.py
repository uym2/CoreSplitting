# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:06:33 2016

@author: uym2
"""

import numpy as np
from sklearn.decomposition import PCA

def scatter_img(dither_img):
# transform a dithered image into a list of
# 2-D scattered points that represent the x and y coordinates of 
# non-zero pixels in the original img
    L = []
    h,w = dither_img.shape
    
    for y in range(h):
        for x in range(w):
            if dither_img[y][x]>0:
                L.append([x,y])                
    return np.mat(L)

def condense_into_img(N):
# N is a matrix where each row is a 2-D point
# corresponding to the coordinates of a non-zero pixel in the image
    h_trans = int(N[:,1].max()+1)
    w_trans = int(N[:,0].max()+1)
    
    L = N.tolist()
    reform_img = np.zeros((h_trans,w_trans),np.uint8)
    
    for p in L:
        reform_img[int(p[1])][int(p[0])]=255
    return reform_img
    
def shift(M,shift_vec):
    M[:,0] = M[:,0]+shift_vec[0]
    M[:,1] = M[:,1]+shift_vec[1]

def project(M,trans_mat):
    return M*trans_mat

def standard_transform(dither_img):
# M is a matrix where each row is a 2-D point
# this function finds the 2 axises that best represent data in M
# it returns the transformed data, the axises, and the shift-vector
    L = scatter_img(dither_img)
    M = np.mat(L)
    # perform PCA
    pca = PCA(n_components=2)
    pca.fit(M)
    # get the axises

    trans_mat = np.mat(pca.components_).transpose() # * r_mat # v has 2 columns corresponding to the 2 axises

# MAD@ them
    print trans_mat
    if max(abs(trans_mat[0,0]), abs(trans_mat[0,1])) > 0.99: # below 8 degrees
        print "No need!"
        trans_mat = np.eye(2) #np.mat([[1, 0], [0, 1]])
    else:
        if trans_mat[0,0] < 0:
            trans_mat = -trans_mat

        if trans_mat[0,0] < abs(trans_mat[0,1]):
            sign = np.sign(trans_mat[0,1])
            if sign == 0:
                sign = 1
            trans_mat = trans_mat * np.mat([[0, -sign], [sign, 0]])
    print trans_mat
# End MAD@ them
    
    N = project(M,trans_mat)
    # shift data so that min_x = 0 and min_y = 0
    # to avoid the case when min_x < 0 then we lose the data points that have x<0
    Min_x = N[:,0].min()
    Min_y = N[:,1].min()
    
    shift_vec = [-Min_x,-Min_y]
    shift(N,shift_vec)
    
    transform_img = condense_into_img(N)
        
    return transform_img, trans_mat, shift_vec
    
def map_back_img(transform_img,trans_mat,shift_vec):
    N = np.mat(scatter_img(transform_img))
    shift(N,[-shift_vec[0],-shift_vec[1]])
    M = project(N,trans_mat.transpose())
    return condense_into_img(M)
    
def map_back_point(p,trans_mat,shift_vec):
    p1 = np.mat(p)
    shift(p1,[-t for t in shift_vec])
    return project(p1,trans_mat.transpose()).tolist()[0]
    
