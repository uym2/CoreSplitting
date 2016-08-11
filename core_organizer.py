# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:09:36 2016

@author: uym2
"""

import numpy as np
import core_splitting_lib as split
from sklearn import linear_model

def is_blocked(objIdx,objList,sIdxList,flagList,dim):
    if dim == 'r':
        s = 0
        e = 1
    elif dim == 'c':
        s = 2
        e = 3
    ctr = (objList[objIdx][s] + objList[objIdx][e])/2
    
    for i in sIdxList:
        if i == objIdx:
            return False
        if flagList[i] and objList[i][s] < ctr and objList[i][e] > ctr:
            return True
    return False

def get_unblocked_idx(objList,sIdxList,flagList,dim):
    unblkIdxList = []
    for i in sIdxList:
        if flagList[i] and not is_blocked(i,objList,sIdxList,flagList,dim):
            unblkIdxList.append(i)
            
    for i in unblkIdxList:
        flagList[i] = False
        
    return unblkIdxList
    
def get_row_unblocked_idx(objList,sIdxList,flagList):
    return get_unblocked_idx(objList,sIdxList,flagList,'r')
    
def get_col_unblocked_idx(objList,sIdxList,flagList):
    return get_unblocked_idx(objList,sIdxList,flagList,'c')

def group2Rows(objList):
    sIdxList = np.argsort([(obj[2]+obj[3])/2 for obj in objList])
    flagList = [True]*len(sIdxList)
    
    #objCtrs_x = [(obj[0]+obj[1])/2 for obj in objList]
    #objCtrs_y = [-(obj[2]+obj[3])/2 for obj in objList]
    #plt.plot(objCtrs_x, objCtrs_y, '.y')
    
    #colors = ['.r','.g','.b']
    
    objSize = split.typical_obj_size(objList)
    rowIdxList = []
    
    #color_idx = 0
    
    while sum(flagList):
        idxList = get_row_unblocked_idx(objList,sIdxList,flagList)
        
        if len(idxList) == 1:
            rowIdxList.append(idxList)
            continue
        
        unblk_x = np.mat([(objList[i][0]+objList[i][1])/2 for i in idxList]).transpose()
        unblk_y = np.array([-(objList[i][2]+objList[i][3])/2 for i in idxList])
        
        
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=objSize[0]/2)
        model_ransac.fit(unblk_x, unblk_y)
        
        #line_x = np.arange(min(objCtrs_x), max(objCtrs_x))
        #line_y_ransac = model_ransac.predict(line_x[:, np.newaxis])
        
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        
        row = [idxList[i] for i in range(len(inlier_mask)) if inlier_mask[i]]        
        
        for i in range(len(outlier_mask)):
            if outlier_mask[i]:
                if model_ransac.predict(unblk_x[i,np.newaxis])>unblk_y[i]+objSize[0]/2:
                    flagList[idxList[i]] = True
                else:
                    row.append(idxList[i])
        rowIdxList.append(row)
        #plt.plot(unblk_x[inlier_mask], unblk_y[inlier_mask],colors[color_idx%len(colors)])
        #plt.plot(line_x, line_y_ransac, '-b', label='RANSAC regressor')
        #color_idx = color_idx + 1
    return rowIdxList