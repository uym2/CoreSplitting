# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:09:36 2016

@author: uym2
"""

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