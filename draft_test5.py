# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:35:40 2016

@author: uym2
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import core_splitting_lib as split
import core_organizer as org
from sklearn import linear_model

import sys
from PyQt4 import QtGui

app = QtGui.QApplication(sys.argv)
fname = QtGui.QFileDialog.getOpenFileName(None,
                    "Load Query", '.',
                    "Image Files (*.png; *.jpg; *.jpeg)")
if fname == "":
    sys.exit()
app.quit()

img = cv2.imread(fname)
cv2.imshow("img",img)

dither_img = split.naive_dithering(img)
objList = split.label_cores(dither_img)
objList = split.remove_tiny_objs(objList)
objList = split.handle_giant(objList,dither_img)

sIdxList = np.argsort([(obj[2]+obj[3])/2 for obj in objList])
flagList = [True]*len(sIdxList)

objCtrs_x = [(obj[0]+obj[1])/2 for obj in objList]
objCtrs_y = [-(obj[2]+obj[3])/2 for obj in objList]
plt.plot(objCtrs_x, objCtrs_y, '.y')

colors = ['.r','.g','.b']

objSize = split.typical_obj_size(objList)

color_idx = 0

while sum(flagList):
    idxList = org.get_row_unblocked_idx(objList,sIdxList,flagList)
    
    unblk_x = np.mat([(objList[i][0]+objList[i][1])/2 for i in idxList]).transpose()
    unblk_y = np.array([-(objList[i][2]+objList[i][3])/2 for i in idxList])
    
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=objSize[0]/2)
    model_ransac.fit(unblk_x, unblk_y)
    
    line_x = np.arange(min(objCtrs_x), max(objCtrs_x))
    line_y_ransac = model_ransac.predict(line_x[:, np.newaxis])
    
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    for i in range(len(outlier_mask)):
        if outlier_mask[i] and model_ransac.predict(unblk_x[i,np.newaxis])>unblk_y[i]+objSize[0]/2:
            flagList[idxList[i]] = True
    
    plt.plot(unblk_x[inlier_mask], unblk_y[inlier_mask],colors[color_idx%len(colors)])
    plt.plot(line_x, line_y_ransac, '-b', label='RANSAC regressor')
    color_idx = color_idx + 1

