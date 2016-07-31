# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:07:55 2016

@author: uym2
"""

import core_splitting_lib as split
import pca_transform as trans
#import sip
#sip.setapi("QString", 2)
#sip.setapi("QVariant", 2)

import cv2
import numpy as np

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

dither_img = split.naive_dithering(img)

transform_img,trans_mat,shift = trans.standard_transform(dither_img)
objs = []
h_trans,w_trans = transform_img.shape
cv2.imshow("transform",transform_img)
cv2.waitKey(0)
split.find_all_splits(transform_img,objs,0,w_trans,0,h_trans)
objs = split.remove_tiny_objs(objs)
rowIdx = split.group2rowNsort(objs)
colIdx = split.group2colNsort(objs)
#matIdx = split.infer_missing_idx(rowIdx,colIdx)
# map rowList to the original axises
#rowList_trans = [] # rowList standard

#for row in rowList:
#    row_trans = []
objs_trans = []
for obj in objs:
    if obj:
        p1 = [obj[0],obj[2]]
        p2 = [obj[1],obj[3]]
        p1_trans = trans.map_back_point(p1,trans_mat,shift)
        p2_trans = trans.map_back_point(p2,trans_mat,shift)
        objs_trans.append([int(t) for t in [p1_trans[0],p2_trans[0],p1_trans[1],p2_trans[1]]])
#rgb = cv2.cvtColor(transform_img,cv2.COLOR_GRAY2BGR)
#split.show_objs_by_matrix(matIdx,objs_trans,img)        
split.show_objs_by_dim(colIdx,objs_trans,img)
#split.show_objs(objs_trans,img)