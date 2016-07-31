# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 08:25:30 2016

@author: Mai Anh Dung
"""

import core_splitting_lib as split
import pca_transform as trans
import cv2
import pylab
import numpy as np
from scipy.ndimage import measurements,morphology
from PIL import Image

import sys
from PyQt4 import QtGui

DITHER_THRES = 205

def main(fname):
    
    img = cv2.imread(fname)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #labels, nbr_objects = measurements.label(1 * (bw_img < DITHER_THRES))
    im_close = morphology.binary_closing(1 * (bw_img < DITHER_THRES), np.ones((3,3)),iterations=1)
    labels, nbr_objects = measurements.label(im_close)
    out = Image.fromarray(255 * (labels == 0))
    pylab.imshow(out)
    pylab.show()
    print labels.shape, nbr_objects
    h, w = labels.shape
    
    labels_dict = {}
    for i in range(h):
        for j in range(w):
            key = labels[i,j]
            if key == 0:
                continue
            info_list = labels_dict.get(key, [0, 99999, 0, 99999, 0])
            info_list[0] += 1

            if info_list[1] > j:
                info_list[1] = j
    
            if info_list[2] < j:
                info_list[2] = j
    
            if info_list[3] > i:
                info_list[3] = i
    
            if info_list[4] < i:
                info_list[4] = i
            labels_dict[key] = info_list
            
    num_big = num_500 = num_400 = num_300 = num_200 = num_130 = num_100 = num_50 = \
    num_40 = num_30 = num_20 = num_10 = 0
    val_list = []
    for key in labels_dict.keys():
        val = labels_dict[key][0]
        val_list.append(val)
        if val < 10:
            num_10 += 1
        elif val < 20:
            num_20 += 1
        elif val < 30:
            num_30 += 1
        elif val < 40:
            num_40 += 1
        elif val < 50:
            num_50 += 1
        elif val < 100:
            num_100 += 1;
        elif val < 130:
            num_130 += 1
        elif val < 200:
            num_200 += 1
        elif val < 300:
            num_300 += 1
        elif val < 400:
            num_400 += 1
        elif val < 500:
            num_500 += 1
        else:
            num_big += 1
    
    mean_val = np.mean(val_list) / 2
    order_list = labels_dict.keys()
    sorted(order_list)
    
    objs = []
    for key in order_list:
        l = labels_dict[key]
        if l[0] < mean_val:
            continue
        x_start = l[1]
        x_end = l[2]
        y_start = l[3]
        y_end = l[4]
        objs.append([x_start,x_end,y_start,y_end])
        #cv2.rectangle(img,(x_start,y_start),(x_end,y_end),(0, 0, 255),2)

        #cv2.imshow("split lines",img)
        #cv2.waitKey(500)
        
    print " < 10 : %d" % num_10
    print " < 20 : %d" % num_20
    print " < 30 : %d" % num_30
    print " < 40 : %d" % num_40
    print " < 50 : %d" % num_50
    print " < 100 : %d" % num_100
    print " < 130 : %d" % num_130
    print " < 200 : %d" % num_200
    print " < 300 : %d" % num_300
    print " < 400 : %d" % num_400
    print " < 500 : %d" % num_500
    print " > 500 : %d" % num_big

    print "Mean : %f" % np.mean(val_list)
    print "Median : %f" % np.median(val_list)
    
    #rowIdx = split.group2rowNsort(objs)
    split.show_objs(objs,img)
    cv2.imwrite("output.jpeg",img)

app = QtGui.QApplication(sys.argv)
fname = QtGui.QFileDialog.getOpenFileName(None,
                    "Load Query", '.',
                    "Image Files (*.png; *.jpg; *.jpeg)")
if fname == "":
    sys.exit()
app.quit()
main(fname)
sys.exit()