# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 08:25:30 2016

@author: Mai Anh Dung
"""

import cv2
import numpy as np
from scipy.ndimage import measurements,morphology

import sys
from PyQt4 import QtGui
import core_splitting_lib as clib

DITHER_THRES = 205

def imfill(im_in):
    im_floodfill = im_in.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out
    


def main(fname):
    
    img = cv2.imread(fname)
    max0 = np.max(img[:,:,0])
    max1 = np.max(img[:,:,1])
    max2 = np.max(img[:,:,2])
    min0 = np.min(img[:,:,0])
    min1 = np.min(img[:,:,1])
    min2 = np.min(img[:,:,2])
    img[:,:,0] = np.uint8(255./(max0 - min0) * (img[:,:,0] - min0))
    img[:,:,1] = np.uint8(255./(max1 - min1) * (img[:,:,1] - min1))
    img[:,:,2] = np.uint8(255./(max2 - min2) * (img[:,:,2] - min2))
#    img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
    cv2.imshow("Origine",img)
    
#    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(img,3,75,75)
#    blur= cv2.medianBlur(img,5)
    cv2.imshow("Blur",blur)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #dst = clib.naive_dithering(gray)
    dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    #dst = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
#    ret2,dst = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    dst = cv2.Canny(blur, 100, 255, L2gradient = True)
    cv2.imshow("Dst",dst)
    im_close = morphology.binary_closing(dst==0, np.ones((3,3)),iterations=1)

    cv2.imshow("Im_Close",np.uint8(255 * (im_close > 0)))
    
    im_fill = imfill(np.uint8(im_close))
    cv2.imshow("Im_Fill",np.uint8(255 * (im_fill > 0)))
    cv2.waitKey(0)
#    return

#    bw_img = 1 * (dst > 0)
    #labels, nbr_objects = measurements.label(1 * (bw_img < DITHER_THRES))
#    im_close = morphology.binary_closing(1 * (bw_img < DITHER_THRES), np.ones((3,3)),iterations=1)
#    im_close = morphology.binary_closing(bw_img, np.ones((3,3)),iterations=1)
#    bw_img = np.uint8(255 * (dst > 0))
#    im_close = imfill(bw_img)
    labels, nbr_objects = measurements.label(im_fill)
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
    
#    mean_val = np.mean(val_list) / 2
    mean_val = np.max(val_list) / 16
    order_list = labels_dict.keys()
    sorted(order_list)
    

    for key in order_list:
        l = labels_dict[key]
        if l[0] < mean_val:
            continue
        x_start = l[1]
        x_end = l[2]
        y_start = l[3]
        y_end = l[4]
        cv2.rectangle(img,(x_start,y_start),(x_end,y_end),(0, 0, 255),2)

        cv2.imshow("split lines",img)
        cv2.waitKey(100)
        
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

app = QtGui.QApplication(sys.argv)
fname = QtGui.QFileDialog.getOpenFileName(None,
                    "Load Query", '.',
                    "Image Files (*.png; *.jpg; *.jpeg)")
if fname == "":
    sys.exit()
app.quit()
main(fname)
sys.exit()