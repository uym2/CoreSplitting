# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:19:29 2016

@author: uym2
"""

import StepMiner as sm
import cv2
import core_splitting_lib as split
import numpy as np

def step_miner_thres(data):
    fit_result = sm.fitStepSimple(data)
    return fit_result[6]


img = cv2.imread("img1.jpeg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = np.concatenate(gray_img)
data.sort()
print data
thres = int(step_miner_thres(data))
print thres
dither_img = split.naive_dithering(img,thres)

cv2.imshow("gray",gray_img)
cv2.imshow("dither",dither_img)
cv2.waitKey(0)