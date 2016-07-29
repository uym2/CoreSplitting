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


gray_img = cv2.imread("img5.jpeg",0)
thres = step_miner_thres(gray_img.tolist())
dither_img = split.naive_dithering(gray_img,thres)

cv2.imshow("gray",gray_img)
cv2.imshow("dither",dither_img)
cv2.waitKey(0)