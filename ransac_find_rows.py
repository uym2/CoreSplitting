# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:00:12 2016

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

rowIdxList = org.group2Rows(objList)
print len(rowIdxList)
split.show_objs_by_dim(rowIdxList,objList,img)