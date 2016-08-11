# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 22:13:01 2016

@author: uym2
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import core_splitting_lib as split

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
objCtrs_x = [(obj[0]+obj[1])/2 for obj in objList]
objCtrs_y = [-(obj[2]+obj[3])/2 for obj in objList]

plt.scatter(objCtrs_x,objCtrs_y)
plt.show()


