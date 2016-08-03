# Test for giant object detection

import sys
from PyQt4 import QtGui

app = QtGui.QApplication(sys.argv)
fname = QtGui.QFileDialog.getOpenFileName(None,
                    "Load Query", '.',
                    "Image Files (*.png; *.jpg; *.jpeg)")                    
if fname == "":
    sys.exit()
app.quit()

import core_splitting_lib as util
import pca_transform as trans
import cv2
import numpy as np

image = cv2.imread(fname)
dither_img = util.naive_dithering(image)

objs = []
h,w = dither_img.shape
util.find_all_splits(dither_img,objs,0,w,0,h)
util.show_objs(objs,image,waitTime=2)

# the function below pick-out the giant objects
objs = util.remove_tiny_objs(objs)
giantList = util.get_giant_objs(objs)
util.show_objs(giantList,image,color=(255,255,0))