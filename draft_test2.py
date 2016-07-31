#import sip
#sip.setapi("QString", 2)
#sip.setapi("QVariant", 2)

# Test for stepminer's performance on classify_by_size

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
cv2.imshow("img",image)
dither_img = util.naive_dithering(image)
#cv2.imshow("Dither", dither_img)
#cv2.waitKey(0)

objs = []
h,w = dither_img.shape
util.find_all_splits(dither_img,objs,0,w,0,h)
# the function below use stepminer to classify objects into tiny, normal, and giant lists
tinyList,normalList,giantList = util.classify_by_size(objs)
util.show_obj_by_size(tinyList,normalList,giantList,image)