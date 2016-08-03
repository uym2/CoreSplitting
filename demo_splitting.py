import core_splitting_lib as util
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

image = cv2.imread(fname)
#cv2.imshow("img",image)
dither_img = util.naive_dithering(image)
#cv2.imshow("Dither", dither_img)
#cv2.waitKey(0)

#objs = []
#h,w = dither_img.shape
#util.find_all_splits(dither_img,objs,0,w,0,h)
objs = util.label_cores(dither_img)
#objs = util.remove_tiny_objs(objs)
giantList = util.get_giant_objs(objs)
giantSplitted = util.split_giant_objs(giantList,dither_img,util.typical_obj_size(objs),open_win=3,iterNum=1)
#objs = util.remove_tiny_objs(objs)

#rowIdx = util.group2rowNsort(objs)
#colIdx = util.group2colNsort(objs)

#print(colIdx[0][0])
#print(colIdx[0][0])
#util.column_alignment(rowList)
#idxMat = util.infer_missing_idx(rowIdx,colIdx)
#print idxMat
<<<<<<< HEAD
util.show_objs_by_matrix(idxMat,objs,image)
#util.show_objs_by_dim(rowIdx,objs,image)
#util.show_objs(objs,image)