import core_splitting_lib as clib

#import sip
#sip.setapi("QString", 2)
#sip.setapi("QVariant", 2)

import cv2

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
objList = clib.extract_cores(image)
clib.show_objs(objList,image)