import core_splitting_lib as util
import cv2
import argparse

# construct the argument parser and parse the arguments
# to run: python core_split_demo.py -i img1.jpeg

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-o", "--output", help = "Output image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
dither_img = util.naive_dithering(image)
#cv2.imshow("Dither", dither_img)
#cv2.waitKey(0)

objs = []
h,w = dither_img.shape
util.find_all_splits(dither_img,objs,0,w,0,h)
objs = util.remove_tiny_objs(objs)

rowIdx = util.group2rowNsort(objs)
colIdx = util.group2colNsort(objs)

#print(colIdx[0][0])
#print(colIdx[0][0])
#util.column_alignment(rowList)
idxMat = util.infer_missing_idx(rowIdx,colIdx)
#print idxMat
util.show_objs_by_matrix(idxMat,objs,image)
#util.show_objs_by_dim(colIdx,objs,image)
#util.show_objs(objs,image)

# output
if args["output"]:
	filename = args["output"]
else:
	filename = "splitted.jpeg"
cv2.imwrite(filename,image)