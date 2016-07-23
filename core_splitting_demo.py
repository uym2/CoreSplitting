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

objs = []
h,w = dither_img.shape
util.find_all_splits(dither_img,objs,0,w,0,h)
objs = util.remove_tiny_objs(objs)

orgList = util.organize_obj(objs,w,h)

util.show_objs_by_row(orgList,image)

# output
if args["output"]:
	filename = args["output"]
else:
	filename = "splitted.jpeg"
cv2.imwrite(filename,image)

# illustration
cv2.imshow("split lines",image)
cv2.waitKey(0)