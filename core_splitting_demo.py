import function_utility as support
import cv2
import argparse

# construct the argument parser and parse the arguments
# to run: python core_split_demo.py -i img1.jpeg

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-o", "--output", help = "Output image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
dither_img = support.naive_dithering(image)

objs = []
h,w = dither_img.shape
support.find_all_splits(dither_img,objs,0,w,0,h)

for i in range(len(objs)):
		a_obj = objs[i]
		x_start = a_obj[0]
		x_end = a_obj[1]
		y_start = a_obj[2]
		y_end = a_obj[3]
		cv2.rectangle(image,(x_start,y_start),(x_end,y_end),(0,255,0),2)

# output
if args["output"]:
	filename = args["output"]
else:
	filename = "splitted.jpeg"
cv2.imwrite(filename,image)

# illustration
cv2.imshow("split lines",image)
cv2.waitKey(0)