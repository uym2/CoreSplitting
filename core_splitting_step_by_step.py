# import the necessary packages
import numpy as np
import argparse
import cv2
import core_splitting_lib as util

def find_splits(dither_img,lines,x_start,x_end,y_start,y_end):
	h,w = dither_img.shape
	# stop recursion
	#print h,w,np.sum(dither_img), np.mean(dither_img)
	if h < 2 or w < 2 or np.mean(dither_img)<255*0.01:
		return
	y = util.find_split_line(dither_img,'H')
	if (y>0):
		lines.append((y+y_start,x_start,x_end,'H'))
		# recursive calls
		find_splits(dither_img[:y-1,:],lines,x_start,x_end,y_start,y_start+y)
		find_splits(dither_img[y+1:,:],lines,x_start,x_end,y_start+y+1,y_end)
	else:
		x = util.find_split_line(dither_img,'V')
		if (x <= 0):
			return
		lines.append((x+x_start,y_start,y_end,'V'))
		# recursive calls
		find_splits(dither_img[:,:x-1],lines,x_start,x_start+x,y_start,y_end)
		find_splits(dither_img[:,x+1:],lines,x_start+x+1,x_end,y_start,y_end)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
dither_img = util.naive_dithering(image)
cv2.imshow("dither",dither_img)
cv2.waitKey()

lines = []
h,w = dither_img.shape
find_splits(dither_img,lines,0,w,0,h)
#y1 = find_split_line(dither_img)
#y2 = find_split_line(dither_img[y1+1:,:])
#lines = [y1,y1+1+y2]

for i in range(len(lines)):
	l = lines[i]
	if l[3] == 'H':
		cv2.line(image,(l[1],l[0]),(l[2],l[0]),(255,0,0),2)
	else:
		cv2.line(image,(l[0],l[1]),(l[0],l[2]),(255,0,0),2)
	cv2.imshow("split lines",image)
	cv2.waitKey(0)