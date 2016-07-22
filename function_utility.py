# import the necessary packages
import numpy as np
import cv2

DITHER_THRES = 200
MIN_INTENSITY = 0
MAX_INTENSITY = 255
NOISE_TORL_RATIO = 0.01 # noise torlerance ration: maximum proportion of the pixels in the splitting line that are noise (noises that was failed to be dithered out)
NOISE_TORL = MAX_INTENSITY*NOISE_TORL_RATIO

def naive_dithering(img):
	# input: an image
	# output: a black-white image reflecting the same content

	# convert to grayscale
	output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h,w = output_img.shape
	
	for i in range(h):
		for j in range(w):
			if (output_img[i,j]>DITHER_THRES):
				output_img[i,j] = MIN_INTENSITY
			else:
				output_img[i,j] = MAX_INTENSITY
	return output_img			

def is_split_line_of(dither_img,line_type,intercept):
	if line_type == 'H':
		return np.mean(dither_img[intercept,:]) < NOISE_TORL
	elif line_type == 'V':
		return np.mean(dither_img[:,intercept]) < NOISE_TORL

def find_split_lines(dither_img,line_type):
	# input: a dithered image (black-white only)
	# output: two lines (horizontal or vertical) that splits the image into 3 segments:
		# + upper and lower parts contains all objects without overlapping 
		# + the middle part is blank
	# (a horizontal/vertical line can be determined by its y-intercept/x-intercepts)

	h,w = dither_img.shape
	# start from the middle
	if line_type == 'H':
		dim = h
	else:
		dim = w

	l1 = dim/2
	l2 = dim/2

	if (is_split_line_of(dither_img,line_type,l1)):
		while l1>0 and is_split_line_of(dither_img,line_type,l1-1):
			l1 = l1-1
		while l2<dim-1 and is_split_line_of(dither_img,line_type,l2+1):
			l2 = l2+1
		return [l1,l2]
	for i in range(1,dim/2):
		if (is_split_line_of(dither_img,line_type,l2-i)):
			l2 = l2-i
			l1 = l2
			while l1>0 and is_split_line_of(dither_img,line_type,l1-1):
				l1 = l1-1
			return [l1,l2]
		elif (is_split_line_of(dither_img,line_type,l1+i)):
			l1 = l1+i
			l2 = l1
			while l2<dim-1 and is_split_line_of(dither_img,line_type,l2+1):
				l2 = l2+1
			return [l1,l2]
	return None # not found

def find_split_line(dither_img,line_type):
	# input: a dithered image (black-white only)
	# output: a line (horizontal or vertical) that splits the image into 2 halves without any overlapping object
	# (a horizontal line can be determined by its y-intercept)

	h,w = dither_img.shape
	# start from the middle
	if line_type == 'H':
		dim = h
	else:
		dim = w
	intercept = dim/2
	if (is_split_line_of(dither_img,line_type,intercept)):
		return intercept
	for i in range(dim/2):
		if (is_split_line_of(dither_img,line_type,intercept-i)):
			return intercept-i
		elif (is_split_line_of(dither_img,line_type,intercept+i)):
			return intercept+i
	return -1 # not found


def find_all_splits(dither_img,objs,x_start,x_end,y_start,y_end):
	h,w = dither_img.shape
	# stop recursion
	if h < 2 or w < 2 or np.mean(dither_img)<NOISE_TORL:
		return
	horr_split_lines = find_split_lines(dither_img,'H')
	if horr_split_lines:
		y1 = horr_split_lines[0]
		y2 = horr_split_lines[1]
		# recursive calls
		find_all_splits(dither_img[:y1,:],objs,x_start,x_end,y_start,y_start+y1)
		find_all_splits(dither_img[y2:,:],objs,x_start,x_end,y_start+y2,y_end)
	else:
		ver_split_lines = find_split_lines(dither_img,'V')
		if not ver_split_lines:
			objs.append([x_start,x_end,y_start,y_end]) # add the found object
			return
		x1 = ver_split_lines[0]
		x2 = ver_split_lines[1]
		# recursive calls
		find_all_splits(dither_img[:,:x1],objs,x_start,x_start+x1,y_start,y_end)
		find_all_splits(dither_img[:,x2:],objs,x_start+x2+1,x_end,y_start,y_end)
