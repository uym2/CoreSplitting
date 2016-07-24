# import the necessary packages
import numpy as np
import cv2

DITHER_THRES = 200
MIN_INTENSITY = 0
MAX_INTENSITY = 255
NOISE_TORL_RATIO = 0.01 # noise torlerance ratio: maximum proportion of the pixels in the splitting line that are noise (noises that was failed to be dithered out)
NOISE_TORL = MAX_INTENSITY*NOISE_TORL_RATIO
MIN_TO_TPC = 0.5 # the minimum ratio of an obj to the "typical" in an objectList
OFFSET_RATIO = 0.5 # the maximum offset of objects in a column/row (relative to object widh/height)

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


# each object is located by a rectangular bounding box
# a rectangle is located by 4 coordinates: x_start, x_end, y_start, y_end
# therefore, each element in the objList passed to this function has the form 
# [x_start, x_end, y_start, y_end]
def typical_bounding_area(objList,method='avg'):
    # assume a majority of objects in objList are similar in size
    # this function return the area of a "typical" object in the list
    # method: 
    #       avg: default option; take the average to represent the "typical"
    #       med: take the median to represent the "typical"
    # intuitively, taking the median as the typical is a better method
    # however, computing median in a list takes more computing time
    # therefore, average was chosen as the default method
    if method == 'med':
        tpc_width = np.median([abs(obj[1]-obj[0]) for obj in objList])
        tpc_height = np.median([abs(obj[3]-obj[2]) for obj in objList])
        return tpc_width*tpc_height
    else:
        tpc_width = np.mean([abs(obj[1]-obj[0]) for obj in objList])
        tpc_height = np.mean([abs(obj[3]-obj[2]) for obj in objList])
        return tpc_width*tpc_height
        
def obj_area(obj):
    return abs((obj[3]-obj[2])*(obj[1]-obj[0]))
        
def remove_tiny_objs(objList):
    # the splitting algorithm produces some tiny pieces that are not real objects
    # this function remove those from the objList
    tpc_area = typical_bounding_area(objList)
    return [obj for obj in objList if obj_area(obj)/tpc_area >= MIN_TO_TPC]
 
def find_centers(objList):
    return [[(obj[1]+obj[0])/2,(obj[3]+obj[2])/2] for obj in objList]

def organize_obj(objList):
    # sort objects by y-coordinate and x-coordinate
    # after sorting, the objects can be read from left-> right and top->bottom

    # find object's center    
    c = find_centers(objList)
    THRES = OFFSET_RATIO*np.mean([abs(obj[3]-obj[2]) for obj in objList])
    #THRES = OFFSET_RATIO*imgH

    # sort by vertical dimension
    sIdx = np.argsort([ctr[1] for ctr in c])
    
    # after sorting, objects in the same row are clusterred together
    # traverse the sorted list to split the clusters (in this stage: cluster = row)
    # then sort objects in each cluster by column
    organized_list = []
    i = 0    
    for j in range(1,len(sIdx)):
        if abs(c[sIdx[j]][1]-c[sIdx[j-1]][1]) > THRES:
            # sort by x-coordinate for objects within a row
            sIdx_row = sIdx[i:j]
            sortRow_idx = np.argsort([c[k][0] for k in sIdx_row])
            organized_list.append([objList[sIdx_row[k]] for k in sortRow_idx])
            i = j
            
    # add the last row
    sIdx_row = sIdx[i:]
    sortRow_idx = np.argsort([c[k][0] for k in sIdx_row])
    organized_list.append([objList[sIdx_row[k]] for k in sortRow_idx])
 
    return organized_list
            
def show_objs_by_row(orgList,image):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    i = 0
    for row in orgList:
        for obj in row:
        		x_start = obj[0]
        		x_end = obj[1]
        		y_start = obj[2]
        		y_end = obj[3]
        		cv2.rectangle(image,(x_start,y_start),(x_end,y_end),colors[i%len(colors)],2)
        i = i+1
    # illustration
    cv2.imshow("split lines",image)
    cv2.waitKey(0)
 
def showSteps_objs_byRow(orgList,image):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    i = 0
    for row in orgList:
        for obj in row:
            x_start = obj[0]
            x_end = obj[1]
            y_start = obj[2]
            y_end = obj[3]
            cv2.rectangle(image,(x_start,y_start),(x_end,y_end),colors[i%len(colors)],2)
            # illustration
            cv2.imshow("split lines",image)
            cv2.waitKey(500)
        i = i+1
  
def show_objs(objList,image):
    for obj in objList:
        x_start = obj[0]
        x_end = obj[1]
        y_start = obj[2]
        y_end = obj[3]
        cv2.rectangle(image,(x_start,y_start),(x_end,y_end),(255,0,0),2)
    cv2.imshow("show",image)
    cv2.waitKey(0)