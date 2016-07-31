# import the necessary packages
import numpy as np
import cv2
from scipy.ndimage import morphology, measurements
import StepMiner

DITHER_THRES = 200
MIN_INTENSITY = 0
MAX_INTENSITY = 255
NOISE_TORL_RATIO = 0.01 # noise torlerance ratio: maximum proportion of the pixels in the splitting line that are noise (noises that was failed to be dithered out)
NOISE_TORL = MAX_INTENSITY*NOISE_TORL_RATIO
MIN_TO_TPC = 0.5 # the minimum ratio of an obj to the "typical" in an objectList
MAX_TO_TPC = 1.8 # the maximum ration of an obj to the "typical"

def step_miner_thres(data):
    fit_result = StepMiner.fitStepSimple(data)
    return fit_result[6]

def naive_dithering(img,dither_thres=DITHER_THRES,inv=True):
	# input: an image
	# output: a black-white image reflecting the same content

	# convert to grayscale
	output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	logical_idx = (output_img <= dither_thres) if inv else (output_img > dither_thres)
	output_img = MAX_INTENSITY*logical_idx
	return output_img			


def component_labeling(img):
    # input: an image
    # output: a black-white image reflecting the same content
    # MAD

    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = 1 * (gray_img < DITHER_THRES)
    im_close = morphology.binary_closing(bw_img, np.ones((3,3)),iterations=1)
#    im_close = 1 * (gray_img < DITHER_THRES)
    labels, nbr_objects = measurements.label(im_close)
    	
    h, w = labels.shape
    
    labels_dict = {}
    for i in range(h):
        for j in range(w):
            key = labels[i,j]
            if key == 0:
                continue
            labels_dict[key] = labels_dict.get(key, 0) + 1

    mean_val = np.mean(labels_dict.values()) * MIN_TO_TPC
    print mean_val, np.median(labels_dict.values())

    total_labels = rem_num = 0
    for i in range(h):
        for j in range(w):
            key = labels[i,j]
            if key > 0:
                total_labels += 1
                if  labels_dict[key] < mean_val:
                    labels[i,j] = 0
                    rem_num += 1
    print total_labels, rem_num

    return labels

def otsu_dithering(img,inv=True):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if inv:
        thr,binary_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    else:
        thr,binary_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return binary_img

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
def typical_obj_size(objList,method='avg'):
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
        return [tpc_width,tpc_height]
    else:
        tpc_width = np.mean([abs(obj[1]-obj[0]) for obj in objList])
        tpc_height = np.mean([abs(obj[3]-obj[2]) for obj in objList])
        return [tpc_width,tpc_height]
        
def obj_area(obj):
    return abs((obj[3]-obj[2])*(obj[1]-obj[0]))
    
def obj_width(obj):
    return abs((obj[1]-obj[0]))
    
def obj_height(obj):
    return abs((obj[3]-obj[2])) 
        
def remove_tiny_objs(objList):
    # the splitting algorithm produces some tiny pieces that are not real objects
    # this function removes those from the objList
    tpc_size = typical_obj_size(objList)
    tpc_area = tpc_size[0]*tpc_size[1]
    return [obj for obj in objList if obj_area(obj)/tpc_area >= MIN_TO_TPC]

# not a great solution, but not yet come up with a better one at this stage
def get_giant_objs(objList):
    # (I think) it's better to remove tiny objects to eliminate lower outliners
    # BEFORE calling this function so that finding the typical object is more reliable
    tpc_size = typical_obj_size(objList)
    width_thres = tpc_size[0]*MAX_TO_TPC
    height_thres = tpc_size[1]*MAX_TO_TPC
    return [obj for obj in objList if obj_width(obj)>width_thres or obj_height(obj)>height_thres]

# wrote here just for testing purpose
# not clear how to find a "good" percentage of tiny or giant objects 
def remove_outliers(objList,tiny_rm_percent,giant_rm_percent):
    area_sort_idx = np.argsort([obj_area(obj) for obj in objList])
    length = len(objList)
    startIdx = int(length*tiny_rm_percent/100)
    endIdx = int(length*(100-giant_rm_percent)/100)
    return [objList[idx] for idx in area_sort_idx[startIdx:endIdx]]
 
 
# using step_miner to find thresholds
# under developing 
def classify_by_size(objList):
    # based on the object's area
    # classify them as normal, tiny, or giant
    areaList = [obj_area(obj) for obj in objList]
    sorted(areaList)
    thres_tiny = step_miner_thres(areaList)
    thres_giant = step_miner_thres([a for a in areaList if a >= thres_tiny])
    
    tinyList = [obj for obj in objList if obj_area(obj) < thres_tiny]
    normalList = [obj for obj in objList if obj_area(obj)>=thres_tiny and obj_area(obj)<=thres_giant]
    giantList = [obj for obj in objList if obj_area(obj) > thres_giant] 
 
    return tinyList, normalList, giantList  

def find_centers(objList):
    return [[(obj[1]+obj[0])/2,(obj[3]+obj[2])/2] for obj in objList]

def clusterInOneDim(objList,dim):
    # sort objects by y-coordinate (if group by row) or x-coordinate (if group by column)
    # after sorting, the objects can be read from left-> right or top->bottom
    
    # sort in one dimension
    
    if dim == 'row':
        start = 2
        end = 3
        start_ops = 0
    else:
        start = 0
        end = 1
        start_ops = 2
    
    sIdx = np.argsort([obj[end] for obj in objList])
        
    # after sorting, objects in the same row (column) are clusterred together
    # traverse the sorted list to split the clusters
    # then sort objects in each cluster
    clusters = []
    i = 0    
    for j in range(1,len(sIdx)):
        this_obj = objList[sIdx[j]]
        neighbor_obj = objList[sIdx[i]]
        d_min = abs(this_obj[start_ops]-neighbor_obj[start_ops])
        
        # find the closest neighbor
        for k in range(i+1,j):
            d = abs(this_obj[start_ops] - objList[sIdx[k]][start_ops])
            if d<d_min:
                d_min = d
                neighbor_obj = objList[sIdx[k]]
        
        # check if this_obj belong to current row/column
        if (neighbor_obj[end]-this_obj[start]<7):        
            #add the new cluster and split 
            clusters.append([obj_idx for obj_idx in sIdx[i:j]])
            i = j
            
    # add the last row/column
    clusters.append([obj_idx for obj_idx in sIdx[i:]])     
 
    return clusters
    
def group2rowNsort(objList):
    clusters = clusterInOneDim(objList,'row')
    #adjacent_exchange(clusters,objList,'row')
    rowIdx = []
    for cl in clusters:
        cl_sort = np.argsort([objList[idx][0] for idx in cl])
        rowIdx.append([cl[idx] for idx in cl_sort])
    return rowIdx

def group2colNsort(objList):
    clusters = clusterInOneDim(objList,'col')
    #adjacent_exchange(clusters,objList,'col')
    colIdx = []
    for cl in clusters:
        cl_sort = np.argsort([objList[idx][2] for idx in cl])
        colIdx.append([cl[idx] for idx in cl_sort])
    return colIdx

def adjacent_exchange(clusters,objList,dim):
    if dim == 'row':
        start = 2
        end = 3
    else:
        start = 0
        end = 1

    curr_clst_pos = np.mean([(objList[idx][start]+objList[idx][end])/2 for idx in clusters[0]])        
    for i in range(1,len(clusters)):
        # Note: clusters[i-1] is the "current"
        #       clusters[i] is the "adjacent"
        adj_clst_pos = np.mean([(objList[idx][start]+objList[idx][end])/2 for idx in clusters[i]])        
        
        for idx in clusters[i-1]:
            # check each obj in current cluster
            pos = (objList[idx][start]+objList[idx][end])/2
            d1 = abs(pos-curr_clst_pos)
            d2 = abs(pos-adj_clst_pos)
            if d1 > d2:
                # move to the adjacent cluster
                clusters[i-1].remove(idx)
                clusters[i].append(idx)
                
        for idx in clusters[i]:
            # check each obj in adjacent cluster
            pos = (objList[idx][start]+objList[idx][end])/2
            d1 = abs(pos-curr_clst_pos)
            d2 = abs(pos-adj_clst_pos)
            if d1 < d2:
                # move to the current cluster
                clusters[i].remove(idx)
                clusters[i-1].append(idx)                
                
        curr_clst_pos = adj_clst_pos
        
def column_alignment(rowList):
    j = 0
    while j< max([len(row) for row in rowList]):
        col = []
        for row in rowList:
            if j < len(row):
                col.append(row[j])
            else:
                col.append(None)
        #col = [row[j] for row in rowList if j < len(row)]
        #THRES = np.max([abs(obj[1]-obj[0]) for obj in col if obj])
        col_leftLimit = np.min([max(obj[0],obj[1]) for obj in col if obj])

        i = 0
        for obj in col:
            if not obj:
                # reach the end of this row, add virtual objs from now on
                rowList[i].append(None)
            else:
                if obj[1]-col_leftLimit == 0:
                    print i,j
                if col_leftLimit-min(obj[0],obj[1]) < 0:
                #print i,j
                #print col_leftLimit, obj[0]
                # add a virtual object
                    rowList[i].insert(j,None)
            
            i = i+1
        j = j+1
   
def infer_missing_idx(rowIdx,colIdx):
    for i in range(len(rowIdx)):
        for j in range(len(colIdx)):
            if len(rowIdx[i])<=j:
                #print i,j
                rowIdx[i].append(None)
                colIdx[j].insert(i,None)
            elif len(colIdx[j])<=i:
                #print i,j
                colIdx[j].append(None)
                rowIdx[i].insert(j,None)
            elif rowIdx[i][j] != colIdx[j][i]:
                #print i,j
                rowIdx[i].insert(j,None)
                colIdx[j].insert(i,None)
    return np.array(rowIdx)
            
def show_objs_by_dim(idxList,objList,image):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    i = 0
    for cluster in idxList:
        for idx in cluster:
                 obj = objList[idx]
                 x_start = obj[0]
                 x_end = obj[1]
                 y_start = obj[2]
                 y_end = obj[3]
                 cv2.rectangle(image,(x_start,y_start),(x_end,y_end),colors[i%len(colors)],2)
        i = i+1
    cv2.imshow("Objects grouped by one dimension",image)
    cv2.waitKey(0)

def show_objs_by_matrix(idxMat,objList,image):
    real_color = [255,0,0]
    virtual_color = [0,0,255]
    
    h,w = idxMat.shape
    for i in range(h):
        for j in range(w):
            if not idxMat[i,j] is None:
                obj = objList[idxMat[i,j]]
                x_start = obj[0]
                x_end = obj[1]
                y_start = obj[2]
                y_end = obj[3]
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),real_color,2)
            else:
                rowObjs = [objList[k] for k in idxMat[i,:] if k]
                colObjs = [objList[k] for k in idxMat[:,j] if k]
                x_start = int(np.mean([obj[0] for obj in colObjs]))
                x_end = int(np.mean([obj[1] for obj in colObjs]))
                y_start = int(np.mean([obj[2] for obj in rowObjs]))
                y_end = int(np.mean([obj[3] for obj in rowObjs]))
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),virtual_color,2)
        
    cv2.imshow("infer objs",image)
    cv2.waitKey(0)

#########################################################
###### obsolete function, should be removed soon ########
def show_objs_by_column(alignedList,image):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    virtual_color = (0,255,255)
    c = 0
    
    for j in range(len(alignedList[0])):
        for i in range(len(alignedList)):
            obj = alignedList[i][j]
            if obj:
                x_start = obj[0]
                x_end = obj[1]
                y_start = obj[2]
                y_end = obj[3]
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),colors[c%len(colors)],2)
            else:
                x_start = int(np.median([r[j][0] for r in alignedList if r[j]]))
                x_end = int(np.median([r[j][1] for r in alignedList if r[j]]))
                y_start = int(np.median([oj[2] for oj in alignedList[i] if oj]))
                y_end = int(np.median([oj[3] for oj in alignedList[i] if oj]))
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),virtual_color,2)
        c = c+1
    # illustration
    cv2.imshow("split lines",image)
    cv2.waitKey(0)

#########################################################
###### obsolete function, should be removed soon ######## 
def showSteps_objs_byRow(orgList,image):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    i = 0
    for row in orgList:
        for obj in row:
            if obj:
                x_start = obj[0]
                x_end = obj[1]
                y_start = obj[2]
                y_end = obj[3]
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),colors[i%len(colors)],2)
                # illustration
                cv2.imshow("split lines",image)
                cv2.waitKey(500)
        i = i+1
  
def show_objs(objList,image,color=(255,0,0),linewidth=2,showWindow=True,waitTime=0):
    for obj in objList:
        if obj:
            x_start = obj[0]
            x_end = obj[1]
            y_start = obj[2]
            y_end = obj[3]
            cv2.rectangle(image,(x_start,y_start),(x_end,y_end),color,linewidth)
    if showWindow:
        cv2.imshow("show",image)
        cv2.waitKey(waitTime)
    
def show_obj_by_size(tinyList,normalList,giantList,image):
    show_objs(tinyList,image,color=(255,0,0),showWindow=False)
    show_objs(normalList,image,color=(0,255,0),showWindow=False)
    show_objs(giantList,image,color=(0,0,255),showWindow=True)