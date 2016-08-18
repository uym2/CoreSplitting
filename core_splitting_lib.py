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
MAX_TO_TPC = 1.5 # the maximum ratio of an obj to the "typical"

def extract_cores(img):
    #im_filled = preprocess(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dither_img = naive_dithering(gray)
    im_filled = morphology.binary_closing(dither_img, np.ones((3,3)),iterations=1)
    objList = label_cores(im_filled)
    objList = postprocess(objList,im_filled)
    return objList

def imfill(im_in):
    im_floodfill = im_in.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out
    
def preprocess(img):
    # standardize color channels
    max0 = np.max(img[:,:,0])
    max1 = np.max(img[:,:,1])
    max2 = np.max(img[:,:,2])
    min0 = np.min(img[:,:,0])
    min1 = np.min(img[:,:,1])
    min2 = np.min(img[:,:,2])
    img[:,:,0] = np.uint8(255./(max0 - min0) * (img[:,:,0] - min0))
    img[:,:,1] = np.uint8(255./(max1 - min1) * (img[:,:,1] - min1))
    img[:,:,2] = np.uint8(255./(max2 - min2) * (img[:,:,2] - min2))
    
    # blur to remove noise
    blur = cv2.bilateralFilter(img,3,75,75)

    # convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    dither = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    #dither = naive_dithering(gray)
    im_close = morphology.binary_closing(dither==0, np.ones((2,2)),iterations=2)
    
    im_fill = imfill(np.uint8(im_close))
    return im_fill

def postprocess(objList,filled_img):
    objList = remove_tiny_objs(objList)
    objList = handle_giant(objList,filled_img)
    return objList

def step_miner_thres(data):
    fit_result = StepMiner.fitStepSimple(data)
    return fit_result[6]

def stepminer_dithering(gray_img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = np.concatenate(gray_img)
    data.sort()
    thres = int(step_miner_thres(data))
    print thres
    if thres < (MAX_INTENSITY-MIN_INTENSITY)/2:
        dither_type = cv2.THRESH_BINARY
    else:
        dither_type = cv2.THRESH_BINARY_INV
    ret,dither_img = cv2.threshold(gray_img,thres,255,dither_type)
    return dither_img

def naive_dithering(gray_img,dither_thres=DITHER_THRES,inv=True):
	# input: an image
	# output: a black-white image reflecting the same content

	# convert to grayscale
	#output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	logical_idx = (gray_img <= dither_thres) if inv else (gray_img > dither_thres)
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

def otsu_dithering(gray_img,inv=True):
    #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if inv:
        thr,binary_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    else:
        thr,binary_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    return binary_img

def is_split_line_of(dither_img,line_type,intercept,noise_torl=NOISE_TORL):
	if line_type == 'H':
		return np.mean(dither_img[intercept,:]) < noise_torl
	elif line_type == 'V':
		return np.mean(dither_img[:,intercept]) < noise_torl

def find_split_lines(dither_img,line_type,noise_torl=NOISE_TORL):
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

	if (is_split_line_of(dither_img,line_type,l1,noise_torl=noise_torl)):
		while l1>0 and is_split_line_of(dither_img,line_type,l1-1):
			l1 = l1-1
		while l2<dim-1 and is_split_line_of(dither_img,line_type,l2+1,noise_torl=noise_torl):
			l2 = l2+1
		return [l1,l2]
	for i in range(1,dim/2):
		if (is_split_line_of(dither_img,line_type,l2-i)):
			l2 = l2-i
			l1 = l2
			while l1>0 and is_split_line_of(dither_img,line_type,l1-1):
				l1 = l1-1
			return [l1,l2]
		elif (is_split_line_of(dither_img,line_type,l1+i,noise_torl=noise_torl)):
			l1 = l1+i
			l2 = l1
			while l2<dim-1 and is_split_line_of(dither_img,line_type,l2+1,noise_torl=noise_torl):
				l2 = l2+1
			return [l1,l2]
	return None # not found

def find_split_line(dither_img,line_type,noise_torl=NOISE_TORL):
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
		if (is_split_line_of(dither_img,line_type,intercept-i,noise_torl=noise_torl)):
			return intercept-i
		elif (is_split_line_of(dither_img,line_type,intercept+i,noise_torl=noise_torl)):
			return intercept+i
	return -1 # not found


def find_all_splits(dither_img,objs,x_start,x_end,y_start,y_end,noise_torl=NOISE_TORL):
	h,w = dither_img.shape
	# stop recursion
	if h < 2 or w < 2 or np.mean(dither_img)<noise_torl:
		return
	horr_split_lines = find_split_lines(dither_img,'H',noise_torl=noise_torl)
	if horr_split_lines:
		y1 = horr_split_lines[0]
		y2 = horr_split_lines[1]
		# recursive calls
		find_all_splits(dither_img[:y1,:],objs,x_start,x_end,y_start,y_start+y1,noise_torl)
		find_all_splits(dither_img[y2:,:],objs,x_start,x_end,y_start+y2,y_end,noise_torl)
	else:
		ver_split_lines = find_split_lines(dither_img,'V',noise_torl=noise_torl)
		if not ver_split_lines:
			objs.append([x_start,x_end,y_start,y_end]) # add the found object
			return
		x1 = ver_split_lines[0]
		x2 = ver_split_lines[1]
		# recursive calls
		find_all_splits(dither_img[:,:x1],objs,x_start,x_start+x1,y_start,y_end,noise_torl)
		find_all_splits(dither_img[:,x2:],objs,x_start+x2+1,x_end,y_start,y_end,noise_torl)

# alternative to find_all_split
# using connected component to label objs
# MAD's work
def label_cores(filled_img):
    #im_close = morphology.binary_closing(dither_img, np.ones((3,3)),iterations=1)
    labels, nbr_objects = measurements.label(filled_img)
    
    h, w = labels.shape
    
    labels_dict = {}
    for i in range(h):
        for j in range(w):
            key = labels[i,j]
            if key == 0:
                continue
            info_list = labels_dict.get(key, [0, 99999, 0, 99999, 0])
            info_list[0] += 1

            if info_list[1] > j:
                info_list[1] = j
    
            if info_list[2] < j:
                info_list[2] = j
    
            if info_list[3] > i:
                info_list[3] = i
    
            if info_list[4] < i:
                info_list[4] = i
            labels_dict[key] = info_list
            
    val_list = []
    for key in labels_dict.keys():
        val = labels_dict[key][0]
        val_list.append(val)
    
    mean_val = np.mean(val_list) / 2
    order_list = labels_dict.keys()
    sorted(order_list)
    
    objs = []
    for key in order_list:
        l = labels_dict[key]
        if l[0] < mean_val:
            continue
        x_start = l[1]
        x_end = l[2]
        y_start = l[3]
        y_end = l[4]
        objs.append([x_start,x_end,y_start,y_end])
    return objs

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
        
def remove_tiny_objs(objList,tpc_size=None):
    # the splitting algorithm produces some tiny pieces that are not real objects
    # this function removes those from the objList
    if not tpc_size:
        tpc_size = typical_obj_size(objList,'med')
    tpc_area = tpc_size[0]*tpc_size[1]
    return [obj for obj in objList if obj_area(obj)/tpc_area >= MIN_TO_TPC]

# notyet  a great solution, but not yet come up with a better one at this stage
def get_giant_objs(objList,tpc_size=None):
    # it's better to remove tiny objects to eliminate lower outliners
    # BEFORE calling this function so that finding the typical object is more reliable
    # this functions find giant objs, put them to the giantList and 
    # AT THE SAME TIME remove those objs from objList
    if not tpc_size:
        tpc_size = typical_obj_size(objList,'med')
    width_thres = tpc_size[0]*MAX_TO_TPC
    height_thres = tpc_size[1]*MAX_TO_TPC
    giantList = []
    for obj in objList:
        if obj_width(obj)>width_thres or obj_height(obj)>height_thres:
            # add to giantList
            giantList.append(obj)
            # remove from objList
            objList.remove(obj)
            # also remove all objs that are inside the giant obj
            for obj1 in objList:
                if ( obj1[0]>=obj[0] and obj1[1]<=obj[1] and
                     obj1[2]>=obj[2] and obj1[3]<=obj[3] ):
                         objList.remove(obj1)
    return giantList
        

def split_giant_objs(giantList,filled_img,tpc_size,open_win=3,iterNum=1):
    # use morphology opening to split sticky objects
    # (the reason for that they form giant objs in the first round)
    objList = []
    #dither_img = morphology.binary_closing(dither_img, 
    #                                   np.ones((3,3)),iterations=iterNum )
    while giantList:
        for obj in giantList:
                x_start = obj[0]
                x_end = obj[1]
                y_start = obj[2]
                y_end = obj[3]
                subImg = filled_img[y_start:y_end,x_start:x_end]
                subImg = morphology.binary_opening( subImg, 
                                           np.ones((open_win,open_win)),iterations=iterNum )
                subObjs = label_cores(subImg) # subOjbs: a list of objects in subImg
                giantSplitted = [[obj[0]+x_start,obj[1]+x_start,obj[2]+y_start,obj[3]+y_start] for obj in subObjs]
                #objList = merge_objList_N_giantSplitted(objList,giantSplitted)
                objList = objList + giantSplitted
        objList = remove_tiny_objs(objList,tpc_size)
        giantList = get_giant_objs(objList,tpc_size)
        open_win = open_win+2
        if open_win > tpc_size[0]:
            break
    
    return objList                          

def is_overlapped(obj1,obj2):
    return ( (obj1[1]-obj2[0])*(obj2[1]-obj1[0]) >= 0 and
             (obj1[3]-obj2[2])*(obj2[3]-obj1[2]) >= 0 )

def merge_objList_N_giantSplitted(objList,giantSplitted):
    # an obj in the giantSplitted is added to objList ONLY IF 
    # it is not overlapped with any obj in objList
    validList = []
    for s_obj in giantSplitted:
        has_overlap = False
        for obj in objList:
            if is_overlapped(s_obj,obj):
                has_overlap = True
                break
        if not has_overlap:    
            validList.append(s_obj)
    return objList + validList

def handle_giant(objList,dither_img):
    # find abnormally large objs in objList (the giants)
    # split them into smaller objs and add those splitted objs 
    # back to the objList. There are some details in the implementation
    # refer to the functions called below for details
    giantList = get_giant_objs(objList)
    giantSplitted = split_giant_objs(giantList,dither_img,typical_obj_size(objList))
    return merge_objList_N_giantSplitted(objList,giantSplitted)
    
# wrote here just for testing purpose
# not clear how to find a "good" percentage for tiny and giant objects 
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
    
    sIdx = np.argsort([obj[start] for obj in objList])
    print sIdx
        
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
        if (neighbor_obj[end]-this_obj[start]<0):        
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

#########################################################
###### obsolete function, should be removed soon ########        
def column_alignment(rowList):
    j = 0
    while j< max([len(row) for row in rowList]):
        col = []
        for row in rowList:
            if j < len(row):
                col.append(row[j])
            else:
                col.append(np.nan)
        #col = [row[j] for row in rowList if j < len(row)]
        #THRES = np.max([abs(obj[1]-obj[0]) for obj in col if obj])
        col_leftLimit = np.min([max(obj[0],obj[1]) for obj in col if obj])

        i = 0
        for obj in col:
            if not obj:
                # reach the end of this row, add virtual objs from now on
                rowList[i].append(np.nan)
            else:
                if obj[1]-col_leftLimit == 0:
                    print i,j
                if col_leftLimit-min(obj[0],obj[1]) < 0:
                #print i,j
                #print col_leftLimit, obj[0]
                # add a virtual object
                    rowList[i].insert(j,np.nan)
            
            i = i+1
        j = j+1
   
def infer_missing_idx(rowIdx,colIdx):
    for i in range(len(rowIdx)):
        for j in range(len(colIdx)):
            if len(rowIdx[i])<=j:
                #print i,j
                rowIdx[i].append(-1)
                colIdx[j].insert(i,-1)
            elif len(colIdx[j])<=i:
                #print i,j
                colIdx[j].append(-1)
                rowIdx[i].insert(j,-1)
            elif rowIdx[i][j] != colIdx[j][i]:
                #print i,j
                rowIdx[i].insert(j,-1)
                colIdx[j].insert(i,-1)
    return np.array(rowIdx)
            
def show_objs_by_dim(idxList,objList,image,showWin=True):
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
    if showWin:
        cv2.imshow("Objects grouped by one dimension",image)
        cv2.waitKey(0)
                
def show_objs_by_matrix(idxMat,objList,image,showWin=True):
    real_color = [255,0,0]
    virtual_color = [0,0,255]
    h,w = idxMat.shape
    for i in range(h):
        for j in range(w):
            if idxMat[i,j]>=0:
                obj = objList[idxMat[i,j]]
                x_start = obj[0]
                x_end = obj[1]
                y_start = obj[2]
                y_end = obj[3]
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),real_color,2)
            else:
                rowObjs = [objList[k] for k in idxMat[i,:] if k>0]
                colObjs = [objList[k] for k in idxMat[:,j] if k>0]
                x_start = int(np.mean([obj[0] for obj in colObjs]))
                x_end = int(np.mean([obj[1] for obj in colObjs]))
                y_start = int(np.mean([obj[2] for obj in rowObjs]))
                y_end = int(np.mean([obj[3] for obj in rowObjs]))
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),virtual_color,2)
    if showWin:    
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