# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:42:58 2016

@author: uym2
"""

import argparse
import openslide as ops
import json


def get_core_pos_by_lev (obj,slide_props,bound_w=1,bound_h=1,input_lev=3,target_lev=1):
    # to be consistent with the core_splitting_lib, a core is called an "obj"
    # obj is identified by a 4-element list of [x_start,x_end,y_start,y_end]
    # therefore, w = obj[1]-obj[0] and h = obj[3]-obj[2]
	lev_factor = (4**(input_lev-target_lev))
	core_w = (obj[1]-obj[0]+2*bound_w)*lev_factor # get core's width at target_lev
	core_h = (obj[3]-obj[2]+2*bound_h)*lev_factor # ________'s height ____________
	
	x_start = int(slide_props[ops.PROPERTY_NAME_BOUNDS_X]) + (obj[0]-bound_w)*(4**input_lev)
	y_start = int(slide_props[ops.PROPERTY_NAME_BOUNDS_Y]) + (obj[2]-bound_h)*(4**input_lev)
	return x_start,y_start,core_w,core_h

def coreImgs_from_objList(ops_slide,objList,output_path,input_lev=3,target_lev=1):
    for i in range(len(objList)):
        print 'core: ', i
        x_start,y_start,core_w,core_h = get_core_pos_by_lev(objList[i],ops_slide.properties)
        core_img = ops_slide.read_region((x_start,y_start),target_lev,(core_w,core_h))
        output_file = output_path + 'core_' + str(i+1) + '.jpg'
        core_img.save(output_file)
        
ops_slide = ops.OpenSlide('D:\\Coding\\CoreSplitViewer\\static\\TMA-scan\\NCICDP_Colon_Cancer_1C__CDX2.scn')
output_path = 'D:\\Coding\\CoreSplitViewer\\static\\TMA-scan\\NCICDP_Colon_Cancer_1C__CDX2_cores\\'
objfile = ('D:\\Coding\\CoreSplitViewer\\static\\TMA-scan\\openslide_views_lev3\\NCICDP_Colon_Cancer_1C__CDX2__lev3_cores.json')
objList = json.loads(open(objfile).read())
coreImgs_from_objList(ops_slide,objList,output_path)

