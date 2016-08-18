import argparse
import openslide as ops

def get_core_pos_by_lev (core_pos,slide_props,target_lev):
	lev_factor = (4**(core_pos[0]-target_lev))
	core_w = core_pos[3]*lev_factor # get core's width at target_lev
	core_h = core_pos[4]*lev_factor # ________'s height ____________
	
	x_start = int(slide_props[ops.PROPERTY_NAME_BOUNDS_X]) + core_pos[1]*(4**core_pos[0])
	y_start = int(slide_props[ops.PROPERTY_NAME_BOUNDS_Y]) + core_pos[2]*(4**core_pos[0])
	return x_start,y_start,core_w,core_h

ap = argparse.ArgumentParser()
ap.add_argument('-i','--input',required = True, help = 'Input slide')
ap.add_argument('-p','--position',required=True,help='Core position: specified by a list [lev core_x_start core_y_start core_w core_h]')
ap.add_argument('-l','--level', help = 'Zooming level from which the core will be extracted')
ap.add_argument('-o','--output',help = 'Output image')
args = vars(ap.parse_args())


# 0 is the highest level (highest resolution)
# each successive level is downsampled by 4 times in EACH dimension
# 16*h2 = 4*h1 = h0  and  16*w2 = 4*w1 = w0

if not args['level']:
	target_level = 3
else:
	target_level = int(args['level'])


slide = ops.OpenSlide(args['input'])

# x_start and y_start are those OF LEVEL 0
# w,h are those of TARGET LEVEL
print args['position']
core_pos = [int(x) for x in args['position'][1:-1].split(",")]
x_start,y_start,w,h = get_core_pos_by_lev(core_pos,slide.properties,target_level)

if target_level > slide.level_count:
	print "specified level exceeds the max level"
	print "getting max level possible, which is: level ", slide.level_count
	level = slide.level_count

output_img = slide.read_region((x_start,y_start),target_level,(w,h))

print "level ", target_level, " size: ", output_img.size

if not args['output']:
	output_file = 'core.jpg'
else:
	output_file = args['output']

output_img.save(output_file)

