#!/usr/bin/env python3
"""
Based upon a data file, an annotation and a mask file this script will
will generate a set of crops from within the annotated regions.

Masks should be pregenerated using create_masks.py


elie.debrauwer@barco.com - 20170801

"""

import argparse
import numpy as np
from scipy import misc

        
# Append ASAP to PYHONPATH prior to import
import sys
sys.path.append("/opt/ASAP/bin")

import multiresolutionimageinterface as mir

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="The file to operate on")
parser.add_argument("--annotation", required=True, help="The annotation file")
parser.add_argument("--mask", required=True, help="The file which contains the mask")
parser.add_argument("--dim", type=int, default=299, help="The default crop dimension in pixel")
parser.add_argument("--stridex", type=int, default=0, help="The horizontal offset (can be negative) between two consecutive crops (default: nonoverlapping)")
parser.add_argument("--stridey", type=int, default=0, help="The vertical offset (can be negative) between two consecutive crops (default: nonoverlapping)")

args = parser.parse_args()

reader = mir.MultiResolutionImageReader()
mr_image = reader.open(args.data)
assert (mr_image.valid() == True), "Failed to open data: %s" % args.data

reader_mask  = mir.MultiResolutionImageReader()
mr_mask = reader.open(args.mask)
assert (mr_image.valid() == True), "Failed to open mask: %s" % args.data

annotation_list = mir.AnnotationList()
xml_repo = mir.XmlRepository(annotation_list)
xml_repo.setSource(args.annotation)
assert (xml_repo.load() == True), "Failed to open annotation %s" % args.annotation

crop_cnt = 0
annotation_cnt = 0
for annotation in annotation_list.getAnnotations():
    if annotation.getGroup().getName() == "metastases":
        annotation_cnt += 1
        bounding_box = annotation.getImageBoundingBox()
        print("Found metatstatis annotation %d: (%d, %d) to (%d, %d)" %
              (annotation_cnt, 
               bounding_box[0].getX(), bounding_box[0].getY(),
               bounding_box[1].getX(), bounding_box[1].getY()))
        assert(annotation.getType() == mir.Annotation.POLYGON), "Polygon annotation expected"
        assert(annotation.getArea() > 0), "Expected non-zero annotation area"

        start_x = bounding_box[0].getX()
        start_y = bounding_box[0].getY()
        end_x = bounding_box[1].getX()
        end_y = bounding_box[1].getY()
        y = start_y - args.dim

        while y < end_y:
            x = start_x - args.dim
            while x < end_x:
                crop_cnt += 1
                x += args.dim + args.stridex
                print("Taking crop %d at %d %d" % (crop_cnt, x, y))
                patch = np.array(mr_image.getUCharPatch(int(x), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                mask = np.array(mr_mask.getUCharPatch(int(x), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                mask_grey = np.repeat(mask * 255, 3, axis=2) # Convert the mask in to a black-n-white array.
                misc.imsave("patch_%05d_%d_%d_patch.png" % (crop_cnt, x, y),  patch)
                misc.imsave("patch_%05d_%d_%d_mask.png" % (crop_cnt, x, y),  mask_grey)

                metastases_area = np.count_nonzero(mask) / (299. * 299.)
                print("Metastasis area %5.2f%%" % (metastases_area * 100))
                
            y += args.dim + args.stridey
    else:
        assert(False), "File has normal region, not supported in this version !"



print("Took %d crops from %d annotations" % (crop_cnt, annotation_cnt))

