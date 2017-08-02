#!/usr/bin/env python3
"""
Creates mask files based upon the original data and the annotation.

The generate mask will have different colors for 'not-annotated', 
'normal' and 'metastasis' regions.

elie.debrauwer@barco.com - 20170801

"""


# Append ASAP to PYTHONPATH prior to import
import sys
sys.path.append("/opt/ASAP/bin")

import multiresolutionimageinterface as mir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="The file to operate on")
parser.add_argument("--annotation", required=True, help="The annotation file")
parser.add_argument("--mask", required=True, help="The mask file to generate")
args = parser.parse_args()

reader = mir.MultiResolutionImageReader()
mr_image = reader.open(args.data)
assert (mr_image.valid() == True), "Failed to open %s" % args.data

annotation_list = mir.AnnotationList()
xml_repo = mir.XmlRepository(annotation_list)
xml_repo.setSource(args.annotation)
assert (xml_repo.load() == True), "Failed to open annotation %s" % args.annotation


annotation_mask = mir.AnnotationToMask()    
label_map = {'metastases': 1, 'normal': 2}
print("Calculating the mask, this will take about 5 minutes...")
annotation_mask.convert(annotation_list, args.mask, mr_image.getDimensions(), mr_image.getSpacing(), label_map)


