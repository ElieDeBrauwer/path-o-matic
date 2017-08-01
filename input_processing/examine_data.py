#!/usr/bin/env python3
"""
Examines the data files and the annotation and prints
some information to the console.


elie.debrauwer@barco.com - 20170801

"""


# Append ASAP to PYHONPATH prior to import
import sys
sys.path.append("/opt/ASAP/bin")

import multiresolutionimageinterface as mir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="The file to operate on")
parser.add_argument("--annotation", required=True, help="The annotation file")
args = parser.parse_args()

reader = mir.MultiResolutionImageReader()
mr_image = reader.open(args.data)
print("Image valid:", mr_image.valid())
print("Number of z planes: %d" % mr_image.getNumberOfZPlanes())
levels = mr_image.getNumberOfLevels()
print("Color type (RGB=%d) %d" % (mir.RGB, mr_image.getColorType()))
print("Data type (UCHAR=%d) %d" % (mir.UChar, mr_image.getDataType()))
print("Pixel spacing (um)", mr_image.getSpacing())
print("Number of levels: %d" % mr_image.getNumberOfLevels())
for level in range(levels):
    dims = mr_image.getLevelDimensions(level)
    sample_fact = mr_image.getLevelDownsample(level)
    print("Level %d: dimensions %6dx%6d downsample factor %6.2f" % (level, dims[0], dims[1], sample_fact ))
print("Base dimension (level 0):", mr_image.getDimensions())

annotation_list = mir.AnnotationList()
xml_repo = mir.XmlRepository(annotation_list)
xml_repo.setSource(args.annotation)
xml_repo.load()
for annotation in annotation_list.getAnnotations():
    print(annotation.getName(), "part of group", annotation.getGroup().getName())
    print(" * Number of points", annotation.getNumberOfPoints())
    print(" * Area", annotation.getArea())
    print(" * Center (%d, %d)" % (annotation.getCenter().getX(), annotation.getCenter().getY()) )
    bounding_box = annotation.getImageBoundingBox()
    print(" * Bounding box (%d, %d) to  (%d, %d)" % (bounding_box[0].getX(), bounding_box[0].getY(),
                                                     bounding_box[1].getX(), bounding_box[1].getY()))
    print(" * Type (Polygon=%d) %d" % (mir.Annotation.POLYGON, annotation.getType()))

