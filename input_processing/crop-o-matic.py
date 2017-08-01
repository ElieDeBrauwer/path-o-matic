#!/usr/bin/env python3
"""
Based upon a data file, an annotation and a mask file this script will
will generate a set of crops from within the annotated regions.

Masks should be pregenerated using create_masks.py


elie.debrauwer@barco.com - 20170801

"""

import argparse
import numpy as np
import os
from scipy import misc
import sys
import tensorflow as tf

# Append ASAP to PYTHONPATH prior to import
sys.path.append("/opt/ASAP/bin")

import multiresolutionimageinterface as mir

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="The file to operate on")
parser.add_argument("--annotation", required=True, help="The annotation file")
parser.add_argument("--mask", required=True, help="The file which contains the mask")
parser.add_argument("--dim", type=int, default=299, help="The default crop dimension in pixel of a quadrant")
parser.add_argument("--stridex", type=int, default=598, help="The horizontal offset (can be negative) between two consecutive crops (default: nonoverlapping)")
parser.add_argument("--stridey", type=int, default=598, help="The vertical offset (can be negative) between two consecutive crops (default: nonoverlapping)")
parser.add_argument("--tfrecord", default="./tfrecord.tf", help="The TFRecord to write to")
parser.add_argument("--imageout", default="./", help="The folder where the images will be written to")
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

writer = tf.python_io.TFRecordWriter(args.tfrecord)

for annotation in annotation_list.getAnnotations():
    if annotation.getGroup() is None:
        # Sometimes the group of a bogus annotation is set to None
        continue
    if annotation.getGroup().getName() == "metastases":
        annotation_cnt += 1
        bounding_box = annotation.getImageBoundingBox()
        print("Found metastatis annotation %d: (%d, %d) to (%d, %d)" %
              (annotation_cnt,
               bounding_box[0].getX(), bounding_box[0].getY(),
               bounding_box[1].getX(), bounding_box[1].getY()))
        assert(annotation.getType() == mir.Annotation.POLYGON), "Polygon annotation expected"
        assert(annotation.getArea() > 0), "Expected non-zero annotation area"

        start_x = bounding_box[0].getX()
        start_y = bounding_box[0].getY()
        end_x = bounding_box[1].getX()
        end_y = bounding_box[1].getY()
        y = start_y

        while y < end_y:
            x = start_x
            while x < end_x:
                crop_cnt += 1
                print("Taking crop %d at %d %d" % (crop_cnt, x, y))
                # A B
                # C D

                # Grab patches.
                patch_a = np.array(mr_image.getUCharPatch(int(x), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                patch_b = np.array(mr_image.getUCharPatch(int(x + args.dim), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                patch_c = np.array(mr_image.getUCharPatch(int(x), int(y + args.dim), args.dim, args.dim, 0), dtype=np.uint8)
                patch_d = np.array(mr_image.getUCharPatch(int(x + args.dim), int(y + args.dim), args.dim, args.dim, 0), dtype=np.uint8)
                patch_full = np.array(mr_image.getUCharPatch(int(x), int(y), 2 * args.dim, 2 * args.dim, 0), dtype=np.uint8)
                # This should work, but doesn't. No worky :(
                # Let's hack around this.
                # patch_low = np.array(mr_image.getUCharPatch(int(x * 2), int(y * 2), args.dim, args.dim, 1), dtype=np.uint8)
                patch_low = misc.imresize(patch_full, 0.5, interp="bilinear")
                print(patch_low)
                print(patch_low.shape)

                # Grab masks.
                mask_a = np.array(mr_mask.getUCharPatch(int(x), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                mask_b = np.array(mr_mask.getUCharPatch(int(x + args.dim), int(y), args.dim, args.dim, 0), dtype=np.uint8)
                mask_c = np.array(mr_mask.getUCharPatch(int(x), int(y + args.dim), args.dim, args.dim, 0), dtype=np.uint8)
                mask_d = np.array(mr_mask.getUCharPatch(int(x + args.dim), int(y + args.dim), args.dim, args.dim, 0), dtype=np.uint8)
                mask_full = np.append(
                    np.append( mask_a, mask_b, axis=1),
                    np.append( mask_c, mask_d, axis=1), axis=0)
                mask_full_bw = np.repeat(mask_full * 255, 3, axis=2) # Convert the mask in to a black-n-white array.

                # Write images.
                fname = os.path.basename(args.data).split(".")[0]
                misc.imsave(args.imageout + "/%s_%05d_A_%d_%d.png" % (fname, crop_cnt, x, y),  patch_a)
                misc.imsave(args.imageout + "/%s_%05d_B_%d_%d.png" % (fname, crop_cnt, x + args.dim, y),  patch_b)
                misc.imsave(args.imageout + "/%s_%05d_C_%d_%d.png" % (fname, crop_cnt, x, y + args.dim),  patch_c)
                misc.imsave(args.imageout + "/%s_%05d_D_%d_%d.png" % (fname, crop_cnt, x + args.dim, y + args.dim),  patch_d)
                misc.imsave(args.imageout + "/%s_%05d_%d_%d_low.png" % (fname, crop_cnt, x, y),  patch_low)
                misc.imsave(args.imageout + "/%s_%05d_%d_%d_full.png" % (fname, crop_cnt, x, y),  patch_full)
                misc.imsave(args.imageout + "/%s_%05d_%d_%d_mask.png" % (fname, crop_cnt, x, y),  mask_full_bw)

                # Calculate areas
                metastases_area_a = np.count_nonzero(mask_a) / (args.dim * args.dim)
                metastases_area_b = np.count_nonzero(mask_b) / (args.dim * args.dim)
                metastases_area_c = np.count_nonzero(mask_c) / (args.dim * args.dim)
                metastases_area_d = np.count_nonzero(mask_d) / (args.dim * args.dim)
                metastases_area = np.count_nonzero(mask_full) / (args.dim * args.dim * 4.0) # 4 quadrants
                print("Metastasis area %5.2f%%" % (metastases_area * 100))

                # Append to TFRecord.
                data = tf.train.Example(features=tf.train.Features(feature={
                    'crop_top_left': _bytes_feature(patch_a.tostring()),
                    'crop_top_right': _bytes_feature(patch_b.tostring()),
                    'crop_bottom_left': _bytes_feature(patch_c.tostring()),
                    'crop_bottom_right': _bytes_feature(patch_d.tostring()),
                    'image_low_res': _bytes_feature(patch_low.tostring()),
                    'x': _int64_feature(int(x)),
                    'y': _int64_feature(int(y)),
                    "area": _int64_feature(int(metastases_area * 100)),
                    "area_a": _int64_feature(int(metastases_area_a * 100)),
                    "area_b": _int64_feature(int(metastases_area_b * 100)),
                    "area_c": _int64_feature(int(metastases_area_c * 100)),
                    "area_d": _int64_feature(int(metastases_area_d * 100)),
                    "image_name": _bytes_feature(bytes(os.path.basename(args.data), "utf-8")), #TODO: URI
                    "label": _int64_feature(int( metastases_area > 0.10)), # Threshold set to 10%.
                    }))
                writer.write(data.SerializeToString())

                x += args.stridex
            y += args.stridey
    else:
        assert(False), "File has normal region, not supported in this version !"

writer.close()

print("Took %d crops from %d annotations" % (crop_cnt, annotation_cnt))
