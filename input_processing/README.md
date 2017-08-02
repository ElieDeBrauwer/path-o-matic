# path-o-matic - Data preprocessing junkyard

## examine_data.py

examine_data.py takes a tiff and its annotation and prints some basic information about the data file and the annotation. It could be use for exploration and for testing that the library has been correctly installed.

## create_masks.py and create_masks_runner.py

The purpose of these scripts is to create mask files based on a data file and an annotation. The python script does the work an operates on a single file. The runner performs the operation in bulk on an entire folder.


## crop-o-matic.py and crop-o-matic_runner.py

Based upon the mask files, the data files and the annotations the crop-o-matic will create a tfrecord and some images. The runner performs this operation in bulk on an entire folder.