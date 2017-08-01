#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Requires the data and annotation containing folder as a first argument"
    exit
fi

FOLDER=`realpath $1`

if [ ! -d "$FOLDER" ]; then
    echo "Requires the data and annotation containing folder as a first argument"
    exit
fi

for TIF in `ls $FOLDER/patient*tif`; do
    echo "Processing " $TIF
    echo "============================="
    BASE=`echo $TIF | sed 's/....$//'`
    time ./create_masks.py --data=${BASE}.tif --annotation=${BASE}.xml --mask=${BASE}_mask.tif
done
