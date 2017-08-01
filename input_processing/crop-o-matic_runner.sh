#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Requires the data folder as a first argument"
    exit
fi

FOLDER=`realpath $1`

if [ ! -d "$FOLDER" ]; then
    echo "Requires the data folder as a first argument"
    exit
fi

for TIF in `ls $FOLDER/patient*tif`; do
    echo "Extracting images from " $TIF
    echo "============================="
    BASE=`echo $TIF | sed 's/....$//'`
    time ./crop-o-matic.py --data=${BASE}.tif --annotation=${BASE}.xml --mask=${BASE}_mask.tif --tfrecord=${BASE}.tfrecord --imageout=${FOLDER}
done
