#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up

# Now that we are set up, we can start processing some images. We'll show how
# easy it is to swap in another set of images, instead of using the original
# 'flowers' set.

# Usage: USER=<user> DATA=<date> pathomatic_preproc.sh gs://your-bucket-name
# substituting 'your-bucket-name' with the name of the GCS bucket to use.

if [ -z "$1" ]
  then
    echo "No bucket supplied."
    exit 1
fi

declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r JOB_ID="pathomatic_${USER}_${DATE}"
declare -r DFJOB_ID="pathomatic-`echo ${DATE} | sed 's/_/-/g'`"
declare -r BUCKET=$1
declare -r OUTPUT_PATH="${BUCKET}/${USER}/preproc/${JOB_ID}"

echo
echo "Using job id: " $JOB_ID
echo "Using OUTPUT_PATH: " $OUTPUT_PATH
echo "Using eval input path: " "${BUCKET}/input_path/eval"
echo "Using train input path: " "${BUCKET}/input_path/train"
echo "Using eval output path: " "${OUTPUT_PATH}/eval"
echo "Using train output path: " "${OUTPUT_PATH}/train"

set -v -e

# We'll use Dataflow (Beam) to do our preprocessing. The preprocessing will use
# the trained Inception 3 model, and run our images through it to derive so-
# called 'bottleneck' embeddings for the images. The embedding info will be
# saved in the form of TFRecords, in GCS. We'll be able to use these records to
# train our own model.

# It takes a bit of time to preprocess all the images.

# If run in the cloud, these calls are asynchronous (by default).
# You can track job progress in the Dataflow console.
# The following uses 20 cores each, you can edit to use more / less
# if your project quota is sufficient to do so.
python trainer/preprocess.py \
  --input_path "${BUCKET}/input_path/eval/patient_*.tfrecord" \
  --output_path "${OUTPUT_PATH}/eval" \
  --job_name "${DFJOB_ID}e" \
  --num_workers 20 \
  --cloud \
  --requirements_file requirements.txt

python trainer/preprocess.py \
  --input_path "${BUCKET}/input_path/train/patient_*.tfrecord" \
  --output_path "${OUTPUT_PATH}/train" \
  --job_name "${DFJOB_ID}t" \
  --num_workers 20 \
  --cloud \
  --worker_machine_type n1-standard-8 \
  --requirements_file requirements.txt

set +v
echo
echo "Generated job id: " $JOB_ID
echo "Using OUTPUT_PATH: " $OUTPUT_PATH
echo "Using eval output path: " "${OUTPUT_PATH}/eval"
echo "Using train output path: " "${OUTPUT_PATH}/train"
echo "Dataflow jobs are running at ${DFJOB_ID}e and ${DFJOB_ID}t"
