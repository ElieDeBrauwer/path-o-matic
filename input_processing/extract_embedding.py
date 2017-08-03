#!/usr/bin/env python3
"""
extract_embedding will read some TFRecords and extract the embeddings
and the filenames. This will be the input for the similar image search.
"""

import argparse
import os
import pickle
import subprocess
import tensorflow as tf


def persist_embeddings(embeddings, output):
    """
    Write the embedding to output.
    """
    print("Writing embeddings to ", output)
    with open(output, 'wb') as file:
        pickle.dump(embeddings, file, protocol=2)


def learn_image_name_from_gs(name, x, y):
    """
    When creating the embedding we stored the original filename, in order to get the
    similar image name from the embeddings we need to look it up in GCS :/.
    """
    query = "gs://path-o-matic/%s_*_%d_%d_low.png" % (name[:-4], x, y)
    return(subprocess.getoutput("gsutil ls %s" % query).split("/")[-1])

def extract_embeddings(file):
    """
    Extract the embeddings from a given files.
    Returns an array of (file, label, embedding) tuples.
    """
    print("Going to extract embeddings from", file)
    tuples = []
    cnt = 1
    for serialized_example in tf.python_io.tf_record_iterator(file):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image = example.features.feature["image_name"].bytes_list.value[0].decode("utf-8")
        x = example.features.feature["x"].float_list.value[0]
        y = example.features.feature["y"].float_list.value[0]
        label = example.features.feature["label"].float_list.value[0]
        embeddings = []
        for i in range(2048):
            embeddings.append(example.features.feature["emb_low_res"].float_list.value[i])
        # The following is the proper workaround
        #image_name = learn_image_name_from_gs(image, x, y)
        # And this is the ugly hack which achieves the same :-], but assumes that
        # Jonas' dataflow magic works more or less synchronous.
        image_name = "%s_%05d_%d_%d_low.png" % (image[:-4], cnt,  x, y)
        cnt += 1

        tuples.append( (image_name, label, embeddings) )
        
    return tuples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Location where to write the result to")
    parser.add_argument("--input", required=True, nargs="+", help="The input TFRecords")
    args = parser.parse_args()

    embeddings = []
    for file in args.input:
        emb_file = extract_embeddings(file)
        for e in emb_file:
            embeddings.append(e)
        print("Found %d embeddings in this file" % len(emb_file))


    print("Found %d embeddings in total" % len(embeddings))
    persist_embeddings(embeddings, args.output)
    
if __name__ == "__main__":
    main()
    
