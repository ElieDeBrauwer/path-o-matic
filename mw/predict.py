import sys
import tensorflow as tf
from socket import *
import mnist_features as features

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if len(sys.argv) < 4:
    print('Usage: python ' +  sys.argv[0] + ' <model_dir> <model_name> <pngfile>')
    sys.exit(1)

model_dir = sys.argv[1]
model_name = sys.argv[2]
model_meta = model_dir + "/" + model_name + ".ckpt.meta"
filename = sys.argv[3]

session = tf.Session()
model_saver = tf.train.import_meta_graph(model_meta)
model_saver.restore(session, tf.train.latest_checkpoint(model_dir))
print("Model restored.")

def predict(filename):
    x = features.get_features(filename)
    y_predict_val = session.run("prediction:0", feed_dict={"x:0":[x]})
    return str(y_predict_val)

predict_val  = predict(filename)
print("predict:val", predict_val)
