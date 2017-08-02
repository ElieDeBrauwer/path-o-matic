import tensorflow as tf
import mnist_features as features

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

session = tf.Session()
prediction_version = "prediction:0"

def restore(model_dir, model_name, prediction_version_val):
    print("Restoring model_dir:", model_dir, " model_name:", model_name, " prediction_version:", prediction_version_val);
    model_meta = model_dir + "/" + model_name + ".ckpt.meta"
    model_saver = tf.train.import_meta_graph(model_meta)
    model_saver.restore(session, tf.train.latest_checkpoint(model_dir))
    #prediction_version = prediction_version_val
    print("Model restored.")

def predict(filename):
    x = features.get_features(filename)
    y_predict_val = session.run(prediction_version, feed_dict={"x:0":[x]})
    return str(y_predict_val)

