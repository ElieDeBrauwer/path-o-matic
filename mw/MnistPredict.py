import sys
import tensorflow as tf
from PIL import Image, ImageFilter
from socket import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_x():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MINST_data', one_hot=True)
    TRAIN_SIZE = 5500
    x_train = mnist.train.images[:TRAIN_SIZE,:]
    index = 3
    return x_train[index]

def get_features(filename):
    im = Image.open(filename).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    #print(tva)



session = tf.Session()
model_saver = tf.train.import_meta_graph("saved_models/Images.ckpt.meta")
model_saver.restore(session, tf.train.latest_checkpoint("saved_models/"))
print("Model restored.")

def predict(filename):
    x = get_features(filename)
    y_predict_val = session.run("prediction:0", feed_dict={"x:0":[x]})
    return str(y_predict_val)

#filename = sys.argv[1]
#predict_val  = predict(filename)
#print("predict:val", predict_val)
