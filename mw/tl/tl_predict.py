import sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

#model_in = InceptionV3(weights="imagenet", include_top=True)
modelFile="./tf_saved_models"
model_tl = load_model(modelFile)

from PIL import Image

target_size_in = (299, 299)

def predict_in(model_x, img, target_size, top_n=3):
    if(img.size != target_size):
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    preds = model_x.predict(x)
    return preds
#    return decode_predictions(preds, top=top_n)[0]

def predict_tl(model_x, img, target_size, top_n=3):
    if(img.size != target_size):
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    preds = model_x.predict(x)
    return preds
#    return decode_predictions(preds, top=top_n)[0]

#img = Image.open("./African_Bush_Elephant.jpg")
if(len(sys.argv) < 2):
    print("Usage: ", sys.argv[0], " <imgFilename>")
    exit()
imgFile = sys.argv[1]
img = Image.open(imgFile)
#prediction = predict_in(model_in, img, target_size_in)
prediction = predict_tl(model_tl, img, target_size_in)
print("imgFile:", imgFile, " prediction:", prediction)
