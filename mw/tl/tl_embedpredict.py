import sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

#model_in = InceptionV3(weights="imagenet", include_top=True)
modelFile="./tf_embed_saved_models"
model_tl = load_model(modelFile)

from PIL import Image

target_size_in = (299, 299)

embedding_model = InceptionV3(weights='imagenet', include_top=False)
def predict_tl(model_x, img, target_size, top_n=3):
    if(img.size != target_size):
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    bottleneck_features = embedding_model.predict(x)
    print(bottleneck_features)
    preds = model_x.predict(bottleneck_features)
    return preds

if(len(sys.argv) < 2):
    print("Usage: ", sys.argv[0], " <imgFilename>")
    exit()

imgFile = sys.argv[1]
img = Image.open(imgFile)
prediction = predict_tl(model_tl, img, target_size_in)
print("imgFile:", imgFile, " prediction:", prediction)
