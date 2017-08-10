# Cats abd Dogs
import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

#from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
#from PIL import Image
#from keras.models import Model
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras.optimizers import SGD

train_dir = "./cats_and_dogs_filtered/train"
validation_dir = "./cats_and_dogs_filtered/validation"
#train_dir = "./small/train"
#validation_dir = "./small/validation"
IM_WIDTH=299
IM_HEIGHT=299
BATCH_SIZE=16
#BATCH_SIZE=2
FC_SIZE=1024
NB_IV3_LAYERS_TO_FREEZE = 172
NB_EPOCHS = 50
#NB_EPOCHS = 1
SAVED_MODEL_DIR="./tf_embed_saved_models"

def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
    print("count:", count)
    return count

def save_bottleneck_features():

  model = InceptionV3(weights='imagenet', include_top=False)

  nb_train_samples = get_nb_files(train_dir)
  train_datagen = ImageDataGenerator(
    rescale=1. /255
  )
  train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
  )
  bottleneck_features_train = model.predict_generator(
    train_generator, nb_train_samples // BATCH_SIZE)
  np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
  print("saved train embeddings:", bottleneck_features_train.shape)

  nb_validation_samples = get_nb_files(validation_dir)
  validation_datagen = ImageDataGenerator(
    rescale=1. /255
  )
  validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
  )
  bottleneck_features_validation = model.predict_generator(
    validation_generator, nb_validation_samples // BATCH_SIZE)
  np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
  print("saved validation embeddings:", bottleneck_features_validation)

def train_top_model():
  train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
  nb_train_samples = get_nb_files(train_dir)
  train_labels = np.array(
	[0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
  validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
  nb_validation_samples = get_nb_files(validation_dir)
  validation_labels = np.array(
	[0] * int(nb_validation_samples/2) + [1] * int(nb_validation_samples/2))
  model = Sequential()
  model.add(Flatten(input_shape=train_data.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

  model.fit(
	train_data,
	train_labels,
	batch_size=BATCH_SIZE,
	validation_data = (validation_data, validation_labels)
  )

  model.save(SAVED_MODEL_DIR)
  print("saved model")
  

save_bottleneck_features()
train_top_model()
