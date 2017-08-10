# Cats abd Dogs
import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from PIL import Image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

#train_dir = "./cats_and_dogs_filtered/train"
#validation_dir = "./cats_and_dogs_filtered/validation"
train_dir = "./small/train"
validation_dir = "./small/validation"
IM_WIDTH=299
IM_HEIGHT=299
#BATCH_SIZE=32
BATCH_SIZE=2
FC_SIZE=1024
NB_IV3_LAYERS_TO_FREEZE = 172
#NB_EPOCHS = 3
NB_EPOCHS = 1
SAVED_MODEL_DIR="./tf_saved_models"

def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
    return count

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE
)

def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

base_model = InceptionV3(weights='imagenet', include_top=False)
nb_classes = 2
model = add_new_last_layer(base_model, nb_classes)
setup_to_transfer_learn(model, base_model)
nb_train_samples = get_nb_files(train_dir)
nb_val_samples = get_nb_files(validation_dir)
history_tl = model.fit_generator(
    train_generator,
    nb_epoch = NB_EPOCHS,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')
#print("TL: loss", history_tl['loss'], " val_loss", history_tl['val_loss'])
print("TL: loss", history_tl)
setup_to_finetune(model)
history_ft = model.fit_generator(
    train_generator,
    nb_epoch = NB_EPOCHS,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')
#print("FT: loss", history_ft['loss'], " val_loss", history_ft['val_loss'])
print("FT: loss", history_ft)
model.save(SAVED_MODEL_DIR)
