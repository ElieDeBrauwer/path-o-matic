There are 2 sets of model/predict files

1. Common
   - expects a directory structure as follows
   cats_and_dogs_filtered/
      train/
        cats/
        dogs/
      validation/
        cats/
        dogs/
   The total number of train images is 2000
   The total number of validation images is 800
   The zip file can be downloaded from TBD but should work for other images

2. tl_model.py 
   - Generates a model with fine tuning from InceptionV3 model

3. tl_predict.py
   - Predicts the class of an image using the fine tune model generated

4. tl_embedmodel.py
   - Generates embeddings(bottleneck features) which are predictions from InceptionV3
   - Builds a whole new model with the embeddings as features

5. tl_embedpredict.py
   - Translates the image to bottleneck features using InceptionV3
   - Predicts the class of an image using the new model generated
