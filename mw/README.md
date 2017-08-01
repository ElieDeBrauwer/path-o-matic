The middleware files are used to prototype a simple end to end MNIST model and serv setup. The following files are used in the setup
1. MnistModel.py : Used to build the MNIST model and save to disk
   The model is written to idirectory ./saved_models
2. MnistPredict.py : Used to read the model and predict/classify an image
   Reads ./saved_models/Images.ckpt.meta to build the graph
   Reads ./saved_models/* to read the model values 
3. web.py : Handle POST web requests on port 8080 to classify an input image
   Invokes the predict method in MnistPredict.py
4. postcloud.sh : shell script uses curl to send a classify request 
   Can used on an external ubuntu system to post an image for classification
