#Middleware for Mnist classify service prototype

The middleware files are used to prototype a simple end to end MNIST model and serv setup. The directory names used are to make the integration points clear

The following files are used in the setup
1. MnistModel.py : Used to build the MNIST model and save to disk
   The model is written to directory ./saved_models
2. MnistPredict.py : Used to read the model and predict/classify an image
   Reads ./saved_models/Images.ckpt.meta to build the graph
   Reads ./saved_models/* to read the model values 
3. web.py : Handle POST web requests on port 8080 to classify an input image
   Invokes the predict method in MnistPredict.py
4. postcloud.sh : shell script uses curl to send a classify request 
   Can used on an external ubuntu system to post an image for classification

These set of files are copied onto a VM instance
# NGINX
The instance is updated to include an nginx service
The nginx config has a proxy_pass to redirect /app/classify to the web.py server on port 8080
The nginx server is secured using instructions available 
1. ssl cert generation at https://www.digitalocean.com/community/tutorials/how-to-create-a-self-signed-ssl-certificate-for-nginx-in-ubuntu-16-04
2. authentication at https://www.digitalocean.com/community/tutorials/how-to-set-up-basic-http-authentication-with-nginx-on-ubuntu-14-04
The default username/password is available in the postcloud.sh script

# web.py server
The web.py server is started using a screen command so that it stays up
https://www.mattcutts.com/blog/a-quick-tutorial-on-screen/
