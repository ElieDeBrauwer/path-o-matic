# Web Server for analyzing images for classification


The webserver is implemented as a python app.

Code is based on sample webserver implementation from
https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_sections/transfer_learning/cloudml/web_server

## Installing dependencies 


If pip is not already installed, you can install it using:

```shell
sudo apt-get install python-setuptools python-dev build-essential
sudo easy_install pip 
```
 
To install the dependencies run:
```shell
sudo pip install -r requirements.txt
```

## Starting the app

To use self hosted prediction service you need to provide direct_url option and username/password:

```shell
python predict_server.py --model_name dummy --project dummy --dict static/dict.txt --direct_url='https://35.197.102.39/app/classify' --username=barco --password='somepassword'
```

To use Google Cloud ML micro service for prediction provide correct model_name and project:


```shell
python predict_server.py --model_name somemodel --project someproject --dict static/dict.txt 
```

The app listens on port 5000 by default.
You can expose it as such for internal testing or expose it via Apache/Nginx.

You can specify alternate port using --port argument
If you want to run on port 80, use sudo to invoke the app:

```shell
sudo python predict_server.py --model_name somemodel --project someproject --dict static/dict.txt --port 80
```




