# Copyright 2016 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a little example web server that shows how to access the Cloud ML
API for prediction, using the google api client libraries. It lets the user
upload an image, then encodes that image and sends off the prediction request,
then displays the results.
It uses the default version of the specified model.  Some of the web app's
UI is hardwired for the "hugs" image set, but would be easy to modify for other
models.
See the README.md for how to start the server.
"""

import argparse
import base64
from cStringIO import StringIO
import numpy as np
import os
import pickle
import sys

from flask import Flask, redirect, render_template, request, url_for
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from PIL import Image
from werkzeug import secure_filename


desired_width = 598
desired_height = 598

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'png', 'PNG'])


args = None
labels = None

with open("embeddings.pickle", "rb") as file:
    embeddings=pickle.load(file)
    print "Loaded %s embeddings" % len(embeddings)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


current_idx = 1

def get_similar_images(pred_embeddings):
    """
    Get similar images based on the embeddings. Not this is not very
    space efficients. A sliding window version would be more interesting.
    """
    input = np.array(pred_embeddings)
    vals = []
    for entry in embeddings:
        cur = np.array(entry[2])
        distance = np.sum(np.square(input - cur))
        vals.append( (distance, entry[0]) )
    vals.sort(key=lambda val: val[0])
    res = '%s,%s,%s,%s,%s' % (vals[0][1][:-8], vals[1][1][:-8], vals[2][1][:-8], vals[3][1][:-8], vals[4][1][:-8])
    print "Similar images:", res
    return res

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global current_idx
    print "upload file"
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):
            label, score = '', ''

            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'],
                      filename))
            fname = "%s/%s" % (UPLOAD_FOLDER, filename)

            # Use Google's ML service for prediction
            ml_client = create_client()
            result = get_prediction(ml_client, args.project,
                                    args.model_name, fname)

            predictions = result['predictions']
            prediction = predictions[0]
            print("prediction: %s" % prediction["scores"])
            label_idx = prediction['prediction']
            score = prediction['scores'][label_idx]
            label = labels[label_idx]
            image_list = get_similar_images(prediction["embeddings"])
            print(label, score)
            return redirect(url_for('show_result',
                                    filename=filename,
                                    label=label,
                                    score=score,
                                    image_list=image_list))

    return render_template('index.html')


@app.route('/result')
def show_result():
    print "show_result"

    filename = request.args['filename']
    label = request.args['label']
    score = request.args['score']
    image_list = request.args['image_list']

    # This result handling logic is hardwired for the "hugs/not-hugs"
    # example, but would be easy to modify for some other set of
    # classification labels.
    if label == 'malignant':
        return render_template('jresults.html',
                               filename=filename,
                               label="Malignant",
                               score=score,
                               border_color="#B20000",
                               image_list=image_list)

    elif label == 'benign':
        return render_template('jresults.html',
                               filename=filename,
                               label="Benign",
                               score=score,
                               border_color="#00FF48",
                               image_list=image_list)
    else:
        return render_template('error.html',
                               message="Something went wrong.")


@app.route('/similar_images')
def show_similar_images():

    image_list = request.args['image_list']
    images = image_list.split(',')

    print "Similar Images: ", image_list

    return render_template('similarimages.html', image_list=images)

@app.route('/display_image')
def display_image():
    image_name = request.args['image_name']
    print "Display Image:", image_name
    return render_template('display_image.html', image_name=image_name)

def create_client():
  credentials = GoogleCredentials.get_application_default()
  ml_service = discovery.build(
      'ml', 'v1', credentials=credentials)
  return ml_service


def get_prediction(ml_service, project, model_name, input_image):
  request_dict = make_request_json(input_image)
  body = {'instances': [request_dict]}

  # This request will use the default model version.
  parent = 'projects/{}/models/{}'.format(project, model_name)
  request = ml_service.projects().predict(name=parent, body=body)
  result = request.execute()
  return result

def make_request_json(input_image):
  """..."""

  image = Image.open(input_image)
  resized_handle = StringIO()
  is_too_big = ((image.size[0] * image.size[1]) >
                (desired_width * desired_height))
  if is_too_big:
    image = image.resize(
        (desired_width, desired_height), Image.BILINEAR)

  image.save(resized_handle, format='PNG')
  encoded_contents = base64.b64encode(resized_handle.getvalue())

  image_json = {'key': input_image,
                'image_bytes': {'b64': encoded_contents}}
  return image_json


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_name', type=str, required=True,
      help='The name of the model.')
  parser.add_argument(
      '--dict',
      type=str,
      required=True,
      help='Path to dictionary file.')
  parser.add_argument(
      '--project', type=str, required=True,
      help=('The project name to use.'))
  parser.add_argument(
      '--port', type=int,
      help=('Specify port to run on (default 5000'))
  args, _ = parser.parse_known_args(sys.argv)
  return args


def read_dictionary(path):
  with open(path) as f:
    return f.read().splitlines()


if __name__ == "__main__":

    args = parse_args()
    labels = read_dictionary(args.dict)
    print("labels: %s" % labels)

    # Runs on port 5000 by default.
    if args.port and args.port > 0:
        app.run(host='0.0.0.0', port=args.port, debug=True)
    else:
        app.run(host='0.0.0.0', debug=True)


