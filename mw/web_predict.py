#!/usr/bin/env python
 
from os import curdir
from os.path import join as pjoin
from http.server import BaseHTTPRequestHandler, HTTPServer

import sys
import predict as predict
 
# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

  store_path = pjoin(curdir, 'store.png')
 
  # GET
  def do_GET(self):
        # Send response status code
        self.send_response(200)
 
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
  def do_GET(self):
        # Send response status code
        self.send_response(200)
 
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

  def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        with open(self.store_path, "wb") as fh:
          fh.write(post_data)
          fh.close()
        predict_val = predict.predict(self.store_path)

        self.send_header('Content-type','text/html')
        self.end_headers()
        message = "predict_val:" + predict_val
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

 
def run():
  print('starting server...')
 
  port = 8080
  server_address = ('127.0.0.1', 8080)
  httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
  print('listening on port:', port)
  httpd.serve_forever()
 
 
if len(sys.argv) < 4:
    print('Usage: python ' +  sys.argv[0] + ' <model_dir> <model_name> <prediction_version>')
    sys.exit(1)

model_dir = sys.argv[1]
model_name = sys.argv[2]
prediction_version_val = sys.argv[3]
predict.restore(model_dir, model_name, prediction_version_val)
run()
