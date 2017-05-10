#!flask/bin/python
import sys
from flask import Flask, jsonify, request

import imageops
import rsd
import numpy as np

import uuid
import socket

import cv2

volume_dir = '/tmp'
haarCascade = cv2.CascadeClassifier('road-signs-haar-cascade.xml')
train_image_paths = rsd.read_paths("data")
CV_LOAD_IMAGE_COLOR = 1

if len(sys.argv) == 2:
    volume_dir = sys.argv[1]

print "Data dir is: " + volume_dir
print("Training images: {}".format(train_image_paths))

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'uuid': uuid.uuid4(),'hostname': socket.gethostname()})

@app.route('/s2img', methods=['POST'])
def s2img():
    json_data = imageops.get_request_rest_image_data(request, volume_dir)
    image_arr = np.fromstring (json_data['image'], np.uint8)
    image = cv2.imdecode(image_arr, CV_LOAD_IMAGE_COLOR)
    images_all = np.concatenate(rsd.recognise_road_signs_on_image(image, haarCascade), axis=1)
    return jsonify(imageops.create_response("-", images_all))

@app.route('/s2speed', methods=['POST'])
def s2speed():
    json_data = imageops.get_request_rest_image_data(request, volume_dir)
    image_arr = np.fromstring (json_data['image'], np.uint8)
    image = cv2.imdecode(image_arr, CV_LOAD_IMAGE_COLOR)
    images_cropped = rsd.recognise_road_signs_on_image(image, haarCascade)

    speed_all = ''
    speed_min = 300
    good_images = []
    for img in images_cropped:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        speed = rsd.get_speed_for_image(img_gray, train_image_paths)
        print("Found speed: " + speed)
        if speed != '-':
            speed_min = min(speed_min, int(speed))
            speed_all = speed_all + speed + ","
            good_images.append(img)

    images_all_raw = None
    if len(good_images) != 0:
        images_all = np.concatenate(good_images, axis=1)
        images_all_raw = cv2.imencode('.jpg',images_all)[1]
    speed_str = "{}({})".format(speed_min, speed_all)
    return jsonify(imageops.create_response(speed_str, images_all_raw))

@app.route('/fs2rest', methods=['POST'])
def fs2rest():
    json_data = imageops.get_request_rest_image_data(request, volume_dir)
    image_location = json_data['image_filename']
    image = json_data['image']
    return jsonify(imageops.create_response(image_location, image))

@app.route('/rest2fs', methods=['POST'])
def rest2fs():
    json_data = imageops.get_request_rest_image_data(request, volume_dir)
    image_location = imageops.save_image_to_disk(volume_dir, json_data)
    return jsonify(imageops.create_response(image_location))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9001)
