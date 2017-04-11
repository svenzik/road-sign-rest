#!flask/bin/python
from flask import Flask, jsonify, request

import imageops

import uuid
import socket

app = Flask(__name__)

tasks = [
    {
        'id': 1,
	'uuid': uuid.uuid4(),
	'hostname': socket.gethostname(),
        'name': u'First'
    }
]

@app.route('/')
def index():
    return jsonify({'uuid': uuid.uuid4(),'hostname': socket.gethostname()})

@app.route('/fs2fs')
def fs2fs():
    return jsonify({'uuid': uuid.uuid4(),'hostname': socket.gethostname()})

@app.route('/fs2rest', methods=['POST'])
def fs2rest():
    json_data = imageops.get_request_rest_image_data(request)
    image_location = json_data['image_filename']
    image = json_data['image']
    return jsonify(imageops.create_response(image_location, image))

@app.route('/rest2fs', methods=['POST'])
def rest2fs():
    json_data = imageops.get_request_rest_image_data(request)
    image_location = imageops.save_image_to_disk(json_data)
    return jsonify(imageops.create_response(image_location))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9001)
