import base64
import uuid
import socket

#import cv2

def get_request_rest_image_data(request):
    json_data = request.get_json()
    filename = json_data['filename']

    if 'image' in json_data:
        image_data = base64.b64decode(json_data['image'])
    else:
        image_data = read_image_from_disk(filename)

    result = {
        'image_filename': filename,
        'image': image_data
    }
    #if json_data['image_filename_new'] is not None:
    #    result['image_filename_new'] = json_data['image_filename_new']
    return result

def create_response(image_location, image = None):
    result = {
        'id': uuid.uuid4(),
        'hostname': socket.gethostname(),
        'image_path': image_location
    }
    if image is not None:
        print "Found image from " + image_location + " with length " + str(len(image))
        result['image'] = base64.b64encode(image)
    return result

def save_image_to_disk(image_data):
    file_path = '/tmp/' + image_data['image_filename']
    print "Write image to " + file_path
    f = open(file_path, 'wb')
    f.write(image_data['image'])
    f.close()
    return file_path

def read_image_from_disk(file_path):
    print "Read image from: " + file_path
    f = open(file_path, 'rb')
    image_data = f.read()
    f.close()
    return image_data
    # return cv2.imread(file_path)
