import numpy as np
import cv2
import matplotlib.pyplot as plt
# import cv

import re
import sys
import os

import rsd

from sklearn.externals import joblib
from skimage.feature import hog


def read_paths(path):
    """Returns a list of files in given path"""
    paths = []
    for dirname, dirnames, _ in os.walk(path):
        for subdirname in dirnames:
            filepath = os.path.join(dirname, subdirname)
            for filename in os.listdir(filepath):
                imgpath = str(os.path.join(filepath, filename))
                paths.append(imgpath)
    return paths

def main(argv):
    haarCascade = cv2.CascadeClassifier('road-signs-haar-cascade.xml')
    train_image_paths = read_paths("data")

    if len(argv) > 0:
        for image_path in argv:
            print("Recognising images in {}".format(image_path))
            image = cv2.imread(image_path)
            #show_image(rsd.recognise_road_signs_on_image_and_draw_rectangles(image, haarCascade))

            images_cropped = rsd.recognise_road_signs_on_image(image, haarCascade)
            #show_image(np.concatenate(images_cropped, axis=1))
            for img in images_cropped:
                show_image(img)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                speed = rsd.get_speed_for_image(img_gray, train_image_paths)
                print("Speed = " + speed)




def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imshow('frame',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
   main(sys.argv[1:])

# camera = cv2.VideoCapture(0)
# while(1):
#     retval, image = camera.read()
#     recognise_road_signs_on_image_and_show(image, haarCascade)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# del(camera)
