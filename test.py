import numpy as np
import cv2
import cv

import re
import sys
import os

import rsd

from sklearn.externals import joblib
from skimage.feature import hog

def read_paths(path):
    """Returns a list of files in given path"""
    images = [[] for _ in range(2)]
    for dirname, dirnames, _ in os.walk(path):
        for subdirname in dirnames:
            filepath = os.path.join(dirname, subdirname)
            for filename in os.listdir(filepath):
                try:
                    imgpath = str(os.path.join(filepath, filename))
                    images[0].append(imgpath)
                    limit = re.findall('[0-9]+', filename)
                    images[1].append(limit[0])
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    return images

def load_images(imgpath):
    """Loads images in given path and returns
     a list containing image and keypoints"""
    images = read_paths(imgpath)
    imglist = [[], [], [], []]
    cur_img = 0
    # sift = cv2.SIFT()
    # sift = cv2.FeatureDetector_create("SIFT")
    # sift = cv2.ORB_create()
    sift = cv2.ORB()
    for i in images[0]:
        img = cv2.imread(i, 0)
        imglist[0].append(img)
        imglist[1].append(images[1][cur_img])
        cur_img += 1
        keypoints, des = sift.detectAndCompute(img, None)
        imglist[2].append(keypoints)
        imglist[3].append(des)
    return imglist

def main(argv):
    haarCascade = cv2.CascadeClassifier('road-signs-haar-cascade.xml')
    # clf = joblib.load("digits_cls.pkl")

    # haarCascade = cv2.CascadeClassifier('lbpCascade.xml')
    compare_images = load_images("data")
    train_image_paths = read_paths("data")[0]

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
                #comp, amnt  = rsd.run_bf(img_gray, compare_images)
                #print("ROad sign result: {} {}".format(comp, amnt))

                # match_speed_image(img_gray, 'data/50/50.jpg')
                # match_speed_image(img_gray, 'data/80/80.jpg')
                # match_speed_image(img_gray, 'data/90/90-est.jpg')
                # match_speed_image(img_gray, 'data/90/90-fra.jpg')
                # match_speed_image(img_gray, 'data/120/120.jpg')
                # nr = recognise_numbers(img_gray, clf)
                # print nr
                # id_features(cv.LoadImage('data/50/50.jpg', 0), cv.fromarray(img_gray))
                # show_image(img_gray)

                speed = rsd.get_speed_for_image(img_gray, train_image_paths)
                print("Speed = " + speed)




def show_image(img):
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
