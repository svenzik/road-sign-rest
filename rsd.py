# RSD - road sign detection

import numpy as np
import cv2
#import matplotlib.pyplot as plt
import os
import re

# haarCascade = cv2.CascadeClassifier('road-signs-haar-cascade.xml')

orb = None
try:
    orb = cv2.ORB_create()
except AttributeError:
    orb = cv2.ORB()

def detect_road_signs(image, haarCascade):
    return haarCascade.detectMultiScale(image, 1.1, 3)

def draw_rectangles_to_image(image, rectangles):
    for rectangle in rectangles:
        #print(rectangle)
        x, y, w, h = rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255))

def crop_rectangles_from_image(image, rectangles):
    images = []
    for rectangle in rectangles:
        #print(rectangle)
        x, y, w, h = rectangle
        c = 0
        x = x+c
        y = y+c
        w = w-2*c
        h = h-2*c
        cropped_image = image[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (128, 128))
        images.append(resized_image)
    return images


def recognise_road_signs_on_image(image, haarCascade):
    recognisedObjectRectangles = detect_road_signs(image, haarCascade)
    cropped_images = crop_rectangles_from_image(image, recognisedObjectRectangles)
    return cropped_images

def recognise_road_signs_on_image_and_draw_rectangles(image, haarCascade):
    recognisedObjectRectangles = detect_road_signs(image, haarCascade)
    print("Detected signs: {}".format(len(recognisedObjectRectangles)))
    img = image.copy()
    draw_rectangles_to_image(img, recognisedObjectRectangles)
    return img

def show_image(img):
    print("Not showing image")
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.show()

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')
    show_image(out)

    # Also return the image if you'd like a copy
    return out

def drawMatchesKnn(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatchesKNN as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0
    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.
    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through a KNN
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for (img1_idx,mat) in enumerate(matches):
        (x1,y1) = kp1[img1_idx].pt # Modified for KNN
        for mat2 in mat: # Modified for KNN
            # Get the matching keypoints for each of the images
            img2_idx = mat2.trainIdx
            # x - columns
            # y - rows
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    show_image(out)

    return out # Return the image

def detectAndComputeSURF(image, parameter = None):
    return cv2.xfeatures2d.SURF_create().detectAndCompute(image, None)

    surfDetector = cv2.FeatureDetector_create("SURF")
    # surfDetector.setDouble("hessianThreshold", 1000)
    surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
    keypoints = surfDetector.detect(image)
    (keypoints, descriptors) = surfDescriptorExtractor.compute(image, keypoints)
    return (keypoints, descriptors)

def detectAndComputeSIFT(image, parameters = None):
    return cv2.xfeatures2d.SIFT_create().detectAndCompute(image, None)

    siftDetector = cv2.FeatureDetector_create("SIFT")
    siftDescriptorExtractor = cv2.DescriptorExtractor_create("SIFT")
    keypoints = siftDetector.detect(image)
    (keypoints, descriptors) = siftDescriptorExtractor.compute(image, keypoints)
    return (keypoints, descriptors)

def get_image_matches(img1, img2):
    if img1 is None:
        print("Image1 not found")
    if img2 is None:
        print("Image2 not found")

    # Initiate SIFT orb
    orb = orb

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    # img3 = drawMatches(img1,kp1,img2,kp2,matches[:10])
    img3 = drawMatches(img1,kp1,img2,kp2,matches)

    return matches;

def get_image_knn_matches(img1, img2):
    if img1 is None:
        print("Image1 not found")
    if img2 is None:
        print("Image2 not found")

    blur = cv2.GaussianBlur(img1,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    img1 = cv2.bitwise_not(thresh)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    #FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    # index_params= dict(algorithm = FLANN_INDEX_LSH,
    #                    table_number = 12, # 12
    #                    key_size = 20,     # 20
    #                    multi_probe_level = 2) #2
    search_params = dict()   # or pass empty dictionary checks=50

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # return matches
    # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #        # Removed the brackets around m
    #        good.append(m)

    # Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for i,mn in enumerate(matches):
        if len(mn) != 2:
            continue
        (m,n) = mn
        # good.append(m)
        # print("{}<0.7*{}".format(m.distance, n.distance))
        if m.distance < 0.7*n.distance:
        # if m.distance < 50:
        # if m.distance < 0.7*n.distance and m.distance < 70:
            good.append(m)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    # img3 = drawMatches(img1,kp1,img2,kp2,matches[:10])

    # img3 = drawMatches(img1,kp1,img2,kp2,good)

    #img3 = drawMatchesKnn(img1,kp1,img2,kp2,matches)

    return good

def get_image_knn_matches_sift(img1, img2):
    if img1 is None:
        print("Image1 not found")
    if img2 is None:
        print("Image2 not found")

    # blur = cv2.GaussianBlur(img1,(5,5),0)
    # thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    # thresh = cv2.dilate(thresh, None, iterations=2)
    # img1 = cv2.bitwise_not(thresh)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detectAndComputeSIFT(img1,None)
    kp2, des2 = detectAndComputeSIFT(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for x in enumerate(matches):
        i,mn = x
        if len(mn) != 2:
            continue
        (m,n) = mn
        # good.append(m)
        # print("{}<0.7*{}".format(m.distance, n.distance))
        if m.distance < 0.8*n.distance:
            good.append(m)

    img3 = drawMatches(img1,kp1,img2,kp2,good)
    #img3 = drawMatchesKnn(img1,kp1,img2,kp2,matches)

    return good

def get_image_knn_matches_surf(img1, img2):
    if img1 is None:
        print("Image1 not found")
    if img2 is None:
        print("Image2 not found")

    # blur = cv2.GaussianBlur(img1,(5,5),0)
    # thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    # img1 = cv2.bitwise_not(thresh)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detectAndComputeSURF(img1,None)
    kp2, des2 = detectAndComputeSURF(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # return matches
    # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #        # Removed the brackets around m
    #        good.append(m)

    # Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for x in enumerate(matches):
        i,mn = x
        if len(mn) != 2:
            continue
        (m,n) = mn
        # good.append(m)
        # print("{}<0.7*{}".format(m.distance, n.distance))
        if m.distance < 0.7*n.distance:
        # if m.distance < 50:
        # if m.distance < 0.7*n.distance and m.distance < 70:
            good.append(m)

    img3 = drawMatches(img1,kp1,img2,kp2,good)
    #img3 = drawMatchesKnn(img1,kp1,img2,kp2,matches)

    return good;

def get_speed_for_image(img_gray, img_trian_array):
    results = dict()
    for img_path in img_trian_array:
        try:
            speed_number = re.search('/(.+)/', img_path).group(1)
        except AttributeError:
            # AAA, ZZZ not found in the original string
            speed_number = '-' # apply your error handling

        matches = match_speed_image(img_gray, img_path)
        results[speed_number] = len(matches)

    # speeds_hits_sorted = sorted(results.items(), key=lambda x: x[1])
    speeds_hits_sorted = sorted(results.items(), key=lambda(k, v): (v, int(k)))
    print("speeds_hits_sorted:" + str(speeds_hits_sorted))
    speed,count = speeds_hits_sorted[len(speeds_hits_sorted)-1]
    if count <= 3:
        return '-'
    return speed

def match_speed_image(img_gray, img_train_path):
    try:
        speed_number = re.search('/(.+)/', img_train_path).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        speed_number = '-' # apply your error handling

    img_train = cv2.imread(img_train_path, 0)
    matches = get_image_knn_matches(img_gray, img_train)
    sum = 0
    count = len(matches)

    distances = []
    for moo in matches:
        distances.append(round(moo.distance,1))
        sum = sum + moo.distance

    if count == 0:
        avg = 0
    else:
        avg = round(sum/count,1)

    print("Matches count for {}: {}, avg: {} {}".format(speed_number, count, avg, distances))
    return matches

def read_paths(path):
    images = []
    for dirname, dirnames, _ in os.walk(path):
        for subdirname in dirnames:
            filepath = os.path.join(dirname, subdirname)
            for filename in os.listdir(filepath):
            # try:
                imgpath = str(os.path.join(filepath, filename))
                images.append(imgpath)
            # except IOError, (errno, strerror):
            #     print "I/O error({0}): {1}".format(errno, strerror)
            # except:
            #     print "Unexpected error:", sys.exc_info()[0]
            #     raise
    return images
