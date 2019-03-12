# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:49:34 2019

@author: haanrj
"""


# import time
import cv2
import numpy as np
import geojson


def detectCorners(filename, method = 'standard'):
    """
    functiebeschrijving
    """
    img = cv2.imread(filename) # load image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert image to grayscale
    if method == 'adaptive':
        blur = cv2.medianBlur(gray, 5) # flur grayscale image
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imwrite('Adaptive_threshold.jpg', thresh) # store threshold image
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # flur grayscale image
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # threshold grayscale image as binary
        cv2.imwrite('Standard_threshold.jpg', thresh)  # store threshold image

    # detect corner circles in the image (min/max radius ensures only finding those)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 200, param1 = 50, param2 = 22, minRadius = 50, maxRadius = 70)

    # ensure at least some circles were found, such falesafes (also for certain error types) should be build in in later versions
    if circles is None:
        print('no circles')
        return
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    canvas = circles[:, :2]

    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 4) # draw circle around detected corner
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) # draw rectangle at center of detected corner
        #canvas.append([x, y]) # store corner coordinates

    # cv2.imwrite('CornersDetected.jpg', img) # store the corner detection image
    """
    # storing corners for the perspective warp, disabled

    with open('circles_real.txt', 'w') as f:
        for item in circles:
            f.write("%s\n" % item)
    print('stored')
    """
    return canvas, thresh


def rotateGrid(canvas, img):
    # get index of one of the two highest corners, store it and delete from array
    lowest_y = int(np.argmin(canvas, axis=0)[1:])
    top_corner1 = canvas[lowest_y]
    x1 = top_corner1[0]
    canvas = np.delete(canvas, (lowest_y), axis=0)

    # get index of the second top corner, store it and delete from array
    lowest_y = int(np.argmin(canvas, axis=0)[1:])
    top_corner2 = canvas[lowest_y]
    x2 = top_corner2[0]
    canvas = np.delete(canvas, (lowest_y), axis=0)

    # store the two bottom corners
    bottom_corner1 = canvas[0]
    x3 = bottom_corner1[0]
    bottom_corner2 = canvas[1]
    x4 = bottom_corner2[0]

    # sort corners along top left, top right, bottom left, bottom right
    if x1 > x2:
        top_left = top_corner2
        top_right = top_corner1
    else:
        top_left = top_corner1
        top_right = top_corner2
    if x3 > x4:
        bottom_left = bottom_corner2
        bottom_right = bottom_corner1
    else:
        bottom_left = bottom_corner1
        bottom_right = bottom_corner2

    # match image points to new corner points according to known ratio
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

    # this value needs changing according to image size
    img_y = 3000  # warped image height
    ratio = 1.3861874976470018770202169598726  # height/width ratio given current grid
    img_x = int(round(img_y * ratio))  # warped image width
    pts2 = np.float32([[0, 0],[img_x, 0],[img_x, img_y],[0, img_y]])  # size for warped image
    perspective = cv2.getPerspectiveTransform(pts1, pts2)  # get perspective to warp image

    # warp image according to the perspective transform and store image
    # warped = cv2.warpPerspective(img, perspective, (img_x, img_y))
    # cv2.imwrite('warpedGrid.jpg', warped)
    origins, radius, features = calcGrid(img_y, img_x)
    return perspective, img_x, img_y, origins, radius, pts1, features


def calcGrid(height, width):
    # determine size of grid circles from image and step size in x direction
    radius = int(round(height / 10))
    x_step = np.cos(np.deg2rad(30)) * radius
    # print(x_step)
    origins = []

    # determine x and y coordinates of gridcells midpoints
    for a in range(1, 16):  # range reflects gridsize in x direction
        x = int(round(x_step * a))
        for b in range(1, 11):  # range reflects gridsize in y direction
            if a % 2 == 0:
                if b == 10:
                    continue
                y = int(round(radius * b))
            else:
                y = int(round(radius * (b - 0.5)))
            origins.append([x, y])
    features = createFeatures(origins, radius)
    return np.array(origins), radius, features


def createFeatures(origins, radius):
    radius = radius/2
    dist = radius/np.cos(np.deg2rad(30))
    x_jump = dist/2
    y_jump = radius
    features = []

    for i, (x, y) in enumerate(origins):
        point1 = [x+dist, y]
        point2 = [x+x_jump, y+y_jump]
        point3 = [x-x_jump, y+y_jump]
        point4 = [x-dist, y]
        point5 = [x-x_jump, y-y_jump]
        point6 = [x+x_jump, y-y_jump]
        polygon = geojson.Polygon([[point1, point2, point3, point4, point5, point6, point1]])
        feature = geojson.Feature(id=i, geometry=polygon)
        features.append(feature)  # features['polygon_%d'%d] = [point1, point2, point3, point4, point5, point6]
    return features  


"""
# this function draws the grid as calculated by calcGrid function. Not called in the script

def drawMask(origins, img):
    global count
    global radius
    r = int(round(radius / 2))
    for (x, y, count) in origins:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        #cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.putText(img, str(count), (x - 50, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)   
    # save image with grid
    cv2.imwrite('drawGrid.jpg', img)
    print('success')
    return  
"""