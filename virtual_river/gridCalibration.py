# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:49:34 2019

@author: haanrj
"""


import json
import cv2
import numpy as np
import geojson
import sandbox_fm.calibrate
from sandbox_fm.calibration_wizard import NumpyEncoder


def create_calibration_file(img_x, img_y, cut_points):
    """
    Function that creates the calibration file (json format) and returns the
    transforms that can be used by other functions.
    """
    calibration = {}
    # model points following SandBox implementation; between [-600, -400] and [600, 400] 
    calibration['model_points'] = ([-400, 300 ], [400, 300], [400, -300], [-400, -300])
    # resolution camera; FullHD
    calibration['img_points'] = [0, 0], [1920, 0], [1920, 1080], [0, 1080]
    # calibration points used to cut images
    calibration['img_pre_cut_points'] = cut_points.tolist()
    # corners of image after image cut
    calibration['img_post_cut_points'] = [0, 0], [img_x, 0], [img_x, img_y],  [0, img_y]
    # tygron project creation; empty world coordinates
    calibration['tygron_export'] = [0, 0], [1000, 0], [1000, -750],  [0, -750]
    # tygron project update; world coordinates once created
    calibration['tygron_update'] = [0, 0], [1000, 0], [1000, 750],  [0, 750]
    # height range
    calibration['z'] = [0.0, 9.0]
    # height of game pieces; may be subject to change after interpolation
    calibration['z_values'] = [0, 5]
    # box == beamer
    calibration['box'] = [0, 0], [640, 0], [640, 480], [0, 480]
    transforms = sandbox_fm.calibrate.compute_transforms(calibration)
    calibration.update(transforms)
    with open('calibration.json', 'w') as f:
        json.dump(calibration, f, sort_keys=True, indent=2, cls=NumpyEncoder)
    return transforms


def detect_corners(filename, method='standard'):
    """
    Function that detects the corners of the board (the four white circles)
    and returns their coordinates as a 2D array.
    """
    img = cv2.imread(filename)  # load image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
    if method == 'adaptive':
        blur = cv2.medianBlur(gray, 5)  # flur grayscale image
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite('Adaptive_threshold.jpg', thresh)  # store threshold image
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # flur grayscale image
        # threshold grayscale image as binary
        ret, thresh = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('Standard_threshold.jpg', thresh)  # store threshold image

    # detect corner circles in the image (min/max radius ensures only
    # finding those)
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 200, param1=50,
                               param2=22, minRadius=50, maxRadius=70)

    # ensure at least some circles were found, such falesafes (also for certain
    # error types) should be build in in later versions
    if circles is None:
        print('no circles')
        return
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    canvas = circles[:, :2]

    for (x, y, r) in circles:
        # draw circle around detected corner
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        # draw rectangle at center of detected corner
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imwrite('CornersDetected.jpg', img) # store the corner detection image
    return canvas, thresh


def rotate_grid(canvas, img):
    """
    Function that sorts the four corners in the right order (top left, top
    right, bottom right, bottom left) and returns the perspective transform
    to be used throughout the session.
    """
    # get index of one of the two top corners, store it and delete from array
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
    # height/width ratio given current grid
    ratio = 1.3861874976470018770202169598726
    img_x = int(round(img_y * ratio))  # warped image width
    # size for warped image
    pts2 = np.float32([[0, 0],[img_x, 0],[img_x, img_y],[0, img_y]])
    # get perspective to warp image
    perspective = cv2.getPerspectiveTransform(pts1, pts2)

    # warp image according to the perspective transform and store image
    # warped = cv2.warpPerspective(img, perspective, (img_x, img_y))
    # cv2.imwrite('warpedGrid.jpg', warped)
    #origins, radius = calc_grid(img_y, img_x)
    features, origins, radius = create_features(img_y, img_x)
    #origins = np.array(origins)
    return perspective, img_x, img_y, origins, radius, pts1, features


def create_features(height, width):
    """
    Function that calculates the midpoint coordinates of each hexagon in the
    transformed picture.
    """
    # determine size of grid circles from image and step size in x direction
    radius = (height / 10)
    x_step = np.cos(np.deg2rad(30)) * radius
    origins = []
    column = []
    # determine x and y coordinates of gridcells midpoints
    for a in range(1, 16):  # range reflects gridsize in x direction
        x = (x_step * a)
        for b in range(1, 11):  # range reflects gridsize in y direction
            if a % 2 == 0:
                if b == 10:
                    continue
                y = (radius * b)
            else:
                y = (radius * (b - 0.5))
            origins.append([x, y])
            column.append(a)
    origins = np.array(origins)
    dist = (radius/2)/np.cos(np.deg2rad(30))
    x_jump = dist/2
    y_jump = radius/2
    features = []
    for i, (x, y) in enumerate(origins):
        point1 = [x+dist, y]
        point2 = [x+x_jump, y+y_jump]
        point3 = [x-x_jump, y+y_jump]
        point4 = [x-dist, y]
        point5 = [x-x_jump, y-y_jump]
        point6 = [x+x_jump, y-y_jump]
        polygon = geojson.Polygon([[point1, point2, point3, point4, point5,
                                    point6, point1]])
        feature = geojson.Feature(id=i, geometry=polygon, properties={"column": column[i]})
        #feature.properties["column"] = column[i]
        if i == 1:
            print(feature)
        features.append(feature)
    return features, origins, radius


"""
def create_features(origins, radius):
    # Function that creates Polygon features (hexagon shaped) for all hexagons.
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
        polygon = geojson.Polygon([[point1, point2, point3, point4, point5,
                                    point6, point1]])
        feature = geojson.Feature(id=i, geometry=polygon)
        features.append(feature)
    return features
"""


def drawMask(origins, img):
    """
    Function that can be called to draw the mask and print hexagon numbers.
    This function is currently not called. Can be removed at a later stage.
    """
    global count
    global radius
    r = int(round(radius / 2))
    for (x, y, count) in origins:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        #cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.putText(img, str(count), (x - 50, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)
    # save image with grid
    cv2.imwrite('drawGrid.jpg', img)
    print('success')
    return
