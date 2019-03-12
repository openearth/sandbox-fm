# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:49:15 2019

@author: haanrj
"""

import json
import time
import cv2
import geojson
import numpy as np
import shapely.geometry
import gridCalibration as cali
import sandbox_fm.calibrate
from sandbox_fm.calibration_wizard import NumpyEncoder


def detectMarkers(file, pers, img_x, img_y, origins, r, features, method = 'rgb'):
    # load and process the new image (game state)
    img = cv2.imread(file)
    warped = cv2.warpPerspective(img, pers, (img_x, img_y))  # warp image it to calibrated perspective
    blur = cv2.GaussianBlur(warped, (25, 25), 0)  # blur image
    if method == 'hsv':
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # convert image to HSV
        # cv2.imwrite('hsv.jpg', hsv)  # save HSV file to show status, unnecessary for script to work
    
        # set ranges for blue and red to create masks, may need updating
        lower_blue = (100, 0, 0)  # lower bound for each channel
        upper_blue = (255, 120, 175)  # upper bound for each channel
        lower_red = (0, 100, 100)  # lower bound for each channel
        upper_red = (100, 255, 255)  # upper bound for each channel
    
        # create the mask and use it to change the colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
    elif method == 'lab':
        print('no lab method written')
    else:
        # set ranges for blue and red to create masks, may need updating
        lower_blue = 240
        upper_blue = 255
        lower_red = 240
        upper_red = 255
        
        # create the mask and use it to change the colors
        blue_mask = cv2.inRange(blur[:,:,0], lower_blue, upper_blue)
        red_mask = cv2.inRange(blur[:,:,2], lower_red, upper_red)

    # dilate resulting images to remove minor contours
    kernel = np.ones((10,10), np.uint8)  # kernel for dilate function
    red_dilate = cv2.dilate(red_mask, kernel, iterations = 1)  # dilate red
    blue_dilate = cv2.dilate(blue_mask, kernel, iterations = 1)  # dilate blue
    cv2.imwrite('red_mask_dilated.jpg', red_dilate)  # save dilated red, unnecessary for script to work
    cv2.imwrite('blue_mask_dilated.jpg', blue_dilate)  # save dilated blue, unnecessary for script to work

    # process input variables
    r = int(round(r / 2))  # convert diameter to actual radius as int value
    safe_r = int(round(r*0.95))  # slightly lower radius to create a mask that removes contours from slight grid misplacement
    mask = np.zeros((2*r, 2*r), dtype = "uint8")  # empty mask of grid cell size
    cv2.circle(mask, (r, r), safe_r, (255, 255, 255), -1)  # circle for mask, centered with safe radius

    # for loop that analyzes all grid cells
    for i, feature in enumerate(features):
        geometry = shapely.geometry.asShape(feature.geometry)
        x = int(geometry.centroid.x)
        y = int(geometry.centroid.y)
        # get region of interest (ROI) for red (geometry) and analyse number of contours (which identifies the markers) found
        roiGeo = red_dilate[y-r:y+r, x-r:x+r]  # region on interest for cell number d
        maskedImgGeo = cv2.bitwise_and(roiGeo, roiGeo, mask = mask)  # mask the ROI to eliminate adjacent grid cells
        im1, contoursGeo, hierarchy1 = cv2.findContours(maskedImgGeo,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)  # find contours within masked ROI
        # cv2.drawContours(maskedImgGeo, contoursGeo, -1, (130,130,130), 3)  # draw contours, not necessary for script to work

        # same as above for blue (ecotopes) as above
        roiEco = blue_dilate[y-r:y+r, x-r:x+r]
        maskedImgEco = cv2.bitwise_and(roiEco, roiEco, mask=mask)
        im2, contoursEco, hierarchy2 = cv2.findContours(maskedImgEco, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(maskedImgEco, contoursEco, -1, (130, 130, 130), 3)

        # store each analyzed ROI, not necessary for the script to work
        # filenameGeo = 'geo_roi_%d.jpg'%d
        # filenameEco = 'eco_roi_%d.jpg'%d
        # cv2.imwrite(filenameGeo, maskedImgGeo)
        # cv2.imwrite(filenameEco, maskedImgEco)
        feature.properties["z"] = len(contoursGeo)
        feature.properties["landuse"] = len(contoursEco)
    with open('hexagons.geojson', 'w') as f:
        geojson.dump(geojson.FeatureCollection(features), f, sort_keys=True, indent=2)
        
    transformed_features = []
    for feature in features:
        
        pts = np.array(feature.geometry['coordinates'][0], dtype='float32')
        # points should be channels
        # pts = np.c_[pts, np.zeros_like(pts[:, 0])]
        
        transform = transforms['img_post_cut2model']
        x, y = pts[:, 0], pts[:, 1]
        x_t, y_t = sandbox_fm.calibrate.transform(x, y, transform)
        xy_t = np.c_[x_t, y_t]
        new_feature = geojson.Feature(id=feature.id, geometry=geojson.Polygon([xy_t.tolist()]), properties=feature.properties)
        transformed_features.append(new_feature)
    transformed_features = geojson.FeatureCollection(transformed_features)
    
    with open('hexagons_features_transformed.geojson', 'w') as f:
        geojson.dump(transformed_features, f, sort_keys=True, indent=2)

if __name__ == '__main__':
    tic = time.time()  # start performance timer
    filename = 'prototype_new2.jpg'
    canvas, thresh = cali.detectCorners(filename, method = 'adaptive')  # image name for calibration (would be first image pre-session)
    pers, img_x, img_y, origins, radius, cut_points, features = cali.rotateGrid(canvas, thresh)  # store calibration values as global variables
    calibration = {}
    calibration['model_points'] = [-400, 300 ], [400, 300], [400, -300], [-400, -300]  # model points following SandBox implementation; between [-600, -400] and [600, 400] 
    calibration['img_points'] = [0, 0], [1920, 0], [1920, 1080], [0, 1080]  # resolution camera; FullHD
    calibration['img_pre_cut_points'] = cut_points.tolist()  # calibration points used to cut images
    calibration['img_post_cut_points'] = [0, 0], [img_x, 0], [img_x, img_y],  [0, img_y]  # corners of image after image cut
    calibration['z'] = [0.0, 9.0]  # height range
    calibration['z_values'] = [0, 5]  # height of game pieces; may be subject to change after interpolation
    calibration['box'] = [0, 0], [640, 0], [640, 480], [0, 480] # box == beamer
    transforms = sandbox_fm.calibrate.compute_transforms(calibration)
    calibration.update(transforms)

    with open('calibration.json', 'w') as f:
        json.dump(calibration, f, sort_keys=True, indent=2, cls=NumpyEncoder)
    with open('hexagons_features.json', 'w') as g:
        json.dump(features, g, sort_keys=True, indent=2)
    tac = time.time()  # calibration performance timer
    detectMarkers(filename, pers, img_x, img_y, origins, radius, features)  # initiate the image processing function
    toc = time.time()  # end performance timer
    print('calibration time:', tac-tic)  # print calibration performance time
    print('image processing time:', toc-tac)  # print image processing performance time
