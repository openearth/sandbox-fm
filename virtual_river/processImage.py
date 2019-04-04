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
import gridCalibration as cali
import sandbox_fm.calibrate
from shapely.geometry import asShape


def detect_markers(file, pers, img_x, img_y, origins, r, features,
                   method='rgb'):
    # load and process the new image (game state)
    img = cv2.imread(file)
    # warp image it to calibrated perspective
    warped = cv2.warpPerspective(img, pers, (img_x, img_y))
    # blur image
    blur = cv2.GaussianBlur(warped, (25, 25), 0)
    if method == 'hsv':
        # convert image to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # save HSV file to show status, unnecessary for script to work
        #cv2.imwrite('hsv.jpg', hsv)  
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
        blue_mask = cv2.inRange(blur[:, :, 0], lower_blue, upper_blue)
        red_mask = cv2.inRange(blur[:, :, 2], lower_red, upper_red)

    # dilate resulting images to remove minor contours
    kernel = np.ones((10, 10), np.uint8)  # kernel for dilate function
    red_dilate = cv2.dilate(red_mask, kernel, iterations=1)  # dilate red
    blue_dilate = cv2.dilate(blue_mask, kernel, iterations=1)  # dilate blue
    # save dilated red and blue, unnecessary for script to work
    cv2.imwrite('red_mask_dilated.jpg', red_dilate)
    cv2.imwrite('blue_mask_dilated.jpg', blue_dilate)

    # convert diameter to actual radius as int value
    r = int(round(r / 2))
    # slightly lower radius to create a mask that removes contours from
    # slight grid misplacement
    safe_r = int(round(r*0.95))
    # empty mask of grid cell size
    mask = np.zeros((2*r, 2*r), dtype="uint8")
    # circle for mask, centered with safe radius
    cv2.circle(mask, (r, r), safe_r, (255, 255, 255), -1)

    # for loop that analyzes all grid cells
    for i, feature in enumerate(features):
        geometry = asShape(feature.geometry)
        x = int(geometry.centroid.x)
        y = int(geometry.centroid.y)
        # get region of interest (ROI) for red (geometry) and analyse number
        # of contours (which identifies the markers) found
        roiGeo = red_dilate[y-r:y+r, x-r:x+r]
        # mask the ROI to eliminate adjacent grid cells
        maskedImgGeo = cv2.bitwise_and(roiGeo, roiGeo, mask = mask)
        # find contours within masked ROI
        im1, contoursGeo, hierarchy1 = cv2.findContours(maskedImgGeo,
                                                        cv2.RETR_CCOMP,
                                                        cv2.CHAIN_APPROX_SIMPLE)
        # draw contours, not necessary for script to work
        # cv2.drawContours(maskedImgGeo, contoursGeo, -1, (130,130,130), 3)

        # same as above for blue (ecotopes) as above
        roiEco = blue_dilate[y-r:y+r, x-r:x+r]
        maskedImgEco = cv2.bitwise_and(roiEco, roiEco, mask=mask)
        im2, contoursEco, hierarchy2 = cv2.findContours(maskedImgEco,
                                                        cv2.RETR_CCOMP,
                                                        cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(maskedImgEco, contoursEco, -1, (130, 130, 130), 3)

        # store each analyzed ROI, not necessary for the script to work
        # filenameGeo = 'geo_roi_%d.jpg'%d
        # filenameEco = 'eco_roi_%d.jpg'%d
        # cv2.imwrite(filenameGeo, maskedImgGeo)
        # cv2.imwrite(filenameEco, maskedImgEco)
        feature.properties["id"] = i
        feature.properties["tygron_id"] = i
        feature.properties["z"] = len(contoursGeo)
        feature.properties["landuse"] = len(contoursEco)
    with open('hexagons.geojson', 'w') as f:
        geojson.dump(geojson.FeatureCollection(features), f, sort_keys=True,
                     indent=2)
    return geojson.FeatureCollection(features)


def transform(features, transforms, export=None):
    """
    Function that transforms geojson files to new coordinates based on where
    the geojson needs to be transformed to (e.g. from the image processed to
    the model: 'img_post_cut2model').
    """
    transformed_features = []
    waterbodies = []
    landbodies = []
    if export == "sandbox":
        transform = transforms['img_post_cut2model']
    elif export == "tygron_initialize":
        transform = transforms['img_post_cut2tygron_export']
    elif export == "tygron":
        transform = transforms['img_post_cut2tygron_update']
    else:
        print("unknown export method, current supported are: 'sandbox', "
              "'tygron' & 'tygron_initialize'")
        return features
    for feature in features.features:
        pts = np.array(feature.geometry["coordinates"][0], dtype="float32")
        # points should be channels
        # pts = np.c_[pts, np.zeros_like(pts[:, 0])]
        x, y = pts[:, 0], pts[:, 1]
        x_t, y_t = sandbox_fm.calibrate.transform(x, y, transform)
        xy_t = np.c_[x_t, y_t]
        new_feature = geojson.Feature(id=feature.id,
                                      geometry=geojson.Polygon([xy_t.tolist()]),
                                      properties=feature.properties)
        if export == "tygron":
            if feature.properties["z"] < 2:
                waterbodies.append(new_feature)
            else:
                landbodies.append(new_feature)
        transformed_features.append(new_feature)
    if export == "sandbox":
        transformed_features = geojson.FeatureCollection(transformed_features)
        with open('hexagons_sandbox_transformed.geojson', 'w') as f:
            geojson.dump(transformed_features, f, sort_keys=True, indent=2)
        return transformed_features
    elif export == "tygron_initialize":
        crs = {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::3857"
                }
            }
        transformed_features = geojson.FeatureCollection(transformed_features,
                                                         crs=crs)
        with open('hexagons_tygron_transformed.geojson', 'w') as f:
            geojson.dump(transformed_features, f, sort_keys=True, indent=2)
        return transformed_features
    else:
        transformed_features = geojson.FeatureCollection(transformed_features)
        with open('hexagons_tygron_update_transformed.geojson', 'w') as f:
            geojson.dump(transformed_features, f, sort_keys=True, indent=2)
        waterbodies = geojson.FeatureCollection(waterbodies)
        with open('waterbodies_tygron_transformed.geojson', 'w') as f:
            geojson.dump(waterbodies, f, sort_keys=True, indent=2)
        landbodies = geojson.FeatureCollection(landbodies)
        with open('landbodies_tygron_transformed.geojson', 'w') as f:
            geojson.dump(landbodies, f, sort_keys=True, indent=2)
        return waterbodies, landbodies


if __name__ == '__main__':
    tic = time.time()  # start performance timer
    filename = 'board_image0.jpg'
    # image name for calibration (would be first image pre-session)
    canvas, thresh = cali.detect_corners(filename, method='adaptive')
    # store calibration values as global variables
    pers, img_x, img_y, origins, radius, cut_points, \
        features = cali.rotate_grid(canvas, thresh)
    transforms = cali.create_calibration_file(img_x, img_y, cut_points)

    with open('hexagons_features.json', 'w') as g:
        json.dump(features, g, sort_keys=True, indent=2)
    tac = time.time()  # calibration performance timer
    # initiate the image processing function
    hexagon_current = detect_markers(filename, pers, img_x, img_y, origins,
                                     radius, features, transforms)
    toc = time.time()  # end performance timer
    # print calibration and image processing performance time
    print('calibration time:', tac-tic)
    print('image processing time:', toc-tac)
