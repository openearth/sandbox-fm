# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:07:11 2019

@author: HaanRJ
"""
import json
import cv2
import geojson
import numpy as np
from scipy import interpolate
import rtree
import shapely.geometry
import netCDF4
import bmi.wrapper


def read_calibration():
    with open('calibration.json') as f:
        calibration = json.load(f)
    # convert to transform matrix
    transform = cv2.getPerspectiveTransform(
            np.array(calibration['img_post_cut_points'], dtype='float32'),
            np.array(calibration['model_points'], dtype='float32')
    )
    calibration['image_post_cut2model'] = transform
    return calibration


def read_hexagons():
    with open('hexagons_features_transformed.geojson') as f:
        features = geojson.load(f)
    return features


def read_grid():
    ds = netCDF4.Dataset(r'D:\Werkzaamheden map\Onderzoek\Design 2018\Models\300x200_2_net.nc')
    x = ds.variables['NetNode_x'][:]
    y = ds.variables['NetNode_y'][:]
    ds.close()

    xy = np.c_[x, y]
    features = []
    for i, xy_i in enumerate(xy):
        pt = geojson.Point(coordinates=list(xy_i))
        feature = geojson.Feature(geometry=pt, id=i)
        features.append(feature)
    feature_collection = geojson.FeatureCollection(features)
    with open('grid.geojson', 'w') as f:
        geojson.dump(feature_collection, f, sort_keys=True, indent=2)
    return feature_collection, x, y


def hex_to_points(hexagons, grid, xi, yi, method='nearest'):
    idx = rtree.index.Index()
    for feature in hexagons.features:
        shape = shapely.geometry.asShape(feature.geometry)
        idx.insert(feature.id, shape.bounds, obj=feature.properties)

    model2hex = {}
    hexagons_by_id = {feature.id: feature for feature in hexagons.features}

    for feature in grid.features:
        # find nearest cell
        # store in feature
        shape = shapely.geometry.asShape(feature.geometry)
        nearest = next(idx.nearest(shape.bounds, objects=True))
        model2hex[feature.id] = nearest.id

    for feature in grid.features:
        hexagon = hexagons_by_id[model2hex[feature.id]]
        feature.properties['z'] = hexagon.properties['z']

    if method == 'griddata':
        x = []
        y = []
        z = []
        for feature in hexagons['features']:
            shape = shapely.geometry.asShape(feature.geometry)
            x_hex = int(shape.centroid.x)
            y_hex = int(shape.centroid.y)
            z_hex = feature.properties["z"]
            x.append(x_hex)
            y.append(y_hex)
            z.append(z_hex)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        z_interpolate = interpolate.griddata((x, y), z, (xi, yi), method='linear')
        #z_extrapolate = interpolate.griddata((xi,yi),z_interpolate,(xi,yi),method='nearest')
        #z_nearest = interpolate.griddata((x,y),z,(xi,yi),method='linear')

        for i, feature in enumerate(grid.features):
            if np.isnan(z_interpolate[i]):
                continue
            else:
                feature.properties['z'] = z_interpolate[i]

    with open('grid_with_z.geojson', 'w') as f:
        geojson.dump(grid, f, sort_keys=True, indent=2)

    """
    for feature in grid.features:
        # find nearest cell
        # store in feature
        shape = shapely.geometry.asShape(feature.geometry)
        nearest = next(idx.nearest(shape.bounds, objects=True))
        # nearest = next(idx.nearest(shape.bounds, objects=True))
        # model2hex[feature.id] = nearest.id
        # print('no interpolation method developed yet')
    else:
        for feature in grid.features:
            # find nearest cell
            # store in feature
            shape = shapely.geometry.asShape(feature.geometry)
            nearest = next(idx.nearest(shape.bounds, objects=True))
            model2hex[feature.id] = nearest.id
        """
    """
    for feature in grid.features:
        hexagon = hexagons_by_id[model2hex[feature.id]]
        feature.properties['z'] = hexagon.properties['z']

    with open('grid_with_z.geojson', 'w') as f:
        geojson.dump(grid, f, sort_keys=True, indent=2)
    """


if __name__ == "__main__":
    calibration = read_calibration()
    hexagons = read_hexagons()
    grid, xi, yi = read_grid()
    hex_to_points(hexagons, grid, xi, yi, method='griddata')
    model = bmi.wrapper.BMIWrapper('dflowfm')
    model.initialize(r'C:\Users\HaanRJ\Documents\GitHub\sandbox-fm\models\sandbox\Waal_schematic\waal_with_side.mdu')
    print('model initialized')
