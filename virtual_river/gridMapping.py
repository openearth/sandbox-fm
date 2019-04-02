# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:07:11 2019

@author: HaanRJ
"""

import time
import json
import cv2
import geojson
import numpy as np
import netCDF4
import bmi.wrapper
from scipy.spatial import cKDTree
from shapely import geometry
from shapely.ops import unary_union


def read_calibration():
    """
    function that loads and returns the calibration. Currently not called in
    the control script as calibration transforms are stored internally.
    """
    with open('calibration.json') as f:
        calibration = json.load(f)
    # convert to transform matrix
    sandbox_transform = cv2.getPerspectiveTransform(
            np.array(calibration['img_post_cut_points'], dtype='float32'),
            np.array(calibration['model_points'], dtype='float32')
    )
    tygron_transform = cv2.getPerspectiveTransform(
            np.array(calibration['img_post_cut_points'], dtype='float32'),
            np.array(calibration['tygron_export'], dtype='float32')
    )
    calibration['image_post_cut2model'] = sandbox_transform
    calibration['image_post_cut2tygron'] = tygron_transform
    return calibration


def read_hexagons():
    """
    function that loads and returns the hexagons. Currently not called in
    the control script as the hexagons are stored internally.
    """
    with open('hexagons_sandbox_transformed.geojson') as f:
        features = geojson.load(f)
    return features


def read_grid():
    """
    function that loads and returns the grid.
    """
    loc = r"D:\Werkzaamheden map\Onderzoek\Design 2018\Models\300x200_2_net.nc"
    ds = netCDF4.Dataset(loc)
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
    return feature_collection


def hex_to_points(hexagons, grid, changed_hex=None, start=False, turn=0):
    """
    Interpolation function when starting (update entire board) and during a
    session (update selection only). Updates grid

    Requires:
        - hexagons: all hexagons (current state)
        - grid: (previous state)

    Optional:
        - changed_hex (hexagons with changed "z" values compared to previous
          state, needed if start=False)
        - start (differentiate between starting and updating, default=False)
        - turn (used to store files under the correct turn number, default = 0)
    """
    if start:
        # create a cKDTree for all the hexagons.
        hex_coor = []
        polygons = []
        for feature in hexagons.features:
            shape = geometry.asShape(feature.geometry)
            x_hex = shape.centroid.x
            y_hex = shape.centroid.y
            hex_coor.append([x_hex, y_hex])
            polygons.append(shape)
        hex_coor = np.array(hex_coor)
        hex_locations = cKDTree(hex_coor)
        multipolygon = geometry.MultiPolygon(polygons)
        board_as_polygon = unary_union(multipolygon)
        board_shapely = geometry.mapping(board_as_polygon)
        board_feature = geojson.Feature(geometry=board_shapely)
        board_featurecollection = geojson.FeatureCollection([board_feature])
        with open('board_border.geojson', 'w') as f:
            geojson.dump(board_featurecollection, f, sort_keys=True, indent=2)
        line = list(geojson.utils.coords(board_feature))

        # determine the bounding box coordinates of the board.
        minx = 0.0
        miny = 0.0
        maxx = 0.0
        maxy = 0.0
        for x, y in line:
            if x < minx:
                minx = x
            elif x > maxx:
                maxx = x
            else:
                continue
            if y < miny:
                miny = y
            elif y > maxy:
                maxy = y
            else:
                continue
        bbox = geometry.Polygon([(minx, maxy), (maxx, maxy), (maxx, miny),
                                 (minx, miny), (minx, maxy)])

        # determine whether or not a point in the grid falls inside or outside.
        # of the board bbox.
        inside_id = []
        inside_coor = []
        for feature in grid.features:
            point = geometry.asShape(feature.geometry)
            if bbox.contains(point):
                feature.properties["board"] = True
                feature.properties["changed"] = True
                inside_id.append(feature.id)
                x_point = point.centroid.x
                y_point = point.centroid.y
                inside_coor.append([x_point, y_point])
            else:
                feature.properties["board"] = False
                feature.properties["changed"] = False

        # create a cKDTree of all the points that fall within the board bbox.
        inside_coor = np.array(inside_coor)
        inside_locations = cKDTree(inside_coor)

        # index the all the grid points to either up to the nearest three
        # hexagons in case the grid point falls within the board bbox.
        # Weighting factors are stored in case two or three hexagons are
        # indexed that can be accessed in case one or more of these hexagons
        # change after a board state update. Otherwise index the grid point to
        # the closest neighbour that falls within the board bbox.
        for feature in grid.features:
            shape = geometry.asShape(feature.geometry)
            x_hex = shape.centroid.x
            y_hex = shape.centroid.y
            xy = np.array([x_hex, y_hex])

            # the block of code below specifies the rules on how many hexagons
            # are indixed for each point that is positioned within the game
            # board, based on the location of each point and the distance to
            # the nearest three hexagons.
            if feature.properties["board"]:
                dist, indices = hex_locations.query(xy, k=3)
                if dist[0] > 35:
                    if dist[1] <= 60:
                        if dist[2] <= 60:
                            weights = 1 / np.power(dist, 2)
                            weights_sum = sum(weights)
                            feature.properties["nearest"] = indices.tolist()
                            feature.properties["weight"] = weights.tolist()
                            feature.properties["weight_sum"] = weights_sum
                        else:
                            weights = 1 / np.power(dist[0:2], 2)
                            weights_sum = sum(weights)
                            feature.properties["nearest"] = indices[0:2].tolist()
                            feature.properties["weight"] = weights.tolist()
                            feature.properties["weight_sum"] = weights_sum
                    else:
                        feature.properties["nearest"] = indices[0].tolist()
                elif dist[1] > 45:
                    feature.properties["nearest"] = indices[0].tolist()
                elif dist[2] > 45:
                    weights = 1 / np.power(dist[0:2], 2)
                    weights_sum = sum(weights)
                    feature.properties["nearest"] = indices[0:2].tolist()
                    feature.properties["weight"] = weights.tolist()
                    feature.properties["weight_sum"] = weights_sum
                else:
                    weights = 1 / np.power(dist, 2)
                    weights_sum = sum(weights)
                    feature.properties["nearest"] = indices.tolist()
                    feature.properties["weight"] = weights.tolist()
                    feature.properties["weight_sum"] = weights_sum
            # if the grid point is not within the board bbox, index to the
            # nearest grid point that is on located within the board bbox.
            else:
                dist, indices = inside_locations.query(xy)
                feature.properties["nearest"] = inside_id[indices]

                # change this section to finding the nearest neighbour on the
                # horizontal axis + another rule if no nearest neighbour on the
                # horizontal axis
    else:
        # In case the method is called as an update, determine which grid
        # points require updating based on which hexagons are changed. This
        # way, only the grid points that need updating are updated, speeding
        # up the updating process.
        indices_updated = []
        counter = 0
        for feature in changed_hex.features:
            indices_updated.append(feature.id)
        for feature in grid.features:
            if feature.properties["board"]:
                if type(feature.properties["nearest"]) is int:
                    if feature.properties["nearest"] in indices_updated:
                        feature.properties["changed"] = True
                        counter += 1
                    else:
                        feature.properties["changed"] = False
                elif any((True for x in feature.properties["nearest"]
                          if x in indices_updated)):
                    feature.properties["changed"] = True
                    counter += 1
                else:
                    feature.properties["changed"] = False
            else:
                continue
        print("Hexagons updated are: "+str(indices_updated))
        print("Number of gridpoints inside the board to update: "+str(counter))

    # block of code that calculates the z variable for each grid point, based
    # on stored indices and, if applicable, weight factors. Distinguishes
    # between start (updates all as all are changed) and update (updates only
    # points connected to changed hexagons).
    hexagons_by_id = {feature.id: feature for feature in hexagons.features}
    for feature in grid.features:
        if feature.properties["board"]:
            if feature.properties["changed"]:
                nearest = feature.properties["nearest"]
                if type(nearest) is int:
                    hexagon = hexagons_by_id[nearest]
                    feature.properties['z'] = hexagon.properties['z']
                else:
                    if len(nearest) == 2:
                        weights = feature.properties["weight"]
                        weights_sum = feature.properties["weight_sum"]
                        hexagon1 = hexagons_by_id[nearest[0]]
                        hexagon2 = hexagons_by_id[nearest[1]]
                        feature.properties['z'] = \
                            round(hexagon1.properties['z'] * (weights[0] /
                                  weights_sum) + hexagon2.properties['z'] *
                                  (weights[1] / weights_sum), 5)
                    else:
                        weights = feature.properties["weight"]
                        weights_sum = feature.properties["weight_sum"]
                        hexagon1 = hexagons_by_id[nearest[0]]
                        hexagon2 = hexagons_by_id[nearest[1]]
                        hexagon3 = hexagons_by_id[nearest[2]]
                        feature.properties['z'] = \
                            round(hexagon1.properties['z'] * (weights[0] /
                                  weights_sum) + hexagon2.properties['z'] *
                                  (weights[1] / weights_sum) +
                                  hexagon3.properties['z'] * (weights[2] /
                                  weights_sum), 5)
            else:
                continue
        else:
            continue

    # block of code that sets the z variable for each grid point outside of the
    # game board by setting the z value equal to the z value of the nearest
    # grid point on the board.
    grid_by_id = {feature.id: feature for feature in grid.features}
    counter = 0
    for feature in grid.features:
        if not feature.properties["board"]:
            nearest = feature.properties["nearest"]
            inside_point = grid_by_id[nearest]
            if inside_point.properties["changed"]:
                feature.properties['z'] = inside_point.properties['z']
                counter += 1
            else:
                continue
        else:
            continue
    if not start:
        print("Number of gridpoints outside the board updated: "+str(counter))
    tec = time.time()
    filename = 'grid_with_z_triangulate_%d.geojson' % turn
    with open(filename, 'w') as f:
        geojson.dump(grid, f, sort_keys=True, indent=2)
    return grid, tec


def create_geotiff(grid):
    """
    Function that creates a GeoTIFF from the grid as constructed in the
    hex_to_points function
    """
    columns = []
    rows = []
    z = []
    for feature in grid.features:
        if feature.properties["board"]:
            shape = geometry.asShape(feature.geometry)
            x_point = shape.centroid.x
            y_point = shape.centroid.y
            if x_point not in columns:
                columns.append(x_point)
            if y_point not in rows:
                rows.append(y_point)
            z.append(feature.properties["z"])
        else:
            continue
    z = np.array(z)
    z = np.reshape(z, (len(rows),len(columns)))
    #z = np.flip(z, 0)
    #z = z.T
    with open('z_array_test.txt', 'w') as f:
        for item in z:
            f.write("%s\n" % item)
    print(columns)
    print(rows)
    #print(len(columns), len(rows))
    return


if __name__ == "__main__":
    tic = time.time()
    calibration = read_calibration()
    hexagons = read_hexagons()
    grid = read_grid()
    tac = time.time()
    grid_triangulate = hex_to_points(hexagons, grid, start=True)
    #create_geotiff(grid_triangulate)
    model = bmi.wrapper.BMIWrapper('dflowfm')
    model.initialize(r'C:\Users\HaanRJ\Documents\GitHub\sandbox-fm\models\sandbox\Waal_schematic\waal_with_side.mdu')
    print('model initialized')
    toc = time.time()
    print('loading time:', tac-tic)
    print('interpolation time:', toc-tac)
