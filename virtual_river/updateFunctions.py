# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:26:52 2019

@author: HaanRJ
"""

import geojson
import tygronInterface as tygron


def compare_hex(token, hexagons_old, hexagons_new):
    """
    compares the current state of each location to the previous state, should
    handle:
        - isolate changed hexagons
        - if hexagons_old >= 2 && hexagons_new < 2 --> update hexagon to water
        - if hexagons_old < 2 && hexagons_new >= 2 --> update hexagon to land
        - if hexagons_old.properties["landuse"] ||
          hexagons_new.properties["landuse"] --> update hexagon land use type
          (including Tygron functionID)
        - grid interpolation --> only update grid points with indices to the
          changed hexagons?
        - initiate cost module to calculate the costs of all changes
    """
    z_changed = []
    becomes_water = []
    becomes_land = []
    landuse_changed = []
    for feature in hexagons_new.features:
        reference_hex = hexagons_old[feature.id]
        if feature.properties["z"] is not reference_hex.properties["z"]:
            print("hexagon " + str(feature.id) + " z value changed")
            z_changed.append(feature)
            if (feature.properties["z"] < 2 and
                    reference_hex.properties["z"] >= 2):
                becomes_water.append(feature)
            elif (feature.properties["z"] >= 2 and
                    reference_hex.properties["z"] < 2):
                becomes_land.append(feature)
        if (feature.properties["landuse"] is not
                reference_hex.properties["landuse"]):
            print("hexagon " + str(feature.id) + " land use value changed")
            landuse_changed.append(feature)
    if becomes_water:
        waterbodies = geojson.FeatureCollection(becomes_water)
        tygron.set_terrain_type(token, waterbodies, terrain_type="water")
    if becomes_land:
        landbodies = geojson.FeatureCollection(becomes_land)
        tygron.set_terrain_type(token, landbodies, terrain_type="land")
    z_changed = geojson.FeatureCollection(z_changed)
    print("changed cells")
    return z_changed


if __name__ == '__main__':
    with open('hexagons_tygron_update_transformed_test2.geojson') as f:
        hexagons_old = geojson.load(f)
    with open('hexagons_tygron_update_transformed_test1.geojson') as g:
        hexagons_new = geojson.load(g)
    with open(r'C:\Users\HaanRJ\Documents\Storage\username.txt', 'r') as f:
        username = f.read()
    with open(r'C:\Users\HaanRJ\Documents\Storage\password.txt', 'r') as g:
        password = g.read()
    token = tygron.join_session(username, password)
    if token is None:
        print("logging in to Tygron failed, unable to make changes in Tygron")
    else:
        token = "token=" + token
        z = compare_hex(token, hexagons_old, hexagons_new)
