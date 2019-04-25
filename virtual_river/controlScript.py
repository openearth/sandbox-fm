# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:56:24 2019

@author: HaanRJ
"""

import time
import geojson
import keyboard
import tygronInterface as tygron
import gridCalibration as cali
import processImage as detect
import gridMapping as gridmap
import updateFunctions as compare


def main_menu():
    """
    Main script that continues to run during a session. Initiates the
    calibration and any updates thereafter. Stores all data while running
    """
    # boolean in order to only run initialize method once
    start = True
    # empty string to store Tygron api session token
    token = ""
    # turn tracker
    turn = 0
    # snapshot filename
    filename = 'board_image%d.jpg' % turn
    # current hexagon state on the board, transformed to sandbox, geojson
    # featurecollection (multipolygon)
    hex_sandbox = None
    # hex_sandbox state of previous update, geojson featurecollection
    # (multipolygon)
    hex_sandbox_prev = None
    # current hexagon state on the board, transformed to tygron (this may
    # need to be changed to another transform), geojson featurecollection
    # (multipolygon)
    hex_tygron = None
    # hex_tygron state of previous update, geojson featurecollection
    # (multipolygon)
    hex_tygron_prev = None
    # current hexagons on the board that are water (z < 2), transformed to
    # tygron, geojson featurecollection (multipolygon)
    hex_water = None
    # hex_water state of previous update, geojson featurecollection
    # (multipolygon)
    hex_water_prev = None
    # current hexagon state on the board, transformed to tygron (this may need
    # to be changed to), geojson featurecollection (multipolygon)
    hex_land = None
    # hex_land state of previous update, geojson featurecollection
    # (multipolygon)
    hex_land_prev = None
    # all necessary transforms
    transforms = None
    # sandbox grid, geojson featurecollection (points)
    grid = None
    # interpolated grid (with z), geojson featurecollection (points)
    grid_interpolated = None
    print("Virtual River started, press '5' to calibrate, "
          "press '1' to update and press '0' to quit")
    while(True):
        a = keyboard.read_key()
        if a == '5':
            if start:
                print("Calibrating Virtual River, preparing SandBox "
                      "and logging into Tygron")
                """
                to add:
                    - initiate camera
                    - take new picture
                    - send picture to initialize or store picture and send
                      correct filename
                """
                tic = time.time()
                #try:
                token, hex_sandbox, hex_tygron, hex_water, hex_land, \
                grid, grid_interpolated, transforms = initialize(filename)
                #except TypeError:
                    #print("Calibration failed, closing application")
                    #time.sleep(2)
                    #break
                with open('token.txt', 'w') as f:
                    f.write(token)
                hex_sandbox_prev = hex_sandbox
                hex_tygron_prev = hex_tygron
                hex_water_prev = hex_water
                hex_land_prev = hex_land
                print("stored initial board state")
                toc = time.time()
                print("Start up and calibration time: "+str(toc-tic))
                """
                this section will also need scripts to:
                    - to create the grid for sandbox
                    - to store indices and weighing factors for interpolation
                    - to execute the interpolation
                    - to convert the grid to geotiff
                    - to initialize all the changes in Tygron (water/land/
                      buildings)
                    - to initiate the hydrodynamic model
                """
                start = False
                print("Initializing complete, waiting for next command")
            else:
                print("Already running")
                pass
        elif a == '1':
            if not start:
                print("Updating board state")
                turn += 1
                filename = 'board_image%d.jpg' % turn  # snapshot filename
                with open('hexagons_tygron_update_transformed_test4.geojson') as f:
                    hexagons_old = geojson.load(f)
                with open('hexagons_tygron_update_transformed_test3.geojson') as g:
                    hexagons_new = geojson.load(g)
                tic = time.time()
                hexagons_new = tygron.update_hexagons_tygron_id(token,
                                                                hexagons_new)
                z_changed = compare.compare_hex(token, hexagons_old,
                                                hexagons_new)
                tac = time.time()
                grid_interpolated = gridmap.hex_to_points(hexagons_new,
                                                          grid_interpolated,
                                                          changed_hex=z_changed,
                                                          turn=turn)
                toc = time.time()
                print("Updated to " + str(turn) +
                      ". Comparison update time: " + str(tac-tic) +
                      ". Interpolation update time: " + str(toc-tac) +
                      ". Total update time: " + str(toc-tic))
            else:
                print("No calibration run, please first calibrate Virtual"
                      "River by pressing '5'")
                pass
        elif a == '0':
            print("Exiting\n")
            time.sleep(1)
            exit(0)
            break
        else:
            print("Unrecognized keystroke\n")
            exit(0)
        time.sleep(0.3)


def initialize(filename):
    """
    Function that handles all the startup necessities:
    - calls logging into Tygron
    - initiates and calibrates the camera
    - get, perform and store the necessary transforms
    - process initial board state
    - initiate sandbox model
    - run sandbox model
    - update Tygron
    - return all variables that need storing
    """
    with open(r'C:\Users\HaanRJ\Documents\Storage\username.txt', 'r') as f:
        username = f.read()
    with open(r'C:\Users\HaanRJ\Documents\Storage\password.txt', 'r') as g:
        password = g.read()
    token = tygron.join_session(username, password)
    if token is None:
        print("logging in to Tygron failed, unable to make changes in Tygron")
    else:
        token = "token=" + token
        print("logged in to Tygron")
        # camera calibration --> to do: initiation
        canvas, thresh = cali.detect_corners(filename, method='adaptive')
        # store calibration values as global variables
        pers, img_x, img_y, origins, radius, cut_points, features \
            = cali.rotate_grid(canvas, thresh)
        # create the calibration file for use by other methods and store it
        transforms = cali.create_calibration_file(img_x, img_y, cut_points)
        print("calibrated camera")
        hexagons = detect.detect_markers(filename, pers, img_x, img_y,
                                         origins, radius, features)
        print("processed initial board state")
        hexagons = tygron.update_hexagons_tygron_id(token, hexagons)
        hexagons_sandbox = detect.transform(hexagons, transforms,
                                            export="sandbox")
        hexagons_tygron = detect.transform(hexagons, transforms,
                                           export="tygron_initialize")
        hexagons_water, hexagons_land = detect.transform(hexagons, transforms,
                                                         export="tygron")
        print("prepared geojson files")
        grid = gridmap.read_grid()
        print("loaded grid")
        grid_interpolated = gridmap.hex_to_points(hexagons_sandbox, grid,
                                                       start=True)
        print("executed grid interpolation")
        """
        This section is not very efficient, once the system is up and running
        replace this to either also check which hexagons need to change or make
        an empty project area that is either land or water only.
        """
        tygron.set_terrain_type(token, hexagons_water, terrain_type="water")
        tygron.set_terrain_type(token, hexagons_land, terrain_type="land")
        print("updated Tygron")
    try:
        return token, hexagons_sandbox, hexagons_tygron, hexagons_water, \
            hexagons_land, grid, grid_interpolated, transforms
    except UnboundLocalError:
        print("logging in to Tygron failed, closing application")
        quit()


def update():
    """
    function that initiates and handles all update steps. Returns all update
    variables
    """
    return


if __name__ == '__main__':
    main_menu()
