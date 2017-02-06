# -*- coding: utf-8 -*-

import pathlib
import logging
import time
import json
import functools
import itertools

try:
    from itertools import izip as zip
except ImportError:
    # python3 has it builtin
    pass

import skimage.io
import cv2    
import tqdm
import click
import numpy as np
import matplotlib.path
import matplotlib.backend_bases
import matplotlib.pyplot as plt

import bmi.wrapper

HAVE_MPI = False
try:
    import mpi4py.MPI
    HAVE_MPI = True
except ImportError:
    pass

from .depth import (
    depth_images,
    calibrated_height_images,
    video_images
)
from .calibrate import (
    transform,
    compute_transforms
)
from .calibration_tool import Calibration
from .plots import (
    Visualization,
    process_events
)

from .sandbox_fm import (
    update_delft3d_initial_vars,
    update_delft3d_vars
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# initialize mpi
if HAVE_MPI:
    mpi4py.MPI.COMM_WORLD
else:
    logging.warn('MPI not initialized')

@click.group()
def cli():
    """
    keys:
     - 1, 2, 3 -> switch views
     - f -> fullscreen
     - c -> toggle currents
     - p -> make picture
     - r -> reset bathymethry
     - b -> set bed level
    """
    pass

@cli.command()
def record():
    """record 10 frames, for testing"""
    videos = video_images()
    raws = depth_images(raw=True)
    for i, (video, raw) in enumerate(zip(videos, raws)):
        skimage.io.imsave("video_%06d.png" % (i, ), video)
        raw.dump("raw_%06d.npy" % (i, ))
        if i > 5:
            break


@cli.command()
def anomaly():
    """calibrate the kinect anomaly for a flat surface"""

    raws = depth_images(raw=True)
    raw = next(raws)
    anomaly = raw - raw.mean()
    anomaly.dump('anomaly.npy')


@cli.command()
@click.argument('schematization', type=click.File('rb'))
def calibrate(schematization):
    """calibrate the sandbox by selecting both 4 points in box and in model"""

    schematization_path = pathlib.Path(schematization.name)
    path = schematization_path.with_name('calibration.json').absolute()
    # raw images
    videos = video_images()
    raws = depth_images(raw=True)
    # start the model (changes directory)
    model = bmi.wrapper.BMIWrapper('dflowfm')
    # this stores current path

    # this changes directory
    model.initialize(str(schematization_path.absolute()))

    calibration = Calibration(path, videos, raws, model)
    calibration.run()

@cli.command()
def view():
    """view raw kinect images"""
    with open("calibration.json") as f:
        calibration = json.load(f)
    images = calibrated_height_images(calibration["z_values"], calibration["z"])
    origin = 'bottom'

    fig, ax = plt.subplots(frameon=False)
    manager = plt.get_current_fig_manager()
    try:
        manager.resize(*manager.window.maxsize())
    except AttributeError:
        # no resize available
        pass
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()
    ax.set_frame_on(False)
    im = ax.imshow(next(images), origin='upper', cmap='terrain')
    plt.ion()
    plt.show(block=False)
    for img in tqdm.tqdm(images):
        im.set_data(img)
        fig.canvas.draw()




@cli.command()
@click.argument('schematization', type=click.File('rb'))
def run(schematization):
    """Console script for sandbox_fm

    keys:
     - 1, 2, 3 -> switch views
     - f -> fullscreen
     - c -> toggle currents
     - p -> make picture
     - r -> reset bathymethry
     - b -> set bed level
    """
    click.echo("Make sure you start the SARndbox first")

    schematization_name = pathlib.Path(schematization.name)
    # keep absolute path so model can change directory
    calibration_name = schematization_name.with_name('calibration.json').absolute()
    config_name = schematization_name.with_name('config.json').absolute()
    anomaly_name = pathlib.Path('anomaly.npy').absolute()

    # calibration info
    data = {}
    with open(str(calibration_name)) as f:
        calibration = json.load(f)
    data.update(calibration)
    data.update(compute_transforms(data))
    with open(str(config_name)) as f:
        configuration = json.load(f)
    data.update(configuration)

    # model
    model = bmi.wrapper.BMIWrapper('dflowfm')
    # initialize model schematization, changes directory
    background_name = pathlib.Path(schematization.name).with_suffix('.jpg').absolute()
    data['background_name'] = background_name
    model.initialize(str(schematization_name.absolute()))
    update_delft3d_initial_vars(data, model)
    dt = model.get_time_step()

    # compute the model bounding box that is shown on the screen
    model_bbox = matplotlib.path.Path(data['model_points'])
    # create an index to see which points/cells are visualized
    data['node_in_box'] = model_bbox.contains_points(np.c_[data['xk'], data['yk']])
    data['cell_in_box'] = model_bbox.contains_points(np.c_[data['xzw'], data['yzw']])

    img_bbox = matplotlib.path.Path([
        (40, 40),
        (40, 480),
        (600, 480),
        (600, 40)
    ])
    xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2box'])
    xk_box, yk_box = transform(data['xk'], data['yk'], data['model2box'])
    print(xzw_box.min(), xzw_box.max())

    # for transformed coordinates see if they are on the screen
    data['cell_in_img_bbox'] = img_bbox.contains_points(np.c_[xzw_box, yzw_box])
    data['node_in_img_bbox'] = img_bbox.contains_points(np.c_[xk_box, yk_box])

    if data.get('debug'):
        plt.scatter(data['xzw'], data['yzw'], c=data['cell_in_img_bbox'], edgecolor='none')
        plt.show()
        plt.scatter(data['xzw'], data['yzw'], c=data['cell_in_box'], edgecolor='none')
        plt.show()

    # images
    heights = calibrated_height_images(
        calibration["z_values"],
        calibration["z"],
        anomaly_name=anomaly_name
    )
    videos = video_images()
    # load model library
    height = next(heights)
    video = next(videos)

    data['height'] = height
    data['video'] = video
    data['zk_original']=data['zk'].copy()
    data['height_original']=data['height'].copy()


    vis = Visualization()
    update_delft3d_vars(data, model)
    vis.initialize(data)

    vis.subscribers.append(
        # fill in the data parameter and subscribe to events
        functools.partial(process_events, data=data, model=model, vis=vis)
    )

    # start model and run for a bit
    # for i in range(5):
    #     model.update(dt)


    for i, (video, height) in enumerate(tqdm.tqdm(zip(videos, heights))):

        tic = time.time()
        # Get data from model
        update_delft3d_vars(data, model)
        # update kinect
        data['height'] = height
        data['video'] = video
        toc = time.time()
        time_get=toc-tic

        # update visualization
        tic = time.time()
        vis.update(data)
        toc = time.time()
        time_vis=toc-tic

        # update model
        tic = time.time()
        model.update(dt)
        toc = time.time()
        time_model=toc-tic
        print('get',time_get,';vis',time_vis,';model',time_model)

if __name__ == "__main__":
    import sandbox_fm.cli
    sandbox_fm.cli.cli()
