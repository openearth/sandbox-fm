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
    update_initial_vars,
    update_vars
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# initialize mpi
if HAVE_MPI:
    mpi4py.MPI.COMM_WORLD
    logger.info("MPI initialized")
else:
    logger.warn('MPI not initialized')

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
    logger.info("Welcome to the sandbox software.")

@cli.command()
def record():
    """record 10 frames, for testing"""
    videos = video_images()
    raws = depth_images()
    for i, (video, raw) in enumerate(zip(videos, raws)):
        skimage.io.imsave("video_%06d.png" % (i, ), video)
        raw.dump("raw_%06d.npy" % (i, ))
        if i > 5:
            break


@cli.command()
def anomaly():
    """calibrate the kinect anomaly for a flat surface"""

    raws = depth_images()
    raw = next(raws)
    anomaly = raw - raw.mean()
    anomaly.dump('anomaly.npy')


@cli.command()
@click.argument('schematization', type=click.File('rb'))
@click.option('--engine', default='dflowfm', type=click.Choice(['dflowfm', 'xbeach']))
def calibrate(schematization, engine):
    """calibrate the sandbox by selecting both 4 points in box and in model"""

    schematization_path = pathlib.Path(schematization.name)
    path = schematization_path.with_name('calibration.json').absolute()
    # raw images
    videos = video_images()
    raws = depth_images()
    # start the model (changes directory)
    model = bmi.wrapper.BMIWrapper(engine)
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
@click.option('--engine', default='dflowfm', type=click.Choice(['dflowfm', 'xbeach']))
@click.option('--max-iterations', default=0, type=int)
def run(schematization, engine, max_iterations):
    """Console script for sandbox_fm

    keys:
     - 1, 2, 3 -> switch views
     - f -> fullscreen
     - c -> toggle currents
     - p -> make picture
     - r -> reset bathymethry
     - b -> set bed level
    """

    schematization_name = pathlib.Path(schematization.name)
    # keep absolute path so model can change directory
    calibration_name = schematization_name.with_name('calibration.json').absolute()
    config_name = schematization_name.with_name('config.json').absolute()
    anomaly_name = pathlib.Path('anomaly.npy').absolute()

    # calibration info
    data = {}
    data['schematization'] = schematization_name
    with open(str(calibration_name)) as f:
        calibration = json.load(f)
    data.update(calibration)
    data.update(compute_transforms(data))
    # if we have a configuration file
    if config_name.exists():
        # open it
        with open(str(config_name)) as f:
            configuration = json.load(f)
    else:
        # default empty
        configuration = {}
    data.update(configuration)

    # model
    model = bmi.wrapper.BMIWrapper(engine)
    # initialize model schematization, changes directory
    background_name = pathlib.Path(schematization.name).with_suffix('.jpg').absolute()
    data['background'] = background_name
    model.initialize(str(schematization_name.absolute()))
    update_initial_vars(data, model)
    dt = model.get_time_step()

    # compute the model bounding box that is shown on the screen
    model_bbox = matplotlib.path.Path(data['model_points'])
    # create an index to see which points/cells are visualized
    data['node_in_box'] = model_bbox.contains_points(np.c_[data['X_NODES'].ravel(), data['Y_NODES'].ravel()])
    data['cell_in_box'] = model_bbox.contains_points(np.c_[data['X_CELLS'].ravel(), data['Y_CELLS'].ravel()])

    img_bbox = matplotlib.path.Path([
        (40, 40),
        (40, 440),
        (600, 440),
        (600, 40)
    ])
    x_nodes_box, y_nodes_box = transform(data['X_NODES'].ravel(), data['Y_NODES'].ravel(), data['model2box'])
    x_cells_box, y_cells_box = transform(data['X_CELLS'].ravel(), data['X_CELLS'].ravel(), data['model2box'])

    # for transformed coordinates see if they are on the screen
    data['node_in_img_bbox'] = img_bbox.contains_points(np.c_[x_nodes_box, y_nodes_box])
    data['cell_in_img_bbox'] = img_bbox.contains_points(np.c_[x_cells_box, y_cells_box])

    if data.get('debug'):
        plt.scatter(data['X_CELLS'], data['Y_CELLS'], c=data['cell_in_img_bbox'], edgecolor='none')
        plt.show()
        plt.scatter(data['X_CELLS'], data['Y_CELLS'], c=data['cell_in_box'], edgecolor='none')
        plt.show()

    # images
    kinect_heights = calibrated_height_images(
        calibration["z_values"],
        calibration["z"],
        anomaly_name=anomaly_name
    )
    kinect_images = video_images()
    # load model library
    kinect_height = next(kinect_heights)
    kinect_image = next(kinect_images)

    data['kinect_height'] = kinect_height
    data['kinect_image'] = kinect_image
    data['height_cells_original'] = data['HEIGHT_CELLS'].copy()
    data['kinect_height_original'] = data['kinect_height'].copy()

    vis = Visualization()
    update_vars(data, model)
    vis.initialize(data)
    vis.subscribers.append(
        # fill in the data parameter and subscribe to events
        functools.partial(process_events, data=data, model=model, vis=vis)
    )
    iterator = enumerate(tqdm.tqdm(zip(kinect_images, kinect_heights)))
    for i, (kinect_image, kinect_height) in iterator:

        # Get data from model
        update_vars(data, model)

        # update kinect
        data['kinect_height'] = kinect_height
        data['kinect_image'] = kinect_image

        # update visualization
        vis.update(data)
        dt = model.get_time_step()
        # HACK: fix unstable timestep in xbeach
        if model.engine == 'xbeach':
            dt = 60
        # update model
        import time
        tic = time.time()
        for i in range(data.get('iterations.per.visualization', 1)):
            model.update(dt)
        toc = time.time()
        logger.info("elapsed %s, t: %s", toc-tic, model.get_current_time())

        if max_iterations and i > max_iterations:
            break

if __name__ == "__main__":
    import sandbox_fm.cli
    sandbox_fm.cli.cli()
