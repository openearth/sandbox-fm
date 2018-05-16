# -*- coding: utf-8 -*-

import pathlib
import logging
import time
import json
import functools

try:
    from itertools import izip as zip
except ImportError:
    # python3 has it builtin
    pass

import tqdm
import click
import numpy as np
import matplotlib.path
import matplotlib.backend_bases
import matplotlib.pyplot as plt

import bmi.wrapper
from mmi.mmi_client import MMIClient
from mmi import recv_array

HAVE_MPI = False
try:
    import mpi4py.MPI
    HAVE_MPI = True
except ImportError:
    pass

# don't import before MPI, otherwise segfault under OSX
import skimage.io

from sandbox_fm.depth import (
    depth_images,
    calibrated_height_images,
    video_images
)
from sandbox_fm.calibrate import (
    transform,
    compute_transforms
)

from sandbox_fm.calibration_wizard import Calibration
from sandbox_fm.plots import (
# from sandbox_fm.plots_cv2 import (
    Visualization,
    process_events,
    default_config
)
from sandbox_fm.variables import (
    update_initial_vars,
    update_vars,
    update_with_message,
    run_update_bedlevel
)
from sandbox_fm.gestures import (
    recognize_gestures
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


def tic_report(tics):
    items = list(tics.items())
    items.sort(key=lambda x:x[1])
    # value of t0
    prev = items[0][1]
    assert items[0][0] == 't0'
    msgs = []
    for item, curr in items[1:]:
        diff = curr - prev
        msgs.append("%s: %.2f" % (item, diff))
        prev = curr
    return " ".join(msgs)


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
    logging.basicConfig(level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    logger.info("Welcome to the sandbox software.")


@cli.command()
def record():
    """record 5 frames, for testing"""
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
@click.argument(
    'schematization',
    type=click.File('rb')
)
@click.option(
    '--engine',
    default='dflowfm',
    type=click.Choice(['dflowfm', 'xbeach']),
    help='BMI compatible model engine'

)
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
#    origin = 'bottom'

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
@click.argument(
    'schematization',
    type=click.File('rb')
)
@click.option(
    '--engine',
    default='dflowfm',
    type=click.Choice(['dflowfm', 'xbeach']),
    help='BMI comptabile model engine'
)
@click.option(
    '--max-iterations',
    default=0,
    type=int,
    help='maximum number of iterations'
)
@click.option(
    '--mmi',
    type=str,
    help='mmi zmq address for example tcp://localhost:62000'
)
def run(schematization, engine, max_iterations, mmi):
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
            configuration_read = json.load(f)
        # combine with defaults
        configuration = default_config.copy()
        configuration.update(configuration_read)
    else:
        # default empty
        configuration = default_config
    with open(str(config_name), 'w') as f:
        json.dump(configuration, f, indent=2)
    data.update(configuration)

    # model
    if not mmi:
        model = bmi.wrapper.BMIWrapper(engine)
    else:
        model = MMIClient(mmi)
        model.engine = engine
        logger.info('Connected to MMI: {}'.format(mmi))


    # initialize model schematization, changes directory

    # search for a background or overlay image
    for layer in ['background', 'overlay', 'background_mask']:
        known_background_paths = [
            pathlib.Path(schematization.name).with_name(layer + '.jpg'),
            pathlib.Path(schematization.name).with_name(layer + '.png')
        ]
        for path in known_background_paths:
            if path.exists():
                data[layer + '_name'] = str(path.absolute())
                break
        else:
            data[layer + '_name'] = None

    # mmi model is already initialized
    if not mmi:
        model.initialize(str(schematization_name.absolute()))
    update_initial_vars(data, model)

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
    x_nodes_box, y_nodes_box = transform(
        data['X_NODES'].ravel(),
        data['Y_NODES'].ravel(),
        data['model2box']
    )
    x_cells_box, y_cells_box = transform(
        data['X_CELLS'].ravel(),
        data['X_CELLS'].ravel(),
        data['model2box']
    )

    # for transformed coordinates see if they are on the screen
    data['node_in_img_bbox'] = img_bbox.contains_points(np.c_[x_nodes_box, y_nodes_box])
    data['cell_in_img_bbox'] = img_bbox.contains_points(np.c_[x_cells_box, y_cells_box])

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
    tics = {}
    last_bed_update = 0  # Time since last automatic bed level update
    if mmi:
        sub_poller = model.subscribe()
        # synchronize data when received
        model.remote('play')
    for i, (kinect_image, kinect_height) in iterator:
        tics['t0'] = time.time()

        # Get data from model
        if not mmi:
            update_vars(data, model)
        else:
            # listen for at most 10 miliseconds for incomming data (flush the queue)
            for sock, n in sub_poller.poll(10):
                for i in range(n):
                    message = recv_array(sock)
                    update_with_message(data, model, message)

        # update kinect
        data['kinect_height'] = kinect_height
        data['kinect_image'] = kinect_image
        tics['update_vars'] = time.time()

        gestures = recognize_gestures(data['kinect_height'])
        data['gestures'] = gestures
        tics['gestures'] = time.time()

        # update visualization
        vis.update(data)
        # visualization can trigger an exit
        if vis.quitting:
            break
        tics['vis'] = time.time()


        if not mmi:
            dt = model.get_time_step()
            # HACK: fix unstable timestep in xbeach
            if model.engine == 'xbeach':
                dt = 60

            model.update(dt)
        tics['model'] = time.time()

        if data['auto_bedlevel_update_interval']:
            time_since_bed_update = (time.time() - last_bed_update)
            if time_since_bed_update >  data['auto_bedlevel_update_interval']:
                run_update_bedlevel(data, model)
                last_bed_update = time.time()

        logger.info("tics: %s", tic_report(tics))

        if max_iterations and i > max_iterations:
            break

if __name__ == "__main__":
    import sandbox_fm.cli
    sandbox_fm.cli.cli()
