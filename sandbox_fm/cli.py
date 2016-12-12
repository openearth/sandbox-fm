# -*- coding: utf-8 -*-

import pathlib
import logging
import time
import json

import scipy.interpolate
import cv2
import tqdm
import click
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt

import mpi4py.MPI
import bmi.wrapper

from .depth import (
    depth_images,
    calibrated_depth_images,
    video_images
)
from .calibrate import (
    compute_affines
)
from .plots import Visualization
from .sandbox_fm import (
    update_delft3d_initial_vars,
    update_delft3d_vars,
    compute_delta_bl,
    compute_delta_zk
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    pass

@cli.command()
def record():
    """record 10 frames, for testing"""
    videos = video_images()
    depths = depth_images()
    for i, (video, depth) in enumerate(zip(videos, depths)):
        plt.imsave("video_%06d.png" % (i, ), video)
        plt.imsave("depth_%06d.png" % (i, ), depth, cmap='Greys')
        if i > 10:
            break



@cli.command()
@click.argument('schematization', type=click.File('rb'))
def calibrate(schematization):
    """calibrate the sandbox by selecting both 4 points in box and in model"""

    # raw images
    videos = video_images()
    depths = depth_images()
    raws = depth_images(raw=True)
    # get video and depth image
    video = next(videos)
    depth = next(depths)
    raw = next(raws)

    # save the current working directory
    curdir = pathlib.Path.cwd()

    fig, axes = plt.subplots(2, 2)
    # sic show instructions in the title
    fig.suptitle('select 4 points (counter clockwise in both figures)')

    # show the depth and video in the left window
    axes[0, 0].imshow(video)
    axes[0, 0].imshow(depth, alpha=0.3, cmap='Reds')
    axes[1, 0].imshow(raw)

    # keep track of the selected points
    img_points = []
    model_points = []
    height_points = []

    # define fixed box coordinate system (what will be on the screen)
    box = np.array([
        [0, 0],
        [0, 480],
        [640, 480],
        [640, 0]
    ], dtype='float32')

    # pointer event
    pid = None

    # define the point selector
    def picker(event):
        if (event.inaxes == axes[0, 0] and len(img_points) < 4):
            img_points.append((event.xdata, event.ydata))
            event.inaxes.set_title('%s points selected' % (
                len(img_points), )
            )
        elif (event.inaxes == axes[0, 1] and len(model_points) < 4):
            model_points.append((event.xdata, event.ydata))
            event.inaxes.set_title('%s points selected' % (
                len(model_points), )
            )
        elif (event.inaxes == axes[1, 0] and len(height_points) < 4):
            height_points.append((event.xdata, event.ydata))
            title = "%s points selected" % (len(height_points), )
            if (len(height_points) == 0):
                title = "select a point at -8m"
            elif (len(height_points) == 1):
                title = "select a point at 0m"
            elif (len(height_points) == 2):
                title = "select a point at 12m"
            event.inaxes.set_title(title)
            event.inaxes.plot(event.xdata, event.ydata, 'ko')
        if len(img_points) == 4 and len(model_points) == 4 and len(height_points) == 3:
            # stop listening we're done
            fig.canvas.mpl_disconnect(pid)
            save_and_show_result()

    def save_and_show_result():
        # we should have results by now
        model2box = cv2.getPerspectiveTransform(
            np.array(model_points, dtype='float32'),
            box
        )
        img2box = cv2.getPerspectiveTransform(
            np.array(img_points, dtype='float32'),
            box
        )
        img2model = cv2.getPerspectiveTransform(
            np.array(img_points, dtype='float32'),
            np.array(model_points, dtype='float32')
        )
        model2img = cv2.getPerspectiveTransform(
            np.array(model_points, dtype='float32'),
            np.array(img_points, dtype='float32')
        )
        box2model = cv2.getPerspectiveTransform(
            np.array(box, dtype='float32'),
            np.array(model_points, dtype='float32')
        )
        box2img = cv2.getPerspectiveTransform(
            np.array(box, dtype='float32'),
            np.array(img_points, dtype='float32')
        )

        values = raw[np.array(height_points, dtype='int')[:, 1], np.array(height_points, dtype='int')[:, 0]].filled().astype('double')
        z = np.array([-12, 0, 8], dtype='double')
        idx = np.argsort(values)
        # parameters of interpolation
        tck_a, tck_b, tck_c = scipy.interpolate.splrep(values[idx], z[idx], k=1, s=0)


        comment = """
        This file contains calibrations for model %s.
        It is generated with the perspective transform from opencv.
        """ % (schematization_path, )
        result = {
            "model2box": model2box.tolist(),
            "img2box": img2box.tolist(),
            "img2model": img2model.tolist(),
            "model2img": model2img.tolist(),
            "box2model": box2model.tolist(),
            "box2img": box2img.tolist(),
            "img_points": img_points,
            "model_points": model_points,
            "height_points": height_points,
            "tck_a": tck_a.tolist(),
            "tck_b": tck_b.tolist(),
            "tck_c": tck_c,
            "_comment": comment

        }
        # save the calibration info
        with open(str(curdir / 'calibration.json'), 'w') as f:
            json.dump(result, f, indent=2)


        # now for showing results
        fig, ax = plt.subplots()
        xy_nodes_in_img = np.squeeze(
            cv2.perspectiveTransform(
                np.dstack([
                    xy_node[:, np.newaxis, 0],
                    xy_node[:, np.newaxis, 1]
                ]).astype('float32'),
                model2box
            )
        )
        # scatter plot
        ax.scatter(
            xy_nodes_in_img[:, 0],
            xy_nodes_in_img[:, 1],
            c=data['zk'].ravel(),
            cmap='Greens',
            edgecolor='none',
            s=20,
            alpha=0.5
        )
        # transformed video on top
        ax.imshow(
            cv2.warpPerspective(
                video,
                img2box,
                (640, 480)
            ),
            cmap='Reds',
            alpha=0.5
        )
        plt.show()

    # start the model (changes directory)
    model = bmi.wrapper.BMIWrapper('dflowfm')
    schematization_path = pathlib.Path(schematization.name)
    model.initialize(str(schematization_path.absolute()))
    data = {}
    update_delft3d_initial_vars(data, model)
    # convert to array we can feed into opencv
    xy_node = np.c_[
        data['xk'],
        data['yk'],
        np.ones_like(data['xk'])
    ].astype('float32')


    axes[0, 1].scatter(data['xk'].ravel(), data['yk'].ravel(), c=data['zk'].ravel(), cmap='Greens', edgecolor='none')
    plt.ion()
    pid = fig.canvas.mpl_connect('button_press_event', picker)
    plt.show(block=True)


@cli.command()
def view():
    """view raw kinect images"""
    with open("calibration.json") as f:
        calibration = json.load(f)
    tck = (
        np.array(calibration['tck_a']),
        np.array(calibration['tck_b']),
        calibration['tck_c']
    )
    images = calibrated_depth_images(tck)
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
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")

    # calibration info
    data = {}
    with open('calibration.json') as f:
        calibration = json.load(f)
    data.update(calibration)

    # model
    model = bmi.wrapper.BMIWrapper('dflowfm')
    # initialize model schematization, changes directory
    model.initialize(str(pathlib.Path(schematization.name).absolute()))
    update_delft3d_initial_vars(data, model)
    dt = model.get_time_step()

    model_bbox = matplotlib.path.Path(data['model_points'])
    data['node_in_box'] = model_bbox.contains_points(np.c_[data['xk'], data['yk']])
    data['cell_in_box'] = model_bbox.contains_points(np.c_[data['xzw'], data['yzw']])

    tck = (
        np.array(calibration['tck_a']),
        np.array(calibration['tck_b']),
        calibration['tck_c']
    )

    # images
    depths = calibrated_depth_images(tck)
    # load model library
    depth = next(depths)
    depth = cv2.warpPerspective(
        depth,
        np.array(data['img2box'], dtype='float32'),
        (640, 480)
    )

    data['kinect_0'] = depth.copy()
    data['kinect'] = depth


    vis = Visualization()
    update_delft3d_vars(data, model)
    vis.initialize(data)

    for i in range(10):
        model.update(dt)


    for i, depth in enumerate(tqdm.tqdm(depths)):
        update_delft3d_vars(data, model)
        depth = cv2.warpPerspective(
            depth,
            np.array(data['img2box'], dtype='float32'),
            (640, 480)
        )
        data['kinect'] = depth

        # only change bathymetry of wet cells
        idx = np.logical_and(data['cell_in_box'], data['is_wet']) #
        # idx = data['cell_in_box']

        if i % 100 == 0:
            # data['bl'][idx] += compute_delta_bl(data, idx)
            idx = data['node_in_box']
            zk_copy = data['zk'].copy()
            zk_copy[idx] += compute_delta_zk(data, idx)
            # replace the part that changed
            for i in np.where(idx)[0]:
                if data['zk'][i] != zk_copy[i]:
                    # TODO: bug in zk
                    model.set_var_slice('zk', [i+1], [1], zk_copy[i:i+1])

        vis.update(data)
        tic = time.time()
        model.update(dt)
        toc = time.time()
        print(toc - tic)

if __name__ == "__main__":
    main()
