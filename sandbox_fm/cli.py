# -*- coding: utf-8 -*-

import os
import pathlib
import logging
import time
import json

import cv2
import tqdm
import click
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt

import bmi.wrapper

from .depth import (
    depth_images,
    calibrated_depth_images,
    percentile_depth_images,
    video_images
)
from .calibrate import (
    compute_affines
)
from .plots import Visualization
from .sandbox_fm import (
    update_delft3d_initial_vars,
    update_delft3d_vars
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    pass

@cli.command()
@click.argument('schematization', type=click.File('rb'))
def calibrate(schematization):
    videos = video_images()
    depths = depth_images()

    curdir = pathlib.Path.cwd()

    fig, axes = plt.subplots(1, 2)
    # sic
    fig.suptitle('select 4 points (counter clockwise in both figures)')

    img_points = []
    model_points = []
    # define fixed box coordinates
    box = np.array([
        [0, 0],
        [0, 480],
        [640, 480],
        [640, 0]
    ], dtype='float32')

    pid = None

    video = next(videos)
    depth = next(depths)



    def picker(event):
        if (event.inaxes == axes[0] and len(img_points) < 4):
            img_points.append((event.xdata, event.ydata))
            event.inaxes.set_title('%s points selected' % (len(img_points), ))
        elif (event.inaxes == axes[1] and len(model_points) < 4):
            model_points.append((event.xdata, event.ydata))
            event.inaxes.set_title('%s points selected' % (len(model_points), ))
        if len(img_points) == 4 and len(model_points) == 4:
            fig.canvas.mpl_disconnect(pid)
            show_result()



    axes[0].imshow(video)
    axes[0].imshow(depth, alpha=0.3, cmap='Reds')

    model = bmi.wrapper.BMIWrapper('dflowfm')
    schematization_path = pathlib.Path(schematization.name)
    model.initialize(str(schematization_path.absolute()))
    data = {}
    update_delft3d_initial_vars(data, model)
    xy_node = np.c_[
        data['xk'],
        data['yk'],
        np.ones_like(data['xk'])
    ].astype('float32')

    def show_result():
        fig, ax = plt.subplots()
        model2box = cv2.getPerspectiveTransform(np.array(model_points, dtype='float32'), box)
        img2box = cv2.getPerspectiveTransform(np.array(img_points, dtype='float32'), box)
        result = {
            "model2box": model2box.tolist(),
            "img2box": img2box.tolist()
        }
        with open(str(curdir / 'calibration.json'), 'w') as f:
            json.dump(result, f)

        xy_nodes_in_img = np.squeeze(
            cv2.perspectiveTransform(
                np.dstack([
                    xy_node[:,np.newaxis,0],
                    xy_node[:,np.newaxis,1]
                ]).astype('float32'),
                model2box
            )
        )
        ax.scatter(xy_nodes_in_img[:, 0], xy_nodes_in_img[:, 1], c=data['zk'].ravel(), cmap='Greens', edgecolor='none', alpha=0.5)
        ax.imshow(cv2.warpPerspective(video, img2box, (640, 480)), cmap='Reds', alpha=0.5)
        plt.show()

    axes[1].scatter(data['xk'].ravel(), data['yk'].ravel(), c=data['zk'].ravel(), cmap='Greens', edgecolor='none')
    plt.ion()
    pid = fig.canvas.mpl_connect('button_press_event', picker)
    plt.show(block=True)


@cli.command()
def view():
    """view raw kinect images"""
    images = calibrated_depth_images()
    origin = 'bottom'

    fig, ax = plt.subplots(frameon=False)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()
    ax.set_frame_on(False)
    im = ax.imshow(next(images), origin='upper', cmap='terrain')
    plt.ion()
    plt.show()
    for img in tqdm.tqdm(images):
        im.set_data(img)
        fig.canvas.draw()




@cli.command()
@click.argument('image', type=click.File('rb'))
@click.argument('schematization', type=click.File('rb'))
@click.option(
    '--max-iterations',
    default=0,
    help='Maximum number of iterations (0=no limit)'
)
@click.option(
    '--random-bathy',
    default=0.0,
    help='Raise or lower the bathymetry every 30 timesteps by at most x meter at random locations'
)
def run(image, schematization, max_iterations, random_bathy):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")
    vis = Visualization()
    # load model library
    images = calibrated_depth_images()

    model = bmi.wrapper.BMIWrapper('dflowfm')
    img = next(images)
    data = dict(
        kinect_0=img.copy(),
        kinect=img
    )

    # initialize model schematization, changes directory
    model.initialize(schematization.name)

    dt = model.get_time_step()
    update_delft3d_initial_vars(data, model)

    # coordinates of the image in the model
    img_in_model = np.array([
        [7.10000000e+04,   4.49500000e+05],
        [7.44420895e+04,   4.53660772e+05],
        [7.16682419e+04,   4.55955498e+05],
        [6.82261523e+04,   4.51794726e+05]
    ])

    # compute the transformation
    img2model, model2img = compute_affines(img_in_model, img.shape)

    # create xy1 matrices for model and img
    xy1_model = np.c_[
        data['xzw'],
        data['yzw'],
        np.ones_like(data['xzw'])
    ]

    img_in_model_path = matplotlib.path.Path(img_in_model)
    in_sandbox = img_in_model_path.contains_points(xy1_model[:, :2])
    data['in_sandbox'] = in_sandbox

    Y_img, X_img = np.mgrid[:img.shape[0], :img.shape[1]]
    xy1_img = np.c_[X_img.ravel(), Y_img.ravel(), np.ones_like(X_img.ravel())]
    data['xy1_img'] = xy1_img

    # image in model coordinates
    data['xy1_img_in_model'] = np.dot(xy1_img, img2model.T)
    # multiply [x, y, 1] with affine' gives [xt, yt, 1] (B'A' = (AB)')
    data['xy1_model_in_img'] = np.dot(xy1_model, model2img.T)

    vis.initialize(data)

    change = 0
    s = np.s_[0:0, 0:0]

    for i, img in enumerate(images):
        update_delft3d_vars(data, model)
        print(i)
        if max_iterations and i > max_iterations:
            break
        data["kinect"] = img

        if random_bathy :
            # generate a random slice
            if ((i % 30) == 0):
                i_0 = np.random.randint(img.shape[0])
                i_1 = np.random.randint(i_0, img.shape[0])
                j_0 = np.random.randint(img.shape[1])
                j_1 = np.random.randint(j_0, img.shape[1])
                s = np.s_[i_0:i_1, j_0:j_1]
                change = np.random.uniform(-random_bathy, random_bathy)
            data["kinect"][s] = change
            is_wet = data['bl'] < data['s1']

            is_selected = np.random.random(size=data['bl'].shape[0]) > 1/100.0
            idx = np.logical_and(is_wet, is_selected)

        # comput if wet
        data['bl'][idx] += change

        img_diff = diff = data["kinect_0"] - data["kinect"]

        # diff = img - imgprevious
        # bathydiff = interp(diff)
        # data["bathydiff"] = bathydiff
        vis.update(data)
        tic = time.time()
        model.update(dt)
        toc = time.time()
        print(toc - tic)

if __name__ == "__main__":
    main()
