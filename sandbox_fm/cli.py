# -*- coding: utf-8 -*-

import os
import logging

import click
import numpy as np
import shapely.geometry

import bmi.wrapper

from .depth import depth_images
from .plots import Visualization
from .sandbox_fm import (
    update_delft3d_initial_vars,
    update_delft3d_vars,
    compute_affines
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@click.command()
@click.argument('image', type=click.File('rb'))
@click.argument('schematization', type=click.File('rb'))
@click.option(
    '--max-iterations',
    default=0,
    help='Maximum number of iterations (0=no limit)'
)
def main(image, schematization, max_iterations):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")
    vis = Visualization()
    # load model library
    images = depth_images(os.path.abspath(image.name))

    model = bmi.wrapper.BMIWrapper('dflowfm')
    img = next(images)
    data = dict(kinect=img)

    # initialize model schematization, changes directory
    model.initialize(schematization.name)

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
    Y_img, X_img = np.mgrid[:img.shape[0], :img.shape[1]]
    xy1_img = np.c_[X_img.ravel(), Y_img.ravel(), np.ones_like(X_img.ravel())]
    data['xy1_img'] = xy1_img

    # image in model coordinates
    data['xy1_img_in_model'] = np.dot(xy1_img, img2model.T)
    # multiply [x, y, 1] with affine' gives [xt, yt, 1] (B'A' = (AB)')
    data['xy1_model_in_img'] = np.dot(xy1_model, model2img.T)
    vis.initialize(data)


    for i, img in enumerate(images):
        update_delft3d_vars(data, model)
        print(i)
        if max_iterations and i > max_iterations:
            break
        data["kinect"] = img

        # diff = img - imgprevious
        # bathydiff = interp(diff)
        # data["bathydiff"] = bathydiff
        vis.update(data)
        model.update()

if __name__ == "__main__":
    main()
