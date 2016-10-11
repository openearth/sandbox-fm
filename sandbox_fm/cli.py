# -*- coding: utf-8 -*-
import click

import numpy as np
from .depth import depth_images
from .plots import Visualization


@click.command()
@click.argument('image', type=click.File('rb'))
@click.argument('schematization', type=click.File('rb'))
@click.option('--max-iterations', default=0, help='Maximum number of iterations (0=no limit)')
def main(image, schematization, max_iterations):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")
    vis = Visualization()
    # load model library
    model = BMIWrapper('dflowfm')
    images = depth_images(image)
    img = next(images)
    data = dict(kinect=img)

    vis.initialize(data)

    # initialize model schematization
    model.initialize(schematization)

    for i, img in enumerate(images):
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
