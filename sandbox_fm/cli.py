# -*- coding: utf-8 -*-
import click

import numpy as np
from .depth import depth_images
from .plots import Visualization


@click.command()
@click.argument('image', type=click.File('rb'))
def main(image):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")


    vis = Visualization()
    # load model library
    # model = BMIWrapper('dflowfm')
    images = depth_images(image)
    img = next(images)
    data = dict(kinect=img)

    vis.initialize(data)
    # initialize model schematization
    # model.initialize("configfile.mdu")
    # interp = scipy.interpolate.LinearND(model.variables['xz'], model.variables['yz'])

    for img in images:
        data["kinect"] = img
        # diff = img - imgprevious
        # bathydiff = interp(diff)
        # data["bathydiff"] = bathydiff
        vis.update(data)

if __name__ == "__main__":
    main()
