# -*- coding: utf-8 -*-
import click

import time
import numpy as np
from .depth import depth_images
from .plots import Visualization

@click.command()
@click.argument('image', type=click.File('rb'))
def main(image):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")
    vis = Visualization()
    images = depth_images(image)
    img = next(images)
    data = dict(kinect=img)
    vis.initialize(data)
    for img in images:
        data["kinect"] = img
        print(np.max(img))
        vis.update(data)
    
if __name__ == "__main__":
    main()
