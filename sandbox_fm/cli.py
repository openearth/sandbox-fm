# -*- coding: utf-8 -*-
import click

import time

from .depth import read_depth_image
import sandbox_fm.plots

@click.command()
@click.argument('image', type=click.File('rb'))
def main(image):
    """Console script for sandbox_fm"""
    click.echo("Make sure you start the SARndbox first")
    vis = sandbox_fm.plots.Visualization()
    while True:
        t = time.time()
        data = dict(kinect=read_depth_image(image.name))
        vis.update(data)
        elapsed = time.time() - t
        time.sleep(0.01)
        click.echo("elapsed %s" % (elapsed, ))
if __name__ == "__main__":
    main()
