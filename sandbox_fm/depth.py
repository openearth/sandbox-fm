import logging

import freenect
import numpy as np
import scipy.interpolate

from .calibrate import HEIGHT, WIDTH, depth2xyzuv

logger = logging.getLogger(__name__)


def uint11_to_uint8(arr):
    """convert uint11 to uint8"""
    np.clip(arr, 0, 2**10 - 1, arr)
    arr >>= 2
    arr = arr.astype(np.uint8)
    return arr


def depth_images():
    """generate depth images"""
    while True:
        depth, _ = freenect.sync_get_depth()
        depth = uint11_to_uint8(depth)
        yield depth


def calibrated_depth_images(extent=None):
    """generate depth images on a fixed grid """
    v, u = np.mgrid[:HEIGHT, :WIDTH]
    images = depth_images()
    if not extent:
        depth_0 = next(images)
        xyz_0, uv_0 = depth2xyzuv(depth_0, u, v)
        xmin = xyz_0[:, 0].min() + 0.01
        xmax = xyz_0[:, 0].max() - 0.01
        ymin = xyz_0[:, 1].min() + 0.01
        ymax = xyz_0[:, 1].max() - 0.01
    else:
        xmin, xmax, ymin, ymax = extent
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, num=320),
        np.linspace(ymin, ymax, num=240)
    )

    for depth in images:
        xyz, uv = depth2xyzuv(depth, u, v)
        F = scipy.interpolate.NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
        z = F(X.ravel(), Y.ravel())
        Z = z.reshape(X.shape)
        yield Z
