import logging
import collections
import itertools
import pathlib
import functools


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from .calibrate import HEIGHT, WIDTH, depth2xyzuv

logger = logging.getLogger(__name__)


class MockupFreenect(object):
    """mockup freenect in case you have no connection"""
    def __init__(self):
        data_dir = pathlib.Path(__file__).parent.parent / 'data'

        # TODO: pkgutil.get_data('data', '*')
        self.videos = itertools.cycle(
            data_dir.glob('video_*.png')
        )
        self.depths = itertools.cycle(
            data_dir.glob('raw_*.npy')
        )

    def sync_get_video(self):
        """keep yielding videos"""
        video = next(self.videos)
        return plt.imread(str(video)), 3

    def sync_get_depth(self):
        """keep yielding depths"""
        depth = next(self.depths)
        arr = np.load(str(depth))
        return arr, 3


HAVE_FREENECT = False
try:
    import freenect
    test_depth = freenect.sync_get_depth()
    if test_depth is not None:
        HAVE_FREENECT = True
except ImportError:
    pass

if not HAVE_FREENECT:
    logging.warn('Using mockup keenect')
    # try if we can read images, use mockup if not
    freenect = MockupFreenect()


def uint11_to_uint8(arr):
    """convert uint11 to uint8"""
    np.clip(arr, 0, 2**10 - 1, arr)
    arr >>= 2
    arr = arr.astype(np.uint8)
    return arr


def percentile_depth_images(buffer_size=25, q=25):
    """"compute running percentile images"""
    buffer = collections.deque(maxlen=buffer_size)
    for img in depth_images():
        buffer.append(img)
        perc = np.percentile(buffer, q=q, axis=0)
        yield perc


def video_images():
    while True:
        img, _ = freenect.sync_get_video()
        yield img


def depth_images(raw=False):
    """generate depth images"""
    while True:
        depth, _ = freenect.sync_get_depth()
        if not raw:
            depth = uint11_to_uint8(depth)
        else:
            depth = np.ma.masked_equal(depth, (2 ** 11) - 1)

        yield depth


def calibrated_height_images(values, z):
    """convert values (from kinect 11 bit) to z values in m"""
    assert values[0] > values[1], "please click deeper point first next time"

    def values2height(x, values, z):
        return (x - (values[0] + values[1])/2.0) / (values[1] - values[0])*(z[1] - z[0]) + (z[0] + z[1])/2.0
    f = functools.partial(values2height, z=z, values=values)

    for raw in depth_images(raw=True):
        height = f(raw)
        yield height


def calibrated_depth_images_old(extent=None):
    """generate depth images on a fixed grid """
    v, u = np.mgrid[:HEIGHT, :WIDTH]
    images = percentile_depth_images()
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
