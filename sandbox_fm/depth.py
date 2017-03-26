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
        arr = np.load(str(depth), encoding='latin1')
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


def depth_images():
    """generate depth images"""
    while True:
        depth, _ = freenect.sync_get_depth()
        depth = np.ma.masked_equal(depth, (2 ** 11) - 1)

        # return as double because we might compute something with it
        yield depth.astype('double')


def calibrated_height_images(depth_max_min, z_min_max, anomaly_name='anomaly.npy'):
    """convert depth values (from kinect 11 bit) to z values in m.
    The depth are expected in max and min and the corresponding z (height) are expected in min and max"""
    if depth_max_min[0] < depth_max_min[1]:
        raise ValueError("Calibration Error. Please click deeper point first next time. Got depths {} - {}".format(depth_max_min[0], depth_max_min[1]))
    if z_min_max[0] > z_min_max[1]:
        raise ValueError("Calibration error. Expected lower higher point. Got height values {} - {}.".format(z_min_max[0], z_min_max[1]))
    anomaly = 0.0
    try:
        anomaly = np.load(str(anomaly_name))
    except:
        logger.exception('Cannot read anomaly file %s', anomaly_name)


    def values2height(x, depth_max_min, z_min_max):
        """convert the values in """
        depth_max, depth_min = depth_max_min
        depth_range = depth_max - depth_min
        z_min, z_max = z_min_max
        z_range = z_max - z_min
        depth2scaled_height = (x - depth_min)/depth_range
        height = depth2scaled_height * z_range + z_min
        return height
    f = functools.partial(values2height, z_min_max=z_min_max, depth_max_min=depth_max_min)

    for raw in depth_images():
        # correct for anomaly
        height = f(raw - anomaly)
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
