import os
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
logger.setLevel(logging.DEBUG)


class MockupFreenect(object):
    """mockup freenect in case you have no connection"""
    def __init__(self):
        data_dir = pathlib.Path(__file__).parent.parent / 'data'
        # Allow to use environment to use record images
        if 'FREENECT_RECORD_DIR' in os.environ:
            data_dir = pathlib.Path(os.environ['FREENECT_RECORD_DIR'])

        # TODO: pkgutil.get_data('data', '*')
        video_files = list(data_dir.glob('video_*.png'))
        depth_files = list(data_dir.glob('raw_*.npy'))

        if not depth_files:
            raise ValueError('No depth files found in %s' % (data_dir, ))

        videos = ([
            plt.imread(str(video_file))
            for video_file
            in video_files
        ])
        depths = ([
            np.load(str(depth_file), encoding='latin1')
            for depth_file
            in depth_files
        ])
        self.videos = itertools.cycle(videos)
        self.depths = itertools.cycle(depths)

    def sync_get_video(self):
        """keep yielding videos"""
        video = next(self.videos)
        return video, 3

    def sync_get_depth(self):
        """keep yielding depths"""
        depth = next(self.depths)
        return depth, 3


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


def percentile_depth_images(buffer_size=50, q=25):
    """"compute running percentile images"""
    buffer = collections.deque(maxlen=buffer_size)
    for img in depth_images():
        logging.info('Adding depth mask to buffer')
        buffer.append(img)
        perc = np.percentile(buffer, q=q, axis=0)
        perc = np.ma.masked_equal(perc, (2 ** 11) - 1)
        yield perc

def exponential_average_depth_images(xi = 0.99):
    """"compute exponential average images"""
    for k, img in enumerate(depth_images()):
        if k == 0:
            prev_img = img
        img_exp_ave = prev_img*xi + (1-xi)*img
        img_exp_ave = np.ma.masked_equal(img_exp_ave, (2 ** 11) - 1)
        yield img_exp_ave

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

def depth_images_choice():
    kinect_depth_mode = 0  # 0 = get measured depths
                           # 1 = get percentile of measured depths
                           # 2 = use exponential averaged measured depth
    """generate a chosen depth image type"""
    if kinect_depth_mode == 1:
        logger.info("Using averaged kinect depth")
        for raw in percentile_depth_images():
            yield raw
    elif kinect_depth_mode == 2:
        logger.info("Using exponential averaged kinect depth")
        for raw in exponential_average_depth_images():
            yield raw
    else:
        logger.info("Using unfiltered kinect depth")
        for raw in depth_images():
            yield raw

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
        # scaled depth (1 = deep, 0=undeep)
        scaled_depth = (x - depth_min)/depth_range
        # scaled height
        scaled_height = (1.0 - scaled_depth)
        height = scaled_height * z_range + z_min
        return height
    f = functools.partial(values2height, z_min_max=z_min_max, depth_max_min=depth_max_min)

    for raw in depth_images_choice():
        # correct for anomaly
        height = f(raw - anomaly)
        yield height
