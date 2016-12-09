import logging

import matplotlib.pyplot as plt
import matplotlib
import cmocean
import scipy.interpolate
import numpy as np

matplotlib.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)

class Visualization():
    def __init__(self):
        # create figure and axes
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )
        self.ax.axis('tight')
        self.ax.axis('off')
        plt.ion()
        plt.show(block=False)

        self.L = None

    def initialize(self, data):
        # create plots here
        self.im_kinect = self.ax.imshow(
            data['kinect'],
            cmap='Greys',
            vmin=0.0,
            vmax=1.5,
            alpha=0.5
        )
        # get the xlim from the kinect image
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # xy of model in image coordinates
        points = data['xy1_model_in_img'][:,:2]
        logger.info("bl_0.shape: %s, points.shape: %s", data['bl_0'].shape, points.shape)

        self.L = scipy.interpolate.NearestNDInterpolator(points, data['bl_0'])

        # interpolate the bl_0 on image points
        pts = np.ascontiguousarray(data['xy1_img'][:,:2].copy())

        bl_in_img = self.L(pts)
        bl_img = bl_in_img.reshape(data['kinect'].shape)

        # set the values to s1_0 (same grid)
        self.L.values = np.ascontiguousarray(data['s1_0'][:, np.newaxis])
        # water levels (vector)
        s1_in_img = self.L(pts)
        # #reshape to img coordinates
        s1_img = s1_in_img.reshape(data['kinect'].shape)

        self.im_bl = self.ax.imshow(
            np.ma.masked_less(bl_img, s1_img),
            cmap='gist_earth',
            alpha=0.2
        )

        self.im_s1 = self.ax.imshow(
            np.ma.masked_less_equal(s1_img, bl_img),
            cmap=cmocean.cm.balance,
            alpha=0.5,
            vmin=-0.25,
            vmax=-0.2
        )

        self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        self.ax.axis('off')


    def update(self, data):
        self.im_kinect.set_data(data['kinect'])



        # interpolate water levels
        pts = np.ascontiguousarray(data['xy1_img'][:,:2].copy())
        # set the values to s1
        self.L.values = np.ascontiguousarray(data['bl'][:, np.newaxis])
        # water levels (vector)
        bl_in_img = np.ma.masked_invalid(self.L(pts))
        # #reshape to img coordinates
        bl_img = bl_in_img.reshape(data['kinect'].shape)
        # interpolate water levels
        # set the values to s1
        self.L.values = np.ascontiguousarray(data['s1'][:, np.newaxis])
        # water levels (vector)
        s1_in_img = np.ma.masked_invalid(self.L(pts))
        # #reshape to img coordinates
        s1_img = s1_in_img.reshape(data['kinect'].shape)

        # show the bathymetry where it is equal or bigger as water level
        values = np.ma.masked_less(bl_img, s1_img)
        self.im_bl.set_data(values)
        # show the water level where it is bigger than bathymetry
        values = np.ma.masked_less_equal(s1_img, bl_img)
        self.im_s1.set_data(values)
        self.im_s1.set_clim(values.min(), values.max())

        # Something like this would be faster
        # self.im.set_data(data['kinect'])
        # self.contour.set_data(data['kinect'])
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
