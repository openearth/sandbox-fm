import logging

import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate

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

    def initialize(self, data):
        # create plots here
        self.im_kinect = self.ax.imshow(
            data['kinect'],
            cmap='gist_earth',
            vmin=0.0,
            vmax=1.5
        )
        # get the xlim from the kinect image
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # xy of model in image coordinates
        points = data['xy1_model_in_img'][:,:2]
        logger.info("bl0.shape: %s, points.shape: %s", data['bl0'].shape, points.shape)

        L = scipy.interpolate.NearestNDInterpolator(points, data['bl0'])

        # interpolate the bl0 on image points
        bl0_in_img = L(data['xy1_img'][:,:2])
        bl0_img = bl0_in_img.reshape(data['kinect'].shape)

        self.im_bl0 = self.ax.imshow(
            bl0_img,
            cmap='jet'
        )

        self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        self.ax.axis('off')


    def update(self, data):
        self.im_kinect.set_data(data['kinect'])

        # Something like this would be faster
        # self.im.set_data(data['kinect'])
        # self.contour.set_data(data['kinect'])
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
