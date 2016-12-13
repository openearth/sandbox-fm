import logging
import itertools

import cv2
import matplotlib.pyplot as plt
import matplotlib
import cmocean
import scipy.interpolate
import numpy as np
import skimage.draw

from .cm import terrajet

from .calibrate import (
    transform,
    HEIGHT,
    WIDTH
)

matplotlib.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)


def warp_flow(img, flow):
    """tansform image with flow field"""
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR, borderValue=(1.0, 1.0, 1.0, 0.0))
    return res


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
        self.ax.axis('off')
        plt.ion()
        plt.show(block=False)
        self.lic = None
        self.counter = itertools.count()

    def initialize(self, data):
        # create plots here (not sure why shape is reversed)
        warped_height = cv2.warpPerspective(
            data['height'],
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )

        # rgba image
        self.lic = np.ones(
            data['height'].shape + (4, ),
            dtype='float32'
        )
        # transparent, white background
        self.lic[..., 3] = 0.0
        self.im_height = self.ax.imshow(
            warped_height,
            cmap=terrajet,
            alpha=1
        )
        # get the xlim from the height image
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # row, column indices
        v, u = np.mgrid[:HEIGHT, :WIDTH]

        # xy of model in image coordinates
        xzw_box, yzw_box = transform(
            data['xzw'],
            data['yzw'],
            data['model2box']
        )
        # transform vectors
        xzw_ucx_box, yzw_ucy_box = transform(
            data['xzw'] + data['ucx'],
            data['yzw'] + data['ucy'],
            data['model2box']
        )
        ucx_in_img = xzw_ucx_box - xzw_box
        ucy_in_img = yzw_ucy_box - yzw_box

        u_t, v_t = transform(
            u.ravel().astype('float32'),
            v.ravel().astype('float32'),
            data['box2model']
        )
        tree = scipy.spatial.cKDTree(np.c_[data['xzw'], data['yzw']])
        _, ravensburger_cells = tree.query(np.c_[u_t, v_t])
        data['ravensburger_cells'] = ravensburger_cells.reshape(HEIGHT, WIDTH)
        tree = scipy.spatial.cKDTree(np.c_[data['xk'], data['yk']])
        _, ravensburger_nodes = tree.query(np.c_[u_t, v_t])
        data['ravensburger_nodes'] = ravensburger_nodes.reshape(HEIGHT, WIDTH)

        s1_img = data['s1'][data['ravensburger_cells']]
        ucx_img = ucx_in_img[data['ravensburger_cells']]
        ucy_img = ucy_in_img[data['ravensburger_cells']]
        bl_img = data['bl'][data['ravensburger_cells']]
        zk_img = data['zk'][data['ravensburger_nodes']]


        self.im_zk = self.ax.imshow(
            # np.ma.masked_less(bl_img, s1_img),
            zk_img,
            cmap='gist_earth',
            alpha=0,
            vmin=-12,
            vmax=24

        )

        self.im_s1 = self.ax.imshow(
            np.ma.masked_less_equal(s1_img, bl_img),
            cmap=cmocean.cm.deep,
            alpha=0.0,
            vmin=1.3,
            vmax=1.7
        )

        self.im_flow = self.ax.imshow(
            self.lic,
            alpha=0.8,
            interpolation='none'
        )

        # self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        # self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        # self.ax.axis('off')
        self.fig.canvas.draw()

    def update(self, data):

        i = next(self.counter)
        warped_height = cv2.warpPerspective(
            data['height'],
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )

        self.im_height.set_data(data['height'])

        xzw_box, yzw_box = transform(
            data['xzw'],
            data['yzw'],
            data['model2box']
        )

        # transform vectors
        xzw_ucx_box, yzw_ucy_box = transform(
            data['xzw'] + data['ucx'],
            data['yzw'] + data['ucy'],
            data['model2box']
        )
        ucx_in_img = xzw_ucx_box - xzw_box
        ucy_in_img = yzw_ucy_box - yzw_box

        warped_kinect = cv2.warpPerspective(
            data['kinect'],
            np.array(data['img2box']),
            (640, 480)
        )
        warped_kinect = np.ma.masked_less(
            np.ma.masked_greater(warped_kinect, 100),
            -3
        )
        self.im_kinect.set_data(warped_kinect)
        print(warped_kinect.min(), warped_kinect.max())
        self.im_kinect.set_clim(-0.5, 1.5)
        zk_img = data['zk'][data['ravensburger_nodes']]

        s1_img = data['s1'][data['ravensburger_cells']]
        ucx_img = ucx_in_img[data['ravensburger_cells']]
        ucy_img = ucy_in_img[data['ravensburger_cells']]
        bl_img = data['bl'][data['ravensburger_cells']]
        self.im_s1.set_data(np.ma.masked_less_equal(s1_img, bl_img))
        self.im_zk.set_data(zk_img)

        scale = 50.0
        flow = np.dstack([ucx_img, ucy_img]) * scale
        self.lic = warp_flow(self.lic.astype('float32'), flow.astype('float32'))
        # fade out
        # self.lic[..., 3] -= 0.01
        # but not < 0
        self.lic[..., 3][self.lic[..., 3] < 0] = 0
        self.im_flow.set_data(self.lic)

        for u, v in zip(np.random.random(4), np.random.random(4)):
            hue = np.random.random()
            rgb = matplotlib.colors.hsv_to_rgb((hue, 0.2, 1.0))
            rgb = (1.0, 1.0, 1.0)
            # make sure outline has the same color
            # create a little dot
            r, c = skimage.draw.circle(v * HEIGHT, u * WIDTH, 4, shape=(HEIGHT, WIDTH))
            self.lic[r, c, :] = tuple(rgb) + (1, )
        self.lic[bl_img >= s1_img, 3] = 0.0

        # TODO: this can be faster, this also redraws axis
        self.fig.canvas.draw()
        # for artist in [self.im_zk, self.im_s1, self.im_flow]:
        #     self.ax.draw_artist(artist)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.ax.redraw_in_frame()
        # interact with window and click events
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
