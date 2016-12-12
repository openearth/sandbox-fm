import logging
import itertools

import cv2
import matplotlib.pyplot as plt
import matplotlib
import cmocean
import scipy.interpolate
import numpy as np
import skimage.draw

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
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
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
        self.ax.axis('tight')
        self.ax.axis('off')
        plt.ion()
        plt.show(block=False)
        self.L_nodes = None
        self.L_cells = None
        self.lic = None
        self.counter = itertools.count()

    def initialize(self, data):
        # create plots here
        warped_kinect = cv2.warpPerspective(data['kinect'], np.array(data['img2box']), data['kinect'].shape[::-1])

        # rgba image
        self.lic = np.ones(data['kinect'].shape + (4, ), dtype='float32')
        # transparent, white background
        self.lic[...,3] = 0.0

        self.im_kinect = self.ax.imshow(
            warped_kinect,
            cmap='Greys',
            alpha=1
        )
        # get the xlim from the kinect image
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # row, column indices
        v, u = np.mgrid[:HEIGHT, :WIDTH]

        # xy of model in image coordinates
        # xk_box, yk_box = transform(data['xk'], data['yk'], data['model2box'])
        # self.L_nodes = scipy.interpolate.NearestNDInterpolator(
        #     np.c_[xk_box, yk_box],
        #     data['zk']
        # )
        xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2box'])
        # self.L_cells = scipy.interpolate.NearestNDInterpolator(
        #     np.c_[xzw_box, yzw_box],
        #     data['bl']
        # )
        # transform vectors
        xzw_ucx_box, yzw_ucy_box = transform(
            data['xzw'] + data['ucx'],
            data['yzw'] + data['ucy'],
            data['model2box']
        )
        ucx_in_img = xzw_ucx_box - xzw_box
        ucy_in_img = yzw_ucy_box - yzw_box

        # cell_vars = np.c_[data['s1'], ucx_in_img, ucy_in_img, data['bl']]
        # self.L_cells.values = cell_vars

        # values_in_img = self.L_cells(np.c_[u.ravel(), v.ravel()])

        # s1_img = values_in_img[:, 0].reshape((HEIGHT, WIDTH))
        # ucx_img = values_in_img[:, 1].reshape((HEIGHT, WIDTH))
        # ucy_img = values_in_img[:, 2].reshape((HEIGHT, WIDTH))
        # bl_img = values_in_img[:, 3].reshape((HEIGHT, WIDTH))

        u_t, v_t = transform(u.ravel().astype('float32'), v.ravel().astype('float32'), data['box2model'])
        tree = scipy.spatial.cKDTree(np.c_[data['xzw'], data['yzw']])
        _, ravensburger_cells = tree.query(np.c_[u_t, v_t])
        data['ravensburger_cells'] = ravensburger_cells.reshape(HEIGHT, WIDTH)
        tree = scipy.spatial.cKDTree(np.c_[data['xk'], data['yk']])
        _, ravensburger_nodes = tree.query(np.c_[u_t, v_t])
        data['ravensburger_nodes'] = ravensburger_nodes.reshape(HEIGHT, WIDTH)

        s1_img = data['s1'][data['ravensburger_cells']]
        ucx_img = ucx_in_img[data['ravensburger_cells']]
        ucy_img = ucx_in_img[data['ravensburger_cells']]
        bl_img = data['bl'][data['ravensburger_cells']]



        self.im_bl = self.ax.imshow(
            np.ma.masked_less(bl_img, s1_img),
            cmap='gist_earth',
            alpha=1.0
        )

        self.im_s1 = self.ax.imshow(
            np.ma.masked_less_equal(s1_img, bl_img),
            cmap=cmocean.cm.deep,
            alpha=1.0,
            vmin=1.3,
            vmax=1.7
        )

        self.im_flow = self.ax.imshow(
            self.lic,
            alpha=0.5
        )

        self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        # self.ax.axis('off')


    def update(self, data):
        i = next(self.counter)
        self.im_kinect.set_data(data['kinect'])

        xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2box'])

        # transform vectors
        xzw_ucx_box, yzw_ucy_box = transform(
            data['xzw'] + data['ucx'],
            data['yzw'] + data['ucy'],
            data['model2box']
        )
        ucx_in_img = xzw_ucx_box - xzw_box
        ucy_in_img = yzw_ucy_box - yzw_box

        # uc1 = np.c_[data['ucx'], data['ucy'], np.ones_like(data['ucx'])]
        # zero1 = np.c_[np.zeros_like(data['ucx']), np.zeros_like(data['ucx']), np.ones_like(data['ucx'])]
        # uc_in_img = np.dot(uc1, data['model2box']) - np.dot(zero1, data['model2box'])
        # ucx_in_img, ucy_in_img = uc_in_img[:, 0], uc_in_img[:, 1]


        # cell_vars = np.c_[data['s1'], ucx_in_img, ucy_in_img, data['bl']]
        # self.L_cells.values = cell_vars

        # # row, column indices
        # v, u = np.mgrid[:HEIGHT, :WIDTH]
        # values_in_img = self.L_cells(np.c_[u.ravel(), v.ravel()])
        # s1_img = values_in_img[:, 0].reshape((HEIGHT, WIDTH))
        # ucx_img = values_in_img[:, 1].reshape((HEIGHT, WIDTH))
        # ucy_img = values_in_img[:, 2].reshape((HEIGHT, WIDTH))
        # bl_img = values_in_img[:, 3].reshape((HEIGHT, WIDTH))
        s1_img = data['s1'][data['ravensburger_cells']]
        ucx_img = ucx_in_img[data['ravensburger_cells']]
        ucy_img = ucy_in_img[data['ravensburger_cells']]
        bl_img = data['bl'][data['ravensburger_cells']]
        self.im_s1.set_data(np.ma.masked_less_equal(s1_img, bl_img))
        self.im_bl.set_data(np.ma.masked_less(bl_img, s1_img))
        print(np.ma.masked_less_equal(s1_img, bl_img).min(), np.ma.masked_less_equal(s1_img, bl_img).max())
        scale = 10.0
        flow = np.dstack([ucx_img, ucy_img]) * scale
        self.lic = warp_flow(self.lic.astype('float32'), flow.astype('float32'))
        # fade out
        self.lic[..., 3] -= 0.01
        # but not < 0
        self.lic[..., 3][self.lic[..., 3] < 0] = 0
        self.im_flow.set_data(self.lic)

        for u, v in zip(np.random.random(4), np.random.random(4)):
            self.lic[..., 3][skimage.draw.circle(v * HEIGHT, u * WIDTH, 3, shape=(HEIGHT, WIDTH))] = 1.0
        self.lic[bl_img >= s1_img, 3] = 0.0
        # # interpolate water levels
        # pts = np.ascontiguousarray(data['xy_img'][:,:2].copy())
        # # set the values to s1
        # self.L.values = np.ascontiguousarray(data['bl'][:, np.newaxis])
        # # water levels (vector)
        # bl_in_img = np.ma.masked_invalid(self.L(pts))
        # # #reshape to img coordinates
        # bl_img = bl_in_img.reshape(data['kinect'].shape)
        # # interpolate water levels
        # # set the values to s1
        # self.L.values = np.ascontiguousarray(data['s1'][:, np.newaxis])
        # # water levels (vector)
        # s1_in_img = np.ma.masked_invalid(self.L(pts))
        # # #reshape to img coordinates
        # s1_img = s1_in_img.reshape(data['kinect'].shape)

        # # show the bathymetry where it is equal or bigger as water level
        # values = np.ma.masked_less(bl_img, s1_img)
        # self.im_bl.set_data(values)
        # # show the water level where it is bigger than bathymetry
        # values = np.ma.masked_less_equal(s1_img, bl_img)
        # self.im_s1.set_data(values)
        # self.im_s1.set_clim(values.min(), values.max())

        # Something like this would be faster
        # self.im.set_data(data['kinect'])
        # self.contour.set_data(data['kinect'])
        self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
