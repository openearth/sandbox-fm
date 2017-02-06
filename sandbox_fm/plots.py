import logging
import itertools

import cv2
import matplotlib.pyplot as plt
import matplotlib
# import cmocean
import scipy.interpolate
import numpy as np
import skimage.draw
import sys
import time

from .cm import terrajet2
from .sandbox_fm import compute_delta_zk

from .calibrate import (
    transform,
    HEIGHT,
    WIDTH
)

matplotlib.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)


def warp_flow(img, flow):
    """transform image with flow field"""
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR,
                    borderValue=(1.0, 1.0, 1.0, 0.0))
    return res


def process_events(evt, data, model, vis):
    if not isinstance(evt, matplotlib.backend_bases.KeyEvent):
        return
    if evt.key == 'b':  # Set bed level to current camera bed level
        # data['bl'][idx] += compute_delta_bl(data, idx)
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
        zk_copy = data['zk'].copy()
        zk_copy[idx] += compute_delta_zk(data, idx)
        # replace the part that changed
        print(np.where(idx))
        for i in np.where(idx)[0]:
            if data['zk'][i] != zk_copy[i]:
                # TODO: bug in zk
                model.set_var_slice('zk', [i + 1], [1], zk_copy[i:i + 1])
    if evt.key == 'r':  # Reset to original bed level
        for i in range(0, len(data['zk_original'])):
            if data['zk'][i] != data['zk_original'][i]:
                model.set_var_slice('zk', [i + 1], [1],
                                    data['zk_original'][i:i + 1])
    if evt.key == 'p':
        vis.lic[:, :, :3] = 1.0
        vis.lic[:, :, 3] = 0.0
        vis.lic = cv2.warpPerspective(
            data['video'].astype('float32') / 255.0,
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )
        if vis.lic.shape[-1] == 3:
            # add depth channel
            vis.lic = np.dstack([
                vis.lic,
                np.ones_like(vis.lic[:, :, 0])
            ])

    if evt.key == 'c':
        vis.im_flow.set_visible(not vis.im_flow.get_visible())
    if evt.key == 'q':  # Quit (on windows)
        sys.exit()
    if evt.key == '1':  # Visualisation preset 1. Show bed level from camera
        vis.im_s1.set_visible(False)
        vis.im_height.set_visible(True)
        vis.im_zk.set_visible(False)
    if evt.key == '2':  # Visualisation preset 2. Show water level in model
        vis.im_s1.set_visible(True)
        vis.im_height.set_visible(False)
        vis.im_zk.set_visible(False)
    if evt.key == '3':  # Visualisation preset 3. Show bed level in model
        vis.im_s1.set_visible(False)
        vis.im_height.set_visible(False)
        vis.im_zk.set_visible(True)


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
        self.subscribers = []

    def notify(self, event):
        for subscriber in self.subscribers:
            subscriber(event)

    def initialize(self, data):
        # create plots here (not sure why shape is reversed)
        warped_height = cv2.warpPerspective(
            data['height'].filled(0),
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )

        # rgba image
        self.lic = cv2.warpPerspective(
            data['video'].astype('float32'),
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )
        if self.lic.shape[-1] == 3:
            # add depth channel
            self.lic = np.dstack([self.lic, np.zeros_like(self.lic[:, :, 0])])
        # transparent, white background
        # self.lic[..., 3] = 0.0

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

        # Plot scanned height
        self.im_height = self.ax.imshow(
            warped_height,
            'jet',
            #cmap=terrajet2,
 #           cmap=summer,
            alpha=1,
            vmin=data['z'][0],
            vmax=data['z'][-1],
            visible=False
        )

        # Plot waterdepth
        self.im_s1 = self.ax.imshow(
            (s1_img - bl_img),
            cmap='Blues',
            alpha=1,
            vmin=0,
            vmax=1,
            visible=True
        )
        # self.fig.colorbar(self.im_s1)
        # self.im_s1.colorbar(self.im_s1, inline=1, fontsize=10)

        # Plot bed level
        self.im_zk = self.ax.imshow(
            bl_img,
            cmap=terrajet2,  # 'gist_earth',
            alpha=1,
            vmin=data['z'][0],
            vmax=data['z'][-1],
            visible=False
        )

        # Plot contours of scanned height
        # self.ct_height = self.ax.contour(warped_height,
        #     colors='k')

        # Plot contours of height in model
        # self.ct_zk = self.ax.contour(
        #     zk_img,
        #     cmap=terrajet2,
        #     alpha=1,
        #     vmin=data['z'][0],
        #     vmax=data['z'][-1]
        # )

        if data.get('debug'):
            self.ct_zk = self.ax.contour(zk_img, colors='k')
            self.ax.clabel(self.ct_zk, inline=1, fontsize=10)

        # Plot particles
        self.im_flow = self.ax.imshow(
            self.lic,
            alpha=0.8,
            interpolation='none',
            visible=True
        )

        # self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        # self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        # self.ax.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.notify)
        self.fig.canvas.mpl_connect('key_press_event', self.notify)

    #@profile
    def update(self, data):
        i = next(self.counter)

        #############################################
        # Update camera visualisation
        warped_height = cv2.warpPerspective(
            data['height'],
            np.array(data['img2box']),
            data['height'].shape[::-1]
        )

        # Update scanned height
        self.im_height.set_data(warped_height)

        #############################################
        # Update model parameters
        #
        # Transform velocity
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

        # Convert to simple arrays
        zk_img = data['zk'][data['ravensburger_nodes']]
        s1_img = data['s1'][data['ravensburger_cells']]
        ucx_img = ucx_in_img[data['ravensburger_cells']]
        ucy_img = ucy_in_img[data['ravensburger_cells']]
        bl_img = data['bl'][data['ravensburger_cells']]

        # Update raster plots
        self.im_s1.set_data(s1_img - bl_img)
        self.im_zk.set_data(bl_img)

        # # Update contour plots
        # # When particles get disabled, update contour plot
        # if not self.im_flow.get_visible():
        #     try:
        #         for c in self.ct_height.collections:
        #             c.remove()
        #     except:
        #         pass
        #     self.ct_height = self.ax.contour(
        #         warped_height,
        #         levels=np.linspace(-7, 10, num=10)
        #     )
        #     self.ct_zk = self.ax.contour(
        #         warped_height,
        #         levels=np.linspace(-7, 10, num=10)
        #     )

        if data.get('debug'):
            for c in self.ct_zk.collections:
                c.remove()
            self.im_zk.set_data(zk_img)
            self.im_s1.set_data(np.ma.masked_less_equal(s1_img, bl_img))

        #################################################
        # Compute liquid added to the model
        #
        # Multiplier on the flow velocities
        scale = data.get('scale', 10.0)
        flow = np.dstack([ucx_img, ucy_img]) * scale

        # compute new flow timestep
        self.lic = warp_flow(self.lic.astype('float32'),
                             flow.astype('float32'))
        # fade out
        # self.lic[..., 3] -= 0.01
        # but not < 0
        self.lic[..., 3][self.lic[..., 3] < 0] = 0

        # Update liquid
        self.im_flow.set_data(self.lic)

        # Put in new white dots (to be plotted next time step)
        for u, v in zip(np.random.random(4), np.random.random(4)):
            rgb = (1.0, 1.0, 1.0)
            # make sure outline has the same color
            # create a little dot
            r, c = skimage.draw.circle(v * HEIGHT, u * WIDTH, 4,
                                       shape=(HEIGHT, WIDTH))
            # Don't plot on (nearly) dry cells
            if (s1_img[int(v * HEIGHT), int(u * WIDTH)] - zk_img[int(v * HEIGHT), int(u * WIDTH)]) < 0.5:
                continue
            # if zk_img[int(v * HEIGHT), int(u * WIDTH)] > 0:
            #     continue
            self.lic[r, c, :] = tuple(rgb) + (1, )

        # Remove liquid on dry places
        self.lic[bl_img >= s1_img, 3] = 0.0
        self.lic[zk_img >= s1_img, 3] = 0.0
        self.lic[
            np.logical_and.reduce([
                (s1_img - bl_img < 0.05), # almost dry
                (s1_img - zk_img < 0.05), # almost dry
                self.lic[..., 3] > 0.01
            ]),
            3
        ] -= 0.01


        #################################################
        # Draw updated canvas
        #
        # TODO: this can be faster, this also redraws axis
        # self.fig.canvas.draw()
        # for artist in [self.im_zk, self.im_s1, self.im_flow]:
        #     self.ax.draw_artist(artist)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.ax.redraw_in_frame()
        # interact with window and click events
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            self.ax.redraw_in_frame()
