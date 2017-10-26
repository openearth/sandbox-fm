import logging
import itertools
import sys
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib
import cmocean.cm
import scipy.interpolate
import numpy as np
import skimage.draw

from .cm import terrajet2
from .sandbox_fm import compute_delta_height
from .models import (
    available
)


from .calibrate import (
    transform,
    HEIGHT,
    WIDTH
)

from .physics import (
    warp_flow,
    warp_particles,
    warp_waves,
    create_wave
)

matplotlib.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



views = [
    {
        "name": "Kinect",
        "layers": ["height"]
    },
    {
        "name": "Waterlevel",
        "layers": ["background", "waterlevel", "flow", "particles"]
    },
    {
        "name": "Bed level",
        "layers": ["bedlevel"]
    },
    {
        "name": "Flow magnitude",
        "layers": ["flow_mag"]
    },
    {
        "name": "Waves",
        "layers": ["wave_height", "waves"]
    },
    {
        "name": "Erosion",
        "layers": ["erosion", "flow"]
    }

]


def process_events(evt, data, model, vis):
    """handle keystrokes and other interactions"""
    meta = available[model.engine]

    if not isinstance(evt, matplotlib.backend_bases.KeyEvent):
        return

    try:
        new_view_idx = int(evt.key) - 1
        old_view = vis.current_view
        new_view = views[new_view_idx]
        vis.current_view = new_view
        logger.info('switching from %s to %s', old_view, new_view)

    except ValueError:
        pass

    if evt.key == 'b':  # Set bed level to current camera bed level
        # data['bl'][idx] += compute_delta_bl(data, idx)
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
        height_nodes_copy = data['HEIGHT_NODES'].copy()
        height_nodes_copy.ravel()[idx] += compute_delta_height(data, idx)
        # replace the part that changed
        logger.info("updating bathymetry in  %s nodes", np.sum(idx))
        meta['update_nodes'](idx, height_nodes_copy, data, model)
    if evt.key == 'h':  # mark high objects as non erodable ([H]ard structure)
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
        height_nodes_copy = data['HEIGHT_NODES'].copy()
        height_nodes_copy.ravel()[idx] += compute_delta_height(data, idx)
        # at least 3 meter
        idx = np.logical_and(idx, height_nodes_copy.ravel() > data['HEIGHT_NODES'].ravel() + data.get('hard_threshold', 3.0))
        # replace the part that changed
        logger.info("updating structures in  %s nodes", np.sum(idx))
        meta['update_structures'](idx, height_nodes_copy, data, model)
    if evt.key == 'r':  # Reset to original bed level
        for i in range(0, len(data['height_cells_original'])):
            if data['HEIGHT_CELLS'][i] != data['height_cells_original'][i]:
                model.set_var_slice(mappings["HEIGHT_CELLS"], [i + 1], [1],
                                    data['height_cells_original'][i:i + 1])
    if evt.key == 'p':
        vis.lic[:, :, :3] = 1.0
        vis.lic[:, :, 3] = 0.0
        vis.lic = cv2.warpPerspective(
            data['video'].astype('float32') / 255.0,
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )
        if vis.lic.shape[-1] == 3:
            # add height channel
            vis.lic = np.dstack([
                vis.lic,
                np.ones_like(vis.lic[:, :, 0])
            ])

    if evt.key == 'c':
        vis.im_flow.set_visible(not vis.im_flow.get_visible())
    if evt.key == 'q':  # Quit (on windows)
        sys.exit()
    if evt.key == '1':  # Visualisation preset 1. Show bed level from camera
        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(False)
        vis.im_waterlevel.set_visible(False)
        vis.im_height.set_visible(True)
        vis.im_height_cells.set_visible(False)
        vis.im_mag.set_visible(False)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(False)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(False)
    if evt.key == '2':  # Visualisation preset 2. Show water level in model
        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(True)
        vis.im_waterlevel.set_visible(True)
        vis.im_height.set_visible(False)
        vis.im_height_cells.set_visible(False)
        vis.im_mag.set_visible(False)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(False)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(False)
    if evt.key == '3':  # Visualisation preset 3. Show bed level in model
        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(False)
        vis.im_waterlevel.set_visible(False)
        vis.im_height.set_visible(False)
        vis.im_height_cells.set_visible(True)
        vis.im_mag.set_visible(False)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(False)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(False)
    if evt.key == '4':  # Visualisation preset . Show flow magnitude in model
        # views ={
        #     4: ["im_mag"],
        #     3: ["im_height_cells"]
        # }
        # for plot_name in ["im_erosion", "im_wave_height"]:
        #     plot = getattr(vis, plot_name)
        #     if not plot:
        #         continue
        #     if plot in views:
        #         plot.set_visible(True)
        #     else:
        #         plot.set_visible(False)

        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(False)
        vis.im_waterlevel.set_visible(False)
        vis.im_height.set_visible(False)
        vis.im_height_cells.set_visible(False)
        vis.im_mag.set_visible(True)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(False)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(False)
    if evt.key == '5':  # Visualisation preset . Show wave height in model
        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(False)
        vis.im_waterlevel.set_visible(False)
        vis.im_height.set_visible(False)
        vis.im_height_cells.set_visible(False)
        vis.im_mag.set_visible(False)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(True)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(False)
    if evt.key == '6':  # Visualisation preset . Show erosion in model
        if hasattr(vis, 'im_background'):
            vis.im_background.set_visible(False)
        vis.im_waterlevel.set_visible(False)
        vis.im_height.set_visible(True)
        vis.im_height_cells.set_visible(False)
        vis.im_mag.set_visible(False)
        if hasattr(vis, 'im_wave_height'):
            vis.im_wave_height.set_visible(False)
        if hasattr(vis, 'im_erosion'):
            vis.im_erosion.set_visible(True)


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
        self.waves = []
        self.background = None
        self.counter = itertools.count()
        self.current_view = None
        self.subscribers = []

    def notify(self, event):
        for subscriber in self.subscribers:
            subscriber(event)

    def initialize(self, data):
        # create plots here (not sure why shape is reversed)
        warped_height = cv2.warpPerspective(
            data['kinect_height'].filled(0),
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )

        # rgba image
        self.lic = cv2.warpPerspective(
            np.zeros_like(data['kinect_image']).astype('float32'),
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )

        if self.lic.shape[-1] == 3:
            # add height channel
            self.lic = np.dstack([self.lic, np.zeros_like(self.lic[:, :, 0])])

        have_waves = 'WAVE_U' in data
        if have_waves:
            self.waves.append(create_wave(data))


        # transparent, white background
        # self.lic[..., 3] = 0.0
        if 'background_name' in data:
            self.background = plt.imread(data['background_name'])

        # get the xlim from the height image
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # row, column indices, not to be confused with velocities
        v, u = np.mgrid[:HEIGHT, :WIDTH]

        # xy of model in image coordinates
        x_cell_box, y_cell_box = transform(
            data['X_CELLS'].ravel(),
            data['Y_CELLS'].ravel(),
            data['model2box']
        )
        # transform vectors, velocities
        x_cell_u_box, y_cell_v_box = transform(
            (data['X_CELLS'] + data['U']).ravel(),
            (data['Y_CELLS'] + data['V']).ravel(),
            data['model2box']
        )
        u_in_img = x_cell_box - x_cell_u_box
        v_in_img = y_cell_box - y_cell_v_box

        # THESE ARE NOT VELOCITIES
        u_t, v_t = transform(
            u.ravel().astype('float32'),
            v.ravel().astype('float32'),
            data['box2model']
        )

        tree = scipy.spatial.cKDTree(np.c_[data['X_CELLS'].ravel(), data['Y_CELLS'].ravel()])
        distances_cells, ravensburger_cells = tree.query(np.c_[u_t, v_t])
        data['ravensburger_cells'] = ravensburger_cells.reshape(HEIGHT, WIDTH)
        data['distances_cells'] = distances_cells.reshape(HEIGHT, WIDTH)
        tree = scipy.spatial.cKDTree(np.c_[data['X_NODES'].ravel(), data['Y_NODES'].ravel()])
        distances_nodes, ravensburger_nodes = tree.query(np.c_[u_t, v_t])
        data['ravensburger_nodes'] = ravensburger_nodes.reshape(HEIGHT, WIDTH)
        data['distances_nodes'] = distances_nodes.reshape(HEIGHT, WIDTH)

        data['node_mask'] = data['distances_nodes'] > 500
        data['cell_mask'] = data['distances_cells'] > 500

        waterlevel_img = data['WATERLEVEL'].ravel()[data['ravensburger_cells']]
        u_img = u_in_img[data['ravensburger_cells']]
        v_img = v_in_img[data['ravensburger_cells']]
        height_cells_img = data['HEIGHT_CELLS'].ravel()[data['ravensburger_cells']]
        height_nodes_img = data['HEIGHT_NODES'].ravel()[data['ravensburger_nodes']]
        mag_img = np.sqrt(u_img**2 + v_img**2)
        if have_waves:
            wave_height_img = data['WAVE_HEIGHT'].ravel()[data['ravensburger_cells']]
            dissipation_img = data['WAVE_DISSIPATION'].ravel()[data['ravensburger_cells']]
            erosion_img = data['EROSION'].ravel()[data['ravensburger_cells']]

        # plot satellite image background
        if self.background is not None:
            self.im_background = self.ax.imshow(
                self.background,
                extent=[0, 640, 480, 0]
            )

        # Plot scanned height
        self.im_height = self.ax.imshow(
            warped_height,
            'jet',
            alpha=1,
            vmin=data['z'][0],
            vmax=data['z'][-1],
            visible=False
        )


        # Plot waterheight
        # data['hh'] in xbeach
        self.im_waterlevel = self.ax.imshow(
            np.ma.masked_less_equal(waterlevel_img - height_cells_img, 0.1),
            cmap='Blues',
            alpha=1.0,
            vmin=0,
            vmax=3,
            visible=False
        )

        # Plot bed level
        self.im_height_cells = self.ax.imshow(
            height_cells_img,
            cmap=terrajet2,  # 'gist_earth',
            alpha=1,
            vmin=data['z'][0],
            vmax=data['z'][-1],
            visible=False
        )

        # Plot flow magnitude
        self.im_mag = self.ax.imshow(
            mag_img,
            'jet',
            alpha=1,
            vmin=0,
            visible=False
        )

        self.im_wave_height = self.ax.imshow(
            wave_height_img,
            'jet',
            alpha=1,
            vmin=0,
            visible=False
        )

        self.im_erosion = self.ax.imshow(
            erosion_img,
            cmocean.cm.balance_r,     # or balance
            alpha=1,
            vmin=-1,
            vmax=1,
            visible=False
        )

        # Plot particles
        self.im_flow = self.ax.imshow(
            self.lic,
            alpha=0.8,
            interpolation='none',
            visible=True
        )

        # add waves to plot
        for wave in self.waves:
            self.ax.add_collection(wave, autolim=False)

        # self.ax.set_xlim(xlim[0] + 80, xlim[1] - 80)
        # self.ax.set_ylim(ylim[0] + 80, ylim[1] - 80)
        self.ax.axis('tight')
        # self.ax.axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.notify)
        self.fig.canvas.mpl_connect('key_press_event', self.notify)


    def update(self, data):
        i = next(self.counter)

        #############################################
        # Update camera visualisation
        warped_height = cv2.warpPerspective(
            data['kinect_height'],
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )

        #############################################
        # Update model parameters
        #
        # Transform velocity
        x_cells_box, y_cells_box = transform(
            data['X_CELLS'].ravel(),
            data['Y_CELLS'].ravel(),
            data['model2box']
        )

        # transform vectors
        x_cells_u_box, y_cells_v_box = transform(
            data['X_CELLS'].ravel() + data['U'].ravel(),
            data['Y_CELLS'].ravel() + data['V'].ravel(),
            data['model2box']
        )
        # not sure whe don't use U
        u_in_img = x_cells_u_box - x_cells_box
        v_in_img = y_cells_v_box - y_cells_box

        # compute wave velocities
        have_waves = 'WAVE_V' in data

        if have_waves:
            x_cells_wave_u_box, y_cells_wave_v_box = transform(
                data['X_CELLS'].ravel() + data['WAVE_U'].ravel(),
                data['Y_CELLS'].ravel() + data['WAVE_V'].ravel(),
                data['model2box']
            )
            wave_u_in_img = x_cells_wave_u_box - x_cells_box
            wave_v_in_img = y_cells_wave_v_box - y_cells_box



        # Convert to simple arrays
        height_nodes_img = data['HEIGHT_NODES'].ravel()[data['ravensburger_nodes']]
        waterlevel_img = data['WATERLEVEL'].ravel()[data['ravensburger_cells']]
        u_img = u_in_img.ravel()[data['ravensburger_cells']]
        v_img = v_in_img.ravel()[data['ravensburger_cells']]

        height_cells_img = data['HEIGHT_CELLS'].ravel()[data['ravensburger_cells']]
        mag_img = np.sqrt(u_img**2 + v_img**2)
        if have_waves:
            wave_u_img = wave_u_in_img[data['ravensburger_cells']]
            wave_v_img = wave_v_in_img[data['ravensburger_cells']]
            wave_height_img = data['WAVE_HEIGHT'].ravel()[data['ravensburger_cells']]
            dissipation_img = data['WAVE_DISSIPATION'].ravel()[data['ravensburger_cells']]
            erosion_img = data['EROSION'].ravel()[data['ravensburger_cells']]

        # Update scanned height
        self.im_height.set_data(warped_height)
        self.im_waterlevel.set_data(np.ma.masked_less_equal(waterlevel_img - height_cells_img, 0.1))
        self.im_height_cells.set_data(height_cells_img)
        self.im_mag.set_data(mag_img)
        self.im_mag.set_clim(0, 2.5)
        if have_waves:
            self.im_wave_height.set_data(wave_height_img)
            self.im_wave_height.set_clim(0, 7)
            self.im_erosion.set_data(erosion_img)

        #################################################
        # Compute liquid added to the model
        #
        # Multiplier on the flow velocities
        scale = data.get('scale', 10.0)
        flow = np.dstack([u_img, v_img]) * scale

        if have_waves:
            waves_flow = np.dstack([wave_u_img, wave_v_img]) * scale

        # compute new flow timestep
        self.lic = warp_flow(
            self.lic.astype('float32'),
            flow.astype('float32')
        )
        # fade out
        # self.lic[..., 3] -= 0.01
        # but not < 0
        self.lic[..., 3][self.lic[..., 3] < 0] = 0
        self.lic[..., 3][data['cell_mask']] = 0

        # Update liquid
        self.im_flow.set_data(self.lic)

        # update waves

        # Put in new white dots (to be plotted next time step)
        n_dots = data.get('n_dots', 4)
        for u, v in zip(np.random.random(n_dots), np.random.random(n_dots)):
            rgb = (random.random(), random.random(), 1.0)
            # make sure outline has the same color
            # create a little dot
            r, c = skimage.draw.circle(v * HEIGHT, u * WIDTH, 4,
                                       shape=(HEIGHT, WIDTH))
            # Don't plot on (nearly) dry cells
            if (
                    waterlevel_img[int(v * HEIGHT), int(u * WIDTH)] -
                    height_nodes_img[int(v * HEIGHT), int(u * WIDTH)]
            ) < 0.5:
                continue
            # if zk_img[int(v * HEIGHT), int(u * WIDTH)] > 0:
            #     continue
            self.lic[r, c, :] = tuple(rgb) + (1, )

        # Remove liquid on dry places
        self.lic[height_cells_img >= waterlevel_img, 3] = 0.0
        self.lic[height_nodes_img >= waterlevel_img, 3] = 0.0

        # compute new waves
        if have_waves:
            if i % 20 == 0:
                wave = create_wave(data)
                self.waves.append(wave)
                self.ax.add_collection(wave)
            if len(self.waves) > 10:
                wave = self.waves.pop(0)
                # TODO: remove, not implementedc
                wave.set_visible(False)
            self.waves = warp_waves(self.waves, waves_flow, data, wave_height_img, dissipation_img)


        #################################################
        # Draw updated canvas
        #
        # TODO: this can be faster, this also redraws axis
        # self.fig.canvas.draw()
        # for artist in [self.im_height_cells, self.im_s1, self.im_flow]:
        #     self.ax.draw_artist(artist)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.ax.redraw_in_frame()
        # interact with window and click events
        # self.fig.canvas.draw()
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
