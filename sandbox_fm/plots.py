import logging
import itertools
import sys
import random
import pathlib

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.streamplot
import matplotlib
import scipy.interpolate
import numpy as np
import skimage.draw

import sandbox_fm.cm as cm

from .models import (
    available
)

from .variables import (
    run_update_bedlevel
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
    apply_hillshade,
    create_wave
)

matplotlib.rcParams['toolbar'] = 'None'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


views = {
    1: {
        "name": "Kinect",
        "layers": ["kinect_height", "lic"]
    },
    2: {
        "name": "Waterdepth",
        "layers": ["background", "waterdepth", "lic"]
    },
    3: {
        "name": "Bed level",
        "layers": ["height_cells"]
    },
    4: {
        "name": "Flow magnitude",
        "layers": ["background", "mag"]
    },
    5: {
        "name": "Waves",
        "layers": ["background", "wavesurface"]
    },
    6: {
        "name": "Erosion",
        "layers": ["erosion"]
    },
    7: {
        "name": "Bastei",
        "layers": ["kinect_height", "waterdepth", "lic"]
    },
    8: {
        "name": "Bastei2",
        "layers": ["height_depth_combined"]
    },
    9: {
        "name": "Bastei3",
        "layers": ["height_depth_combined","overlay"]
    }
}

default_config = {
    "debug": False,
    "scale": 5.0,  # Multiplication on flow velocity for particle speed
    "height_vmin": 0,  # Color scale bedlevel / kinect height
    "height_vmax": 9,  # Color scale bedlevel / kinect height
    "height_cmap": 'bastei',  # Color map of the bedlevel / kinect height
    "velocities_vmin": 0,  # Color scale velocity
    "velocities_vmax": 2,  # Color scale velocity,
    "velocities_cmap": 'speed',  # Color map of the velocity
    "depth_vmin": 0,  # Color scale waterdepth
    "depth_vmax": 3,  # Color scale waterdepth
    "depth_cmap": 'transparent_water',  # color map of the waterdepth
    'default_view': 1,  # Default view to load
    'wave_cmap': 'ice_r',
    'wave_vmin': -2,
    'wave_vmax': 1,

    'erosion_cmap': 'balance_r',
    'erosion_vmin': -1,
    'erosion_vmax': 1,

    'bedlevel_update_threshold': 0.5,  # Threshold (model meters) at which bed level an update is being done
    'bedlevel_update_maximum': 9999,  # Threshold (model meters) at which level the bed level is no longer updated (used for correcting for 'arms')
    'auto_bedlevel_update_interval': 0,  # Interval (s) at which the bed level is automatically updated
    'average_kinect_height': 1,  # If true the bed level update will use a percentile of a set of images instead of the current kinect height
    'figure_axis': [0, 0, 1, 1],  # Left, Bottom, Right, Top. Can be used for (too large) beamer projections; to limit the part of the figure which is filled with axis.
}


def process_events(evt, data, model, vis):
    """handle keystrokes and other interactions"""

    if not isinstance(evt, matplotlib.backend_bases.KeyEvent):
        return

    # we are switching views if evt.key is a number
    if evt.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        new_view_idx = int(evt.key)
        old_view = vis.current_view
        # remove handles


        for layer in old_view['layers']:
            if isinstance(vis.handles[layer], list):
                # a collection
                for item in vis.handles[layer]:
                    item.remove()
            elif isinstance(vis.handles[layer], matplotlib.streamplot.StreamplotSet):
                streamplot = vis.handles[layer]
                try:
                    vis.handles[layer].lines.rexmove()
                except ValueError:
                    logging.exception('no worries')
                for line in streamplot.lines:
                    line.remove()
                for arrow in streamplot.arrows:
                    arrow.remove()


            else:
                # an artist
                vis.handles[layer].remove()
        # remove references to old handles
        vis.handles.clear()

        new_view = views[new_view_idx]

        for layer in new_view['layers']:
            init = getattr(vis, 'init_' + layer)
            logger.info("initializing layer %s", layer)
            init(data)

        for layer in new_view['layers']:
            add = getattr(vis, 'add_' + layer)
            logger.info("adding layer %s", layer)
            add(data)

        vis.current_view = new_view
        logger.info('switching from %s to %s', old_view, new_view)


    if evt.key == 'b':  # Set bed level to current camera bed level
        run_update_bedlevel(data, model)

    if evt.key == 'h':  # mark high objects as non erodable ([H]ard structure)
        idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
        height_nodes_copy = data['HEIGHT_NODES'].copy()
        height_nodes_copy.ravel()[idx] += compute_delta_height(data, idx)
        # at least 3 meter
        idx = np.logical_and(
            idx,
            height_nodes_copy.ravel() > (
                data['HEIGHT_NODES'].ravel() +
                data.get('hard_threshold', 3.0)
            )
        )
        # replace the part that changed
        logger.info("updating structures in  %s nodes", np.sum(idx))
        meta['update_structures'](idx, height_nodes_copy, data, model)
    if evt.key == 'r':  # Reset to original bed level
        for i in range(0, len(data['height_cells_original'])):
            if data['HEIGHT_CELLS'][i] != data['height_cells_original'][i]:
                model.set_var_slice(
                    mappings["HEIGHT_CELLS"],
                    [i + 1],
                    [1],
                    data['height_cells_original'][i:i + 1]
                )
    if evt.key == 'p':
        data['lic'][:, :, :3] = 1.0
        data['lic'][:, :, 3] = 0.0
        data['lic'] = cv2.warpPerspective(
            data['kinect_image'].astype('float32') / 255.0,
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )
        if data['lic'].shape[-1] == 3:
            # add height channel
            data['lic'] = np.dstack([
                data['lic'],
                np.ones_like(data['lic'][:, :, 0])
            ])
    if evt.key == 's':
        vis.update_streamplot(data)
        vis.seed_streamplot(data)
        vis.add_streamplot(data)

    if evt.key == 'c':
        if 'lic' in vis.handles:
            vis.handles['lic'].set_visible(not vis.handles['lic'].get_visible())
    if evt.key == 'q':  # Quit (on windows)
        vis.quitting = True


class Visualization():
    def __init__(self):
        # create figure and axes
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.set_window_title('Sandbox_figure')

        # This should just work....
        self.fig.set_size_inches((3, 2.4))
        self.fig.set_dpi(100)
        self.fig.patch.set_facecolor('black')
        logger.info('dpi: %s', self.fig.get_dpi())
        logger.info('size in px %s x %s', self.fig.get_figwidth(), self.fig.get_figheight())
        logger.info('size in inches: %s', self.fig.get_size_inches())
        # force low dpi
        self.quitting = False

        self.ax.axis('off')
        plt.ion()
        plt.show(block=False)
        self.counter = itertools.count()

        # handles per view name
        self.handles = {
        }
        self.subscribers = []

    def notify(self, event):
        for subscriber in self.subscribers:
            subscriber(event)

    # Plot scanned height from the Kinect

    def init_kinect_height(self, data):
        warped_height = cv2.warpPerspective(
            data['kinect_height'].filled(0),
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )
        data['kinect_height_img'] = warped_height

    def add_kinect_height(self, data):
        # Plot scanned height
        self.handles['kinect_height'] = self.ax.imshow(
            data['kinect_height_img'],
            getattr(cm, data['height_cmap']),
            vmin=data['height_vmin'],
            vmax=data['height_vmax']
        )

    def update_kinect_height(self, data):
        #############################################
        # Update camera visualisation
        warped_height = cv2.warpPerspective(
            data['kinect_height'],
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )
        data['kinect_height_img'] = warped_height

    def blit_kinect_height(self, data):
        # Update scanned height
        self.handles['kinect_height'].set_data(data['kinect_height_img'])

    # Plot bed level in model

    def init_height_cells(self, data):
        self.update_height_cells(data)

    def add_height_cells(self, data):
        # Plot bed level

        self.handles['height_cells'] = self.ax.imshow(
            data['height_cells_img'],
            cmap=getattr(cm, data['height_cmap']),
            alpha=1,
            vmin=data['height_vmin'],
            vmax=data['height_vmax']
        )

    def update_height_cells(self, data):
        height_cells_img = data['HEIGHT_CELLS'].ravel()[data['ravensburger_cells']]
        data['height_cells_img'] = height_cells_img

    def blit_height_cells(self, data):
        self.handles['height_cells'].set_data(data['height_cells_img'])

    # Bed level at nodes (not plotted)

    def init_height_nodes(self, data):
        self.update_height_nodes()

    def update_height_nodes(self, data):
        # Convert to simple arrays
        height_nodes_img = data['HEIGHT_NODES'].ravel()[data['ravensburger_nodes']]
        data['height_nodes_img'] = height_nodes_img

    # Plot waterdepth in model

    def init_waterdepth(self, data):
        self.update_waterdepth(data)

    def add_waterdepth(self, data):
        self.handles['waterdepth'] = self.ax.imshow(
            data['waterdepth_img'],
            cmap=getattr(cm, data['depth_cmap']),
            alpha=1.0,
            vmin=data['depth_vmin'],
            vmax=data['depth_vmax']
        )

    def update_waterdepth(self, data):
        self.update_height_cells(data)
        waterlevel_img = data['WATERLEVEL'].ravel()[data['ravensburger_cells']]
        height_cells_img = data['height_cells_img']
        waterdepth = waterlevel_img - height_cells_img
        mask = waterdepth < 0.1

        if 'background_img_mask' in data:
            # should be same dimensions
            # if mask > 0, (white painted), always make dry
            mask = np.logical_or(mask, data['background_img_mask'])

        data['watermask'] = mask
        data['waterlevel_img'] = waterlevel_img
        data['waterdepth_img'] = np.ma.masked_array(waterdepth, mask=mask)

    def blit_waterdepth(self, data):
        self.handles['waterdepth'].set_data(data['waterdepth_img'])

    # Combine kinect height and waterdepth in one data plot

    def init_height_depth_combined(self, data):
        self.update_height_depth_combined(data)
        self.init_lic(data)

    def add_height_depth_combined(self, data):
        self.handles['height_depth_combined'] = self.ax.imshow(
            data['depth_height_rgb']
        )

    def seed_height_depth_combined(self, data):
        self.seed_lic(data)

    def update_height_depth_combined(self, data):
        self.update_waterdepth(data)
        self.update_kinect_height(data)

        # Or as colors
        N_water = matplotlib.colors.Normalize(data['depth_vmin'], data['depth_vmax'])
        color_water = getattr(cm, data['depth_cmap'])(N_water(data['waterdepth_img']))
        N_land = matplotlib.colors.Normalize(data['height_vmin'], data['height_vmax'])
        color_land = getattr(cm, data['height_cmap'])(N_land(data['kinect_height_img']))
        color_combined = color_water

        color_combined[data['watermask'], :] = color_land[data['watermask'], :]

        self.update_lic(data)

        for ii in range(3):
            background = color_combined[..., ii] * (1 - data['lic'][..., 3])
            foreground = data['lic'][..., ii] * data['lic'][..., 3]
            color_combined[..., ii] = background + foreground

        data['depth_height_rgb'] = color_combined

    def blit_height_depth_combined(self, data):
        # self.handles['height_depth_combined'].set_data(data['height_depth_img'])
        self.handles['height_depth_combined'].set_data(data['depth_height_rgb'])

    # Plot waves

    def init_wave_features(self, data):
        waves = []
        waves.append(create_wave(data))
        data['wave_features'] = waves

    def add_wave_features(self, data):
        self.handles['wave_features'] = []
        # add waves to plot
        for wave in data['wave_features']:
            self.handles['wave_features'].append(self.ax.add_collection(wave, autolim=False))

    def update_wave_features(self, data):

        x_cells_box, y_cells_box = data['x_cells_box'], data['y_cells_box']
        # compute wave celerities
        x_cells_wave_u_box, y_cells_wave_v_box = transform(
            data['X_CELLS'].ravel() + data['WAVE_U'].ravel(),
            data['Y_CELLS'].ravel() + data['WAVE_V'].ravel(),
            data['model2box']
        )
        wave_u_in_img = x_cells_wave_u_box - x_cells_box
        wave_v_in_img = y_cells_wave_v_box - y_cells_box
        wave_u_img = wave_u_in_img[data['ravensburger_cells']]
        wave_v_img = wave_v_in_img[data['ravensburger_cells']]
        data['wave_u_img'] = wave_u_img
        data['wave_v_img'] = wave_v_img

    def blit_wave_features(self, data):
        wave_u_img, wave_v_img = data['wave_u_img'], data['wave_v_img']
        waves_flow = np.dstack([wave_u_img, wave_v_img])
        data['wave_features'] = warp_waves(data['wave_features'], waves_flow, data)

    def seed_wave_features(self, data):
        wave = create_wave(data)
        data['wave_features'].append(wave)
        self.handles['wave_features'].append(self.ax.add_collection(wave))
        if len(data['wave_features']) > 10:
            wave = data['wave_features'].pop(0)
            # TODO: remove, not implementedc
            wave.set_visible(False)
            wave.remove()

    def init_wave_height(self, data):
        wave_height_img = data['WAVE_HEIGHT'].ravel()[data['ravensburger_cells']]
        dissipation_img = data['WAVE_DISSIPATION'].ravel()[data['ravensburger_cells']]
        data['wave_height_img'] = wave_height_img
        data['dissipation_img'] = dissipation_img

    def add_wave_height(self, data):
        self.handles['wave_height'] = self.ax.imshow(
            data['wave_height_img'],
            'jet',
            alpha=1,
            vmin=0
        )

    def update_wave_height(self, data):
        wave_height_img = data['WAVE_HEIGHT'].ravel()[data['ravensburger_cells']]
        dissipation_img = data['WAVE_DISSIPATION'].ravel()[data['ravensburger_cells']]
        data['wave_height_img'] = wave_height_img
        data['dissipation_img'] = dissipation_img

    def blit_wave_height(self, data):
        self.handles['wave_height'].set_data(data['wave_height_img'])

    def init_wavesurface(self, data):
        self.update_wavesurface(data)

    def add_wavesurface(self, data):
        self.handles['wavesurface'] = self.ax.imshow(
            data['waterlevel_gradient_img'],
            getattr(cm, data['wave_cmap']),
            vmin=data['wave_vmin'],
            vmax=data['wave_vmax']
        )

    def update_wavesurface(self, data):
        self.update_waterdepth(data)
        waterlevel_gradient = np.gradient(data['WATERLEVEL'], axis=1)
        waterlevel_gradient_img = waterlevel_gradient.ravel()[
            data['ravensburger_cells']
        ]
        data['waterlevel_gradient_img'] = np.ma.masked_array(waterlevel_gradient_img, mask=data['watermask'])

    def blit_wavesurface(self, data):
        self.handles['wavesurface'].set_data(data['waterlevel_gradient_img'])

    # Plot erosion

    def init_erosion(self, data):
        erosion_img = data['EROSION'].ravel()[data['ravensburger_cells']]
        data['erosion_img'] = erosion_img

    def add_erosion(self, data):
        self.handles['erosion'] = self.ax.imshow(
            data['erosion_img'],
            getattr(cm, data['erosion_cmap']),     # or balance
            alpha=1,
            vmin=data['erosion_vmin'],
            vmax=data['erosion_vmax']
        )

    def update_erosion(self, data):
        erosion_img = data['EROSION'].ravel()[data['ravensburger_cells']]
        data['erosion_img'] = erosion_img

    def blit_erosion(self, data):
        self.handles['erosion'].set_data(data['erosion_img'])

    # Plot background map
    def init_background(self, data):
        if data['background_name']:
            data['background_img'] = plt.imread(data['background_name'])
            # read the mask where to always show the background
            if data['background_mask_name']:
                background_img_mask = plt.imread(data['background_mask_name'])
                data['background_img_mask'] = background_img_mask > 0

        else:
            # 10 black pixels
            data['background_img'] = np.zeros((10, 10, 3))
            logger.warn('could not find background image: %s, default names background.jpg/png', data['background_name'])

    def add_background(self, data):
        self.handles['background'] = self.ax.imshow(
            data['background_img'],
            extent=[0, WIDTH, HEIGHT, 0]
        )

    def update_background(self, data):
        pass

    def blit_background(self, data):
        pass

    # Plot overlay map
    def init_overlay(self, data):
        if data['overlay_name']:
            data['overlay_img'] = plt.imread(data['overlay_name'])
        else:
            # 10x10 see through pixels
            data['overlay_img'] = np.zeros((10, 10, 4))
            logger.warn('could not find overlay image: %s', data['overlay_name'])

    def add_overlay(self, data):
        self.handles['overlay'] = self.ax.imshow(
            data['overlay_img'],
            extent=[0, WIDTH, HEIGHT, 0]
        )

    def update_overlay(self, data):
        pass

    def blit_overlay(self, data):
        pass


    # Plot flow velocities

    def init_uv(self, data):
        # xy of model in image coordinates

        self.update_uv(data)


    def update_uv(self, data):
        #############################################
        # Update model parameters
        #
        # Transform velocity
        x_cells_box, y_cells_box = data['x_cells_box'], data['y_cells_box']

        # transform vectors
        x_cells_u_box, y_cells_v_box = transform(
            data['X_CELLS'].ravel() + data['U'].ravel(),
            data['Y_CELLS'].ravel() + data['V'].ravel(),
            data['model2box']
        )
        # not sure whe don't use U
        u_in_img = x_cells_u_box - x_cells_box
        v_in_img = y_cells_v_box - y_cells_box

        u_img = u_in_img.ravel()[data['ravensburger_cells']]
        v_img = v_in_img.ravel()[data['ravensburger_cells']]

        data['u_img'] = u_img
        data['v_img'] = v_img


    def init_mag(self, data):
        self.init_uv(data)
        self.update_mag(data)

    def add_mag(self, data):
        # Plot flow magnitude
        self.handles['mag'] = self.ax.imshow(
            data['mag_img'],
            getattr(cm, data['velocities_cmap']),
            alpha=1,
            vmin=data['velocities_vmin'],
            vmax=data['velocities_vmax'],
            animated=True
        )
    def update_mag(self, data):
        self.update_waterdepth(data)
        self.update_uv(data)
        u_img = data['u_img']
        v_img = data['v_img']
        mag_img = np.sqrt(u_img**2 + v_img**2)
        data['mag_img'] = np.ma.masked_array(mag_img, mask=data['watermask'])


    def blit_mag(self, data):
        self.handles['mag'].set_data(data['mag_img'])

    # Plot liquid (photo or particles)

    def init_lic(self, data):
        lic = cv2.warpPerspective(
            np.zeros_like(data['kinect_image']).astype('float32'),
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )

        if lic.shape[-1] == 3:
            # add height channel
            lic = np.dstack([lic, np.zeros_like(lic[:, :, 0])])
        data['lic'] = lic

    def add_lic(self, data):
        # Plot particles
        self.handles['lic'] = self.ax.imshow(
            data['lic'],
            alpha=0.8,
            interpolation='none'
        )

    def update_lic(self, data):
        self.update_waterdepth(data)
        self.update_uv(data)
        #################################################
        # Compute liquid added to the model
        #
        # Multiplier on the flow velocities
        scale = data.get('scale', 1.0)
        u_img = data['u_img']
        v_img = data['v_img']
        flow = np.dstack([u_img, v_img]) * scale
        # compute new flow timestep
        data['lic'] = warp_flow(
            data['lic'].astype('float32'),
            flow.astype('float32')
        )
        # fade out
        data['lic'][..., 3] -= 0.01
        # but not < 0
        data['lic'][..., 3][data['lic'][..., 3] < 0] = 0
        data['lic'][..., 3][data['cell_mask']] = 0
        data['lic'][..., 3][data['watermask']] = 0
        # Update liquid

    def seed_lic(self, data):
        # we need waterheights
        self.update_waterdepth(data)
        self.update_height_nodes(data)

        # Put in new white dots (to be plotted next time step)
        n_dots = data.get('n_dots', 4)
        for u, v in zip(np.random.random(n_dots), np.random.random(n_dots)):
            # random light blue
            rgb = (random.random() * 0.1 + 0.8, random.random() * 0.1 + 0.8, 1.0)
            # make sure outline has the same color
            # create a little dot
            r, c = skimage.draw.circle(v * HEIGHT, u * WIDTH, 4,
                                       shape=(HEIGHT, WIDTH))
            # Don't plot on (nearly) dry cells
            if (
                    data['waterdepth_img'][int(v * HEIGHT), int(u * WIDTH)]
            ) < 0.5:
                continue
            # if zk_img[int(v * HEIGHT), int(u * WIDTH)] > 0:
            #     continue
            data['lic'][r, c, :] = tuple(rgb) + (1, )

        # Remove liquid on dry places
        # data['lic'][data['height_cells_img'] >= data['waterlevel_img'], 3] = 0.0
        # data['lic'][data['height_nodes_img'] >= data['waterlevel_img'], 3] = 0.0
        data['lic'][data['watermask'], 3] = 0

    def blit_lic(self, data):
        self.handles['lic'].set_data(data['lic'])

    # Plot streamplot

    def init_streamplot(self, data):
        self.update_uv(data)
        self.seed_streamplot(data)

    def add_streamplot(self, data):
        self.update_streamplot(data)
        x = data['streamplot_x']
        y = data['streamplot_y']
        seed_x = data['streamplot_seed_x']
        seed_y = data['streamplot_seed_y']
        start_points = np.c_[seed_x, seed_y]
        u = data['u_img']
        v = data['v_img']
        self.handles['streamplot'] = self.ax.streamplot(x, y, u, v) # , start_points=start_points)

    def update_streamplot(self, data):
        self.update_waterdepth(data)
        self.update_uv(data)


    # def blit_streamplot(self, data):
    #     # remove all arrows (and everything else)
    #     old_streamplot = self.handles['streamplot']
    #     # remove old arrows
    #     for patch in self.ax.patches:
    #         if isinstance(patch, matplotlib.patches.FancyArrowPatch):
    #             logger.info('removing %s', patch)
    #             patch.remove()
    #     self.add_streamplot(data)
    #     old_streamplot.lines.remove()
    #     del old_streamplot
    #     self.add_streamplot(data)



    def seed_streamplot(self, data):
        # number of seeds
        N = 100
        # domain
        x_0 = data['box'][0][0]
        x_1 = data['box'][1][0]
        y_0 = data['box'][0][1]
        y_1 = data['box'][2][1]

        # x coordinates for u, v
        data['streamplot_x'] = np.arange(x_0, x_1)
        data['streamplot_y'] = np.arange(y_0, y_1)
        # y coordinates for u, v

        n, m = np.mgrid[:HEIGHT, :WIDTH]
        if (
                'gestures' in data
                and len(data['gestures'])
                and data['gestures'][0]["name"] == "hand"
        ):
            # use gesture if available
            # mask out points that are not the feature and not water
            mask = np.logical_or(
                ~data['gestures'][0]['feature'],
                data['watermask']
            )
            n_filtered = n[~mask]
            m_filtered = m[~mask]
        else:
            # otherwise use watermask
            m_filtered = m[~data['watermask']]
            n_filtered = n[~data['watermask']]
        m_points = m_filtered.ravel()
        n_points = n_filtered.ravel()
        points = np.array(
            random.sample(
                list(np.c_[m_points, n_points]),
                N
            )
        )
        data['streamplot_seed_x'] = points[:, 0]
        data['streamplot_seed_y'] = points[:, 1]


    def init_grid(self, data):
        # column and row numbers
        n, m = np.mgrid[:HEIGHT, :WIDTH]
        # transformed to model coordinates
        m_t, n_t = transform(
            m.ravel().astype('float32'),
            n.ravel().astype('float32'),
            data['box2model']
        )

        # lookup  closest cells
        tree = scipy.spatial.cKDTree(np.c_[data['X_CELLS'].ravel(), data['Y_CELLS'].ravel()])
        distances_cells, ravensburger_cells = tree.query(np.c_[m_t, n_t])
        data['ravensburger_cells'] = ravensburger_cells.reshape(HEIGHT, WIDTH)
        data['distances_cells'] = distances_cells.reshape(HEIGHT, WIDTH)

        # lookup closest nodes
        tree = scipy.spatial.cKDTree(np.c_[data['X_NODES'].ravel(), data['Y_NODES'].ravel()])
        distances_nodes, ravensburger_nodes = tree.query(np.c_[m_t, n_t])
        data['ravensburger_nodes'] = ravensburger_nodes.reshape(HEIGHT, WIDTH)
        data['distances_nodes'] = distances_nodes.reshape(HEIGHT, WIDTH)

        # not sure what this does....
        data['node_mask'] = data['distances_nodes'] > 500
        data['cell_mask'] = data['distances_cells'] > 500

        # cell centers
        x_cells_box, y_cells_box = transform(
            data['X_CELLS'].ravel(),
            data['Y_CELLS'].ravel(),
            data['model2box']
        )
        data['x_cells_box'] = x_cells_box
        data['y_cells_box'] = y_cells_box



    def initialize(self, data):
        """"""
        # Adjusting the subplots
        self.fig.subplots_adjust(
            left=data['figure_axis'][0],
            bottom=data['figure_axis'][1],
            right=data['figure_axis'][2],
            top=data['figure_axis'][3]
        )
        self.init_grid(data)

        self.current_view = views[data['default_view']]

        # initialize data for all layers
        for layer in self.current_view['layers']:
            init = getattr(self, 'init_' + layer)
            logger.info("initializing layer %s", layer)
            init(data)

        for layer in self.current_view['layers']:
            add = getattr(self, 'add_' + layer)
            logger.info("adding layer %s", layer)
            add(data)

        self.ax.axis('tight')
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.notify)
        self.fig.canvas.mpl_connect('key_press_event', self.notify)
        plt.show(block=False)
        plt.ion()

    def update(self, data):
        i = next(self.counter)

        # update
        for layer in self.current_view['layers']:
            update = getattr(self, 'update_' + layer)
            update(data)

        # animate
        for layer in self.current_view['layers']:
            blit = getattr(self, 'blit_' + layer)
            blit(data)

        # reseed
        for layer in self.current_view['layers']:
            if hasattr(self, 'seed_' + layer):
                seed = getattr(self, 'seed_' + layer)
                #logger.info("seeding layer %s", layer)
                seed(data)


        #################################################
        # Draw updated canvas
        #

        # The optimal choice of draw commands depends on the backend...
        # TODO: this can be faster, this also redraws axis
        # self.fig.canvas.draw()
        # for artist in [self.im_height_cells, self.im_s1, self.im_flow]:
        #     self.ax.draw_artist(artist)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.ax.redraw_in_frame()
        # interact with window and click events

        # self.fig.canvas.draw()
        plt.draw()

        # for artist in self.handles.values():
        #     if artist is not None:
        #         self.ax.draw_artist(artist)
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
