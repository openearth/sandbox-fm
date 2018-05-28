import logging
import itertools
import random
import time
import cv2
# import matplotlib.patches
# import matplotlib.streamplot
import matplotlib
import cmocean.cm
import scipy.interpolate
import numpy as np
import skimage.draw
from PIL import Image

from .cm import (
    terrajet2,
    # colombia,
    transparent_water
)

# from .sandbox_fm import compute_delta_height

from .calibrate import (
    transform,
    HEIGHT,
    WIDTH
)

from .physics import (
    warp_flow,
    # warp_particles,
    # warp_waves,
    # apply_hillshade,
    # create_wave
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


views = {
    1: {
        "name": "Kinect",
        "layers": ["kinect_height"]
    },
    2: {
        "name": "Waterdepth",
        "layers": ["waterdepth"]
    }, 3: {
        "name": "Bed level",
        "layers": ["height_cells"]
    }, 4: {
        "name": "Flow magnitude",
        "layers": ["velocities"]
    }, 5: {
        "name": "Kinect + particles",
        "layers": ["kinect_height", "lic"]
    }, 6: {
        "name": "Kinect + particles",
        "layers": ["kinect_height", "waterdepth", "lic"]
    }
}

default_config = {
    "debug": False,
    "scale": 5.0,
    "height_vmin": 0,
    "height_vmax": 9,
    "velocities_vmin": 0,
    "velocities_vmax": 2,
    "depth_vmin": 0,
    "depth_vmax": 3
}


def combine_images(image, image_add, alpha=None):
    '''
    This function combines two figures based on the alpha value of the added images

    image = [m x n x 3 (or 4)]
    image_add = [m x n x 4]
    '''
    if image is None:
        # Nothing to combine
        image = image_add[..., :3]
    elif np.ma.is_masked(image_add):
        # Combine with masked array
        image = np.where(image_add.mask, image[..., :3], image_add[..., :3])
    elif alpha is not None:
        # Combining based on fixed oppacity
        background = image[..., :3] * (1 - alpha)
        foreground = image_add[..., :3] * alpha
        image = background + foreground
    else:
        # Combining based on oppacity
        background = (image[..., :3].transpose([2, 0, 1]) * (1 - image_add[..., 3])).transpose([1, 2, 0])
        foreground = (image_add[..., :3].transpose([2, 0, 1]) * image_add[..., 3]).transpose([1, 2, 0])
        image = background + foreground

        # image1_c = (image * 255)
        # image2_c = (image_add * 255)
        # image1_t = image1_c.astype('uint8')
        # image2_t = image2_c.astype('uint8')
        # image1_I = Image.fromarray(image1_t)
        # image2_I = Image.fromarray(image2_t)
        # image3_I = Image.alpha_composite(image1_I, image2_I)
        # image = np.array(image3_I)
    return image


def process_events(evt, data, model, vis):
    ''' Not sure if this is the implementation to use. Included to not break the other scripts'''
    return None


class Visualization():
    def __init__(self):
        # All initialisation actions that do not require any data
        cv2.namedWindow('projection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('projection', 640, 480)
        # cv2.setWindowProperty("projection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.quitting = False
        self.counter = itertools.count()
        self.current_view = views[1]
        self.subscribers = []  # Not used, but included to not break old scripts

    def update_kinect_height(self, data):
        # Update camera visualisation
        warped_height = cv2.warpPerspective(
            data['kinect_height'],
            np.array(data['img2box']),
            data['kinect_height'].shape[::-1]
        )
        data['kinect_height_img'] = warped_height

    def vis_kinect_height(self, data):
        # image = terrajet2(self.N_height(data['kinect_height_img']))
        image = matplotlib.cm.gnuplot(self.N_height(data['kinect_height_img']))
        return image

    def update_height_cells(self, data):
        height_cells_img = data['HEIGHT_CELLS'].ravel()[
            data['ravensburger_cells']]
        data['height_cells_img'] = height_cells_img

    def vis_height_cells(self, data):
        image = terrajet2(self.N_height(data['height_cells_img']))
        return image

    def update_waterdepth(self, data):
        self.update_height_cells(data)
        waterlevel_img = data['WATERLEVEL'].ravel()[data['ravensburger_cells']]
        height_cells_img = data['height_cells_img']
        waterdepth = waterlevel_img - height_cells_img
        mask = waterdepth < 0.1
        data['watermask'] = mask
        data['waterlevel_img'] = waterlevel_img
        data['waterdepth_img'] = np.ma.masked_array(waterdepth, mask=mask)

    def vis_waterdepth(self, data):
        image = transparent_water(self.N_waterdepth(data['waterdepth_img']))
        return image

    def update_velocities(self, data):
        self.update_waterdepth(data)
        self.update_uv(data)
        u_img = data['u_img']
        v_img = data['v_img']
        mag_img = np.sqrt(u_img**2 + v_img**2)
        data['mag_img'] = np.ma.masked_array(mag_img, mask=data['watermask'])

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
        # not sure why don't use U
        u_in_img = x_cells_u_box - x_cells_box
        v_in_img = y_cells_v_box - y_cells_box

        u_img = u_in_img.ravel()[data['ravensburger_cells']]
        v_img = v_in_img.ravel()[data['ravensburger_cells']]

        data['u_img'] = u_img
        data['v_img'] = v_img

    def vis_velocities(self, data):
        image = matplotlib.cm.inferno(self.N_velocities(data['mag_img']))
        return image

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

    def update_lic(self, data):
        # self.update_waterdepth(data)
        self.update_uv(data)
        #################################################
        # Compute liquid added to the model
        #
        # Multiplier on the flow velocities
        u_img = data['u_img']
        v_img = data['v_img']
        flow = np.dstack([u_img, v_img]) * data['scale']
        # compute new flow timestep
        data['lic'] = warp_flow(
            data['lic'].astype('float32'),
            flow.astype('float32')
        )
        # fade out
        # self.lic[..., 3] -= 0.01
        # but not < 0
        # data['lic'][..., 3][data['lic'][..., 3] < 0] = 0
        # data['lic'][..., 3][data['cell_mask']] = 0
        # data['lic'][..., 3][data['watermask']] = 0
        # Update liquid

        self.seed_lic(data)

    def seed_lic(self, data):
        # we need waterheights
        # self.update_waterdepth(data)
        # self.update_height_nodes(data)

        # Put in new white dots (to be plotted next time step)
        n_dots = data.get('n_dots', 4)
        for u, v in zip(np.random.random(n_dots), np.random.random(n_dots)):
            # random light blue
            rgb = (random.random() * 0.1 + 0.8,
                   random.random() * 0.1 + 0.8, 1.0)
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
        data['lic'][data['watermask'], 3] = 0

    def vis_lic(self, data):
        image = data['lic']
        return image

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
        tree = scipy.spatial.cKDTree(
            np.c_[data['X_CELLS'].ravel(), data['Y_CELLS'].ravel()])
        distances_cells, ravensburger_cells = tree.query(np.c_[m_t, n_t])
        data['ravensburger_cells'] = ravensburger_cells.reshape(HEIGHT, WIDTH)
        data['distances_cells'] = distances_cells.reshape(HEIGHT, WIDTH)
        # lookup closest nodes
        tree = scipy.spatial.cKDTree(
            np.c_[data['X_NODES'].ravel(), data['Y_NODES'].ravel()])
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
        # All initialisation actions that do require data
        self.init_grid(data)

        # Initialise all applied normalisations
        self.N_height = matplotlib.colors.Normalize(
            data['height_vmin'], data['height_vmax'])

        self.N_waterdepth = matplotlib.colors.Normalize(
            data['depth_vmin'], data['depth_vmax'])

        self.N_velocities = matplotlib.colors.Normalize(
            data['velocities_vmin'], data['velocities_vmax'])

        self.init_lic(data)

    def update(self, data):
        # Function that creates a plot with updated data
        next(self.counter)

        # update data layers in current view
        for layer in self.current_view['layers']:
            update = getattr(self, 'update_' + layer)
            update(data)

        # plot data layers in current view
        image = None
        for layer in self.current_view['layers']:
            vis = getattr(self, 'vis_' + layer)
            image_add = vis(data)
            image = combine_images(image, image_add)
            # image = image_add

        image_bgr = image[..., [2, 1, 0]]
        cv2.imshow('projection', image_bgr)

        k = cv2.waitKey(1) & 0xFF
        if k and k < 255:
            self.process_events(data, k)

    def process_events(self, data, k):
        view_id = {ord(str(v)): v for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        if k in view_id:
            view_idx = view_id[k]
            if view_idx in views:
                self.current_view = views[view_idx]
                logger.info('switching to view [%s]: %s', view_idx, self.current_view)
            else:
                logger.info('View does not exist: [%s]', view_idx)
        elif k == ord('b'):
            logger.info('Pressed [b]. Updating bed level.')
        elif k == ord('r'):
            logger.info('Pressed [r]. Resetting bed level.')
        elif k == ord('q') or k == 27:  # 27=escape
            logger.info('Pressed [q]. Quitting.')
            self.quitting = True
        else:
            logger.info('Found keypress %s, but not recognised', k)
