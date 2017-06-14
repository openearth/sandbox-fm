# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.transforms
import scipy.interpolate
import logging
from .calibrate import (
    HEIGHT, WIDTH,
    transform
)

logger = logging.getLogger(__name__)


def update_delft3d_initial_vars(data, model):
    """get the initial variables for the model"""
    # variables on t=0
    for name in [
            'xzw',
            'yzw',
            'xk',
            'yk',
            'zk',
            'ndx',
            'ndxi',             # number of internal points (no boundaries)
            'numk',
            'flowelemnode'
    ]:
        try:
            data[name] = model.get_var(name)
        except AttributeError:
            if name == 'numk':
                numk = data['zk'].shape[0]
                data['numk'] = numk
            else:
                logger.exception("variable %s not found", name)

    for name in ['bl', 's1']:
        data[name + '_0'] = model.get_var(name)[:data['ndxi']].copy()


def update_delft3d_vars(data, model):
    """update the time varying variables in a delft3d model"""
    for name in ['bl', 'ucx', 'ucy', 's1', 'zk']:
        # get data and toss away the boundary points
        arr = model.get_var(name)
        # corner data
        if arr.shape[0] == data['numk']:
            data[name] = arr[:data['numk']]
        elif arr.shape[0] == data['ndx']:
            "should be of shape ndx"
            # ndxi:ndx are the boundary points (See  netcdf write code in unstruc)
            data[name] = arr[:data['ndxi']]
            # data should be off consistent shape now
        else:
            raise ValueError("unexpected data shape %s for variable %s" % (arr.shape, name))
    data['is_wet'] = data['s1'] > data['bl']


def compute_delta_zk(data, idx, mode='absolute'):
    """compute the bed level change, normalized a bit and only for cells in idx"""
    if mode=='absolute':
        height = data['height']

        xk_box, yk_box = transform(data['xk'], data['yk'], data['model2img'])

        u = np.clip(np.round(yk_box[idx]).astype('int'), 0, HEIGHT - 1)
        v = np.clip(np.round(xk_box[idx]).astype('int'), 0, WIDTH - 1)
        # define the interpolation function from depth to meters
        # depth2meters = scipy.interpolate.interp1d([0, 127, 255], [-8, 0, 12])

        # cell_depth = depth2meters(depth[u, v].ravel())
        node_height = height[u, v].ravel()
        delta_zk = node_height - data['zk'][idx]
        return delta_zk
    elif mode=='relative':
        height = np.array(data['height']) - np.array(data['height_original'])

        xk_box, yk_box = transform(data['xk'], data['yk'], data['model2img'])

        u = np.clip(np.round(yk_box[idx]).astype('int'), 0, HEIGHT - 1)
        v = np.clip(np.round(xk_box[idx]).astype('int'), 0, WIDTH - 1)
        # define the interpolation function from depth to meters
        # depth2meters = scipy.interpolate.interp1d([0, 127, 255], [-8, 0, 12])

        # cell_depth = depth2meters(depth[u, v].ravel())
        node_height = height[u, v].ravel()
        delta_zk = node_height
        return delta_zk


def compute_delta_s1(data, idx):
    """compute the bathymetry change, normalized a bit and only for cells in idx"""

    height = data['height']

    xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2img'])

    u = np.clip(np.round(yzw_box[idx]).astype('int'), 0, HEIGHT-1)
    v = np.clip(np.round(xzw_box[idx]).astype('int'), 0, WIDTH-1)
    # define the interpolation function from depth to meters
    cell_height = height[u, v].ravel()

    delta_s1 = np.zeros_like(cell_height)

    delta_s1[cell_height > 20] = (cell_height[cell_height > 20] - 20)/4000.0

    return delta_s1
