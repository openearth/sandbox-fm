# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.transforms
import scipy.interpolate
import logging
from .calibrate import (
    HEIGHT, WIDTH,
    transform
)
import sandbox_fm.models

logger = logging.getLogger(__name__)


def update_initial_vars(data, model):
    """get the initial variables for the model"""
    # variables on t=0
    meta = getattr(sandbox_fm.models, model.engine)
    for name in meta['initial_vars']:
        data[name] = model.get_var(name)

    for name in meta['vars']:
        data[name] = model.get_var(name)
        data[name + '_0'] = model.get_var(name).copy()
    for key, val in meta["mapping"].items():
        data[key] = data[val]
    meta['compute'](data)


def update_vars(data, model):
    """update the time varying variables in a delft3d model"""
    meta = getattr(sandbox_fm.models, model.engine)
    for name in meta['vars']:
        # get data and toss away the boundary points
        arr = model.get_var(name)
    # do some stuff per model
    meta["compute"](data)
    for key, val in meta["mapping"].items():
        data[key] = data[val]


def compute_delta_height(data, idx, mode='absolute'):
    """compute the bed level change, normalized a bit and only for cells in idx"""
    height = data['height']

    x_nodes_box, y_nodes_box = transform(data['X_NODES'], data['Y_NODES'], data['model2box'])

    # pixels
    u = np.clip(np.round(y_nodes_box[idx]).astype('int'), 0, HEIGHT - 1)
    v = np.clip(np.round(x_nodes_box[idx]).astype('int'), 0, WIDTH - 1)

    node_height = height[u, v].ravel()
    delta_zk = node_height - data['HEIGHT_NODES'][idx]
    return delta_zk


def compute_delta_s1(data, idx):
    """compute the bathymetry change, normalized a bit and only for cells in idx"""

    height = data['height']

    xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2box'])

    u = np.clip(np.round(yzw_box[idx]).astype('int'), 0, HEIGHT-1)
    v = np.clip(np.round(xzw_box[idx]).astype('int'), 0, WIDTH-1)
    # define the interpolation function from height to meters
    cell_height = height[u, v].ravel()

    delta_s1 = np.zeros_like(cell_height)

    delta_s1[cell_height > 20] = (cell_height[cell_height > 20] - 20)/4000.0

    return delta_s1
