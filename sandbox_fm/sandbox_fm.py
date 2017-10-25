# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.transforms
import scipy.interpolate
import logging
from .calibrate import (
    HEIGHT, WIDTH,
    transform
)
from .models import (
    available
)

logger = logging.getLogger(__name__)


def update_initial_vars(data, model):
    """get the initial variables for the model"""
    # variables on t=0
    meta = available[model.engine]
    for name in meta['initial_vars']:
        data[name] = model.get_var(name)

    for name in meta['vars']:
        data[name] = model.get_var(name)
        data[name + '_0'] = model.get_var(name).copy()
    meta['compute'](data)
    for key, val in meta["mapping"].items():
        data[key] = data[val]


def update_vars(data, model):
    for name in meta['vars']:
        data[name] = model.get_var(name)
    # do some stuff per model
    meta["compute"](data)
    for key, val in meta["mapping"].items():
        data[key] = data[val]




def compute_delta_height(data, idx):
    """compute the bed level change, normalized a bit and only for cells in idx"""
    kinect_height = data['kinect_height']

    x_nodes_box, y_nodes_box = transform(data['X_NODES'].ravel(), data['Y_NODES'].ravel(), data['model2box'])

    # nearest pixels
    u = np.clip(np.round(y_nodes_box[idx]).astype('int'), 0, HEIGHT - 1)
    v = np.clip(np.round(x_nodes_box[idx]).astype('int'), 0, WIDTH - 1)

    new_node_height = kinect_height[u, v].ravel()
    delta_node_height = new_node_height - data['HEIGHT_NODES'].ravel()[idx]
    return delta_node_height
