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
    """get the variables from the model and put them in the data dictionary"""
    meta = available[model.engine]
    for name in meta['vars']:
        data[name] = model.get_var(name)
    # do some stuff per model
    meta["compute"](data)
    for key, val in meta["mapping"].items():
        data[key] = data[val]

def update_with_event(data, model, event):
    """update the data with an event from a model subscription"""
    arr, meta_msg = event
    meta_model = available[model.engine]
    mapped_name = meta_model['reverse_mapping'][meta_msg['name']]
    data[mapped_name] = arr
    logger.info("updated %s", mapped_name)





def compute_delta_height(data, idx):
    """compute the bed level change, normalized a bit and only for cells in idx """

    kinect_height = data['kinect_height']

    x_nodes_box, y_nodes_box = transform(data['X_NODES'].ravel(), data['Y_NODES'].ravel(), data['model2box'])

    # nearest pixels
    u = np.clip(np.round(y_nodes_box[idx.ravel()]).astype('int'), 0, HEIGHT - 1)
    v = np.clip(np.round(x_nodes_box[idx.ravel()]).astype('int'), 0, WIDTH - 1)

    new_node_height = kinect_height[u, v].ravel()
    delta_node_height = new_node_height - data['HEIGHT_NODES'].ravel()[idx.ravel()]
    return delta_node_height
