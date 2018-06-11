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


def update_with_message(data, model, message):
    """update the data with an event from a model subscription"""
    arr, meta_msg = message
    # lookup the var name in the array
    var_name = meta_msg['name']
    # store it
    data[var_name] = arr
    # lookup the metadata for this model
    meta_model = available[model.engine]
    # compute derivitave variables
    meta_model["compute"](data)
    # store in the "known name"
    mapped_name = meta_model['reverse_mapping'][var_name]
    data[mapped_name] = data[var_name]



def compute_delta_height(data, idx):
    """compute the bed level change, normalized a bit and only for cells in idx """


    if not data['average_kinect_height']:
        kinect_height = data['kinect_height']
    else:
        buffer_quantile = 25
        kinect_height = np.percentile(data['kinect_height_buffer'], q=buffer_quantile, axis=0)

    x_nodes_box, y_nodes_box = transform(data['X_NODES'].ravel(), data['Y_NODES'].ravel(), data['model2img'])

    # nearest pixels
    u = np.clip(np.round(y_nodes_box[idx.ravel()]).astype('int'), 0, HEIGHT - 1)
    v = np.clip(np.round(x_nodes_box[idx.ravel()]).astype('int'), 0, WIDTH - 1)

    new_node_height = kinect_height[u, v].ravel()
    delta_node_height = new_node_height - data['HEIGHT_NODES'].ravel()[idx.ravel()]

    return delta_node_height


def run_update_bedlevel(data, model):
    ''' update the bed level in the model'''

    logger.info('Updating bed level')

    meta = available[model.engine]

    idx = np.logical_and(data['node_in_box'], data['node_in_img_bbox'])
    height_nodes_new = data['HEIGHT_NODES'].copy()
    height_nodes_new.ravel()[idx] += compute_delta_height(data, idx)
    # replace the part within the screen
    logger.info("updating bathymetry in  %s nodes", np.sum(idx))

    meta['update_nodes'](idx, height_nodes_new, data, model)

    # Save the kinect height of the last bed level update
    data['kinect_height_last_update'] = data['kinect_height']
