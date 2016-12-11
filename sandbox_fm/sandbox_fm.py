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
            raise ValueError("unexpected data shape %s for variable %s" % (arr.shape, name) )
    data['is_wet'] = data['s1'] > data['bl']


def compute_delta_bl(data, idx):
    """compute the bathymetry change, normalized a bit and only for cells in idx"""

    depth = data['kinect']
    xzw_box, yzw_box = transform(data['xzw'], data['yzw'], data['model2box'])

    u = np.clip(np.round(yzw_box[idx]).astype('int'), 0, HEIGHT-1)
    v = np.clip(np.round(xzw_box[idx]).astype('int'), 0, WIDTH-1)
    # define the interpolation function from depth to meters
    depth2meters = scipy.interpolate.interp1d([0, 127, 255], [-8, 0, 12])
    cell_depth = depth2meters(depth[u, v].ravel())

    delta_bl = cell_depth - data['bl'][idx]
    max_delta_bl = np.max(np.abs(delta_bl))
    normalized_delta_bl = delta_bl/max_delta_bl
    # limit the bathymetry change per timestep to 0.1
    normalized_delta_bl *= min(max_delta_bl, 0.3)
    return normalized_delta_bl
