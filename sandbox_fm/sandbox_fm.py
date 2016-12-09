# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.transforms
import logging

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
