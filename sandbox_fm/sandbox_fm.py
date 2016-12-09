# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.transforms


def update_delft3d_initial_vars(data, model):
    """get the initial variables for the model"""
    # variables on t=0
    for name in [
            'xzw',
            'yzw',
            'xk',
            'yk',
            'ndx',
            'ndxi',             # number of internal points (no boundaries)
            'numk',
            'flowelemnode'
    ]:
        data[name] = model.get_var(name)
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



def compute_affines(rect, shape):
    """compute the affine transformation from a rectangle back to origin (img)"""
    angle = np.arctan2(rect[1, 1] - rect[0, 1], rect[1, 0] - rect[0, 0])
    translate = rect[0, 0], rect[0, 1]
    # x -> n columns, pythagoras
    scale_x = np.sqrt(
        (rect[1, 0] - rect[0, 0])**2 + (rect[1, 1] - rect[0, 1])**2
    ) / shape[1]
    # y -> n rows
    scale_y = np.sqrt(
        (rect[2, 0] - rect[1, 0])**2 + (rect[2, 1] - rect[1, 1])**2
    ) / shape[0]
    scale = scale_x, scale_y

    img2model = matplotlib.transforms.Affine2D()
    img2model.scale(scale[0], scale[1])
    img2model.rotate(angle)
    img2model.translate(translate[0], translate[1])

    model2img = img2model.inverted()

    return img2model.get_matrix(), model2img.get_matrix()
