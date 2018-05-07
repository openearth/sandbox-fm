#!/usr/bin/env python
import functools

import numpy as np
import mmi
import mmi.mmi_client

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation
from matplotlib.offsetbox import OffsetImage
import threading

WIDTH = 1000
HEIGHT = 1000

HIS_XY = (189020, 430051) # Inflow Boundary
# HIS_XY = (188252, 429208) # Bastei_1

def create_fig():
    """create a figure with axes"""
    fig, ax = plt.subplots()
    fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )
    ax.axis('off')
    ax.autoscale('off')
    img = plt.imread('bps/background.jpg')
    ax.imshow(img)

    ax2 = plt.axes((0, 0, 1, 1))
    img2 = plt.imread('bps/BPS.png')
    ax2.axis('off')

    ax2.imshow(img2)
    return fig, ax2


def init(ln):
    """create a line"""
    ln.set_data([], [])
    return ln,


def update(frame, ln, ax, data):
    """update the line"""
    # noramlize to imshow coordinates TODO: normalize image
    # range 3 - 6 m
    val = HEIGHT - (matplotlib.colors.Normalize(3, 6, clip=True)(data['s1']) * HEIGHT)
    ln.set_data([0, WIDTH], [val, val])
    ax.set_title('Current water level: {:.2f} ({})'.format(data['s1'], data['counter']))
    plt.pause(5)


def connect_model():
    """connect to a running model, continue and subscribe"""
    model = mmi.mmi_client.MMIClient('tcp://localhost:62000')
    model.remote('play')
    poller = model.subscribe()
    return model, poller


def update_data(poller, data):
    """update a shared array"""
    while True:
        for sock, n in poller.poll(1000):
            for i in range(n):
                message = mmi.recv_array(sock)
                arr, meta = message
                if meta['name'] == 's1':
                    # just pick a value
                    data['counter'] += 1
                    data['s1'] = arr[data['id']]


def XY_to_array(model, xy):
    X = model.get_var('xzw')
    Y = model.get_var('yzw')

    distsquared = ((X - xy[0])**2 + (Y - xy[1])**2)
    idmin = np.argmin(distsquared)
    return idmin


if __name__ == '__main__':
    data = {
        'counter': 0,
        's1': 0
    }
    # connect to model
    model, poller = connect_model()

    # Find find index for HIS Data
    data['id'] = XY_to_array(model, HIS_XY)

    # start the data update in the background
    threading.Thread(
        target=functools.partial(
            update_data,
            data=data,
            poller=poller
        )
    ).start()
    fig, ax = create_fig()

    # we need this part global
    ln, = ax.plot(
        [0, WIDTH],
        [HEIGHT/2, HEIGHT/2],
        'k-'
    )

    animation = matplotlib.animation.FuncAnimation(
        fig,
        functools.partial(update, data=data, ln=ln, ax=ax),
        interval=200,
        init_func=functools.partial(init, ln=ln)
    )
    plt.show()
